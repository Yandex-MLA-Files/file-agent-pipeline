import logging
import os
import re
import uuid
import warnings
from pathlib import Path

from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter

from file_agent.document import Block, BlockType, Document
from file_agent.parsers.base import BaseParser
from file_agent.parsers.common import infer_heading_level, list_to_markdown, strip_bullet
from file_agent.parsers.formula_enrichment import ENRICHMENT_PENDING
from file_agent.telemetry import tracer

logger = logging.getLogger(__name__)


def _silence_ocr_backend_noise() -> None:
    """Hide harmless third-party warnings raised by CPU-only OCR backends.

    EasyOCR loads a quantized recognition model and a DataLoader configured for
    GPUs, so on a CPU-only machine PyTorch emits a deprecation notice about
    ``torch.quantize_per_tensor`` and a ``pin_memory`` notice on every run.
    Neither affects OCR output; silencing them keeps real warnings visible.
    """
    warnings.filterwarnings("ignore", message=".*quantize_per_tensor.*", category=UserWarning)
    warnings.filterwarnings("ignore", message=".*pin_memory.*", category=UserWarning)


_silence_ocr_backend_noise()

# Formula and code enrichment re-reads the regions the layout model classified
# as such with a dedicated model (CodeFormula, ~400 MB, downloaded once). It is
# a vision model running per region, which is affordable on a GPU and is not on
# a CPU: an 85-page lecture spends minutes there. ``auto`` (the default)
# therefore enables it only when a CUDA device is visible; ``on`` / ``off``
# force it either way.
DEFAULT_ENRICHMENT = "auto"


def resolve_enrichment() -> bool:
    mode = (os.getenv("PDF_ENRICHMENT") or DEFAULT_ENRICHMENT).strip().lower()
    if mode in {"1", "true", "yes", "on"}:
        return True
    if mode in {"0", "false", "no", "off"}:
        return False
    return gpu_available()


def gpu_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:  # pragma: no cover - torch always present in practice
        return False


# Docling hands the enrichment model five regions per forward pass. Sixteen is
# ~29 % faster on a formula-dense lecture (503 s → 358 s of model time) but
# needs proportionally more GPU memory, and Docling *swallows* an out-of-memory
# error inside the stage: at 32 the same document came back with all 378
# formulas empty, in 67 s, looking exactly like a document without formulas.
# The default therefore stays Docling's, and the size is opt-in.
ENRICHMENT_BATCH_ENV = "PDF_ENRICHMENT_BATCH"


def apply_enrichment_batch_size() -> int | None:
    """Apply ``PDF_ENRICHMENT_BATCH`` to Docling's enrichment stage."""
    raw = (os.getenv(ENRICHMENT_BATCH_ENV) or "").strip()
    if not raw:
        return None
    try:
        size = int(raw)
    except ValueError:
        logger.warning("%s=%r is not a number; ignoring it.", ENRICHMENT_BATCH_ENV, raw)
        return None
    if size < 1:
        logger.warning("%s must be at least 1; ignoring %d.", ENRICHMENT_BATCH_ENV, size)
        return None

    for module_path, class_name in (
        ("docling.models.stages.code_formula.code_formula_vlm_model", "CodeFormulaVlmModel"),
        ("docling.models.code_formula_model", "CodeFormulaModel"),
    ):
        try:
            module = __import__(module_path, fromlist=[class_name])
            getattr(module, class_name).elements_batch_size = size
        except Exception:  # pragma: no cover - depends on the Docling version
            continue
        logger.info("Formula/code enrichment batch size set to %d.", size)
        return size

    logger.warning("Could not find Docling's enrichment stage; %s ignored.", ENRICHMENT_BATCH_ENV)
    return None


class DoclingParser(BaseParser):
    """Structured parser for PDF and DOCX built on Docling.

    Docling reconstructs the reading order and classifies each element (heading,
    table, figure, formula, ...), which lets us build a rich :class:`Document`
    with a real table of contents, per-block page numbers and bounding boxes.

    OCR is expensive and can corrupt clean text, so it is *off by default*. The
    pipeline decides per file whether to enable it (see
    :mod:`file_agent.parsers.routing`) and constructs the parser accordingly.
    """

    #: OCR engine selected by the most recent :meth:`_configure_ocr` call.
    active_ocr_engine: str | None = None
    #: PDF backend that is known to work in this process ("docling" or
    #: "pypdfium"); set once the first conversion succeeds or fails.
    resolved_pdf_backend: str | None = None

    #: Set once enrichment turns out to be unavailable in this process, so the
    #: model download is not retried for every document of a run.
    enrichment_available: bool = True

    def __init__(
        self,
        do_ocr: bool = False,
        ocr_full_page: bool = False,
        enrich: bool | None = None,
        keep_empty_regions: bool = False,
    ) -> None:
        """:param enrich: run Docling's own formula/code models (``None``: decide
            from ``PDF_ENRICHMENT``).
        :param keep_empty_regions: keep formula and code regions that carry no
            text, so a later pass can read them from the page image. Without it
            an empty region is dropped, as a block with no text only costs
            retrieval quality.
        """
        self.do_ocr = do_ocr
        self.ocr_full_page = ocr_full_page
        type(self).active_ocr_engine = None
        self._backend = self._preferred_backend()
        if enrich is None:
            enrich = resolve_enrichment()
        self._enrich = enrich and type(self).enrichment_available
        self._keep_empty_regions = keep_empty_regions
        if self._enrich:
            apply_enrichment_batch_size()
        self._converter = self._build_converter(do_ocr, ocr_full_page, self._backend)
        self.ocr_engine = type(self).active_ocr_engine if do_ocr else None

    @classmethod
    def _preferred_backend(cls) -> str:
        forced = (os.getenv("DOCLING_PDF_BACKEND") or "").strip().lower()
        if forced in ("docling", "pypdfium"):
            return forced
        return cls.resolved_pdf_backend or "docling"

    def _build_converter(
        self, do_ocr: bool, ocr_full_page: bool, backend: str
    ) -> DocumentConverter:
        try:
            from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
            from docling.document_converter import PdfFormatOption

            pipeline_options = PdfPipelineOptions()
            pipeline_options.do_ocr = do_ocr
            pipeline_options.do_table_structure = True
            # ACCURATE TableFormer recovers row/column structure of financial and
            # scientific tables far better than FAST for a ~2x model cost.
            pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE
            # A formula in a PDF's text layer extracts as broken glyph soup
            # ("P(A | B)=P(A)" becomes "PA B PA" or worse), and a code listing
            # loses its line breaks. Docling's enrichment models re-read those
            # regions and return LaTeX and code, which is what makes a lecture
            # with formulas answerable at all.
            if self._enrich:
                pipeline_options.do_formula_enrichment = True
                pipeline_options.do_code_enrichment = True
            if do_ocr:
                self._configure_ocr(pipeline_options, ocr_full_page)

            pdf_format_option = PdfFormatOption(pipeline_options=pipeline_options)

            # The default docling-parse backend gives the best text/table cells
            # (row labels survive), but its glyph resources are broken on Windows
            # in some releases; there the pure-Python pypdfium backend is used,
            # with cell matching off because matching against pypdfium text drops
            # the label column of financial tables.
            if backend == "pypdfium":
                from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend

                pdf_format_option.backend = PyPdfiumDocumentBackend
                pipeline_options.table_structure_options.do_cell_matching = False

            return DocumentConverter(format_options={InputFormat.PDF: pdf_format_option})
        except Exception:  # pragma: no cover - fall back to defaults on API drift
            logger.warning(
                "Could not apply custom Docling pipeline options; using defaults.",
                exc_info=True,
            )
            return DocumentConverter(allowed_formats=[InputFormat.PDF, InputFormat.DOCX])

    def _convert(self, path: Path):
        """Convert with the preferred backend, falling back to pypdfium once."""
        try:
            result = self._converter.convert(path)
        except Exception as exc:
            if self._enrich:
                # The enrichment models are downloaded on first use; without
                # them (offline machine, missing extra) the conversion must
                # still produce a document.
                logger.warning(
                    "Docling enrichment unavailable (%s); converting %s without it.",
                    str(exc).splitlines()[0][:160],
                    path.name,
                )
                type(self).enrichment_available = False
                self._enrich = False
                self._converter = self._build_converter(
                    self.do_ocr, self.ocr_full_page, self._backend
                )
                return self._converter.convert(path)
            if self._backend != "docling" or path.suffix.lower() != ".pdf":
                raise
            logger.warning(
                "docling-parse backend failed for %s (%s); retrying with pypdfium.",
                path.name,
                str(exc).splitlines()[0][:200],
            )
            type(self).resolved_pdf_backend = "pypdfium"
            self._backend = "pypdfium"
            self._converter = self._build_converter(self.do_ocr, self.ocr_full_page, "pypdfium")
            return self._converter.convert(path)
        if path.suffix.lower() == ".pdf" and type(self).resolved_pdf_backend is None:
            type(self).resolved_pdf_backend = self._backend
        return result

    @staticmethod
    def _ocr_languages() -> list[str]:
        raw = os.getenv("OCR_LANGS", "ru,en")
        langs = [code.strip() for code in raw.split(",") if code.strip()]
        return langs or ["ru", "en"]

    @staticmethod
    def _ocr_engine() -> str:
        return os.getenv("OCR_ENGINE", "easyocr").strip().lower()

    @classmethod
    def _configure_ocr(cls, pipeline_options, ocr_full_page: bool) -> None:
        """Select the OCR engine.

        Default is **EasyOCR**: it reads Cyrillic as well as Latin (documents are
        frequently Russian), while RapidOCR's bundled models only recognize
        Latin/CJK. EasyOCR is accurate but slow on CPU, so ``OCR_ENGINE=rapidocr``
        switches to the faster offline engine for Latin-only material. Languages
        come from ``OCR_LANGS`` (default ``ru,en``). ``force_full_page_ocr``
        re-OCRs the whole page (genuine scans); otherwise only bitmap regions
        without a text layer are OCR'd.
        """
        langs = cls._ocr_languages()
        preferred = cls._ocr_engine()
        order = ["rapidocr", "easyocr"] if preferred == "rapidocr" else ["easyocr", "rapidocr"]

        for engine in order:
            options = cls._make_ocr_options(engine, langs)
            if options is not None:
                cls._set_full_page_ocr(options, ocr_full_page)
                pipeline_options.ocr_options = options
                cls.active_ocr_engine = engine
                # Third-party engines log inconsistently (RapidOCR is chatty,
                # EasyOCR is silent), so state the effective configuration
                # ourselves — otherwise there is no way to tell OCR ran.
                logger.info(
                    "OCR enabled: engine=%s languages=%s full_page=%s",
                    engine,
                    ",".join(langs) if engine == "easyocr" else "built-in",
                    ocr_full_page,
                )
                return

        logger.warning("No configurable OCR engine available; using Docling's default.")
        cls.active_ocr_engine = "docling-default"
        cls._set_full_page_ocr(pipeline_options.ocr_options, ocr_full_page)

    @staticmethod
    def _make_ocr_options(engine: str, langs: list[str]):
        try:
            if engine == "easyocr":
                from docling.datamodel.pipeline_options import EasyOcrOptions

                return EasyOcrOptions(lang=langs)
            if engine == "rapidocr":
                from docling.datamodel.pipeline_options import RapidOcrOptions

                return RapidOcrOptions()
        except Exception:  # pragma: no cover - engine not installed
            logger.debug("OCR engine %s unavailable.", engine)
        return None

    @staticmethod
    def _set_full_page_ocr(options, ocr_full_page: bool) -> None:
        try:
            options.force_full_page_ocr = ocr_full_page
        except Exception:  # pragma: no cover - depends on Docling version
            pass

    def parse(self, file_path: Path) -> Document:
        path = Path(file_path)

        with tracer.start_as_current_span("file_agent.docling_parse") as span:
            span.set_attribute("file_agent.file_name", path.name)
            span.set_attribute("file_agent.do_ocr", self.do_ocr)
            if self.ocr_engine:
                span.set_attribute("file_agent.ocr_engine", self.ocr_engine)

            result = self._convert(path)
            docling_doc = result.document

            page_heights = self._page_heights(docling_doc)

            raw_blocks: list[Block] = []
            enriched, empty = 0, 0
            for item, level in docling_doc.iterate_items():
                if self._enrich and _is_enriched_label(getattr(item, "label", "")):
                    enriched += 1
                    empty += 0 if (getattr(item, "text", "") or "").strip() else 1
                block = self._item_to_block(item, level, docling_doc, page_heights, path)
                if block is not None:
                    raw_blocks.append(block)
            _warn_on_lost_enrichment(path, enriched, empty)
            if path.suffix.lower() == ".pdf":
                raw_blocks = repair_column_order(raw_blocks)
            blocks = _postprocess_blocks(raw_blocks)

            if not blocks:
                blocks.append(
                    Block(
                        id="block-empty",
                        text="",
                        type=BlockType.TEXT.value,
                        metadata={
                            "source_file": path.name,
                            "warning": "empty document or no extractable content",
                        },
                        block_type=BlockType.TEXT,
                        page_number=1,
                    )
                )

            native_markdown = self._export_markdown(docling_doc)

            document = Document(
                file_name=path.name,
                file_type=path.suffix.lower().lstrip("."),
                blocks=blocks,
                metadata={
                    "parsing_method": "docling_ocr" if self.do_ocr else "docling",
                    "pdf_backend": self._backend,
                    "ocr_engine": self.ocr_engine,
                    "docling_markdown": native_markdown,
                    "title": document_title(blocks),
                },
            )
            document.build_table_of_contents()

            span.set_attribute("file_agent.block_count", len(blocks))
            logger.info("Docling parsed %s into %d block(s)", path.name, len(blocks))
            return document

    # -- helpers ------------------------------------------------------------

    def _item_to_block(self, item, level, docling_doc, page_heights, path: Path) -> Block | None:
        label = getattr(item, "label", "")
        label_lower = str(label).lower()
        # Running headers/footers repeat on every page (paper title, page number)
        # and only pollute chunks and retrieval; drop them at the source.
        if "page_header" in label_lower or "page_footer" in label_lower:
            return None
        block_type = self._map_label(str(label))
        parent = getattr(item, "parent", None)
        parent_ref = getattr(parent, "cref", None) or ""
        # Captions are folded into their figure/table block (see below); a
        # standalone caption item that belongs to one would only duplicate it.
        if "caption" in label_lower and ("/pictures/" in parent_ref or "/tables/" in parent_ref):
            return None
        content = self._extract_content(item, block_type, docling_doc)
        if content is None:
            # Structural containers (groups, lists wrappers) carry no own text.
            return None

        page_number, bbox = self._extract_prov(item, page_heights)
        metadata = {
            "source_file": path.name,
            "docling_label": str(label),
            "docling_parent": parent_ref,
            "docling_parent_label": self._parent_label(parent, docling_doc),
            "hierarchy_level": level,
        }
        if block_type == BlockType.HEADING:
            metadata["hierarchy_level"] = infer_heading_level(
                content, default=int(getattr(item, "level", 1) or 1)
            )
        if not content and block_type in (BlockType.FORMULA, BlockType.CODE):
            metadata[ENRICHMENT_PENDING] = True
        if block_type in (BlockType.TABLE, BlockType.FIGURE, BlockType.IMAGE):
            caption = self._caption_text(item, docling_doc)
            if caption:
                metadata["caption"] = caption
                content = f"{caption}\n{content}".strip() if content else caption
        image_bytes = None
        if block_type in (BlockType.FIGURE, BlockType.IMAGE) and path.suffix.lower() != ".pdf":
            image_bytes = self._embedded_image(item, docling_doc)

        return Block(
            id=f"block-{uuid.uuid4().hex[:8]}",
            text=content,
            type=block_type.value,
            metadata=metadata,
            block_type=block_type,
            page_number=page_number,
            bbox=bbox,
            image_bytes=image_bytes,
        )

    @staticmethod
    def _parent_label(parent, docling_doc) -> str:
        resolver = getattr(parent, "resolve", None)
        if not callable(resolver):
            return ""
        try:
            node = resolver(docling_doc)
        except Exception:  # pragma: no cover
            return ""
        return str(getattr(node, "label", "") or "")

    @staticmethod
    def _caption_text(item, docling_doc) -> str:
        getter = getattr(item, "caption_text", None)
        if not callable(getter):
            return ""
        try:
            return " ".join(str(getter(docling_doc) or "").split())
        except Exception:  # pragma: no cover - depends on Docling version
            return ""

    @staticmethod
    def _embedded_image(item, docling_doc) -> bytes | None:
        """Return PNG bytes of a picture embedded in DOCX/PPTX (no page to crop)."""
        getter = getattr(item, "get_image", None)
        if not callable(getter):
            return None
        try:
            image = getter(docling_doc)
        except Exception:  # pragma: no cover
            return None
        if image is None:
            return None
        try:
            import io

            buffer = io.BytesIO()
            image.convert("RGB").save(buffer, format="PNG")
            return buffer.getvalue()
        except Exception:  # pragma: no cover
            return None

    def _extract_content(self, item, block_type: BlockType, docling_doc) -> str | None:
        if block_type == BlockType.TABLE:
            markdown = self._export_item_markdown(item, docling_doc)
            if markdown:
                return self._normalize_table_markdown(markdown)
            # Keep the table block even if Markdown export is unavailable.
            return getattr(item, "text", None) or "[table]"

        text = getattr(item, "text", None)
        if text:
            return text

        # Figures/images have no text of their own; keep the block so the VLM
        # enhancer can describe it and so it appears in the Markdown output.
        if block_type in (BlockType.FIGURE, BlockType.IMAGE):
            return ""

        # A formula region without enrichment is empty; keep it when something
        # downstream is going to read it from the page image.
        if self._keep_empty_regions and block_type in (BlockType.FORMULA, BlockType.CODE):
            return ""

        return None

    @staticmethod
    def _normalize_table_markdown(markdown: str) -> str:
        """Strip the alignment padding Docling puts into Markdown tables.

        Docling pads every cell so the columns line up in a monospaced view. That
        padding can triple a table's length without adding information: a row of
        50 tokens can occupy 1000 characters, wasting chunk space and making
        previews unreadable. Markdown does not need the padding to render.
        """
        lines: list[str] = []
        for raw_line in markdown.splitlines():
            stripped = raw_line.strip()
            if not stripped.startswith("|"):
                lines.append(raw_line)
                continue

            cells = [
                re.sub(r"\s{2,}", " ", cell.strip()) for cell in stripped.strip("|").split("|")
            ]
            if cells and all(cell and set(cell) <= set("-: ") for cell in cells):
                cells = ["---"] * len(cells)  # separator row
            lines.append("| " + " | ".join(cells) + " |")
        return "\n".join(lines)

    @staticmethod
    def _export_item_markdown(item, docling_doc) -> str | None:
        exporter = getattr(item, "export_to_markdown", None)
        if not callable(exporter):
            return None
        # Newer Docling requires the owning document; older versions take no args.
        for args in ((docling_doc,), ()):
            try:
                return exporter(*args)
            except TypeError:
                continue
            except Exception:  # pragma: no cover
                return None
        return None

    @staticmethod
    def _export_markdown(docling_doc) -> str | None:
        exporter = getattr(docling_doc, "export_to_markdown", None)
        if not callable(exporter):
            return None
        try:
            return exporter()
        except Exception:  # pragma: no cover
            return None

    @staticmethod
    def _page_heights(docling_doc) -> dict[int, float]:
        heights: dict[int, float] = {}
        pages = getattr(docling_doc, "pages", None) or {}
        items = pages.items() if hasattr(pages, "items") else enumerate(pages, start=1)
        for page_no, page in items:
            size = getattr(page, "size", None)
            height = getattr(size, "height", None)
            if height:
                try:
                    heights[int(page_no)] = float(height)
                except (TypeError, ValueError):
                    continue
        return heights

    @staticmethod
    def _extract_prov(item, page_heights: dict[int, float]):
        prov_list = getattr(item, "prov", None)
        if not prov_list:
            return None, None

        prov = prov_list[0]
        page_number = getattr(prov, "page_no", None)
        raw_bbox = getattr(prov, "bbox", None)
        if raw_bbox is None:
            return page_number, None

        bbox = raw_bbox
        height = page_heights.get(int(page_number)) if page_number else None
        to_top_left = getattr(raw_bbox, "to_top_left_origin", None)
        if callable(to_top_left) and height:
            # Docling PDF coordinates use a bottom-left origin; PyMuPDF (used for
            # cropping figures) expects a top-left origin.
            try:
                bbox = to_top_left(page_height=height)
            except Exception:  # pragma: no cover
                bbox = raw_bbox

        try:
            coords = (float(bbox.l), float(bbox.t), float(bbox.r), float(bbox.b))
        except (AttributeError, TypeError, ValueError):
            return page_number, None

        x0, y0, x1, y1 = coords
        return page_number, (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))

    @staticmethod
    def _map_label(label: str) -> BlockType:
        label_lower = label.lower()
        if "section_header" in label_lower or "title" in label_lower or "heading" in label_lower:
            return BlockType.HEADING
        if "table" in label_lower:
            return BlockType.TABLE
        if "picture" in label_lower or "figure" in label_lower or "image" in label_lower:
            return BlockType.FIGURE
        if "formula" in label_lower or "equation" in label_lower:
            return BlockType.FORMULA
        if "list_item" in label_lower:
            return BlockType.LIST
        if label_lower == "code":
            return BlockType.CODE
        return BlockType.TEXT


# Share of enrichment regions that may come back empty before the run is
# treated as a failure rather than as a few unreadable crops.
LOST_ENRICHMENT_RATIO = 0.5


def _is_enriched_label(label) -> bool:
    text = str(label).lower()
    return "formula" in text or text == "code"


def _warn_on_lost_enrichment(path: Path, enriched: int, empty: int) -> None:
    """Say it out loud when enrichment ran but returned nothing.

    Docling catches every exception inside the enrichment stage, including CUDA
    out-of-memory, and returns empty text for the whole batch. The document
    then parses *faster* than without enrichment and silently arrives without
    its formulas — the failure mode that is easiest to ship by accident.
    """
    if not enriched or empty < max(1, int(enriched * LOST_ENRICHMENT_RATIO)):
        return
    logger.warning(
        "Formula/code enrichment returned no text for %d of %d region(s) in %s; "
        "the document is indexed without them (most often not enough free GPU memory — "
        "lower %s or set PDF_ENRICHMENT=off).",
        empty,
        enriched,
        path.name,
        ENRICHMENT_BATCH_ENV,
    )


# -- post-processing -------------------------------------------------------------

_HEADING_CONTINUES = re.compile(
    r"[,(\-–—:;/]\s*$|\b(и|или|the|of|and|for|по|для|в|на|с)\s*$", re.IGNORECASE
)

# A word broken by justification across a line break: PDF text extraction keeps
# the hyphen and turns the line break into a space ("обыкновен- ных"). Left as
# is, the term is invisible to both BM25 and the encoder — which is exactly how
# a question about "обыкновенных акций" misses the table row that answers it.
# A real compound never has a space after its hyphen, so requiring one is safe.
_HYPHEN_BREAK = re.compile(r"([^\W\d_]{2,})[-‐]\s+([^\W\d_]{2,})")
_SOFT_HYPHEN = "­"


def repair_hyphenation(text: str) -> str:
    """Rejoin words split by end-of-line hyphenation."""
    if _SOFT_HYPHEN in text:
        text = text.replace(_SOFT_HYPHEN, "")
    if "-" not in text and "‐" not in text:
        return text
    repaired = _HYPHEN_BREAK.sub(lambda m: m.group(1) + m.group(2), text)
    # A second pass catches chains ("при- виле- гированных").
    return _HYPHEN_BREAK.sub(lambda m: m.group(1) + m.group(2), repaired)


# Column repair. A page is treated as two-column only when the evidence is
# unambiguous: enough blocks, none of them spanning the middle, and the reading
# order actually jumping between the sides more than a heading or two would.
COLUMN_MIN_BLOCKS = 6
COLUMN_MIN_ALTERNATIONS = 4
# A block is "full width" when it covers this share of the page width; a page
# with several of those is a single-column page with wide figures.
FULL_WIDTH_RATIO = 0.65


def repair_column_order(blocks: list[Block]) -> list[Block]:
    """Re-order the blocks of a two-column page that were read across the gutter.

    The layout model normally recovers columns, but when it does not, the
    result is text that alternates between the left and the right column
    sentence by sentence — unreadable for a human and, worse, silently wrong
    for a chunker that will pack the two halves of two different arguments into
    one passage. Pages that show that pattern are re-sorted column by column;
    every other page is left exactly as the model produced it.
    """
    by_page: dict[int, list[Block]] = {}
    for block in blocks:
        if block.page_number is None or block.bbox is None:
            return blocks  # a stream without geometry: nothing to reason about
        by_page.setdefault(block.page_number, []).append(block)

    repaired: dict[int, list[Block]] = {}
    for page, page_blocks in by_page.items():
        ordered = _repair_page(page_blocks)
        if ordered is not None:
            repaired[page] = ordered
    if not repaired:
        return blocks

    result: list[Block] = []
    emitted: set[int] = set()
    for block in blocks:
        page = block.page_number
        if page in repaired:
            if page not in emitted:
                emitted.add(page)
                result.extend(repaired[page])
            continue
        result.append(block)
    return result


def _repair_page(page_blocks: list[Block]) -> list[Block] | None:
    if len(page_blocks) < COLUMN_MIN_BLOCKS:
        return None

    left_edge = min(b.bbox[0] for b in page_blocks)
    right_edge = max(b.bbox[2] for b in page_blocks)
    width = right_edge - left_edge
    if width <= 0:
        return None
    middle = left_edge + width / 2

    sides: list[int] = []
    for block in page_blocks:
        x0, _, x1, _ = block.bbox
        if (x1 - x0) >= width * FULL_WIDTH_RATIO:
            # A full-width element (title, wide table) means the page is not a
            # clean two-column layout, or it separates two column groups.
            return None
        sides.append(0 if x1 <= middle + width * 0.05 else 1)

    if len(set(sides)) < 2:
        return None
    alternations = sum(1 for a, b in zip(sides, sides[1:], strict=False) if a != b)
    if alternations < COLUMN_MIN_ALTERNATIONS:
        return None  # a couple of crossings is normal reading order, not damage

    return [
        block
        for _, block in sorted(
            zip(sides, page_blocks, strict=True),
            key=lambda pair: (pair[0], pair[1].bbox[1], pair[1].bbox[0]),
        )
    ]


def _postprocess_blocks(blocks: list[Block]) -> list[Block]:
    """Turn Docling's flat item stream into retrieval-friendly blocks.

    - consecutive ``list_item`` items of the same group become one ``LIST``
      block (bullet glyphs stripped, nesting kept as indentation);
    - text fragments Docling emits for one paragraph split by run formatting
      (``inline`` groups in DOCX) are merged back into a single paragraph;
    - a heading that the layout model cut across two lines ("… (группе" /
      "состояний)") is stitched back together.
    """
    merged: list[Block] = []
    for block in blocks:
        if block.text and block.block_type in (
            BlockType.TEXT,
            BlockType.LIST,
            BlockType.HEADING,
            BlockType.TABLE,
        ):
            block.text = repair_hyphenation(block.text)
        previous = merged[-1] if merged else None
        parent = block.metadata.get("docling_parent", "")

        if block.block_type == BlockType.LIST:
            if (
                previous is not None
                and previous.block_type == BlockType.LIST
                and previous.metadata.get("docling_parent") == parent
                and previous.page_number in (None, block.page_number)
            ):
                previous.metadata["_items"].append(_list_item_text(block))
                previous.text = list_to_markdown([(0, t) for t in previous.metadata["_items"]])
                previous.metadata["item_count"] = len(previous.metadata["_items"])
                if block.bbox and previous.bbox:
                    previous.bbox = _union_bbox(previous.bbox, block.bbox)
                continue
            block.metadata["_items"] = [_list_item_text(block)]
            block.metadata["item_count"] = 1
            block.text = list_to_markdown([(0, block.metadata["_items"][0])])
            merged.append(block)
            continue

        if (
            block.block_type == BlockType.TEXT
            and previous is not None
            and previous.block_type == BlockType.TEXT
            and parent
            and block.metadata.get("docling_parent_label") == "inline"
            and previous.metadata.get("docling_parent") == parent
            and previous.page_number in (None, block.page_number)
        ):
            opens = previous.text.endswith(("(", "«"))
            closes = block.text[:1] in ",.;:)»"
            joiner = "" if opens or closes else " "
            previous.text = f"{previous.text}{joiner}{block.text}"
            if block.bbox and previous.bbox:
                previous.bbox = _union_bbox(previous.bbox, block.bbox)
            continue

        if (
            block.block_type == BlockType.HEADING
            and previous is not None
            and previous.block_type == BlockType.HEADING
            and previous.page_number == block.page_number
            and _looks_like_split_heading(previous.text, block.text)
        ):
            previous.text = f"{previous.text} {block.text}".strip()
            if block.bbox and previous.bbox:
                previous.bbox = _union_bbox(previous.bbox, block.bbox)
            continue

        merged.append(block)

    for block in merged:
        block.metadata.pop("_items", None)
    return merged


def _list_item_text(block: Block) -> str:
    return strip_bullet(" ".join(block.text.split()))


def _looks_like_split_heading(first: str, second: str) -> bool:
    if not first or not second:
        return False
    if _HEADING_CONTINUES.search(first):
        return True
    return second[:1].islower() or second[:1] in ")»,;"


def _union_bbox(a, b):
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))


# Structural headings that open a document but say nothing about it. Taking one
# as the title puts "Оглавление" in front of every chunk of the document and in
# every breadcrumb the retriever sees.
_GENERIC_TITLES = frozenset(
    {
        "оглавление",
        "содержание",
        "введение",
        "аннотация",
        "приложение",
        "список литературы",
        "table of contents",
        "contents",
        "introduction",
        "abstract",
        "appendix",
        "references",
        "index",
    }
)


def document_title(blocks: list[Block]) -> str | None:
    """First heading that actually names the document."""
    fallback: str | None = None
    for block in blocks[:12]:
        if block.block_type != BlockType.HEADING:
            continue
        text = " ".join(block.text.split())
        if not text:
            continue
        if text.strip(" .:").lower() in _GENERIC_TITLES:
            continue
        if len(text) > 200:  # a paragraph mislabelled as a heading
            fallback = fallback or text[:200]
            continue
        return text
    return fallback
