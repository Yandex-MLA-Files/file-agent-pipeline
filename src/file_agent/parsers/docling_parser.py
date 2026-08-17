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

    def __init__(self, do_ocr: bool = False, ocr_full_page: bool = False) -> None:
        self.do_ocr = do_ocr
        self.ocr_full_page = ocr_full_page
        type(self).active_ocr_engine = None
        self._backend = self._preferred_backend()
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
            for item, level in docling_doc.iterate_items():
                block = self._item_to_block(item, level, docling_doc, page_heights, path)
                if block is not None:
                    raw_blocks.append(block)
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
                    "title": _document_title(blocks),
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


# -- post-processing -------------------------------------------------------------

_HEADING_CONTINUES = re.compile(
    r"[,(\-–—:;/]\s*$|\b(и|или|the|of|and|for|по|для|в|на|с)\s*$", re.IGNORECASE
)


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


def _document_title(blocks: list[Block]) -> str | None:
    for block in blocks[:5]:
        if block.block_type == BlockType.HEADING and block.text.strip():
            return block.text.strip()
    return None
