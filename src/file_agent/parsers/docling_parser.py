import logging
import uuid
from pathlib import Path

from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter

from file_agent.document import Block, BlockType, Document
from file_agent.parsers.base import BaseParser

logger = logging.getLogger(__name__)


class DoclingParser(BaseParser):
    """Structured parser for PDF and DOCX built on Docling.

    Docling reconstructs the reading order and classifies each element (heading,
    table, figure, formula, ...), which lets us build a rich :class:`Document`
    with a real table of contents, per-block page numbers and bounding boxes.

    OCR is expensive and can corrupt clean text, so it is *off by default*. The
    pipeline decides per file whether to enable it (see
    :mod:`file_agent.parsers.routing`) and constructs the parser accordingly.
    """

    def __init__(self, do_ocr: bool = False, ocr_full_page: bool = False) -> None:
        self.do_ocr = do_ocr
        self.ocr_full_page = ocr_full_page
        self._converter = self._build_converter(do_ocr, ocr_full_page)

    def _build_converter(self, do_ocr: bool, ocr_full_page: bool) -> DocumentConverter:
        try:
            from docling.datamodel.pipeline_options import PdfPipelineOptions
            from docling.document_converter import PdfFormatOption

            pipeline_options = PdfPipelineOptions()
            pipeline_options.do_ocr = do_ocr
            pipeline_options.do_table_structure = True
            if do_ocr:
                self._configure_ocr(pipeline_options, ocr_full_page)

            pdf_format_option = PdfFormatOption(pipeline_options=pipeline_options)

            # Prefer the pypdfium backend: it is pure-Python and avoids the glyph
            # resource lookup in the default docling-parse backend, which is broken
            # on Windows in some docling-parse releases.
            try:
                from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend

                pdf_format_option.backend = PyPdfiumDocumentBackend
            except Exception:  # pragma: no cover - keep default backend if missing
                logger.debug("pypdfium2 backend unavailable; using the default backend.")

            return DocumentConverter(format_options={InputFormat.PDF: pdf_format_option})
        except Exception:  # pragma: no cover - fall back to defaults on API drift
            logger.warning(
                "Could not apply custom Docling pipeline options; using defaults.",
                exc_info=True,
            )
            return DocumentConverter(allowed_formats=[InputFormat.PDF, InputFormat.DOCX])

    @staticmethod
    def _configure_ocr(pipeline_options, ocr_full_page: bool) -> None:
        """Select an OCR engine that is actually installed.

        RapidOCR ships its ONNX models inside the wheel, so it works offline and
        needs no system binary — the safest default on Windows. If it is not
        available we fall back to Docling's built-in default engine and only
        tweak the full-page-OCR flag. ``force_full_page_ocr`` re-OCRs the whole
        page (needed for genuine scans); otherwise only bitmap regions without a
        text layer are OCR'd.
        """
        try:
            from docling.datamodel.pipeline_options import RapidOcrOptions

            options = RapidOcrOptions()
            try:
                options.force_full_page_ocr = ocr_full_page
            except Exception:  # pragma: no cover - depends on Docling version
                pass
            pipeline_options.ocr_options = options
            return
        except Exception:  # pragma: no cover - RapidOCR not installed
            logger.debug("RapidOCR unavailable; using Docling's default OCR engine.")

        try:
            pipeline_options.ocr_options.force_full_page_ocr = ocr_full_page
        except Exception:  # pragma: no cover - depends on Docling version
            pass

    def parse(self, file_path: Path) -> Document:
        path = Path(file_path)
        result = self._converter.convert(path)
        docling_doc = result.document

        page_heights = self._page_heights(docling_doc)

        blocks: list[Block] = []
        for item, level in docling_doc.iterate_items():
            block = self._item_to_block(item, level, docling_doc, page_heights, path)
            if block is not None:
                blocks.append(block)

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
                "docling_markdown": native_markdown,
            },
        )
        document.build_table_of_contents()
        return document

    # -- helpers ------------------------------------------------------------

    def _item_to_block(self, item, level, docling_doc, page_heights, path: Path) -> Block | None:
        label = getattr(item, "label", "")
        block_type = self._map_label(str(label))
        content = self._extract_content(item, block_type, docling_doc)
        if content is None:
            # Structural containers (groups, lists wrappers) carry no own text.
            return None

        page_number, bbox = self._extract_prov(item, page_heights)

        return Block(
            id=f"block-{uuid.uuid4().hex[:8]}",
            text=content,
            type=block_type.value,
            metadata={
                "source_file": path.name,
                "docling_label": str(label),
                "hierarchy_level": level,
            },
            block_type=block_type,
            page_number=page_number,
            bbox=bbox,
        )

    def _extract_content(self, item, block_type: BlockType, docling_doc) -> str | None:
        if block_type == BlockType.TABLE:
            markdown = self._export_item_markdown(item, docling_doc)
            if markdown:
                return markdown
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
        return BlockType.TEXT
