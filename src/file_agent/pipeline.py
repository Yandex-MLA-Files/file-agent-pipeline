import logging
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv

from file_agent.document import BlockType, Document
from file_agent.parsers.docling_parser import DoclingParser
from file_agent.parsers.enhancer import DocumentEnhancer
from file_agent.parsers.html_parser import HTMLParser
from file_agent.parsers.image_parser import ImageParser
from file_agent.parsers.md_parser import MarkdownParser
from file_agent.parsers.pdf_parser import PDFParser
from file_agent.parsers.pptx_parser import PPTXParser
from file_agent.parsers.routing import analyze_pdf
from file_agent.parsers.txt_parser import TXTParser
from file_agent.parsers.xlsx_parser import XLSXParser
from file_agent.telemetry import tracer
from file_agent.vlm.factory import create_vlm_client

load_dotenv()

logger = logging.getLogger(__name__)


FULL_SCAN_RATIO = 0.6

OcrMode = Literal["auto", "on", "off"]


def parse_file(
    file_path: str | Path,
    enable_vlm: bool | None = None,
    enable_ocr: OcrMode = "auto",
) -> Document:

    path = Path(file_path)
    suffix = path.suffix.lower()

    with tracer.start_as_current_span("file_agent.parse_file") as span:
        span.set_attribute("file_agent.file_name", path.name)
        span.set_attribute("file_agent.file_suffix", suffix)
        logger.info("Parsing file %s", path.name)

        document = _parse_by_suffix(path, suffix, enable_vlm=enable_vlm, enable_ocr=enable_ocr)

        span.set_attribute("file_agent.block_count", len(document.blocks))
        logger.info("Parsed %s into %d block(s)", path.name, len(document.blocks))
        return document


def _parse_by_suffix(
    path: Path,
    suffix: str,
    enable_vlm: bool | None,
    enable_ocr: OcrMode,
) -> Document:
    if suffix in {".pdf", ".docx"}:
        return _parse_structured(path, enable_vlm=enable_vlm, enable_ocr=enable_ocr)
    if suffix == ".md":
        return MarkdownParser().parse(path)
    if suffix == ".txt":
        return TXTParser().parse(path)
    if suffix in {".html", ".htm"}:
        return HTMLParser().parse(path)
    if suffix == ".xlsx":
        return XLSXParser().parse(path)
    if suffix in {".jpg", ".jpeg", ".png"}:
        return ImageParser().parse(path)
    if suffix == ".pptx":
        document = PPTXParser().parse(path)
        if enable_vlm is True:
            _enhance_with_vlm(document, path)
        return document

    raise ValueError(f"Unsupported file type: {suffix or '<no extension>'}")


def _parse_structured(path: Path, enable_vlm: bool | None, enable_ocr: OcrMode) -> Document:
    do_ocr, ocr_full_page, analysis = _resolve_ocr_policy(path, enable_ocr)

    if do_ocr:
        pages = analysis.ocr_page_numbers if analysis else []
        logger.info(
            "Parsing %s with OCR (%s of %s pages need it: %s)",
            path.name,
            len(pages),
            len(analysis.pages) if analysis else "?",
            pages or "forced",
        )
    else:
        logger.info("Parsing %s without OCR (text layer present on every page)", path.name)

    try:
        document = DoclingParser(do_ocr=do_ocr, ocr_full_page=ocr_full_page).parse(path)
    except Exception:
        logger.warning(
            "Docling failed to parse %s; falling back to the PyMuPDF parser.",
            path.name,
            exc_info=True,
        )
        if path.suffix.lower() == ".pdf":
            return PDFParser().parse(path)
        raise

    if analysis is not None:
        document.metadata["page_analysis"] = analysis.summary()

    if enable_vlm is True:
        _enhance_with_vlm(document, path)

    return document


def _resolve_ocr_policy(path: Path, enable_ocr: OcrMode):
    """Decide whether to run OCR and return (do_ocr, ocr_full_page, analysis)."""
    if path.suffix.lower() != ".pdf":
        return (enable_ocr == "on"), False, None

    analysis = None
    try:
        analysis = analyze_pdf(path)
    except Exception:
        logger.warning("PDF page analysis failed for %s.", path.name, exc_info=True)

    if enable_ocr == "off":
        return False, False, analysis
    if enable_ocr == "on":
        return True, True, analysis

    # "auto": let the per-page heuristic decide.
    if analysis is None:
        return False, False, None
    return analysis.needs_ocr, analysis.scanned_ratio >= FULL_SCAN_RATIO, analysis


def _enhance_with_vlm(document: Document, path: Path) -> None:

    if path.suffix.lower() not in {".pdf", ".pptx"}:
        return

    has_figures = any(
        block.block_type in (BlockType.FIGURE, BlockType.IMAGE) and block.bbox
        for block in document.blocks
    )
    if not has_figures:
        return

    vlm_client = create_vlm_client()
    if vlm_client is None:
        logger.warning(
            "VLM requested but no backend configured; set VLM_BACKEND to "
            "'smolvlm' (local, free) or 'openai' (endpoint via VLM_BASE_URL)."
        )
        return

    logger.info("VLM enhancement: describing figures in %s", path.name)
    DocumentEnhancer(vlm_client=vlm_client).enhance(document, path)
