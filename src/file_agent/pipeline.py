import logging
import os
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv

from file_agent.document import BlockType, Document
from file_agent.parsers.docling_parser import DoclingParser
from file_agent.parsers.enhancer import DocumentEnhancer
from file_agent.parsers.html_parser import HTMLParser
from file_agent.parsers.md_parser import MarkdownParser
from file_agent.parsers.pdf_parser import PDFParser
from file_agent.parsers.pptx_parser import PPTXParser
from file_agent.parsers.routing import analyze_pdf
from file_agent.parsers.xlsx_parser import XLSXParser
from file_agent.vlm.base import MockVLMClient, VLMClient
from file_agent.vlm.openai_compatible import OpenAICompatibleVLMClient

load_dotenv()

logger = logging.getLogger(__name__)

# When more than this share of pages need OCR, treat the document as a full scan
# and OCR every page instead of only the bitmap regions.
FULL_SCAN_RATIO = 0.6

OcrMode = Literal["auto", "on", "off"]


def parse_file(
    file_path: str | Path,
    enable_vlm: bool = False,
    enable_ocr: OcrMode = "auto",
) -> Document:
    """Parse any supported file into a structured :class:`Document`.

    :param enable_vlm: when True, figures/diagrams in PDFs are described by a VLM
        (requires a reachable VLM endpoint; falls back to a mock otherwise).
    :param enable_ocr: OCR policy for PDFs — ``"auto"`` lets the pipeline decide
        per page (see :mod:`file_agent.parsers.routing`), ``"on"`` forces OCR,
        ``"off"`` disables it. Ignored for formats that carry their own text.
    """
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix in {".pdf", ".docx"}:
        return _parse_structured(path, enable_vlm=enable_vlm, enable_ocr=enable_ocr)
    if suffix == ".md":
        return MarkdownParser().parse(path)
    if suffix in {".html", ".htm"}:
        return HTMLParser().parse(path)
    if suffix == ".xlsx":
        return XLSXParser().parse(path)
    if suffix == ".pptx":
        return PPTXParser().parse(path)

    raise ValueError(f"Unsupported file type: {suffix or '<no extension>'}")


def _parse_structured(path: Path, enable_vlm: bool, enable_ocr: OcrMode) -> Document:
    do_ocr, ocr_full_page, analysis = _resolve_ocr_policy(path, enable_ocr)

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

    if enable_vlm:
        _enhance_with_vlm(document, path)

    return document


def _resolve_ocr_policy(path: Path, enable_ocr: OcrMode):
    """Decide whether to run OCR and return (do_ocr, ocr_full_page, analysis)."""
    if path.suffix.lower() != ".pdf":
        # Routing analysis only applies to PDFs; DOCX has its own text layer.
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
    if path.suffix.lower() != ".pdf":
        return

    has_figures = any(
        block.block_type in (BlockType.FIGURE, BlockType.IMAGE) and block.bbox
        for block in document.blocks
    )
    if not has_figures:
        return

    enhancer = DocumentEnhancer(vlm_client=_build_vlm_client())
    enhancer.enhance(document, path)


def _build_vlm_client() -> VLMClient:
    try:
        return OpenAICompatibleVLMClient(
            base_url=os.getenv("VLM_BASE_URL", "http://localhost:11434/v1"),
            model=os.getenv("VLM_MODEL", "qwen2.5-vl:7b"),
            api_key=os.getenv("VLM_API_KEY", "dummy"),
        )
    except Exception:
        logger.warning("Could not initialize the VLM client; using MockVLMClient.", exc_info=True)
        return MockVLMClient()
