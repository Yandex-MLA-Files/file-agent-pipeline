import logging
import os
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv

from file_agent.document import BlockType, Document
from file_agent.parsers.docling_parser import DoclingParser
from file_agent.parsers.docx_parser import DOCXParser
from file_agent.parsers.enhancer import DocumentEnhancer
from file_agent.parsers.html_parser import HTMLParser
from file_agent.parsers.md_parser import MarkdownParser
from file_agent.parsers.pdf_parser import PDFParser
from file_agent.parsers.pptx_parser import PPTXParser
from file_agent.parsers.routing import analyze_pdf
from file_agent.parsers.txt_parser import TXTParser
from file_agent.parsers.vlm_ocr import VLMPageOCR, merge_ocr_blocks
from file_agent.parsers.xlsx_parser import XLSXParser
from file_agent.telemetry import tracer
from file_agent.vlm.factory import create_vlm_client

load_dotenv()

logger = logging.getLogger(__name__)

# When more than this share of pages need OCR, treat the document as a full scan
# and OCR every page instead of only the bitmap regions.
FULL_SCAN_RATIO = 0.6

OcrMode = Literal["auto", "on", "off"]

# OCR engine for pages without a text layer. ``vlm`` (default) transcribes the
# rendered page with the multimodal chat model (see parsers.vlm_ocr) and falls
# back to EasyOCR when no VLM endpoint is configured; ``easyocr`` / ``rapidocr``
# run the classic engines inside Docling.
DEFAULT_OCR_ENGINE = "vlm"

# ``structured`` (default) uses the format-aware parsers that emit headings,
# lists, tables and figures for every format; ``legacy`` keeps the original
# one-block-per-slide/sheet/file parsers as a fallback implementation.
ParserProfile = Literal["structured", "legacy"]
DEFAULT_PARSER_PROFILE: ParserProfile = "structured"


def parse_file(
    file_path: str | Path,
    enable_vlm: bool | None = None,
    enable_ocr: OcrMode = "auto",
    parser_profile: ParserProfile | None = None,
) -> Document:
    """Parse any supported file into a structured :class:`Document`.

    :param enable_vlm: when True, figures/diagrams are described by a VLM.
        The default (None) defers to configuration: the ``VLM_BACKEND`` env var
        selects a backend (``llm`` / ``openai`` / ``smolvlm`` / ``off``), so the
        whole app gains figure understanding without any code changes.
    :param enable_ocr: OCR policy for PDFs — ``"auto"`` lets the pipeline decide
        per page (see :mod:`file_agent.parsers.routing`), ``"on"`` forces OCR,
        ``"off"`` disables it. Ignored for formats that carry their own text.
    :param parser_profile: ``"structured"`` (default, or ``PARSER_PROFILE`` env)
        or ``"legacy"`` to fall back to the original flat parsers.
    """
    path = Path(file_path)
    suffix = path.suffix.lower()
    profile = resolve_parser_profile(parser_profile)

    with tracer.start_as_current_span("file_agent.parse_file") as span:
        span.set_attribute("file_agent.file_name", path.name)
        span.set_attribute("file_agent.file_suffix", suffix)
        span.set_attribute("file_agent.parser_profile", profile)
        logger.info("Parsing file %s (profile=%s)", path.name, profile)

        document = _parse_by_suffix(
            path, suffix, enable_vlm=enable_vlm, enable_ocr=enable_ocr, profile=profile
        )
        document.metadata.setdefault("parser_profile", profile)

        span.set_attribute("file_agent.block_count", len(document.blocks))
        logger.info("Parsed %s into %d block(s)", path.name, len(document.blocks))
        return document


def resolve_parser_profile(explicit: str | None = None) -> ParserProfile:
    value = (explicit or os.getenv("PARSER_PROFILE") or DEFAULT_PARSER_PROFILE).strip().lower()
    if value not in ("structured", "legacy"):
        raise ValueError(f"PARSER_PROFILE must be 'structured' or 'legacy', got {value!r}")
    return value  # type: ignore[return-value]


def _parse_by_suffix(
    path: Path,
    suffix: str,
    enable_vlm: bool | None,
    enable_ocr: OcrMode,
    profile: ParserProfile,
) -> Document:
    if suffix == ".pdf":
        return _parse_structured(path, enable_vlm=enable_vlm, enable_ocr=enable_ocr)
    if suffix == ".docx":
        if profile == "legacy":
            return _parse_structured(path, enable_vlm=enable_vlm, enable_ocr=enable_ocr)
        return _parse_docx(path, enable_vlm=enable_vlm, enable_ocr=enable_ocr)

    if profile == "legacy":
        from file_agent.parsers import legacy

        parsers = {
            ".md": legacy.MarkdownParser,
            ".txt": legacy.TXTParser,
            ".html": legacy.HTMLParser,
            ".htm": legacy.HTMLParser,
            ".xlsx": legacy.XLSXParser,
            ".pptx": legacy.PPTXParser,
        }
    else:
        parsers = {
            ".md": MarkdownParser,
            ".txt": TXTParser,
            ".html": HTMLParser,
            ".htm": HTMLParser,
            ".xlsx": XLSXParser,
            ".xlsm": XLSXParser,
            ".pptx": PPTXParser,
        }

    parser_class = parsers.get(suffix)
    if parser_class is None:
        raise ValueError(f"Unsupported file type: {suffix or '<no extension>'}")
    document = parser_class().parse(path)
    if enable_vlm is not False:
        _enhance_with_vlm(document, path, forced=enable_vlm is True)
    return document


def _parse_docx(path: Path, enable_vlm: bool | None, enable_ocr: OcrMode) -> Document:
    """DOCX via python-docx (paragraph-faithful); Docling remains the fallback."""
    try:
        document = DOCXParser().parse(path)
    except Exception:
        logger.warning(
            "python-docx failed to parse %s; falling back to Docling.", path.name, exc_info=True
        )
        return _parse_structured(path, enable_vlm=enable_vlm, enable_ocr=enable_ocr)
    if enable_vlm is not False:
        _enhance_with_vlm(document, path, forced=enable_vlm is True)
    return document


def _parse_structured(path: Path, enable_vlm: bool | None, enable_ocr: OcrMode) -> Document:
    do_ocr, ocr_full_page, analysis = _resolve_ocr_policy(path, enable_ocr)
    ocr_pages = _pages_to_ocr(path, analysis, enable_ocr) if do_ocr else []
    vlm_ocr = _vlm_ocr_client() if do_ocr and ocr_pages else None

    # Make the decision observable: without this there is no way to tell whether
    # OCR ran, since a document with a full text layer never starts an engine.
    if do_ocr:
        logger.info(
            "Parsing %s with OCR via %s (%s of %s pages need it: %s)",
            path.name,
            "vlm" if vlm_ocr else resolve_ocr_engine(),
            len(ocr_pages),
            len(analysis.pages) if analysis else "?",
            ocr_pages or "forced",
        )
    else:
        logger.info("Parsing %s without OCR (text layer present on every page)", path.name)

    transcribed = {}
    if vlm_ocr is not None:
        try:
            transcribed = vlm_ocr.transcribe(path, ocr_pages)
        except Exception:
            logger.warning(
                "VLM OCR unavailable for %s; falling back to the classic OCR engine.",
                path.name,
                exc_info=True,
            )
            transcribed = {}

    # With a VLM transcript in hand Docling only needs the text layer; without
    # one it runs the classic OCR engine on the bitmap pages as before.
    docling_ocr = do_ocr and not transcribed
    try:
        parser = DoclingParser(do_ocr=docling_ocr, ocr_full_page=ocr_full_page and docling_ocr)
        document = parser.parse(path)
    except Exception:
        logger.warning(
            "Docling failed to parse %s; falling back to the PyMuPDF parser.",
            path.name,
            exc_info=True,
        )
        if path.suffix.lower() == ".pdf":
            document = PDFParser().parse(path)
            if transcribed:
                document.blocks = merge_ocr_blocks(document.blocks, transcribed)
            return document
        raise

    if transcribed:
        document.blocks = merge_ocr_blocks(document.blocks, transcribed)
        document.metadata["parsing_method"] = "docling+vlm_ocr"
        document.metadata["ocr_engine"] = "vlm"
        document.metadata["vlm_ocr_pages"] = sorted(transcribed)
        document.build_table_of_contents()

    if analysis is not None:
        document.metadata["page_analysis"] = analysis.summary()

    if enable_vlm is not False:
        _enhance_with_vlm(document, path, forced=enable_vlm is True)

    return document


def resolve_ocr_engine() -> str:
    return (os.getenv("OCR_ENGINE") or DEFAULT_OCR_ENGINE).strip().lower()


def _vlm_ocr_client() -> VLMPageOCR | None:
    if resolve_ocr_engine() != "vlm":
        return None
    client = create_vlm_client()
    if client is None:
        logger.info("OCR_ENGINE=vlm but no VLM endpoint is configured; using EasyOCR instead.")
        return None
    return VLMPageOCR(client)


def _pages_to_ocr(path: Path, analysis, enable_ocr: OcrMode) -> list[int]:
    if path.suffix.lower() != ".pdf":
        return []
    if enable_ocr == "on" or analysis is None:
        try:
            import fitz

            with fitz.open(str(path)) as pdf:
                return list(range(1, len(pdf) + 1))
        except Exception:  # pragma: no cover
            return []
    return list(analysis.ocr_page_numbers)


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


def _enhance_with_vlm(document: Document, path: Path, forced: bool) -> None:
    has_figures = any(
        block.block_type in (BlockType.FIGURE, BlockType.IMAGE)
        and (block.image_bytes is not None or (block.bbox and path.suffix.lower() == ".pdf"))
        for block in document.blocks
    )
    if not has_figures:
        return

    vlm_client = create_vlm_client()
    if vlm_client is None:
        if forced:
            logger.warning(
                "VLM requested but no backend configured; set VLM_BACKEND to "
                "'llm' (the chat model endpoint), 'openai' (VLM_BASE_URL) or 'smolvlm'."
            )
        return

    logger.info("VLM enhancement: describing figures in %s", path.name)
    DocumentEnhancer(vlm_client=vlm_client).enhance(document, path)
