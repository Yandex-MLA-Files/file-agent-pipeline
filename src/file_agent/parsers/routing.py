"""Heuristics that let the pipeline decide *by itself* when a PDF page needs OCR.

Born-digital PDFs already carry an extractable text layer, so running OCR on
them only wastes time and can corrupt clean text. Scanned or image-only pages,
on the other hand, have little or no text and must be OCR'd to be usable.

This module inspects a PDF locally (PyMuPDF only, no network, no ML models) and
produces an explainable per-page decision plus a document-level summary. The
pipeline uses it to turn Docling's OCR on only when it is actually required.
"""

from dataclasses import dataclass
from pathlib import Path

import fitz  # PyMuPDF

# A page with fewer characters than this has essentially no usable text layer.
MIN_CHARS_FOR_TEXT_LAYER = 100
# A page is treated as scanned when almost no text is present at all.
SCANNED_CHAR_THRESHOLD = 20
# Fraction of the page area that must be covered by raster images before a
# low-text page is considered an image/scan rather than a sparse text page.
IMAGE_COVERAGE_THRESHOLD = 0.45


@dataclass
class PageAnalysis:
    page_number: int  # 1-indexed
    char_count: int
    image_count: int
    image_area_ratio: float  # fraction of page area covered by raster images
    drawing_count: int  # number of vector drawings (diagrams, schemes, charts)
    needs_ocr: bool
    reason: str

    @property
    def likely_diagram(self) -> bool:
        """Vector-drawing-heavy page (flowchart/scheme) that a VLM can explain."""
        return self.drawing_count >= 5 or (
            self.image_count > 0 and self.char_count < MIN_CHARS_FOR_TEXT_LAYER
        )


@dataclass
class PdfAnalysis:
    pages: list[PageAnalysis]

    @property
    def needs_ocr(self) -> bool:
        """Whether *any* page lacks a usable text layer and should be OCR'd."""
        return any(page.needs_ocr for page in self.pages)

    @property
    def ocr_page_numbers(self) -> list[int]:
        return [page.page_number for page in self.pages if page.needs_ocr]

    @property
    def scanned_ratio(self) -> float:
        if not self.pages:
            return 0.0
        return len(self.ocr_page_numbers) / len(self.pages)

    def summary(self) -> dict[str, object]:
        """Compact, JSON-serializable summary for document metadata."""
        return {
            "total_pages": len(self.pages),
            "needs_ocr": self.needs_ocr,
            "ocr_page_numbers": self.ocr_page_numbers,
            "scanned_ratio": round(self.scanned_ratio, 3),
        }


def _analyze_page(page: fitz.Page, page_number: int) -> PageAnalysis:
    text = page.get_text("text") or ""
    char_count = len(text.strip())

    page_area = abs(page.rect.width * page.rect.height) or 1.0
    image_area = 0.0
    image_count = 0
    try:
        for info in page.get_image_info():
            bbox = info.get("bbox")
            if not bbox:
                continue
            image_count += 1
            rect = fitz.Rect(bbox)
            image_area += abs(rect.width * rect.height)
    except Exception:
        # Older PyMuPDF builds may not expose get_image_info(); fall back to a count.
        image_count = len(page.get_images(full=True))

    image_area_ratio = min(image_area / page_area, 1.0)

    try:
        drawing_count = len(page.get_drawings())
    except Exception:
        drawing_count = 0

    needs_ocr, reason = _decide_ocr(char_count, image_count, image_area_ratio)

    return PageAnalysis(
        page_number=page_number,
        char_count=char_count,
        image_count=image_count,
        image_area_ratio=round(image_area_ratio, 3),
        drawing_count=drawing_count,
        needs_ocr=needs_ocr,
        reason=reason,
    )


def _decide_ocr(char_count: int, image_count: int, image_area_ratio: float) -> tuple[bool, str]:
    """Return (needs_ocr, human-readable reason) for a single page."""
    if char_count < SCANNED_CHAR_THRESHOLD and image_count > 0:
        return True, "almost no text layer but page contains images (likely a scan)"
    if char_count < MIN_CHARS_FOR_TEXT_LAYER and image_area_ratio >= IMAGE_COVERAGE_THRESHOLD:
        return True, "sparse text and large image coverage (image-only page)"
    return False, "usable text layer present"


def analyze_pdf(pdf_path: Path) -> PdfAnalysis:
    """Analyze every page of a PDF and decide which pages require OCR."""
    pages: list[PageAnalysis] = []
    with fitz.open(str(pdf_path)) as pdf:
        for index, page in enumerate(pdf, start=1):
            pages.append(_analyze_page(page, index))
    return PdfAnalysis(pages=pages)
