import io

import fitz
from PIL import Image

from file_agent.parsers.routing import (
    IMAGE_COVERAGE_THRESHOLD,
    _decide_ocr,
    analyze_pdf,
)


def _make_text_pdf(path, text="This is a normal digital page with a real text layer. " * 5):
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), text)
    doc.save(path)
    doc.close()


def _make_scanned_pdf(path):
    """A page whose only content is a full-page raster image and no text layer."""
    doc = fitz.open()
    page = doc.new_page()
    image = Image.new("RGB", (600, 800), color=(128, 128, 128))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    page.insert_image(page.rect, stream=buffer.getvalue())
    doc.save(path)
    doc.close()


def test_digital_pdf_does_not_need_ocr(tmp_path):
    path = tmp_path / "digital.pdf"
    _make_text_pdf(path)

    analysis = analyze_pdf(path)

    assert len(analysis.pages) == 1
    assert analysis.needs_ocr is False
    assert analysis.ocr_page_numbers == []


def test_scanned_pdf_needs_ocr(tmp_path):
    path = tmp_path / "scan.pdf"
    _make_scanned_pdf(path)

    analysis = analyze_pdf(path)

    assert analysis.needs_ocr is True
    assert analysis.ocr_page_numbers == [1]
    assert analysis.scanned_ratio == 1.0
    assert analysis.pages[0].image_area_ratio >= IMAGE_COVERAGE_THRESHOLD


def test_summary_is_serializable(tmp_path):
    path = tmp_path / "digital.pdf"
    _make_text_pdf(path)

    summary = analyze_pdf(path).summary()

    assert summary["total_pages"] == 1
    assert summary["needs_ocr"] is False
    assert summary["ocr_page_numbers"] == []


def test_decide_ocr_rules():
    # No text but images present -> scan.
    needs_ocr, _ = _decide_ocr(char_count=0, image_count=1, image_area_ratio=0.9)
    assert needs_ocr is True

    # Sparse text but large image coverage -> image-only page.
    needs_ocr, _ = _decide_ocr(char_count=30, image_count=1, image_area_ratio=0.8)
    assert needs_ocr is True

    # Plenty of text -> no OCR even if an image is present.
    needs_ocr, _ = _decide_ocr(char_count=2000, image_count=1, image_area_ratio=0.3)
    assert needs_ocr is False
