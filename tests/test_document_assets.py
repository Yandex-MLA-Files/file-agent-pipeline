import fitz

from file_agent.document_assets import InMemoryDocumentAssetStore
from file_agent.utils.image_extractor import extract_image_from_pdf_bytes


def _pdf_bytes() -> bytes:
    document = fitz.open()
    page = document.new_page(width=300, height=200)
    page.draw_rect(fitz.Rect(50, 40, 150, 140), fill=(0, 0, 1))
    contents = document.tobytes()
    document.close()
    return contents


def test_in_memory_asset_store_resolves_names_case_insensitively():
    store = InMemoryDocumentAssetStore()
    store.put("Report.PDF", b"pdf-bytes")

    assert store.get_bytes("report.pdf") == b"pdf-bytes"


def test_in_memory_asset_store_rejects_duplicate_file_names():
    store = InMemoryDocumentAssetStore()
    store.put("report.pdf", b"first")

    try:
        store.put("REPORT.PDF", b"second")
    except ValueError as exc:
        assert "share this file name" in str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("duplicate source file must be rejected")


def test_extract_image_from_pdf_bytes_renders_crop_and_full_page():
    contents = _pdf_bytes()

    crop = extract_image_from_pdf_bytes(
        contents,
        page_number=1,
        bbox=(50, 40, 150, 140),
    )
    full_page = extract_image_from_pdf_bytes(contents, page_number=1)

    assert crop is not None
    assert crop.size == (200, 200)
    assert full_page is not None
    assert full_page.size == (600, 400)


def test_extract_image_from_pdf_bytes_returns_none_for_invalid_page():
    assert extract_image_from_pdf_bytes(_pdf_bytes(), page_number=2) is None


def test_extract_image_from_pdf_bytes_limits_pixels_during_render():
    image = extract_image_from_pdf_bytes(
        _pdf_bytes(),
        page_number=1,
        max_pixels=30_000,
    )

    assert image is not None
    assert image.width * image.height <= 30_500
