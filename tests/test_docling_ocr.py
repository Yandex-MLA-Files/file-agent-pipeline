from docling.datamodel.pipeline_options import (
    EasyOcrOptions,
    PdfPipelineOptions,
    RapidOcrOptions,
)

from file_agent.parsers.docling_parser import DoclingParser


def test_configure_ocr_prefers_easyocr_with_cyrillic(monkeypatch):
    monkeypatch.delenv("OCR_LANGS", raising=False)
    monkeypatch.delenv("OCR_ENGINE", raising=False)
    options = PdfPipelineOptions()

    DoclingParser._configure_ocr(options, ocr_full_page=True)

    # EasyOCR reads Cyrillic (the documents are frequently Russian), so it wins.
    assert isinstance(options.ocr_options, EasyOcrOptions)
    assert "ru" in options.ocr_options.lang and "en" in options.ocr_options.lang
    assert options.ocr_options.force_full_page_ocr is True


def test_ocr_engine_can_be_switched_to_rapidocr(monkeypatch):
    monkeypatch.setenv("OCR_ENGINE", "rapidocr")
    options = PdfPipelineOptions()

    DoclingParser._configure_ocr(options, ocr_full_page=False)

    assert isinstance(options.ocr_options, RapidOcrOptions)


def test_ocr_languages_come_from_env(monkeypatch):
    monkeypatch.setenv("OCR_LANGS", "de, fr")

    assert DoclingParser._ocr_languages() == ["de", "fr"]


def test_docling_parser_builds_with_ocr_enabled():
    # Constructing the converter with OCR on must not require running any model.
    parser = DoclingParser(do_ocr=True, ocr_full_page=False)

    assert parser.do_ocr is True
    assert parser._converter is not None
