from docling.datamodel.pipeline_options import PdfPipelineOptions, RapidOcrOptions

from file_agent.parsers.docling_parser import DoclingParser


def test_configure_ocr_prefers_rapidocr():
    options = PdfPipelineOptions()

    DoclingParser._configure_ocr(options, ocr_full_page=True)

    # RapidOCR is bundled with its models and works offline, so it must be picked.
    assert isinstance(options.ocr_options, RapidOcrOptions)
    assert options.ocr_options.force_full_page_ocr is True


def test_docling_parser_builds_with_ocr_enabled():
    # Constructing the converter with OCR on must not require running any model.
    parser = DoclingParser(do_ocr=True, ocr_full_page=False)

    assert parser.do_ocr is True
    assert parser._converter is not None
