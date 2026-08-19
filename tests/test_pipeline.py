import fitz
import pytest
from openpyxl import Workbook
from pptx import Presentation

from file_agent.pipeline import parse_file


def create_pdf(file_path, text):
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), text)
    document.save(file_path)
    document.close()


def create_xlsx(file_path):
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Data"
    sheet.append(["Name", "Age"])
    sheet.append(["Alice", 20])
    workbook.save(file_path)
    workbook.close()


def create_pptx(file_path):
    presentation = Presentation()
    slide = presentation.slides.add_slide(presentation.slide_layouts[1])
    slide.shapes.title.text = "Project Overview"
    slide.placeholders[1].text = "PowerPoint content"
    presentation.save(file_path)


def test_parse_file_uses_markdown_parser(tmp_path):
    file_path = tmp_path / "example.md"
    file_path.write_text("Hello from pipeline", encoding="utf-8")

    document = parse_file(file_path)

    assert document.file_name == "example.md"
    assert document.file_type == "md"
    assert document.blocks[0].text == "Hello from pipeline"


def test_parse_file_uses_txt_parser(tmp_path):
    file_path = tmp_path / "example.txt"
    file_path.write_text("Hello from TXT", encoding="utf-8")

    document = parse_file(file_path)

    assert document.file_name == "example.txt"
    assert document.file_type == "txt"
    assert document.blocks[0].text == "Hello from TXT"


def test_parse_file_parses_pdf(tmp_path):
    file_path = tmp_path / "example.pdf"
    create_pdf(file_path, "Hello from PDF")

    # PDFs go through Docling by default; if Docling is unavailable the pipeline
    # gracefully falls back to the PyMuPDF parser. Both paths must recover the text.
    document = parse_file(file_path)

    assert document.file_name == "example.pdf"
    assert document.file_type == "pdf"
    assert document.blocks
    assert any("Hello from PDF" in block.text for block in document.blocks)


def test_parse_file_preserves_exact_pdf_file_and_native_metadata(tmp_path):
    file_path = tmp_path / "metadata.pdf"
    pdf = fitz.open()
    first_page = pdf.new_page()
    first_page.insert_text((72, 72), "Metadata test")
    pdf.new_page()
    pdf.new_page()
    pdf.set_metadata(
        {
            "title": "DocBench Annual Report",
            "author": "Ada Lovelace",
            "subject": "Benchmark metadata",
        }
    )
    pdf.save(file_path)
    pdf.close()

    document = parse_file(file_path, enable_vlm=False, enable_ocr="off")

    assert document.metadata["total_pages"] == 3
    assert document.metadata["file_size_bytes"] == file_path.stat().st_size
    assert document.metadata["native_pdf_word_counts_by_page"] == [2, 0, 0]
    assert document.metadata["native_pdf_word_count"] == 2
    assert document.metadata["pdf_metadata"]["title"] == "DocBench Annual Report"
    assert document.metadata["pdf_metadata"]["author"] == "Ada Lovelace"
    assert document.metadata["pdf_metadata"]["subject"] == "Benchmark metadata"


def test_parse_file_uses_html_parser(tmp_path):
    file_path = tmp_path / "example.html"
    file_path.write_text("<p>Hello from HTML</p>", encoding="utf-8")

    document = parse_file(file_path)

    assert document.file_name == "example.html"
    assert document.file_type == "html"
    assert len(document.blocks) == 1
    assert document.blocks[0].type == "html_text"
    assert document.blocks[0].text == "Hello from HTML"


def test_parse_file_uses_xlsx_parser(tmp_path):
    file_path = tmp_path / "example.xlsx"
    create_xlsx(file_path)

    document = parse_file(file_path)

    assert document.file_name == "example.xlsx"
    assert document.file_type == "xlsx"
    assert len(document.blocks) == 1
    assert document.blocks[0].type == "xlsx_sheet"
    assert "Alice\t20" in document.blocks[0].text


def test_parse_file_uses_pptx_parser(tmp_path):
    file_path = tmp_path / "example.pptx"
    create_pptx(file_path)

    document = parse_file(file_path)

    assert document.file_name == "example.pptx"
    assert document.file_type == "pptx"
    assert len(document.blocks) == 1
    assert document.blocks[0].type == "pptx_slide"
    assert "Project Overview" in document.blocks[0].text
    assert "PowerPoint content" in document.blocks[0].text


def test_parse_file_rejects_unsupported_extension(tmp_path):
    file_path = tmp_path / "example.csv"
    file_path.write_text("Unsupported", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported file type"):
        parse_file(file_path)
