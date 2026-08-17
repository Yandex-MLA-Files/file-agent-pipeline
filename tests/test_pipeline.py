import fitz
import pytest
from openpyxl import Workbook
from pptx import Presentation

from file_agent.document import BlockType
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


def test_parse_file_uses_html_parser(tmp_path):
    file_path = tmp_path / "example.html"
    file_path.write_text("<p>Hello from HTML</p>", encoding="utf-8")

    document = parse_file(file_path)

    assert document.file_name == "example.html"
    assert document.file_type == "html"
    assert len(document.blocks) == 1
    assert document.blocks[0].block_type == BlockType.TEXT
    assert document.blocks[0].text == "Hello from HTML"
    assert document.metadata["parser_profile"] == "structured"


def test_parse_file_uses_xlsx_parser(tmp_path):
    file_path = tmp_path / "example.xlsx"
    create_xlsx(file_path)

    document = parse_file(file_path)

    assert document.file_name == "example.xlsx"
    assert document.file_type == "xlsx"
    # A workbook overview opens the document, then the sheet and its table.
    assert [b.block_type for b in document.blocks] == [
        BlockType.TEXT,
        BlockType.HEADING,
        BlockType.TABLE,
    ]
    assert document.blocks[0].metadata["workbook_overview"] is True
    assert "листов — 1" in document.blocks[0].text
    assert "| Alice | 20 |" in document.blocks[2].text


def test_parse_file_uses_pptx_parser(tmp_path):
    file_path = tmp_path / "example.pptx"
    create_pptx(file_path)

    document = parse_file(file_path)

    assert document.file_name == "example.pptx"
    assert document.file_type == "pptx"
    assert document.blocks[0].block_type == BlockType.HEADING
    assert document.blocks[0].text == "Project Overview"
    assert any("PowerPoint content" in b.text for b in document.blocks)


def test_parse_file_legacy_profile_uses_flat_parsers(tmp_path, monkeypatch):
    file_path = tmp_path / "example.pptx"
    create_pptx(file_path)
    monkeypatch.setenv("PARSER_PROFILE", "legacy")

    document = parse_file(file_path)

    assert len(document.blocks) == 1
    assert document.blocks[0].type == "pptx_slide"
    assert document.metadata["parser_profile"] == "legacy"


def test_parse_file_rejects_unknown_profile(tmp_path):
    file_path = tmp_path / "example.md"
    file_path.write_text("# hi", encoding="utf-8")

    with pytest.raises(ValueError, match="PARSER_PROFILE"):
        parse_file(file_path, parser_profile="fancy")


def test_parse_file_rejects_unsupported_extension(tmp_path):
    file_path = tmp_path / "example.csv"
    file_path.write_text("Unsupported", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported file type"):
        parse_file(file_path)
