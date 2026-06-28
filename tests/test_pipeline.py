import pytest
import fitz

from file_agent.pipeline import parse_file


def create_pdf(file_path, text):
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), text)
    document.save(file_path)
    document.close()


def test_parse_file_uses_markdown_parser(tmp_path):
    file_path = tmp_path / "example.md"
    file_path.write_text("Hello from pipeline", encoding="utf-8")

    document = parse_file(file_path)

    assert document.file_name == "example.md"
    assert document.file_type == "md"
    assert document.blocks[0].text == "Hello from pipeline"


def test_parse_file_uses_pdf_parser(tmp_path):
    file_path = tmp_path / "example.pdf"
    create_pdf(file_path, "Hello from PDF")

    document = parse_file(file_path)

    assert document.file_name == "example.pdf"
    assert document.file_type == "pdf"
    assert len(document.blocks) == 1
    assert "Hello from PDF" in document.blocks[0].text
    assert document.blocks[0].metadata["page_number"] == 1


def test_parse_file_uses_html_parser(tmp_path):
    file_path = tmp_path / "example.html"
    file_path.write_text("<p>Hello from HTML</p>", encoding="utf-8")

    document = parse_file(file_path)

    assert document.file_name == "example.html"
    assert document.file_type == "html"
    assert len(document.blocks) == 1
    assert document.blocks[0].type == "html_text"
    assert document.blocks[0].text == "Hello from HTML"


def test_parse_file_rejects_unsupported_extension(tmp_path):
    file_path = tmp_path / "example.txt"
    file_path.write_text("Unsupported", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported file type"):
        parse_file(file_path)
