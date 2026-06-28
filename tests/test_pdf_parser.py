import fitz

from file_agent.parsers.pdf_parser import PDFParser


def create_pdf(file_path, page_texts):
    document = fitz.open()
    for text in page_texts:
        page = document.new_page()
        page.insert_text((72, 72), text)
    document.save(file_path)
    document.close()


def test_pdf_parser_creates_block_for_each_page(tmp_path):
    file_path = tmp_path / "sample.pdf"
    create_pdf(file_path, ["First page", "Second page"])

    document = PDFParser().parse(file_path)

    assert document.file_name == "sample.pdf"
    assert document.file_type == "pdf"
    assert len(document.blocks) == 2
    assert document.blocks[0].id == "page-1"
    assert "First page" in document.blocks[0].text
    assert document.blocks[0].type == "pdf_page"
    assert document.blocks[0].metadata == {
        "page_number": 1,
        "source_file": "sample.pdf",
    }
    assert document.blocks[1].id == "page-2"
    assert "Second page" in document.blocks[1].text
    assert document.blocks[1].metadata["page_number"] == 2
