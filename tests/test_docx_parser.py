from docx import Document as create_docx_document

from file_agent.parsers.docx_parser import DOCXParser


def create_docx(file_path):
    document = create_docx_document()
    document.add_paragraph("Introduction")

    table = document.add_table(rows=2, cols=2)
    table.cell(0, 0).text = "Name"
    table.cell(0, 1).text = "Value"
    table.cell(1, 0).text = "Accuracy"
    table.cell(1, 1).text = "0.85"

    document.add_paragraph("")
    document.add_paragraph("Conclusion")
    document.save(file_path)


def test_docx_parser_preserves_paragraph_and_table_order(tmp_path):
    file_path = tmp_path / "report.docx"
    create_docx(file_path)

    document = DOCXParser().parse(file_path)

    assert document.file_name == "report.docx"
    assert document.file_type == "docx"
    assert [block.type for block in document.blocks] == [
        "docx_paragraph",
        "docx_table",
        "docx_paragraph",
    ]
    assert [block.text for block in document.blocks] == [
        "Introduction",
        "Name\tValue\nAccuracy\t0.85",
        "Conclusion",
    ]


def test_docx_parser_preserves_source_coordinates(tmp_path):
    file_path = tmp_path / "report.docx"
    create_docx(file_path)

    document = DOCXParser().parse(file_path)

    first_paragraph, table, last_paragraph = document.blocks
    assert first_paragraph.id == "paragraph-1"
    assert first_paragraph.metadata == {
        "source_file": "report.docx",
        "paragraph_number": 1,
    }
    assert table.id == "table-1"
    assert table.metadata == {
        "source_file": "report.docx",
        "table_number": 1,
        "rows_count": 2,
        "columns_count": 2,
    }
    assert last_paragraph.id == "paragraph-3"
    assert last_paragraph.metadata == {
        "source_file": "report.docx",
        "paragraph_number": 3,
    }
