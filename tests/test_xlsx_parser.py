from openpyxl import Workbook

from file_agent.parsers.xlsx_parser import XLSXParser


def create_workbook(file_path):
    workbook = Workbook()
    first_sheet = workbook.active
    first_sheet.title = "People"
    first_sheet.append(["Name", "Age", "City"])
    first_sheet.append(["Alice", 20, "Moscow"])
    first_sheet.append(["Bob", 22, "Kazan"])

    second_sheet = workbook.create_sheet("Summary")
    second_sheet.append(["Metric", "Value"])
    second_sheet.append(["Rows", 2])

    workbook.save(file_path)
    workbook.close()


def test_xlsx_parser_creates_block_for_each_sheet(tmp_path):
    file_path = tmp_path / "sample.xlsx"
    create_workbook(file_path)

    document = XLSXParser().parse(file_path)

    assert document.file_name == "sample.xlsx"
    assert document.file_type == "xlsx"
    assert len(document.blocks) == 2
    assert document.blocks[0].type == "xlsx_sheet"
    assert document.blocks[1].type == "xlsx_sheet"


def test_xlsx_parser_extracts_sheet_text(tmp_path):
    file_path = tmp_path / "sample.xlsx"
    create_workbook(file_path)

    document = XLSXParser().parse(file_path)

    assert "Name\tAge\tCity" in document.blocks[0].text
    assert "Alice\t20\tMoscow" in document.blocks[0].text
    assert "Bob\t22\tKazan" in document.blocks[0].text
    assert "Metric\tValue" in document.blocks[1].text


def test_xlsx_parser_preserves_sheet_metadata(tmp_path):
    file_path = tmp_path / "sample.xlsx"
    create_workbook(file_path)

    document = XLSXParser().parse(file_path)

    assert document.blocks[0].metadata == {
        "source_file": "sample.xlsx",
        "sheet_name": "People",
        "max_row": 3,
        "max_column": 3,
    }
    assert document.blocks[1].metadata == {
        "source_file": "sample.xlsx",
        "sheet_name": "Summary",
        "max_row": 2,
        "max_column": 2,
    }
