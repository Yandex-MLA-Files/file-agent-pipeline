from pathlib import Path
from typing import Any

from openpyxl import load_workbook

from file_agent.document import Block, Document
from file_agent.parsers.base import BaseParser


class XLSXParser(BaseParser):
    def parse(self, file_path: Path) -> Document:
        path = Path(file_path)
        blocks: list[Block] = []

        workbook = load_workbook(path, read_only=True, data_only=True)
        try:
            for sheet_index, sheet in enumerate(workbook.worksheets, start=1):
                blocks.append(
                    Block(
                        id=f"sheet-{sheet_index}",
                        text=_sheet_to_text(sheet),
                        type="xlsx_sheet",
                        metadata={
                            "source_file": path.name,
                            "sheet_name": sheet.title,
                            "max_row": sheet.max_row,
                            "max_column": sheet.max_column,
                        },
                    )
                )
        finally:
            workbook.close()

        return Document(
            file_name=path.name,
            file_type="xlsx",
            blocks=blocks,
        )


def _sheet_to_text(sheet: Any) -> str:
    rows: list[str] = []

    for row in sheet.iter_rows(values_only=True):
        values = ["" if value is None else str(value) for value in row]
        if not any(value.strip() for value in values):
            continue
        rows.append("\t".join(values))

    return "\n".join(rows)
