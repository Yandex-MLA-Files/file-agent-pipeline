"""Tabular views of parsed documents for the agent's ``query_table`` tool.

Spreadsheets are parsed into one block per sheet (tab-separated rows) and
Docling renders tables inside PDF/DOCX files as Markdown; both are turned back
into pandas DataFrames here so the model can aggregate, filter and join them
with code instead of reading rows and estimating.
"""

import csv
import re
from dataclasses import dataclass
from typing import Any

from file_agent.document import Block, BlockType, Document

# Rows in a header position that are mostly empty are not headers: spreadsheet
# exports often start with a title line above the real column names.
_MIN_NAMED_HEADER_SHARE = 0.5
_MARKDOWN_SEPARATOR = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$")


@dataclass
class TableView:
    """One DataFrame plus where it came from."""

    name: str
    frame: Any  # pandas.DataFrame
    block: Block
    origin: str  # "sheet" or "table"

    @property
    def metadata(self) -> dict[str, Any]:
        metadata: dict[str, Any] = {"source_file": self.block.metadata.get("source_file", "")}
        if self.origin == "sheet":
            metadata["sheet_name"] = self.name
        else:
            metadata["table"] = self.name
        if self.block.page_number is not None:
            metadata["page_number"] = self.block.page_number
        return metadata


def document_tables(document: Document, header_row: int = 0) -> list[TableView]:
    """Every table of a document as a DataFrame, sheets first."""
    import pandas as pd

    views: list[TableView] = []
    table_index = 0
    for block in document.blocks:
        if block.type == "xlsx_sheet" or block.metadata.get("sheet_name"):
            frame = _sheet_frame(pd, block.text, header_row)
            if frame is not None:
                name = str(block.metadata.get("sheet_name") or f"sheet{len(views) + 1}")
                views.append(TableView(name=name, frame=frame, block=block, origin="sheet"))
            continue
        if block.block_type == BlockType.TABLE or _looks_like_markdown_table(block.text):
            frame = _markdown_frame(pd, block.text, header_row)
            if frame is not None:
                table_index += 1
                views.append(
                    TableView(name=f"table{table_index}", frame=frame, block=block, origin="table")
                )
    return views


def describe_frame(frame: Any, max_columns: int = 12) -> str:
    """``300 rows x 5 columns: id, product, stock, price, category``."""
    rows, columns = frame.shape
    names = [str(column) for column in frame.columns[:max_columns]]
    if columns > max_columns:
        names.append(f"... {columns - max_columns} more")
    return f"{rows} rows x {columns} columns: {', '.join(names)}"


def preview_frame(frame: Any, rows: int = 5) -> str:
    import pandas as pd

    with pd.option_context("display.max_columns", 40, "display.width", 200):
        dtypes = ", ".join(f"{column}={dtype}" for column, dtype in frame.dtypes.items())
        return f"{describe_frame(frame)}\ndtypes: {dtypes}\n{frame.head(rows).to_string()}"


def _sheet_frame(pd: Any, text: str, header_row: int) -> Any | None:
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        return None
    rows = list(csv.reader(lines, delimiter="\t", quoting=csv.QUOTE_NONE))
    return _frame_from_rows(pd, rows, header_row)


def _markdown_frame(pd: Any, text: str, header_row: int) -> Any | None:
    rows: list[list[str]] = []
    for line in text.splitlines():
        if "|" not in line or _MARKDOWN_SEPARATOR.match(line):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        rows.append(cells)
    if len(rows) < 2:
        return None
    return _frame_from_rows(pd, rows, header_row)


def _frame_from_rows(pd: Any, rows: list[list[str]], header_row: int) -> Any | None:
    if not rows:
        return None
    header_row = max(0, min(header_row, len(rows) - 1))
    if header_row == 0:
        header_row = _detect_header_row(rows)
    header = rows[header_row]
    width = max(len(row) for row in rows)
    header = _unique_names([*header, *[""] * (width - len(header))])
    body = [[*row, *[""] * (width - len(row))] for row in rows[header_row + 1 :]]
    frame = pd.DataFrame(body, columns=header)
    frame = frame.replace({"": None})
    # Numbers arrive as text from both sources; convert what converts.
    for column in frame.columns:
        converted = pd.to_numeric(frame[column], errors="coerce")
        if converted.notna().sum() and converted.notna().sum() >= frame[column].notna().sum():
            frame[column] = converted
    return frame


def _detect_header_row(rows: list[list[str]], lookahead: int = 5) -> int:
    """Skip title lines above the real column names."""
    width = max(len(row) for row in rows)
    for index, row in enumerate(rows[:lookahead]):
        named = sum(1 for cell in row if str(cell).strip())
        if width and named / width >= _MIN_NAMED_HEADER_SHARE:
            return index
    return 0


def _unique_names(names: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    result: list[str] = []
    for index, raw in enumerate(names):
        name = str(raw).strip() or f"col{index + 1}"
        if name in seen:
            seen[name] += 1
            name = f"{name}.{seen[name]}"
        else:
            seen[name] = 0
        result.append(name)
    return result


def _looks_like_markdown_table(text: str) -> bool:
    lines = [line for line in text.splitlines() if line.strip()]
    if len(lines) < 3:
        return False
    return any(_MARKDOWN_SEPARATOR.match(line) for line in lines[:3]) and lines[0].count("|") >= 2


def parse_header_row(value: Any) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0
