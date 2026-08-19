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
# Share of a column's non-empty cells that must parse as numbers for the column
# to become numeric (the rest turn into NaN).
_MIN_NUMERIC_SHARE = 0.9
_SUBHEADER_TEXT_SHARE = 0.75
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
    width = max(len(row) for row in rows)
    rows = [[*row, *[""] * (width - len(row))] for row in rows]
    header_row = max(0, min(header_row, len(rows) - 1))
    if header_row == 0:
        header_row = _detect_header_row(rows)
    header = list(rows[header_row])
    body_start = header_row + 1
    # A second header row (platform names above "Bullet / Blitz / Rapid", units
    # under measure names) is merged into the column names instead of becoming
    # a text row that turns every numeric column into strings.
    if _is_subheader_row(rows, header_row):
        filled = ""
        merged: list[str] = []
        for top, sub in zip(header, rows[header_row + 1], strict=True):
            top = str(top).strip()
            sub = str(sub).strip()
            if top:
                filled = top
            merged.append(" ".join(part for part in (filled if top or sub else "", sub) if part))
        header = merged
        body_start = header_row + 2
    header = _unique_names(header)
    frame = pd.DataFrame(rows[body_start:], columns=header)
    frame = frame.replace({"": None})
    # Numbers arrive as text from both sources; convert columns that are
    # (almost) entirely numeric, so a stray label does not keep a column textual.
    for column in frame.columns:
        present = frame[column].notna().sum()
        if not present:
            continue
        converted = pd.to_numeric(frame[column], errors="coerce")
        if converted.notna().sum() >= max(1, _MIN_NUMERIC_SHARE * present):
            frame[column] = converted
    return frame


def _is_subheader_row(rows: list[list[str]], header_row: int) -> bool:
    if len(rows) < header_row + 3:
        return False
    candidate = [str(cell).strip() for cell in rows[header_row + 1]]
    following = [str(cell).strip() for cell in rows[header_row + 2]]
    present = [cell for cell in candidate if cell]
    present_next = [cell for cell in following if cell]
    if not present or not present_next:
        return False
    textual = sum(1 for cell in present if not _is_number(cell)) / len(present)
    numeric_next = sum(1 for cell in present_next if _is_number(cell)) / len(present_next)
    # A labels-only row followed by a numbers-only row is a second header line;
    # a row with a name column and a number column is plain data.
    return textual >= _SUBHEADER_TEXT_SHARE and numeric_next >= _SUBHEADER_TEXT_SHARE


def _is_number(cell: str) -> bool:
    try:
        float(cell.replace(" ", "").replace(",", "."))
    except ValueError:
        return False
    return True


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
