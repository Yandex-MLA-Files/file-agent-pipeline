"""Structured XLSX parser built on openpyxl.

Spreadsheets are read as *tables*, not as text dumps: every sheet becomes a
``HEADING`` block (the sheet name) followed by one ``TABLE`` block per
rectangular data region rendered as Markdown with a real header row. Blank
rows/columns split a sheet into several tables (dashboards often stack a note,
a header block and the data on one sheet), merged cells are filled down/right
so every row is self-describing, formulas yield their cached values, numbers
lose spurious ``.0`` and dates are ISO formatted. Notes above a table (single
cells with prose) become ``TEXT`` blocks so they stay attached to the sheet.

Very large sheets are truncated at ``MAX_ROWS_PER_SHEET`` rows (recorded in
metadata) — beyond that a retrieval index is the wrong tool anyway.
"""

import logging
from pathlib import Path
from typing import Any

from openpyxl import load_workbook

from file_agent.document import BlockType, Document
from file_agent.parsers.base import BaseParser
from file_agent.parsers.common import BlockFactory, format_cell, table_to_markdown
from file_agent.telemetry import tracer

logger = logging.getLogger(__name__)

MAX_ROWS_PER_SHEET = 5000
MAX_COLUMNS = 60
# A row whose non-empty cells are all long free text (and few) is a note, not
# a table row.
_NOTE_MIN_CHARS = 25


class XLSXParser(BaseParser):
    def parse(self, file_path: Path) -> Document:
        path = Path(file_path)
        with tracer.start_as_current_span("file_agent.xlsx_parse") as span:
            span.set_attribute("file_agent.file_name", path.name)

            factory = BlockFactory(source_file=path.name)
            workbook = load_workbook(str(path), read_only=False, data_only=True)
            table_count = 0
            try:
                sheets = workbook.worksheets
                for sheet_index, sheet in enumerate(sheets, start=1):
                    grid, truncated = _read_grid(sheet)
                    if not grid:
                        continue
                    factory.heading(
                        f"Sheet: {sheet.title}",
                        1,
                        page_number=sheet_index,
                        metadata={"sheet_name": sheet.title, "sheet_index": sheet_index},
                    )
                    for region in _split_regions(grid):
                        if _is_note(region):
                            text = "\n".join(
                                " ".join(cell for cell in row if cell) for row in region
                            )
                            factory.add(
                                text,
                                BlockType.TEXT,
                                {"sheet_name": sheet.title},
                                page_number=sheet_index,
                            )
                            continue
                        markdown = table_to_markdown(region, header=_has_header(region))
                        if not markdown:
                            continue
                        table_count += 1
                        factory.add(
                            markdown,
                            BlockType.TABLE,
                            {
                                "sheet_name": sheet.title,
                                "table_index": table_count,
                                "row_count": len(region) - (1 if _has_header(region) else 0),
                                "truncated": truncated,
                            },
                            page_number=sheet_index,
                        )
            finally:
                workbook.close()

            if not factory.blocks:
                factory.add(
                    "",
                    BlockType.TEXT,
                    {"warning": "empty workbook"},
                    page_number=1,
                    skip_empty=False,
                )

            document = Document(
                file_name=path.name,
                file_type="xlsx",
                blocks=factory.blocks,
                metadata={
                    "parsing_method": "openpyxl",
                    "sheet_count": len(workbook.worksheets),
                    "table_count": table_count,
                },
            )
            document.build_table_of_contents()
            span.set_attribute("file_agent.block_count", len(document.blocks))
            logger.info("Parsed %s into %d block(s)", path.name, len(document.blocks))
            return document


def _read_grid(sheet: Any) -> tuple[list[list[str]], bool]:
    """Read the used range as formatted strings, resolving merged cells."""
    max_row = min(sheet.max_row or 0, MAX_ROWS_PER_SHEET)
    max_col = min(sheet.max_column or 0, MAX_COLUMNS)
    truncated = bool((sheet.max_row or 0) > MAX_ROWS_PER_SHEET)
    if max_row == 0 or max_col == 0:
        return [], truncated

    grid = [
        [format_cell(cell) for cell in row]
        for row in sheet.iter_rows(
            min_row=1, max_row=max_row, min_col=1, max_col=max_col, values_only=True
        )
    ]

    # Merged ranges only carry the value in their top-left cell; propagate it
    # so each row still names its category/date after the split into chunks.
    try:
        merged_ranges = list(sheet.merged_cells.ranges)
    except Exception:  # pragma: no cover - read-only sheets
        merged_ranges = []
    for merged in merged_ranges:
        min_r, min_c, max_r, max_c = merged.min_row, merged.min_col, merged.max_row, merged.max_col
        if min_r > max_row or min_c > max_col:
            continue
        value = grid[min_r - 1][min_c - 1]
        if not value:
            continue
        for r in range(min_r, min(max_r, max_row) + 1):
            for c in range(min_c, min(max_c, max_col) + 1):
                if not grid[r - 1][c - 1]:
                    grid[r - 1][c - 1] = value

    # Trim trailing empty rows/columns.
    while grid and not any(grid[-1]):
        grid.pop()
    return grid, truncated


def _split_regions(grid: list[list[str]]) -> list[list[list[str]]]:
    """Split a sheet on blank rows into independent tables/notes."""
    regions: list[list[list[str]]] = []
    current: list[list[str]] = []
    for row in grid:
        if any(cell for cell in row):
            current.append(row)
        elif current:
            regions.append(current)
            current = []
    if current:
        regions.append(current)
    return regions


def _is_note(region: list[list[str]]) -> bool:
    if len(region) > 3:
        return False
    for row in region:
        cells = [cell for cell in row if cell]
        if len(cells) != 1 or len(cells[0]) < _NOTE_MIN_CHARS:
            return False
    return True


def _has_header(region: list[list[str]]) -> bool:
    """A header row is mostly text while the following rows carry numbers."""
    if len(region) < 2:
        return True
    first = [cell for cell in region[0] if cell]
    second = [cell for cell in region[1] if cell]
    if not first:
        return False
    first_numeric = sum(1 for cell in first if _is_number(cell))
    second_numeric = sum(1 for cell in second if _is_number(cell))
    if first_numeric == 0:
        return True
    return first_numeric / len(first) < 0.5 and second_numeric > first_numeric


def _is_number(cell: str) -> bool:
    try:
        float(cell.replace(",", ".").replace(" ", ""))
        return True
    except ValueError:
        return False
