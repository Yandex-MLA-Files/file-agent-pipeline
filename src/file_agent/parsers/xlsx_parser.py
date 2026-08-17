"""Structured XLSX parser built on openpyxl.

Spreadsheets are read as *tables*, not as text dumps: every sheet becomes a
``HEADING`` block (the sheet name) followed by one ``TABLE`` block per
rectangular data region rendered as Markdown with a real header row. Blank
rows/columns split a sheet into several tables (dashboards often stack a note,
a header block and the data on one sheet), merged cells are filled down/right
so every row is self-describing, formulas yield their cached values, numbers
lose spurious ``.0`` and dates are ISO formatted. Notes above a table (single
cells with prose) become ``TEXT`` blocks so they stay attached to the sheet.

Every table with a header additionally gets a **profile** block (``TEXT``):
row count, columns with their types, and for each numeric column min/max
(with the row label), sum and mean, for each low-cardinality column the
distinct count and top values, and the sum of every numeric column grouped by
every low-cardinality column (top groups). Retrieval alone cannot answer
"which region has the highest total revenue" from 500 rows split into
chunks; the profile makes such aggregate facts a first-class passage.

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
from file_agent.parsers.table_profile import profile_table
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
            overview: list[tuple[str, int, list[str]]] = []
            try:
                sheets = workbook.worksheets
                # One sheet is read, converted and released before the next is
                # opened: holding every grid at once would cost hundreds of
                # megabytes on a workbook of several full sheets.
                for sheet_index, sheet in enumerate(sheets, start=1):
                    grid, truncated = _read_grid(sheet)
                    if not grid:
                        continue
                    items = _sheet_items(grid)
                    del grid
                    title = sheet.title
                    for kind, payload in items:
                        if kind == "table":
                            header, body = payload
                            overview.append((title, len(body), header or []))
                            break

                    factory.heading(
                        f"Sheet: {title}",
                        1,
                        page_number=sheet_index,
                        metadata={"sheet_name": title, "sheet_index": sheet_index},
                    )
                    for kind, payload in items:
                        if kind == "note":
                            factory.add(
                                payload,
                                BlockType.TEXT,
                                {"sheet_name": title},
                                page_number=sheet_index,
                            )
                            continue
                        header, body = payload
                        rows = [header, *body] if header else body
                        markdown = table_to_markdown(rows, header=header is not None)
                        if not markdown:
                            continue
                        table_count += 1
                        factory.add(
                            markdown,
                            BlockType.TABLE,
                            {
                                "sheet_name": title,
                                "table_index": table_count,
                                "row_count": len(body),
                                "truncated": truncated,
                                "table_header": list(header) if header else None,
                            },
                            page_number=sheet_index,
                        )
                        profile = profile_table(header, body) if header else ""
                        if profile:
                            factory.add(
                                f"Сводка по таблице «{title}» "
                                f"(вычислена автоматически):\n{profile}",
                                BlockType.TEXT,
                                {"sheet_name": title, "table_profile": True},
                                page_number=sheet_index,
                            )

                # A workbook-level overview answers the questions that no single
                # table can ("how many sheets are there, what is the main one
                # called") and gives retrieval one passage naming every sheet and
                # column. It opens the document, so it is inserted in front.
                summary = _workbook_overview(path.name, len(sheets), overview)
                if summary:
                    block = factory.add(
                        summary,
                        BlockType.TEXT,
                        {"workbook_overview": True},
                        page_number=1,
                    )
                    if block is not None:
                        factory.blocks.remove(block)
                        factory.blocks.insert(0, block)
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
                    # A workbook has no title of its own, and the first sheet
                    # name is a poor stand-in ("Sheet: Comparisons" as the title
                    # of every chunk of every other sheet). The file name is what
                    # questions actually refer to.
                    "title": path.name,
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


def _sheet_items(grid: list[list[str]]) -> list[tuple[str, Any]]:
    """Split a sheet into notes and tables, rejoining continuations.

    Blank rows separate independent tables *and* occur inside a single long
    table (a page break in an export, a visual gap between groups of rows).
    Treating every gap as a new table was actively harmful: one sheet became
    ten fragments, each with its own automatic summary, so "the maximum rating
    in the table" had ten different wrong answers competing in retrieval. A
    region whose columns match the previous table and that has no header of
    its own is therefore appended to that table instead.
    """
    items: list[tuple[str, Any]] = []
    last_table: list[list[str]] | None = None  # body rows of the open table
    last_header: list[str] | None = None

    for region in _split_regions(grid):
        if _is_note(region):
            items.append(("note", "\n".join(_note_line(row) for row in region)))
            last_table = None
            continue

        header, body = _split_header(region)
        continues = (
            header is None
            and last_table is not None
            and last_header is not None
            and _same_width(last_header, body)
        )
        if continues and last_table is not None:
            last_table.extend(body)
            continue

        if header is None and last_header is not None and _same_width(last_header, body):
            header = last_header
        rows = list(body)
        items.append(("table", (header, rows)))
        last_table, last_header = rows, header

    return items


def _workbook_overview(
    file_name: str, sheet_count: int, sheets: list[tuple[str, int, list[str]]]
) -> str:
    if not sheets:
        return ""
    lines = [
        f"Обзор файла {file_name}: листов — {sheet_count}, "
        f"названия листов: {', '.join(f'«{name}»' for name, _, _ in sheets)}."
    ]
    for name, row_count, header in sheets:
        columns = ", ".join(column for column in header if column)
        detail = f"«{name}»: строк с данными — {row_count}"
        if columns:
            detail += f"; столбцы: {columns}"
        lines.append(detail)
    return "\n".join(lines)


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


def _distinct_cells(row: list[str]) -> list[str]:
    """Non-empty cells with merged-cell repeats collapsed."""
    distinct: list[str] = []
    for cell in row:
        if cell and (not distinct or distinct[-1] != cell):
            distinct.append(cell)
    return distinct


def _note_line(row: list[str]) -> str:
    return " ".join(_distinct_cells(row))


def _is_note(region: list[list[str]]) -> bool:
    if len(region) > 3:
        return False
    for row in region:
        cells = _distinct_cells(row)
        if len(cells) != 1 or len(cells[0]) < _NOTE_MIN_CHARS:
            return False
    return True


def _split_header(region: list[list[str]]) -> tuple[list[str] | None, list[list[str]]]:
    """Detect one- or two-row headers; return (header, data rows).

    A header row is mostly text while the rows below carry numbers. Two
    stacked text rows above numeric data (a group row over sub-columns, as
    produced by merged cells) are combined into "Group Sub" column names.
    """
    if not region:
        return None, region
    if len(region) == 1:
        return None, region

    def numeric_share(row: list[str]) -> float:
        cells = [cell for cell in row if cell]
        if not cells:
            return 0.0
        return sum(1 for cell in cells if _is_number(cell)) / len(cells)

    first_share = numeric_share(region[0])
    if first_share >= 0.5:
        return None, region
    if len(region) >= 3 and numeric_share(region[1]) < 0.5 and numeric_share(region[2]) >= 0.5:
        top, sub = region[0], region[1]
        width = max(len(top), len(sub))
        header = []
        for index in range(width):
            group = top[index] if index < len(top) else ""
            leaf = sub[index] if index < len(sub) else ""
            header.append(" ".join(part for part in (group, leaf) if part) or "")
        return header, region[2:]
    if numeric_share(region[1]) > first_share or first_share == 0.0:
        return region[0], region[1:]
    return None, region


def _same_width(header: list[str], body: list[list[str]]) -> bool:
    if not body:
        return False
    header_width = len([cell for cell in header if cell])
    widths = {len(row) for row in body}
    return bool(widths) and max(widths) >= header_width and header_width > 0


def _is_number(cell: str) -> bool:
    try:
        float(cell.replace(",", ".").replace(" ", ""))
        return True
    except ValueError:
        return False
