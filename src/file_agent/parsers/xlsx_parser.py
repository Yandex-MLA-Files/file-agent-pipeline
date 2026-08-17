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
import re
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
# Profile limits: categorical columns with more distinct values than this are
# not grouped; at most this many groups/values are listed per aggregate.
PROFILE_MAX_CATEGORIES = 60
PROFILE_TOP_N = 12
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
                    last_header: list[str] | None = None
                    for region in _split_regions(grid):
                        if _is_note(region):
                            text = "\n".join(_note_line(row) for row in region)
                            factory.add(
                                text,
                                BlockType.TEXT,
                                {"sheet_name": sheet.title},
                                page_number=sheet_index,
                            )
                            continue
                        header, body = _split_header(region)
                        if header is None and last_header and _same_width(last_header, body):
                            # A block of rows below a blank line continues the
                            # previous table; reuse its header instead of col1..colN.
                            header = last_header
                        rows = [header, *body] if header else body
                        markdown = table_to_markdown(rows, header=header is not None)
                        if not markdown:
                            continue
                        if header:
                            last_header = header
                        table_count += 1
                        factory.add(
                            markdown,
                            BlockType.TABLE,
                            {
                                "sheet_name": sheet.title,
                                "table_index": table_count,
                                "row_count": len(body),
                                "truncated": truncated,
                            },
                            page_number=sheet_index,
                        )
                        profile = _profile_table(header, body) if header else ""
                        if profile:
                            factory.add(
                                f"Сводка по таблице «{sheet.title}» "
                                f"(вычислена автоматически):\n{profile}",
                                BlockType.TEXT,
                                {"sheet_name": sheet.title, "table_profile": True},
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


# -- table profile -----------------------------------------------------------------


def _to_number(cell: str) -> float | None:
    if not cell:
        return None
    text = cell.replace("\u00a0", "").replace(" ", "")
    if text.count(",") == 1 and "." not in text:
        text = text.replace(",", ".")
    try:
        return float(text)
    except ValueError:
        return None


_DATE_LIKE = re.compile(r"^\d{4}-\d{2}-\d{2}|^\d{1,2}[./]\d{1,2}[./]\d{2,4}")


def _looks_like_dates(column_values: list[str]) -> bool:
    non_empty = [v for v in column_values if v]
    if not non_empty:
        return False
    return sum(1 for v in non_empty if _DATE_LIKE.match(v)) / len(non_empty) >= 0.8


def _fmt(value: float) -> str:
    if abs(value - round(value)) < 1e-9 and abs(value) < 1e15:
        return f"{int(round(value)):,}".replace(",", " ")
    return f"{value:,.2f}".replace(",", " ")


def _profile_table(header: list[str], body: list[list[str]]) -> str:
    """Describe a table's shape and the aggregates a reader would otherwise compute."""
    if len(body) < 3 or not header:
        return ""
    width = len(header)
    columns = [(header[i] or f"col{i + 1}") for i in range(width)]
    values = [[row[i] if i < len(row) else "" for row in body] for i in range(width)]

    numeric: dict[int, list[float | None]] = {}
    categorical: dict[int, list[str]] = {}
    for index, column_values in enumerate(values):
        non_empty = [v for v in column_values if v]
        if not non_empty:
            continue
        numbers = [_to_number(v) for v in column_values]
        numeric_share = sum(
            1 for n, v in zip(numbers, column_values, strict=True) if v and n is not None
        )
        if numeric_share / len(non_empty) >= 0.9:
            numeric[index] = numbers
        else:
            distinct = set(non_empty)
            if len(distinct) <= PROFILE_MAX_CATEGORIES:
                categorical[index] = column_values

    # An "id"-like numeric column (unique, monotonic) is an identifier, not a measure.
    measures = {}
    for index, numbers in numeric.items():
        clean = [n for n in numbers if n is not None]
        if (
            len(set(clean)) == len(clean)
            and clean == sorted(clean)
            and columns[index].lower()
            in {
                "id",
                "no",
                "№",
                "index",
                "n",
            }
        ):
            continue
        measures[index] = numbers

    # Row label for min/max: the most specific text column (names, titles),
    # never a date column.
    text_columns = [
        i for i in range(width) if i not in numeric and not _looks_like_dates(values[i])
    ]
    label_index = max(text_columns, key=lambda i: len(set(values[i])), default=None)

    lines = [f"строк: {len(body)}; столбцы: {', '.join(columns)}"]
    for index, numbers in measures.items():
        pairs = [(n, r) for r, n in enumerate(numbers) if n is not None]
        if not pairs:
            continue
        total = sum(n for n, _ in pairs)
        low = min(pairs, key=lambda p: p[0])
        high = max(pairs, key=lambda p: p[0])

        def label(row_index: int) -> str:
            if label_index is None:
                return ""
            text = values[label_index][row_index]
            return f" ({text})" if text else ""

        lines.append(
            f"{columns[index]}: минимум {_fmt(low[0])}{label(low[1])}, максимум "
            f"{_fmt(high[0])}{label(high[1])}, сумма {_fmt(total)}, среднее "
            f"{_fmt(total / len(pairs))}"
        )
    for index, column_values in categorical.items():
        if _looks_like_dates(column_values):
            continue
        counts: dict[str, int] = {}
        for value in column_values:
            if value:
                counts[value] = counts.get(value, 0) + 1
        top = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:PROFILE_TOP_N]
        rendered = ", ".join(f"{name} ({count})" for name, count in top)
        suffix = " …" if len(counts) > PROFILE_TOP_N else ""
        lines.append(f"{columns[index]}: уникальных значений {len(counts)}: {rendered}{suffix}")
        for m_index, numbers in measures.items():
            sums: dict[str, float] = {}
            for value, number in zip(column_values, numbers, strict=True):
                if value and number is not None:
                    sums[value] = sums.get(value, 0.0) + number
            if not sums:
                continue
            ranked = sorted(sums.items(), key=lambda item: -item[1])
            shown = ", ".join(f"{name}: {_fmt(total)}" for name, total in ranked[:PROFILE_TOP_N])
            more = " …" if len(ranked) > PROFILE_TOP_N else ""
            lines.append(
                f"сумма {columns[m_index]} по {columns[index]} (по убыванию): {shown}{more}"
            )
    return "\n".join(lines)
