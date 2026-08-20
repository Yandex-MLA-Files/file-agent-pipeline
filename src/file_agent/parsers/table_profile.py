"""Automatic numeric summary of a table.

Retrieval can hand the model the rows of a table, but not the answer to
"which region has the highest total revenue" or "what is the maximum rating in
the table" — that requires reading every row at once, which no chunk does.
A profile block states those facts explicitly: the shape of the table, the
minimum and maximum of each numeric column *with the row it belongs to*, sums
and means, the distinct values of low-cardinality columns and the sum of every
measure grouped by them.

Spreadsheets get their profile from the parser, where the cell values are
still typed; tables extracted from PDF, DOCX, HTML or Markdown are profiled
from their Markdown rendering by the chunker, so a financial statement in a
PDF is as answerable as the same table in a workbook.
"""

import re

# Profile limits: categorical columns with more distinct values than this are
# not grouped; at most this many groups/values are listed per aggregate.
PROFILE_MAX_CATEGORIES = 60
PROFILE_TOP_N = 12
# Values longer than this on average are prose, not categories.
PROFILE_MAX_CATEGORY_CHARS = 60


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


def profile_table(header: list[str], body: list[list[str]]) -> str:
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
            average_length = sum(len(v) for v in non_empty) / len(non_empty)
            # A column of prose (a "term / definition" table) has no categories
            # to count: listing its values would only repeat the table.
            if (
                len(distinct) <= PROFILE_MAX_CATEGORIES
                and average_length <= PROFILE_MAX_CATEGORY_CHARS
            ):
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

    # Without a numeric column there is nothing to aggregate: a list of the
    # values of a text table only repeats the table it came from.
    if not measures:
        return ""

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
