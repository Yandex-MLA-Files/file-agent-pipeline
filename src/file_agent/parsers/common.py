"""Helpers shared by the structured parsers.

Every format parser produces the same :class:`~file_agent.document.Block`
contract (typed blocks, heading levels, Markdown tables), so the routines that
build that contract live here instead of being re-implemented per format.
"""

import datetime as _dt
import re
from pathlib import Path
from typing import Any

from file_agent.document import Block, BlockType

# Bullet glyphs that PDF/DOCX exports leave in list items (Symbol/Wingdings
# private-use characters, typographic bullets, dashes used as bullets).
_BULLET_PREFIX = re.compile(r"^\s*(?:[-•●○◦▪▫■□➢➤►▶✓✔\-–—*·]|o)\s+")
_MULTI_SPACE = re.compile(r"[ \t ]{2,}")
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

# "1.", "1.2", "1.2.3", "1)", "1.2)" — numbered headings whose depth tells the
# hierarchy level directly.
_NUMBERED_HEADING = re.compile(r"^\s*(\d+(?:\.\d+)*)[.)]?\s+\S")
# "Глава 3", "Раздел 2", "Часть I", "Chapter 4", "Part II", "Лекция 5", "Тема 10"
_NAMED_TOP_LEVEL = re.compile(
    r"^\s*(глава|раздел|часть|chapter|part|лекция|lecture|тема|topic|модуль|module)\b",
    re.IGNORECASE,
)
_ROMAN_HEADING = re.compile(r"^\s*[IVXLC]+[.)]\s+\S")

# Trailing punctuation that marks a *complete* sentence rather than a heading.
_SENTENCE_END = (".", "!", "?", ";", ":", ",")

_TEXT_ENCODINGS = ("utf-8-sig", "utf-16", "cp1251", "koi8-r", "cp866", "latin-1")


def clean_text(text: str) -> str:
    """Normalize whitespace and drop control characters, keeping line breaks."""
    text = _CONTROL_CHARS.sub("", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [_MULTI_SPACE.sub(" ", line).strip() for line in text.split("\n")]
    return "\n".join(lines).strip()


def strip_bullet(text: str) -> str:
    """Remove a leading bullet glyph so list items read as plain sentences."""
    return _BULLET_PREFIX.sub("", text, count=1).strip()


def infer_heading_level(text: str, default: int | None = None) -> int | None:
    """Guess a heading's hierarchy level from its numbering.

    ``"1. Introduction"`` -> 1, ``"1.2 Scope"`` -> 2, ``"1.2.3 Details"`` -> 3,
    ``"Глава 2"`` -> 1. Returns ``default`` when the text carries no numbering.
    """
    match = _NUMBERED_HEADING.match(text)
    if match:
        depth = match.group(1).count(".") + 1
        return min(depth, 6)
    if _NAMED_TOP_LEVEL.match(text) or _ROMAN_HEADING.match(text):
        return 1
    return default


def looks_like_heading(text: str, max_chars: int = 120) -> bool:
    """Cheap structural test used when a format has no explicit heading styles."""
    stripped = text.strip()
    if not stripped or len(stripped) > max_chars or "\n" in stripped:
        return False
    if stripped.endswith(_SENTENCE_END) and not stripped.endswith(":"):
        return False
    words = stripped.split()
    return len(words) <= 16


def table_to_markdown(
    rows: list[list[Any]],
    header: bool = True,
    max_rows: int | None = None,
) -> str:
    """Render a rectangular grid as a GitHub-flavoured Markdown table.

    Cells are whitespace-normalized and pipes escaped; empty rows are dropped;
    ragged rows are padded so every row has the same number of columns. When
    ``header`` is False the first row is data and a generic header is generated
    (``col1 | col2 ...``) so the table still splits/repeats correctly downstream.
    """
    cleaned: list[list[str]] = []
    for row in rows:
        cells = [format_cell(value) for value in row]
        if any(cell for cell in cells):
            cleaned.append(cells)
    if not cleaned:
        return ""

    width = max(len(row) for row in cleaned)
    # Drop columns that are empty everywhere (spreadsheets often carry them).
    keep = [
        index for index in range(width) if any(index < len(row) and row[index] for row in cleaned)
    ]
    grid = [[row[index] if index < len(row) else "" for index in keep] for row in cleaned]
    if not grid or not keep:
        return ""

    if header:
        head, body = grid[0], grid[1:]
    else:
        head, body = [f"col{index + 1}" for index in range(len(keep))], grid
    if max_rows is not None:
        body = body[:max_rows]

    lines = [_markdown_row(head), _markdown_row(["---"] * len(head))]
    lines.extend(_markdown_row(row) for row in body)
    return "\n".join(lines)


def _markdown_row(cells: list[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def format_cell(value: Any) -> str:
    """Render a cell value compactly: ``1100.0`` -> ``1100``, dates as ISO."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if value.is_integer() and abs(value) < 1e15:
            return str(int(value))
        return f"{value:.6g}" if abs(value) >= 1e6 or abs(value) < 1e-4 else str(round(value, 6))
    if isinstance(value, _dt.datetime):
        if value.time() == _dt.time(0, 0):
            return value.date().isoformat()
        return value.isoformat(sep=" ", timespec="minutes")
    if isinstance(value, _dt.date):
        return value.isoformat()
    text = str(value)
    text = text.replace("|", "\\|")
    return " ".join(text.split())


def read_text_file(path: Path) -> str:
    """Read a text file, detecting the encoding.

    UTF-8 (with or without BOM) and UTF-16 are recognized first; legacy Russian
    encodings (cp1251, koi8-r, cp866) come next because those files are common
    in Russian corpora and decode "successfully" as latin-1 into mojibake.
    """
    raw = Path(path).read_bytes()
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        return raw.decode("utf-16")
    for encoding in _TEXT_ENCODINGS:
        try:
            text = raw.decode(encoding)
        except UnicodeDecodeError:
            continue
        if encoding == "utf-16" and not _plausible_text(text):
            continue
        return text
    return raw.decode("utf-8", errors="replace")


def _plausible_text(text: str) -> bool:
    sample = text[:2000]
    if not sample:
        return True
    printable = sum(1 for char in sample if char.isprintable() or char in "\n\r\t")
    return printable / len(sample) > 0.95


class BlockFactory:
    """Allocates sequential block ids and stamps common metadata."""

    def __init__(self, source_file: str, prefix: str = "block") -> None:
        self.source_file = source_file
        self.prefix = prefix
        self._counter = 0
        self.blocks: list[Block] = []

    def _next_id(self) -> str:
        self._counter += 1
        return f"{self.prefix}-{self._counter}"

    def add(
        self,
        text: str,
        block_type: BlockType,
        metadata: dict[str, Any] | None = None,
        page_number: int | None = None,
        bbox: tuple[float, float, float, float] | None = None,
        image_bytes: bytes | None = None,
        skip_empty: bool = True,
    ) -> Block | None:
        if skip_empty and not text.strip() and image_bytes is None:
            return None
        meta: dict[str, Any] = {"source_file": self.source_file, "block_type": block_type.value}
        if metadata:
            meta.update(metadata)
        block = Block(
            id=self._next_id(),
            text=text,
            type=block_type.value,
            metadata=meta,
            block_type=block_type,
            page_number=page_number,
            bbox=bbox,
            image_bytes=image_bytes,
        )
        self.blocks.append(block)
        return block

    def heading(
        self,
        text: str,
        level: int,
        page_number: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Block | None:
        meta = {"hierarchy_level": max(1, min(int(level), 6))}
        if metadata:
            meta.update(metadata)
        return self.add(text.strip(), BlockType.HEADING, meta, page_number=page_number)


_VISIBLE_MARKER = re.compile(r"^(\d+[.)]|[a-zа-я][.)]|[ivx]+[.)])\s+", re.IGNORECASE)


def list_to_markdown(items: list[tuple[int, str]], ordered: bool = False) -> str:
    """Render ``(indent_level, text)`` pairs as a Markdown list.

    Items that already start with a visible number/letter marker keep it as-is
    (the source numbering is meaningful, e.g. exam question numbers).
    """
    lines: list[str] = []
    for position, (indent, text) in enumerate(items, start=1):
        text = " ".join(text.split())
        if not text:
            continue
        if _VISIBLE_MARKER.match(text):
            marker = ""
        elif ordered and indent == 0:
            marker = f"{position}. "
        else:
            marker = "- "
        lines.append("  " * max(0, indent) + f"{marker}{text}")
    return "\n".join(lines)
