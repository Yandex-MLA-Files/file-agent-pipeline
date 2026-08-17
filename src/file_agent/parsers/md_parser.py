"""Structured Markdown parser.

Markdown already *is* structure, so the parser mirrors it in blocks:

- ATX (``## Title``) and setext (underlined ``===`` / ``---``) headings become
  ``HEADING`` blocks with their level;
- fenced code blocks (`````` ``` `````` / ``~~~``) become ``CODE`` blocks and are
  never mistaken for headings/tables;
- pipe tables become ``TABLE`` blocks (so the chunker can split them by rows
  and repeat the header);
- bullet/numbered lists become ``LIST`` blocks; blank-line separated
  paragraphs become ``TEXT`` blocks; images become ``FIGURE`` blocks with their
  alt text;
- YAML front matter is parsed into document metadata (``title`` if present)
  instead of being indexed as prose.
"""

import re
from pathlib import Path

from file_agent.document import BlockType, Document
from file_agent.parsers.base import BaseParser
from file_agent.parsers.common import BlockFactory, read_text_file

# ATX heading: 1-6 '#' characters, a space, then the title.
_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
_SETEXT_H1 = re.compile(r"^={3,}\s*$")
_SETEXT_H2 = re.compile(r"^-{3,}\s*$")
_FENCE = re.compile(r"^\s*(```+|~~~+)\s*([\w+-]*)")
_HTML_TAG = re.compile(r"<[^>]+>")
_TABLE_ROW = re.compile(r"^\s*\|.*\|\s*$")
_TABLE_SEPARATOR = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$")
_LIST_ITEM = re.compile(r"^(\s*)([-*+]|\d+[.)])\s+(.*)$")
_IMAGE = re.compile(r"^\s*!\[([^\]]*)\]\(([^)]*)\)\s*$")
_FRONT_MATTER = re.compile(r"^---\s*$")
_INLINE_MARKUP = re.compile(r"(\*\*|__|`)")


class MarkdownParser(BaseParser):
    """Split markdown into typed blocks (headings, text, lists, tables, code)."""

    def parse(self, file_path: Path) -> Document:
        path = Path(file_path)
        text = read_text_file(path)
        lines = text.replace("\r\n", "\n").split("\n")

        factory = BlockFactory(source_file=path.name)
        metadata: dict = {"parsing_method": "markdown"}

        start = _consume_front_matter(lines, metadata)
        _Parser(factory).run(lines[start:])

        if not factory.blocks:
            factory.add(text, BlockType.TEXT, skip_empty=False)

        document = Document(file_name=path.name, file_type="md", blocks=factory.blocks)
        document.metadata.update(metadata)
        document.build_table_of_contents()
        return document


def parse_markdown_blocks(
    text: str,
    source_file: str,
    page_number: int | None = None,
    id_prefix: str = "block",
) -> list:
    """Parse a Markdown string into typed blocks (used for VLM page transcripts)."""
    factory = BlockFactory(source_file=source_file, prefix=id_prefix)
    lines = text.replace("\r\n", "\n").split("\n")
    _Parser(factory).run(lines)
    if page_number is not None:
        for block in factory.blocks:
            block.page_number = page_number
    return factory.blocks


def _consume_front_matter(lines: list[str], metadata: dict) -> int:
    if not lines or not _FRONT_MATTER.match(lines[0]):
        return 0
    for index in range(1, min(len(lines), 60)):
        if _FRONT_MATTER.match(lines[index]):
            for raw in lines[1:index]:
                if ":" in raw:
                    key, value = raw.split(":", 1)
                    key = key.strip().lower()
                    if key in {"title", "author", "date", "description"}:
                        metadata[key] = value.strip().strip("\"'")
            return index + 1
    return 0


class _Parser:
    def __init__(self, factory: BlockFactory) -> None:
        self.factory = factory
        self._paragraph: list[str] = []
        self._list: list[str] = []

    def run(self, lines: list[str]) -> None:
        index = 0
        total = len(lines)
        while index < total:
            line = lines[index]

            fence = _FENCE.match(line)
            if fence:
                self._flush_all()
                index = self._consume_code(lines, index, fence)
                continue

            if (
                _TABLE_ROW.match(line)
                and index + 1 < total
                and _TABLE_SEPARATOR.match(lines[index + 1])
            ):
                self._flush_all()
                index = self._consume_table(lines, index)
                continue

            heading = _HEADING.match(line)
            if heading:
                self._flush_all()
                self._emit_heading(heading.group(2), len(heading.group(1)))
                index += 1
                continue

            # Setext heading: a single text line underlined with === or ---.
            if (
                index + 1 < total
                and line.strip()
                and not self._list
                and len(self._paragraph) == 0
                and (_SETEXT_H1.match(lines[index + 1]) or _SETEXT_H2.match(lines[index + 1]))
                and not _LIST_ITEM.match(line)
            ):
                self._emit_heading(line, 1 if _SETEXT_H1.match(lines[index + 1]) else 2)
                index += 2
                continue

            image = _IMAGE.match(line)
            if image:
                self._flush_all()
                alt = image.group(1).strip()
                self.factory.add(alt or "image", BlockType.FIGURE, {"src": image.group(2)})
                index += 1
                continue

            item = _LIST_ITEM.match(line)
            if item:
                self._flush_paragraph()
                indent = len(item.group(1).replace("\t", "    ")) // 2
                self._list.append("  " * indent + f"{item.group(2)} {item.group(3).strip()}")
                index += 1
                continue

            if not line.strip():
                self._flush_all()
                index += 1
                continue

            if self._list and (line.startswith(("  ", "\t"))):
                # Continuation line of a list item.
                self._list[-1] += " " + line.strip()
                index += 1
                continue

            self._flush_list()
            self._paragraph.append(line.rstrip())
            index += 1

        self._flush_all()

    # -- emitters ---------------------------------------------------------------

    def _emit_heading(self, raw: str, level: int) -> None:
        title = _HTML_TAG.sub("", raw).strip()
        title = _INLINE_MARKUP.sub("", title).strip() or raw.strip()
        if title:
            self.factory.heading(title, level)

    def _consume_code(self, lines: list[str], index: int, fence: re.Match) -> int:
        marker = fence.group(1)[0]
        language = fence.group(2) or ""
        body: list[str] = []
        index += 1
        while index < len(lines):
            if re.match(rf"^\s*{re.escape(marker)}{{3,}}\s*$", lines[index]):
                index += 1
                break
            body.append(lines[index])
            index += 1
        code = "\n".join(body).rstrip()
        if code.strip():
            self.factory.add(code, BlockType.CODE, {"language": language or None})
        return index

    def _consume_table(self, lines: list[str], index: int) -> int:
        rows: list[str] = []
        while index < len(lines) and _TABLE_ROW.match(lines[index]):
            rows.append(lines[index].strip())
            index += 1
        if rows:
            self.factory.add("\n".join(rows), BlockType.TABLE, {"row_count": max(0, len(rows) - 2)})
        return index

    def _flush_paragraph(self) -> None:
        if self._paragraph:
            text = "\n".join(self._paragraph).strip()
            if text:
                self.factory.add(text, BlockType.TEXT)
            self._paragraph = []

    def _flush_list(self) -> None:
        if self._list:
            self.factory.add("\n".join(self._list), BlockType.LIST, {"item_count": len(self._list)})
            self._list = []

    def _flush_all(self) -> None:
        self._flush_paragraph()
        self._flush_list()
