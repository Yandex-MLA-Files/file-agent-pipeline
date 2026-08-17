"""Structured HTML parser built on BeautifulSoup.

Walks the DOM in document order and emits typed blocks instead of one flat
text dump: ``HEADING`` for ``h1``-``h6`` (level from the tag), ``TEXT`` for
paragraphs and other block-level prose, ``LIST`` for ``ul``/``ol`` (nested
items indented), ``TABLE`` for ``table`` rendered as Markdown, ``CODE`` for
``pre`` blocks and ``FIGURE`` for ``img`` with alt text. Boilerplate that hurts
retrieval (scripts, styles, navigation, headers/footers, forms) is dropped and
the ``<title>`` becomes the document title.
"""

import logging
import re
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup, NavigableString, Tag

from file_agent.document import BlockType, Document
from file_agent.parsers.base import BaseParser
from file_agent.parsers.common import (
    BlockFactory,
    clean_text,
    list_to_markdown,
    read_text_file,
    table_to_markdown,
)
from file_agent.telemetry import tracer

logger = logging.getLogger(__name__)

_DROP_TAGS = {"script", "style", "noscript", "nav", "header", "footer", "aside", "form", "svg"}
_HEADING_TAGS = {f"h{level}": level for level in range(1, 7)}
_BLOCK_TAGS = {
    "p",
    "div",
    "section",
    "article",
    "main",
    "blockquote",
    "li",
    "dd",
    "dt",
    "figcaption",
    "address",
    "summary",
    "details",
    "td",
    "th",
}
_INLINE_BREAK = re.compile(r"\n{3,}")


class HTMLParser(BaseParser):
    def parse(self, file_path: Path) -> Document:
        path = Path(file_path)
        with tracer.start_as_current_span("file_agent.html_parse") as span:
            span.set_attribute("file_agent.file_name", path.name)

            html = read_text_file(path)
            soup = BeautifulSoup(html, "html.parser")
            for tag in soup(list(_DROP_TAGS)):
                tag.decompose()

            title = None
            if soup.title and soup.title.string:
                title = clean_text(soup.title.string)

            factory = BlockFactory(source_file=path.name)
            root = soup.body or soup
            walker = _Walker(factory)
            walker.walk(root)
            walker.flush_text()

            if not factory.blocks:
                text = clean_text(soup.get_text(separator="\n", strip=True))
                factory.add(text, BlockType.TEXT, skip_empty=False)

            document = Document(
                file_name=path.name,
                file_type=path.suffix.lower().lstrip("."),
                blocks=factory.blocks,
                metadata={"parsing_method": "html", "title": title},
            )
            document.build_table_of_contents()
            span.set_attribute("file_agent.block_count", len(document.blocks))
            logger.info("Parsed %s into %d block(s)", path.name, len(document.blocks))
            return document


class _Walker:
    def __init__(self, factory: BlockFactory) -> None:
        self.factory = factory
        self._text_parts: list[str] = []

    def flush_text(self) -> None:
        text = clean_text("\n".join(self._text_parts))
        self._text_parts = []
        text = _INLINE_BREAK.sub("\n\n", text)
        if text:
            self.factory.add(text, BlockType.TEXT)

    def walk(self, node: Any) -> None:
        for child in node.children:
            if isinstance(child, NavigableString):
                text = str(child)
                if text.strip():
                    self._text_parts.append(text)
                continue
            if not isinstance(child, Tag):
                continue
            name = child.name.lower()
            if name in _HEADING_TAGS:
                self.flush_text()
                text = clean_text(child.get_text(" ", strip=True))
                if text:
                    self.factory.heading(text, _HEADING_TAGS[name])
            elif name in ("ul", "ol"):
                self.flush_text()
                items = _list_items(child, 0)
                if items:
                    self.factory.add(
                        list_to_markdown(items, ordered=(name == "ol")),
                        BlockType.LIST,
                        {"item_count": len(items)},
                    )
            elif name == "table":
                self.flush_text()
                markdown = _table_markdown(child)
                if markdown:
                    caption = child.find("caption")
                    caption_text = clean_text(caption.get_text(" ", strip=True)) if caption else ""
                    text = f"{caption_text}\n{markdown}" if caption_text else markdown
                    self.factory.add(text, BlockType.TABLE, {"caption": caption_text or None})
            elif name == "pre":
                self.flush_text()
                code = child.get_text()
                if code.strip():
                    self.factory.add(code.rstrip(), BlockType.CODE)
            elif name == "img":
                alt = clean_text(child.get("alt") or "")
                if alt:
                    self.factory.add(alt, BlockType.FIGURE, {"src": child.get("src")})
            elif name == "br":
                self._text_parts.append("\n")
            elif name in _BLOCK_TAGS:
                # Paragraph-level container: its text is one unit, but headings,
                # lists and tables nested inside are still surfaced separately.
                if child.find(list(_HEADING_TAGS) + ["ul", "ol", "table", "pre"]) is not None:
                    self.walk(child)
                else:
                    self.flush_text()
                    text = clean_text(child.get_text(" ", strip=True))
                    if text:
                        self.factory.add(text, BlockType.TEXT)
            else:
                self.walk(child)


def _list_items(list_tag: Tag, depth: int) -> list[tuple[int, str]]:
    items: list[tuple[int, str]] = []
    for li in list_tag.find_all("li", recursive=False):
        nested = li.find_all(["ul", "ol"], recursive=False)
        own_text_parts = []
        for child in li.children:
            if isinstance(child, Tag) and child.name in ("ul", "ol"):
                continue
            own_text_parts.append(
                child.get_text(" ", strip=True) if isinstance(child, Tag) else str(child)
            )
        own_text = clean_text(" ".join(own_text_parts))
        if own_text:
            items.append((depth, own_text))
        for sub in nested:
            items.extend(_list_items(sub, depth + 1))
    return items


def _table_markdown(table: Tag) -> str:
    rows: list[list[str]] = []
    for tr in table.find_all("tr"):
        cells = [clean_text(cell.get_text(" ", strip=True)) for cell in tr.find_all(["th", "td"])]
        if cells:
            rows.append(cells)
    return table_to_markdown(rows)
