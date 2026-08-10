import re
from pathlib import Path

from file_agent.document import Block, BlockType, Document
from file_agent.parsers.base import BaseParser

# ATX heading: 1-6 '#' characters, a space, then the title.
_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
# Fenced code block delimiter (``` or ~~~); '#' lines inside fences are code
# comments, not headings.
_FENCE = re.compile(r"^\s*(```|~~~)")
# Inline HTML tags occasionally used for styling inside headings.
_HTML_TAG = re.compile(r"<[^>]+>")


class MarkdownParser(BaseParser):
    """Split markdown into heading and text blocks.

    Heading blocks carry ``hierarchy_level`` so downstream consumers get
    section-aware chunking, a table of contents and section reading — the
    same structural contract that rich parsers (Docling) provide.
    """

    def parse(self, file_path: Path) -> Document:
        path = Path(file_path)
        text = path.read_text(encoding="utf-8")

        blocks: list[Block] = []
        body_lines: list[str] = []
        in_fence = False

        def flush_body() -> None:
            body = "\n".join(body_lines).strip()
            body_lines.clear()
            if body:
                blocks.append(
                    Block(
                        id=f"block-{len(blocks) + 1}",
                        text=body,
                        type="markdown",
                        metadata={"block_type": "markdown"},
                        block_type=BlockType.TEXT,
                    )
                )

        for line in text.splitlines():
            if _FENCE.match(line):
                in_fence = not in_fence
                body_lines.append(line)
                continue

            heading = None if in_fence else _HEADING.match(line)
            if heading is None:
                body_lines.append(line)
                continue

            flush_body()
            level = len(heading.group(1))
            title = _HTML_TAG.sub("", heading.group(2)).strip() or heading.group(2).strip()
            blocks.append(
                Block(
                    id=f"block-{len(blocks) + 1}",
                    text=title,
                    type=BlockType.HEADING.value,
                    metadata={"block_type": "heading", "hierarchy_level": level},
                    block_type=BlockType.HEADING,
                )
            )

        flush_body()

        if not blocks:
            blocks.append(
                Block(
                    id="block-1",
                    text=text,
                    type="markdown",
                    metadata={"block_type": "markdown"},
                    block_type=BlockType.TEXT,
                )
            )

        document = Document(file_name=path.name, file_type="md", blocks=blocks)
        document.build_table_of_contents()
        document.metadata["parsing_method"] = "markdown"
        return document
