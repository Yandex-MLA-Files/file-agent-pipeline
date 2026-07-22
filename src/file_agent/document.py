from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class BlockType(StrEnum):
    """Structural role of a block.

    Rich parsers (e.g. Docling) classify every block so that downstream
    consumers can render, chunk and route each element appropriately. Legacy
    parsers that only produce free text may leave ``Block.block_type`` unset.
    """

    TEXT = "text"
    HEADING = "heading"
    TABLE = "table"
    FIGURE = "figure"
    IMAGE = "image"
    FORMULA = "formula"
    PDF_PAGE = "pdf_page"


# Heading blocks are wrapped with this many '#' characters when their nesting
# level is unknown or out of range.
_MIN_HEADING_LEVEL = 1
_MAX_HEADING_LEVEL = 6


@dataclass
class Block:
    """A single logical element of a document.

    The first four fields (``id``, ``text``, ``type``, ``metadata``) form the
    stable, backward-compatible interface consumed by chunking, retrieval and
    the Streamlit app. The remaining fields are optional structural annotations
    populated by rich parsers; when unset the block behaves like a plain text
    block.
    """

    id: str
    text: str
    type: str
    metadata: dict[str, Any] = field(default_factory=dict)

    # Optional structural annotations (populated by Docling and other rich parsers).
    block_type: BlockType | None = None
    page_number: int | None = None
    bbox: tuple[float, float, float, float] | None = None  # (x0, y0, x1, y1)
    vlm_description: str | None = None

    def to_markdown(self) -> str:
        """Render this block as Markdown based on its structural type."""
        text = self.text.strip()

        if self.block_type == BlockType.HEADING:
            level = self.metadata.get("hierarchy_level", 1)
            try:
                level = int(level)
            except (TypeError, ValueError):
                level = 1
            level = max(_MIN_HEADING_LEVEL, min(level, _MAX_HEADING_LEVEL))
            return f"{'#' * level} {text}" if text else ""

        if self.block_type in (BlockType.FIGURE, BlockType.IMAGE):
            caption = self.vlm_description or text or "image"
            caption = " ".join(caption.split())
            return f"![{caption}]()"

        if self.block_type == BlockType.FORMULA and text:
            return f"$$\n{text}\n$$"

        # TEXT, TABLE (already Markdown from Docling), PDF_PAGE and untyped blocks.
        return self.text

    def to_dict(self) -> dict[str, Any]:
        """Serialize the block for export to JSON/CSV."""
        return {
            "id": self.id,
            "text": self.text,
            "type": self.type,
            "block_type": self.block_type.value if self.block_type else None,
            "page_number": self.page_number,
            "bbox": self.bbox,
            "vlm_description": self.vlm_description,
            "metadata": self.metadata,
        }


@dataclass
class Document:
    file_name: str
    file_type: str
    blocks: list[Block]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Populate default document-level metadata when it is not provided."""
        self.metadata.setdefault("table_of_contents", [])
        self.metadata.setdefault(
            "total_pages",
            max((b.page_number or 0 for b in self.blocks), default=0),
        )
        self.metadata.setdefault("parsing_method", "unknown")

    def build_table_of_contents(self) -> list[dict[str, Any]]:
        """Build a table of contents from heading blocks and cache it in metadata."""
        toc: list[dict[str, Any]] = []
        for block in self.blocks:
            if block.block_type == BlockType.HEADING:
                toc.append(
                    {
                        "title": block.text.strip(),
                        "level": block.metadata.get("hierarchy_level", 1),
                        "page": block.page_number,
                        "block_id": block.id,
                    }
                )
        self.metadata["table_of_contents"] = toc
        return toc

    def to_markdown(self, include_toc: bool = False) -> str:
        """Render the whole document as a single Markdown string.

        This is the canonical ``.* / .pdf / .docx -> .md`` conversion: every
        parser produces ``Block`` objects and this method flattens them into
        Markdown, so the representation is uniform across input formats.
        """
        parts: list[str] = []

        if include_toc:
            toc = self.metadata.get("table_of_contents") or self.build_table_of_contents()
            if toc:
                parts.append("## Table of contents")
                for entry in toc:
                    indent = "  " * (int(entry.get("level", 1)) - 1)
                    parts.append(f"{indent}- {entry['title']}")
                parts.append("")

        for block in self.blocks:
            rendered = block.to_markdown()
            if rendered.strip():
                parts.append(rendered)

        return "\n\n".join(parts).strip() + "\n"

    def to_dict(self) -> dict[str, Any]:
        """Serialize the document for export."""
        return {
            "file_name": self.file_name,
            "file_type": self.file_type,
            "metadata": self.metadata,
            "blocks": [b.to_dict() for b in self.blocks],
        }
