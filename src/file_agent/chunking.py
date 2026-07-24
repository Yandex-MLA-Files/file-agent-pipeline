from dataclasses import dataclass, field
from typing import Any

from file_agent.document import Block, BlockType, Document

DEFAULT_MAX_CHARS = 1000
DEFAULT_OVERLAP = 100

_SEPARATOR = "\n\n"

# Per-block Docling internals that are meaningless once blocks are packed together.
_SKIP_BLOCK_METADATA = frozenset({"docling_label", "hierarchy_level"})


@dataclass
class Chunk:
    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the chunk for export (CSV/JSON) or storage in a vector DB."""
        return {
            "id": self.id,
            "text": self.text,
            "metadata": self.metadata,
        }


@dataclass
class _Section:
    """A heading and the blocks that belong to it (its reading-order body)."""

    heading: str | None
    blocks: list[Block]

    @property
    def text(self) -> str:
        return _SEPARATOR.join(b.text for b in self.blocks if b.text)


def chunk_document(
    document: Document,
    max_chars: int = DEFAULT_MAX_CHARS,
    overlap: int = DEFAULT_OVERLAP,
    min_chars: int | None = None,
) -> list[Chunk]:
    """Split a document into retrieval-sized, section-coherent chunks.

    Structured parsers (e.g. Docling) emit many small blocks — a heading, a
    short paragraph, a caption. Turning each block into its own chunk starves
    retrieval: the top-k results become a handful of tiny fragments with almost
    no context. Merging blocks blindly is not enough either: it mixes unrelated
    sections into one chunk and mislabels it with a trailing heading.

    So chunking works in two levels:

    1. Blocks are grouped into **sections** (a heading plus its body), which
       guarantees a heading always opens a chunk and never dangles at the end.
    2. Whole sections are packed together up to ``max_chars`` (small adjacent
       slides/sections merge), flushing once a chunk reaches ``min_chars`` so
       chunks stay coherent instead of greedily spanning the whole document. A
       section larger than ``max_chars`` is packed block-by-block, and a single
       oversized block is split into overlapping windows. Tables are kept whole.

    Each chunk records ``section`` (the heading it starts under), ``sections``
    (all headings it covers), page numbers, block ids and any VLM description,
    so retrieval results stay filterable and traceable.
    """
    if max_chars <= 0:
        raise ValueError("max_chars must be greater than 0")
    if overlap < 0:
        raise ValueError("overlap must be greater than or equal to 0")
    if overlap >= max_chars:
        raise ValueError("overlap must be smaller than max_chars")

    if min_chars is None:
        min_chars = max(1, max_chars // 3)
    min_chars = min(min_chars, max_chars)

    chunker = _Chunker(document, max_chars, overlap, min_chars)
    for section in _group_sections(document.blocks):
        chunker.add_section(section)
    return chunker.finish()


def _group_sections(blocks: list[Block]) -> list[_Section]:
    sections: list[_Section] = []
    current = _Section(heading=None, blocks=[])
    for block in blocks:
        if block.block_type == BlockType.HEADING and block.text.strip():
            if current.blocks:
                sections.append(current)
            current = _Section(heading=block.text.strip(), blocks=[block])
        else:
            current.blocks.append(block)
    if current.blocks:
        sections.append(current)
    return sections


class _Chunker:
    def __init__(self, document: Document, max_chars: int, overlap: int, min_chars: int) -> None:
        self._document = document
        self._max_chars = max_chars
        self._overlap = overlap
        self._min_chars = min_chars
        self._chunks: list[Chunk] = []
        self._index = 1
        self._buffer: list[_Section] = []
        self._buffer_len = 0

    def add_section(self, section: _Section) -> None:
        text = section.text
        if not text:
            return

        # A section too large for one chunk is packed block by block.
        if len(text) > self._max_chars:
            self._flush()
            self._pack_blocks(section.blocks, section.heading)
            return

        would_exceed = self._buffer_len + len(_SEPARATOR) + len(text) > self._max_chars
        if self._buffer and (self._buffer_len >= self._min_chars or would_exceed):
            self._flush()

        self._buffer.append(section)
        self._buffer_len += len(text) + (len(_SEPARATOR) if len(self._buffer) > 1 else 0)

    def finish(self) -> list[Chunk]:
        self._flush()
        return self._chunks

    # -- internals ----------------------------------------------------------

    def _flush(self) -> None:
        if not self._buffer:
            return
        blocks = [block for section in self._buffer for block in section.blocks]
        headings = [section.heading for section in self._buffer if section.heading]
        section = headings[0] if headings else None
        self._emit(blocks, section=section, sections=headings)
        self._buffer = []
        self._buffer_len = 0

    def _pack_blocks(self, blocks: list[Block], heading: str | None) -> None:
        buffer: list[Block] = []
        buffer_len = 0

        def flush_buffer() -> None:
            nonlocal buffer, buffer_len
            if not buffer:
                return
            self._emit(
                buffer,
                section=heading,
                sections=[heading] if heading else [],
                prepend_heading=True,
            )
            buffer = self._overlap_seed(buffer)
            buffer_len = sum(len(b.text) + len(_SEPARATOR) for b in buffer)

        for block in blocks:
            text = block.text
            if not text:
                continue

            if block.block_type == BlockType.TABLE or len(text) > self._max_chars:
                flush_buffer()
                buffer, buffer_len = [], 0
                self._emit_oversized(block, text, heading)
                continue

            addition = len(text) + (len(_SEPARATOR) if buffer else 0)
            if buffer and buffer_len + addition > self._max_chars:
                flush_buffer()
                addition = len(text) + (len(_SEPARATOR) if buffer else 0)
            buffer.append(block)
            buffer_len += addition

        flush_buffer()

    def _overlap_seed(self, blocks: list[Block]) -> list[Block]:
        if self._overlap == 0:
            return []
        seed: list[Block] = []
        total = 0
        for block in reversed(blocks):
            length = len(block.text)
            if length == 0 or total + length > self._overlap:
                break
            seed.insert(0, block)
            total += length + len(_SEPARATOR)
        if len(seed) == len(blocks):  # never carry the whole chunk forward
            seed = seed[1:]
        return seed

    def _emit_oversized(self, block: Block, text: str, heading: str | None) -> None:
        sections = [heading] if heading else []

        if block.block_type == BlockType.TABLE:
            # Keep small tables whole; split large ones by rows so each piece fits
            # an embedding window, repeating the header row for standalone meaning.
            for piece in self._split_table(text):
                self._emit(
                    [block], section=heading, sections=sections, text=piece, prepend_heading=True
                )
            return

        if len(text) <= self._max_chars:
            self._emit([block], section=heading, sections=sections, text=text, prepend_heading=True)
            return

        step = self._max_chars - self._overlap
        start = 0
        while start < len(text):
            self._emit(
                [block],
                section=heading,
                sections=sections,
                text=text[start : start + self._max_chars],
                prepend_heading=True,
            )
            start += step

    def _split_table(self, text: str) -> list[str]:
        if len(text) <= self._max_chars:
            return [text]

        lines = text.split("\n")
        header_lines: list[str] = []
        body = lines
        # A Markdown table header is a row followed by a separator like |---|:--|.
        if len(lines) >= 2 and "|" in lines[0] and set(lines[1].strip()) <= set("|-: "):
            header_lines = lines[:2]
            body = lines[2:]
        header = "\n".join(header_lines)

        pieces: list[str] = []
        current: list[str] = []
        current_len = len(header)
        for row in body:
            row_len = len(row) + 1
            if current and current_len + row_len > self._max_chars:
                pieces.append(self._join_table(header, current))
                current = []
                current_len = len(header)
            current.append(row)
            current_len += row_len
        if current:
            pieces.append(self._join_table(header, current))
        return pieces or [text]

    @staticmethod
    def _join_table(header: str, rows: list[str]) -> str:
        body = "\n".join(rows)
        return f"{header}\n{body}" if header else body

    def _emit(
        self,
        blocks: list[Block],
        section: str | None,
        sections: list[str],
        text: str | None = None,
        prepend_heading: bool = False,
    ) -> None:
        chunk_text = text if text is not None else _SEPARATOR.join(b.text for b in blocks if b.text)
        # Give continuation chunks of a long section their heading as context, so
        # every chunk is self-describing for retrieval (a "breadcrumb").
        if prepend_heading and section and section not in chunk_text:
            chunk_text = f"{section}\n\n{chunk_text}"
        self._chunks.append(
            Chunk(
                id=f"{blocks[0].id}-chunk-{self._index}",
                text=chunk_text,
                metadata=self._build_metadata(blocks, section, sections),
            )
        )
        self._index += 1

    def _build_metadata(
        self,
        blocks: list[Block],
        section: str | None,
        sections: list[str],
    ) -> dict[str, Any]:
        block_types: list[str] = []
        for block in blocks:
            value = block.block_type.value if block.block_type else block.type
            if value not in block_types:
                block_types.append(value)

        pages = sorted({b.page_number for b in blocks if b.page_number is not None})
        descriptions = [b.vlm_description for b in blocks if b.vlm_description]

        # Preserve useful source coordinates set by parsers (page_number,
        # slide_number, sheet_name, ...); the first block wins on conflicts.
        metadata: dict[str, Any] = {}
        for block in blocks:
            for key, value in block.metadata.items():
                if key not in _SKIP_BLOCK_METADATA:
                    metadata.setdefault(key, value)

        metadata.update(
            {
                "source_file": metadata.get("source_file", self._document.file_name),
                "file_type": self._document.file_type,
                "block_ids": [b.id for b in blocks],
                "block_type": block_types[0] if len(block_types) == 1 else "mixed",
                "block_types": block_types,
            }
        )
        if pages:
            metadata["page_number"] = pages[0]
            metadata["page_numbers"] = pages
        if section:
            metadata["section"] = section
        if sections:
            metadata["sections"] = sections
        if len(blocks) == 1 and blocks[0].bbox is not None:
            metadata["bbox"] = blocks[0].bbox
        if descriptions:
            metadata["vlm_description"] = " ".join(descriptions)

        return metadata
