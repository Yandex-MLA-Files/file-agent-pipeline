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


def chunk_document(
    document: Document,
    max_chars: int = DEFAULT_MAX_CHARS,
    overlap: int = DEFAULT_OVERLAP,
) -> list[Chunk]:
    """Split a document into retrieval-sized chunks.

    Structured parsers (e.g. Docling) emit many small blocks — a heading, a
    short paragraph, a caption. Turning each block into its own chunk starves
    retrieval: the top-k results become a handful of tiny fragments carrying
    almost no context. Instead we *pack* consecutive blocks together up to
    ``max_chars`` so every chunk is a coherent, reasonably sized passage:

    - short blocks are merged (a heading naturally stays with the section text
      that follows it);
    - a block larger than ``max_chars`` is split into overlapping windows;
    - tables are emitted as their own chunk so their Markdown layout survives;
    - consecutive packed chunks overlap by whole trailing blocks (up to
      ``overlap`` characters) to preserve continuity across boundaries.

    Structural metadata (page numbers, block ids, bounding box, section title,
    VLM descriptions) is carried into each chunk for filtering and tracing.
    """
    if max_chars <= 0:
        raise ValueError("max_chars must be greater than 0")
    if overlap < 0:
        raise ValueError("overlap must be greater than or equal to 0")
    if overlap >= max_chars:
        raise ValueError("overlap must be smaller than max_chars")

    packer = _BlockPacker(document, max_chars, overlap)
    for block in document.blocks:
        packer.add(block)
    return packer.finish()


class _BlockPacker:
    def __init__(self, document: Document, max_chars: int, overlap: int) -> None:
        self._document = document
        self._max_chars = max_chars
        self._overlap = overlap
        self._chunks: list[Chunk] = []
        self._index = 1
        self._buffer: list[Block] = []
        self._buffer_len = 0
        self._dirty = False  # buffer holds fresh (non-overlap) content
        self._section: str | None = None

    def add(self, block: Block) -> None:
        text = block.text
        if block.block_type == BlockType.HEADING and text.strip():
            self._section = text.strip()
        if not text:
            return

        # Tables and oversized blocks are emitted on their own so their layout
        # is preserved and they never bloat a packed chunk.
        if block.block_type == BlockType.TABLE or len(text) > self._max_chars:
            self._flush(keep_overlap=False)
            self._reset_buffer()
            self._emit_standalone(block, text)
            return

        addition = len(text) + (len(_SEPARATOR) if self._buffer else 0)
        if self._buffer and self._buffer_len + addition > self._max_chars:
            self._flush(keep_overlap=True)
            addition = len(text) + (len(_SEPARATOR) if self._buffer else 0)

        self._buffer.append(block)
        self._buffer_len += addition
        self._dirty = True

    def finish(self) -> list[Chunk]:
        self._flush(keep_overlap=False)
        return self._chunks

    # -- internals ----------------------------------------------------------

    def _flush(self, keep_overlap: bool) -> None:
        if not self._buffer or not self._dirty:
            return

        text = _SEPARATOR.join(block.text for block in self._buffer)
        self._append(text, self._buffer)
        self._dirty = False

        if keep_overlap:
            self._buffer = self._overlap_seed(self._buffer)
        else:
            self._buffer = []
        self._buffer_len = sum(len(b.text) + len(_SEPARATOR) for b in self._buffer)

    def _overlap_seed(self, blocks: list[Block]) -> list[Block]:
        """Keep trailing whole blocks (up to ``overlap`` chars) for continuity."""
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
        # Never carry the whole chunk forward, or it would be re-emitted verbatim.
        if len(seed) == len(blocks):
            seed = seed[1:]
        return seed

    def _emit_standalone(self, block: Block, text: str) -> None:
        if block.block_type == BlockType.TABLE or len(text) <= self._max_chars:
            self._append(text, [block])
            return

        step = self._max_chars - self._overlap
        start = 0
        while start < len(text):
            self._append(text[start : start + self._max_chars], [block])
            start += step

    def _reset_buffer(self) -> None:
        self._buffer = []
        self._buffer_len = 0
        self._dirty = False

    def _append(self, text: str, blocks: list[Block]) -> None:
        self._chunks.append(
            Chunk(
                id=f"{blocks[0].id}-chunk-{self._index}",
                text=text,
                metadata=self._build_metadata(blocks),
            )
        )
        self._index += 1

    def _build_metadata(self, blocks: list[Block]) -> dict[str, Any]:
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
        if self._section:
            metadata["section"] = self._section
        if len(blocks) == 1 and blocks[0].bbox is not None:
            metadata["bbox"] = blocks[0].bbox
        if descriptions:
            metadata["vlm_description"] = " ".join(descriptions)

        return metadata
