from dataclasses import dataclass, field
from typing import Any

from file_agent.document import Block, BlockType, Document


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
    max_chars: int = 1000,
    overlap: int = 100,
) -> list[Chunk]:
    if max_chars <= 0:
        raise ValueError("max_chars must be greater than 0")
    if overlap < 0:
        raise ValueError("overlap must be greater than or equal to 0")
    if overlap >= max_chars:
        raise ValueError("overlap must be smaller than max_chars")

    chunks: list[Chunk] = []
    for block in document.blocks:
        chunks.extend(_chunk_block(document, block, max_chars, overlap))

    return chunks


def _chunk_block(
    document: Document,
    block: Block,
    max_chars: int,
    overlap: int,
) -> list[Chunk]:
    if not block.text:
        return []

    # Keep structured tables intact: splitting Markdown tables mid-row would
    # break their layout and make them useless for retrieval and rendering.
    if block.block_type == BlockType.TABLE:
        return [
            Chunk(
                id=f"{block.id}-chunk-1",
                text=block.text,
                metadata=_build_chunk_metadata(document, block),
            )
        ]

    chunks: list[Chunk] = []
    step = max_chars - overlap
    start = 0
    chunk_index = 1

    while start < len(block.text):
        end = start + max_chars
        chunk_text = block.text[start:end]
        metadata = _build_chunk_metadata(document, block)

        chunks.append(
            Chunk(
                id=f"{block.id}-chunk-{chunk_index}",
                text=chunk_text,
                metadata=metadata,
            )
        )

        start += step
        chunk_index += 1

    return chunks


def _build_chunk_metadata(document: Document, block: Block) -> dict[str, Any]:
    metadata = dict(block.metadata)
    metadata["source_file"] = metadata.get("source_file", document.file_name)
    metadata["block_id"] = block.id
    metadata["block_type"] = block.type

    # Structural fields are only populated by rich parsers (e.g. Docling). When
    # present they are propagated so retrieval can filter and trace results by
    # page, region (bbox) or visual modality.
    if block.page_number is not None:
        metadata.setdefault("page_number", block.page_number)
    if block.bbox is not None:
        metadata.setdefault("bbox", block.bbox)
    if block.vlm_description:
        metadata["vlm_description"] = block.vlm_description

    return metadata
