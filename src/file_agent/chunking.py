from dataclasses import dataclass, field
from typing import Any

from file_agent.document import Block, Document


@dataclass
class Chunk:
    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


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
    metadata: dict[str, Any] = {
        "source_file": block.metadata.get("source_file", document.file_name),
        "block_id": block.id,
        "block_type": block.type,
    }

    if "page_number" in block.metadata:
        metadata["page_number"] = block.metadata["page_number"]

    return metadata
