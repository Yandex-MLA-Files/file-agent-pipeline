from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from file_agent.chunking import Chunk, chunk_document, get_embedding_tokenizer
from file_agent.document import Document
from file_agent.pipeline import parse_file
from file_agent.retrieval import Retriever, SearchResult


@dataclass
class RAGResponse:
    answer: str
    sources: list[SearchResult]
    documents_count: int
    chunks_count: int


def load_documents(file_paths: Iterable[str | Path]) -> list[Document]:
    return [parse_file(file_path) for file_path in file_paths]


def chunk_documents(
    documents: list[Document],
    max_chars: int = 1000,
    overlap: int = 100,
) -> list[Chunk]:
    # Budget chunks in the retrieval encoder's own tokens so nothing is silently
    # truncated when they are embedded; falls back to characters when the
    # tokenizer cannot be loaded (e.g. offline).
    tokenizer = get_embedding_tokenizer()
    chunks: list[Chunk] = []

    for document in documents:
        chunks.extend(
            chunk_document(
                document=document,
                max_chars=max_chars,
                overlap=overlap,
                tokenizer=tokenizer,
            )
        )

    return chunks


def index_documents(
    documents: list[Document],
    retriever: Retriever,
    max_chars: int = 1000,
    overlap: int = 100,
) -> list[Chunk]:
    chunks = chunk_documents(
        documents=documents,
        max_chars=max_chars,
        overlap=overlap,
    )
    retriever.index(chunks)
    return chunks
