import logging
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from file_agent.chunking import Chunk, chunk_document, get_embedding_tokenizer
from file_agent.document import Document
from file_agent.pipeline import parse_file
from file_agent.retrieval import Retriever, SearchResult
from file_agent.telemetry import tracer

logger = logging.getLogger(__name__)


@dataclass
class RAGResponse:
    answer: str
    sources: list[SearchResult]
    documents_count: int
    chunks_count: int
    search_queries: list[str] = field(default_factory=list)
    retry_count: int = 0
    stop_reason: str = "answer_generated"
    tool_calls: list[dict[str, Any]] = field(default_factory=list)


def load_documents(file_paths: Iterable[str | Path]) -> list[Document]:
    file_paths = list(file_paths)
    with tracer.start_as_current_span("file_agent.load_documents") as span:
        span.set_attribute("file_agent.file_count", len(file_paths))
        documents = [parse_file(file_path) for file_path in file_paths]
        logger.info("Loaded %d document(s)", len(documents))
        return documents


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
    with tracer.start_as_current_span("file_agent.index_documents") as span:
        span.set_attribute("file_agent.document_count", len(documents))
        chunks = chunk_documents(
            documents=documents,
            max_chars=max_chars,
            overlap=overlap,
        )
        retriever.index(chunks)
        span.set_attribute("file_agent.chunk_count", len(chunks))
        logger.info("Indexed %d chunk(s) from %d document(s)", len(chunks), len(documents))
        return chunks
