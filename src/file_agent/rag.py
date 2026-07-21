import logging
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from file_agent.chunking import Chunk, chunk_document, get_embedding_tokenizer
from file_agent.document import Document
from file_agent.lancedb_retriever import LanceDBRetriever
from file_agent.llm.base import LLMClient
from file_agent.pipeline import parse_file
from file_agent.qa import answer_question_with_context
from file_agent.retrieval import Retriever, SearchResult
from file_agent.telemetry import tracer

logger = logging.getLogger(__name__)


@dataclass
class RAGResponse:
    answer: str
    sources: list[SearchResult]
    documents_count: int
    chunks_count: int


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


def answer_indexed_documents(
    question: str,
    llm_client: LLMClient,
    retriever: Retriever,
    documents_count: int,
    chunks_count: int,
    top_k: int = 5,
) -> RAGResponse:
    with tracer.start_as_current_span("file_agent.answer_indexed_documents") as span:
        span.set_attribute("file_agent.question", question)
        span.set_attribute("file_agent.top_k", top_k)

        results = retriever.search(query=question, top_k=top_k)
        response = answer_with_results(
            question=question,
            results=results,
            llm_client=llm_client,
            documents_count=documents_count,
            chunks_count=chunks_count,
        )

        span.set_attribute("file_agent.result_count", len(results))
        return response


def answer_with_results(
    question: str,
    results: list[SearchResult],
    llm_client: LLMClient,
    documents_count: int,
    chunks_count: int,
) -> RAGResponse:
    answer = answer_question_with_context(
        question=question,
        results=results,
        llm_client=llm_client,
    )

    return RAGResponse(
        answer=answer,
        sources=results,
        documents_count=documents_count,
        chunks_count=chunks_count,
    )


def answer_files(
    file_paths: Iterable[str | Path],
    question: str,
    llm_client: LLMClient,
    top_k: int = 5,
    max_chars: int = 1000,
    overlap: int = 100,
    retriever: Retriever | None = None,
) -> RAGResponse:
    documents = load_documents(file_paths)
    return answer_documents(
        documents=documents,
        question=question,
        llm_client=llm_client,
        top_k=top_k,
        max_chars=max_chars,
        overlap=overlap,
        retriever=retriever,
    )


def answer_documents(
    documents: list[Document],
    question: str,
    llm_client: LLMClient,
    top_k: int = 5,
    max_chars: int = 1000,
    overlap: int = 100,
    retriever: Retriever | None = None,
) -> RAGResponse:
    active_retriever = retriever or LanceDBRetriever()
    chunks = index_documents(
        documents=documents,
        retriever=active_retriever,
        max_chars=max_chars,
        overlap=overlap,
    )
    return answer_indexed_documents(
        question=question,
        llm_client=llm_client,
        retriever=active_retriever,
        documents_count=len(documents),
        chunks_count=len(chunks),
        top_k=top_k,
    )
