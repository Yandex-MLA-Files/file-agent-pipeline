import os
from collections.abc import Iterable
from pathlib import Path
from typing import Literal, cast

from file_agent.chunking import Chunk
from file_agent.document import Document
from file_agent.lancedb_retriever import LanceDBRetriever
from file_agent.llm.base import LLMClient
from file_agent.rag_core import (
    RAGResponse,
    chunk_documents,
    index_documents,
    load_documents,
)
from file_agent.rag_graph import (
    IngestionContext,
    QAContext,
    agentic_qa_graph,
    ingestion_graph,
    qa_graph,
)
from file_agent.retrieval import Retriever, SearchResult

RAGMode = Literal["standard", "agentic"]
DEFAULT_RAG_MODE: RAGMode = "standard"
DEFAULT_AGENTIC_MAX_RETRIES = 2

__all__ = [
    "RAGResponse",
    "RAGMode",
    "answer_documents",
    "answer_files",
    "answer_indexed_documents",
    "answer_with_results",
    "chunk_documents",
    "index_documents",
    "ingest_documents",
    "ingest_files",
    "load_documents",
    "resolve_agentic_max_retries",
    "resolve_rag_mode",
]


def ingest_files(
    file_paths: Iterable[str | Path],
    retriever: Retriever,
    max_chars: int = 1000,
    overlap: int = 100,
) -> tuple[list[Document], list[Chunk]]:
    state = ingestion_graph.invoke(
        {"file_paths": list(file_paths)},
        context=IngestionContext(
            retriever=retriever,
            max_chars=max_chars,
            overlap=overlap,
        ),
    )
    return state["documents"], state["chunks"]


def ingest_documents(
    documents: list[Document],
    retriever: Retriever,
    max_chars: int = 1000,
    overlap: int = 100,
) -> list[Chunk]:
    state = ingestion_graph.invoke(
        {"documents": documents},
        context=IngestionContext(
            retriever=retriever,
            max_chars=max_chars,
            overlap=overlap,
        ),
    )
    return state["chunks"]


def answer_indexed_documents(
    question: str,
    llm_client: LLMClient,
    retriever: Retriever,
    documents_count: int,
    chunks_count: int,
    top_k: int = 5,
    mode: str | None = None,
    max_retries: int | None = None,
) -> RAGResponse:
    active_graph, context = _qa_runtime(
        mode=mode,
        max_retries=max_retries,
        llm_client=llm_client,
        retriever=retriever,
    )
    state = active_graph.invoke(
        {
            "question": question,
            "top_k": top_k,
            "documents_count": documents_count,
            "chunks_count": chunks_count,
        },
        context=context,
    )
    return state["response"]


def answer_with_results(
    question: str,
    results: list[SearchResult],
    llm_client: LLMClient,
    documents_count: int,
    chunks_count: int,
    retriever: Retriever | None = None,
    mode: str | None = None,
    max_retries: int | None = None,
) -> RAGResponse:
    active_graph, context = _qa_runtime(
        mode=mode,
        max_retries=max_retries,
        llm_client=llm_client,
        retriever=retriever,
    )
    state = active_graph.invoke(
        {
            "question": question,
            "results": results,
            "search_queries": [question],
            "documents_count": documents_count,
            "chunks_count": chunks_count,
        },
        context=context,
    )
    return state["response"]


def answer_files(
    file_paths: Iterable[str | Path],
    question: str,
    llm_client: LLMClient,
    top_k: int = 5,
    max_chars: int = 1000,
    overlap: int = 100,
    retriever: Retriever | None = None,
    mode: str | None = None,
    max_retries: int | None = None,
) -> RAGResponse:
    active_retriever = retriever or LanceDBRetriever()
    documents, chunks = ingest_files(
        file_paths=file_paths,
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
        mode=mode,
        max_retries=max_retries,
    )


def answer_documents(
    documents: list[Document],
    question: str,
    llm_client: LLMClient,
    top_k: int = 5,
    max_chars: int = 1000,
    overlap: int = 100,
    retriever: Retriever | None = None,
    mode: str | None = None,
    max_retries: int | None = None,
) -> RAGResponse:
    active_retriever = retriever or LanceDBRetriever()
    chunks = ingest_documents(
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
        mode=mode,
        max_retries=max_retries,
    )


def resolve_rag_mode(mode: str | None = None) -> RAGMode:
    value = (mode or os.getenv("RAG_MODE", DEFAULT_RAG_MODE)).strip().lower()
    if value not in ("standard", "agentic"):
        raise ValueError(f"Unsupported RAG_MODE: {value}")
    return cast(RAGMode, value)


def resolve_agentic_max_retries(max_retries: int | None = None) -> int:
    value: int
    if max_retries is not None:
        value = max_retries
    else:
        raw_value = os.getenv("RAG_MAX_RETRIES", str(DEFAULT_AGENTIC_MAX_RETRIES))
        try:
            value = int(raw_value)
        except ValueError as exc:
            raise ValueError("RAG_MAX_RETRIES must be an integer") from exc

    if value < 0:
        raise ValueError("RAG_MAX_RETRIES must be greater than or equal to zero")
    return value


def _qa_runtime(
    mode: str | None,
    max_retries: int | None,
    llm_client: LLMClient,
    retriever: Retriever | None,
):
    active_mode = resolve_rag_mode(mode)
    if active_mode == "agentic":
        return (
            agentic_qa_graph,
            QAContext(
                llm_client=llm_client,
                retriever=retriever,
                max_retries=resolve_agentic_max_retries(max_retries),
            ),
        )
    return (
        qa_graph,
        QAContext(
            llm_client=llm_client,
            retriever=retriever,
        ),
    )
