from collections.abc import Iterable
from pathlib import Path

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
    ingestion_graph,
    qa_graph,
)
from file_agent.retrieval import Retriever, SearchResult

__all__ = [
    "RAGResponse",
    "answer_documents",
    "answer_files",
    "answer_indexed_documents",
    "answer_with_results",
    "chunk_documents",
    "index_documents",
    "ingest_documents",
    "ingest_files",
    "load_documents",
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
) -> RAGResponse:
    state = qa_graph.invoke(
        {
            "question": question,
            "top_k": top_k,
            "documents_count": documents_count,
            "chunks_count": chunks_count,
        },
        context=QAContext(
            llm_client=llm_client,
            retriever=retriever,
        ),
    )
    return state["response"]


def answer_with_results(
    question: str,
    results: list[SearchResult],
    llm_client: LLMClient,
    documents_count: int,
    chunks_count: int,
) -> RAGResponse:
    state = qa_graph.invoke(
        {
            "question": question,
            "results": results,
            "documents_count": documents_count,
            "chunks_count": chunks_count,
        },
        context=QAContext(llm_client=llm_client),
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
    )
