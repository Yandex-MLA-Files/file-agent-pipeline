from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from file_agent.chunking import Chunk, chunk_document
from file_agent.document import Document
from file_agent.lancedb_retriever import LanceDBRetriever
from file_agent.llm.base import LLMClient
from file_agent.pipeline import parse_file
from file_agent.qa import answer_question_with_context
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
    chunks: list[Chunk] = []

    for document in documents:
        chunks.extend(
            chunk_document(
                document=document,
                max_chars=max_chars,
                overlap=overlap,
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


def answer_indexed_documents(
    question: str,
    llm_client: LLMClient,
    retriever: Retriever,
    documents_count: int,
    chunks_count: int,
    top_k: int = 5,
) -> RAGResponse:
    results = retriever.search(query=question, top_k=top_k)
    return answer_with_results(
        question=question,
        results=results,
        llm_client=llm_client,
        documents_count=documents_count,
        chunks_count=chunks_count,
    )


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
