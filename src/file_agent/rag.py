import os
from collections.abc import Iterable
from pathlib import Path
from typing import Literal, cast

from langchain_core.messages import HumanMessage, SystemMessage

from file_agent.agent_tools import (
    TOOL_AGENT_SYSTEM_PROMPT,
    ToolAgentContext,
)
from file_agent.chunking import Chunk
from file_agent.document import Document
from file_agent.lancedb_retriever import LanceDBRetriever
from file_agent.llm.base import LLMClient, ToolCallingLLMClient
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
    tool_agent_graph,
)
from file_agent.retrieval import Retriever, SearchResult

RAGMode = Literal["standard", "tool_agent"]
DEFAULT_RAG_MODE: RAGMode = "standard"
DEFAULT_MAX_TOOL_ROUNDS = 4

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
    "resolve_max_tool_rounds",
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
    documents: list[Document] | None = None,
    mode: str | None = None,
    max_tool_rounds: int | None = None,
) -> RAGResponse:
    if resolve_rag_mode(mode) == "tool_agent":
        return _answer_with_tool_agent(
            question=question,
            llm_client=llm_client,
            retriever=retriever,
            documents=documents or [],
            documents_count=documents_count,
            chunks_count=chunks_count,
            max_tool_rounds=max_tool_rounds,
        )

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
    retriever: Retriever | None = None,
    documents: list[Document] | None = None,
    mode: str | None = None,
    max_tool_rounds: int | None = None,
) -> RAGResponse:
    if resolve_rag_mode(mode) == "tool_agent":
        if retriever is None:
            raise ValueError("A retriever is required for RAG_MODE=tool_agent")
        return _answer_with_tool_agent(
            question=question,
            llm_client=llm_client,
            retriever=retriever,
            documents=documents or [],
            documents_count=documents_count,
            chunks_count=chunks_count,
            max_tool_rounds=max_tool_rounds,
        )

    state = qa_graph.invoke(
        {
            "question": question,
            "results": results,
            "search_queries": [question],
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
    mode: str | None = None,
    max_tool_rounds: int | None = None,
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
        documents=documents,
        mode=mode,
        max_tool_rounds=max_tool_rounds,
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
    max_tool_rounds: int | None = None,
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
        documents=documents,
        mode=mode,
        max_tool_rounds=max_tool_rounds,
    )


def resolve_rag_mode(mode: str | None = None) -> RAGMode:
    value = (mode or os.getenv("RAG_MODE", DEFAULT_RAG_MODE)).strip().lower()
    if value not in ("standard", "tool_agent"):
        raise ValueError(f"Unsupported RAG_MODE: {value}")
    return cast(RAGMode, value)


def resolve_max_tool_rounds(max_tool_rounds: int | None = None) -> int:
    value: int
    if max_tool_rounds is not None:
        value = max_tool_rounds
    else:
        raw_value = os.getenv("RAG_MAX_TOOL_ROUNDS", str(DEFAULT_MAX_TOOL_ROUNDS))
        try:
            value = int(raw_value)
        except ValueError as exc:
            raise ValueError("RAG_MAX_TOOL_ROUNDS must be an integer") from exc

    if value < 1:
        raise ValueError("RAG_MAX_TOOL_ROUNDS must be greater than zero")
    return value


def _answer_with_tool_agent(
    question: str,
    llm_client: LLMClient,
    retriever: Retriever,
    documents: list[Document],
    documents_count: int,
    chunks_count: int,
    max_tool_rounds: int | None,
) -> RAGResponse:
    if not isinstance(llm_client, ToolCallingLLMClient):
        raise TypeError("The configured LLM client does not support native tool calling")

    state = tool_agent_graph.invoke(
        {
            "messages": [
                SystemMessage(content=TOOL_AGENT_SYSTEM_PROMPT),
                HumanMessage(content=question),
            ],
            "question": question,
            "sources": [],
            "search_queries": [],
            "documents_count": documents_count,
            "chunks_count": chunks_count,
        },
        context=ToolAgentContext(
            llm_client=llm_client,
            retriever=retriever,
            documents=documents,
            max_tool_rounds=resolve_max_tool_rounds(max_tool_rounds),
        ),
    )
    return state["response"]
