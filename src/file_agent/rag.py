import json
import os
import uuid
from collections.abc import Iterable
from pathlib import Path
from typing import Literal, cast

from file_agent.agent_tools import (
    DEFAULT_HISTORY_TURNS,
    ToolAgentContext,
)
from file_agent.chunking import Chunk
from file_agent.document import Document
from file_agent.document_assets import DocumentAssetStore, InMemoryDocumentAssetStore
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
from file_agent.telemetry import tracer
from file_agent.vlm.base import VLMClient

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
    "resolve_history_turns",
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
    thread_id: str | None = None,
    max_history_turns: int | None = None,
    vlm_client: VLMClient | None = None,
    asset_store: DocumentAssetStore | None = None,
) -> RAGResponse:
    with tracer.start_as_current_span("file_agent.answer_indexed_documents") as span:
        span.set_attribute("file_agent.question", question)
        span.set_attribute("file_agent.top_k", top_k)
        span.set_attribute("langfuse.observation.type", "agent")
        span.set_attribute(
            "langfuse.observation.input",
            json.dumps({"question": question, "top_k": top_k}, ensure_ascii=False),
        )

        active_mode = resolve_rag_mode(mode)
        span.set_attribute("langfuse.observation.metadata.rag_mode", active_mode)
        if active_mode == "tool_agent":
            response = _answer_with_tool_agent(
                question=question,
                llm_client=llm_client,
                retriever=retriever,
                documents=documents or [],
                documents_count=documents_count,
                chunks_count=chunks_count,
                max_tool_rounds=max_tool_rounds,
                thread_id=thread_id,
                max_history_turns=max_history_turns,
                vlm_client=vlm_client,
                asset_store=asset_store,
            )
        else:
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
            response = state["response"]

        span.set_attribute("file_agent.result_count", len(response.sources))
        span.set_attribute(
            "langfuse.observation.output",
            json.dumps(
                {
                    "answer": response.answer,
                    "source_chunk_ids": [result.chunk.id for result in response.sources],
                    "stop_reason": response.stop_reason,
                },
                ensure_ascii=False,
            ),
        )
        return response


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
    thread_id: str | None = None,
    max_history_turns: int | None = None,
    vlm_client: VLMClient | None = None,
    asset_store: DocumentAssetStore | None = None,
) -> RAGResponse:
    with tracer.start_as_current_span("file_agent.answer_with_results") as span:
        span.set_attribute("file_agent.question", question)
        span.set_attribute("file_agent.result_count", len(results))
        span.set_attribute("langfuse.observation.type", "agent")
        span.set_attribute(
            "langfuse.observation.input",
            json.dumps(
                {
                    "question": question,
                    "source_chunk_ids": [result.chunk.id for result in results],
                },
                ensure_ascii=False,
            ),
        )

        active_mode = resolve_rag_mode(mode)
        span.set_attribute("langfuse.observation.metadata.rag_mode", active_mode)
        if active_mode == "tool_agent":
            if retriever is None:
                raise ValueError("A retriever is required for RAG_MODE=tool_agent")
            response = _answer_with_tool_agent(
                question=question,
                llm_client=llm_client,
                retriever=retriever,
                documents=documents or [],
                documents_count=documents_count,
                chunks_count=chunks_count,
                max_tool_rounds=max_tool_rounds,
                thread_id=thread_id,
                max_history_turns=max_history_turns,
                vlm_client=vlm_client,
                asset_store=asset_store,
            )
        else:
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
            response = state["response"]

        span.set_attribute(
            "langfuse.observation.output",
            json.dumps(
                {
                    "answer": response.answer,
                    "source_chunk_ids": [result.chunk.id for result in response.sources],
                    "stop_reason": response.stop_reason,
                },
                ensure_ascii=False,
            ),
        )
        return response


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
    thread_id: str | None = None,
    max_history_turns: int | None = None,
    vlm_client: VLMClient | None = None,
    asset_store: DocumentAssetStore | None = None,
) -> RAGResponse:
    paths = list(file_paths)
    active_retriever = retriever or LanceDBRetriever()
    active_asset_store = asset_store
    if active_asset_store is None and resolve_rag_mode(mode) == "tool_agent":
        active_asset_store = InMemoryDocumentAssetStore.from_files(paths)
    documents, chunks = ingest_files(
        file_paths=paths,
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
        thread_id=thread_id,
        max_history_turns=max_history_turns,
        vlm_client=vlm_client,
        asset_store=active_asset_store,
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
    thread_id: str | None = None,
    max_history_turns: int | None = None,
    vlm_client: VLMClient | None = None,
    asset_store: DocumentAssetStore | None = None,
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
        thread_id=thread_id,
        max_history_turns=max_history_turns,
        vlm_client=vlm_client,
        asset_store=asset_store,
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


def resolve_history_turns(max_history_turns: int | None = None) -> int:
    value: int
    if max_history_turns is not None:
        value = max_history_turns
    else:
        raw_value = os.getenv("RAG_HISTORY_TURNS", str(DEFAULT_HISTORY_TURNS))
        try:
            value = int(raw_value)
        except ValueError as exc:
            raise ValueError("RAG_HISTORY_TURNS must be an integer") from exc

    if value < 1:
        raise ValueError("RAG_HISTORY_TURNS must be greater than zero")
    return value


def _answer_with_tool_agent(
    question: str,
    llm_client: LLMClient,
    retriever: Retriever,
    documents: list[Document],
    documents_count: int,
    chunks_count: int,
    max_tool_rounds: int | None,
    thread_id: str | None,
    max_history_turns: int | None,
    vlm_client: VLMClient | None,
    asset_store: DocumentAssetStore | None,
) -> RAGResponse:
    if not isinstance(llm_client, ToolCallingLLMClient):
        raise TypeError("The configured LLM client does not support native tool calling")

    active_thread_id = thread_id.strip() if thread_id is not None else uuid.uuid4().hex
    if not active_thread_id:
        raise ValueError("thread_id must not be empty")

    state = tool_agent_graph.invoke(
        {
            "question": question,
            "documents_count": documents_count,
            "chunks_count": chunks_count,
        },
        config={"configurable": {"thread_id": active_thread_id}},
        context=ToolAgentContext(
            llm_client=llm_client,
            retriever=retriever,
            documents=documents,
            max_tool_rounds=resolve_max_tool_rounds(max_tool_rounds),
            max_history_turns=resolve_history_turns(max_history_turns),
            vlm_client=vlm_client,
            asset_store=asset_store,
        ),
    )
    return state["response"]
