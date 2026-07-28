from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime

from file_agent.chunking import Chunk
from file_agent.document import Document
from file_agent.llm.base import LLMClient
from file_agent.qa import answer_question_with_context
from file_agent.rag_core import (
    RAGResponse,
    chunk_documents,
    load_documents,
)
from file_agent.retrieval import Retriever, SearchResult


class IngestionState(TypedDict, total=False):
    file_paths: list[str | Path]
    documents: list[Document]
    chunks: list[Chunk]
    documents_count: int
    chunks_count: int


@dataclass(frozen=True)
class IngestionContext:
    retriever: Retriever
    max_chars: int = 1000
    overlap: int = 100


class QAState(TypedDict, total=False):
    question: str
    top_k: int
    results: list[SearchResult]
    answer: str
    documents_count: int
    chunks_count: int
    response: RAGResponse


@dataclass(frozen=True)
class QAContext:
    llm_client: LLMClient
    retriever: Retriever | None = None


def load_documents_node(state: IngestionState) -> dict:
    documents = load_documents(state["file_paths"])
    return {
        "documents": documents,
        "documents_count": len(documents),
    }


def chunk_documents_node(
    state: IngestionState,
    runtime: Runtime[IngestionContext],
) -> dict:
    documents = state["documents"]
    chunks = chunk_documents(
        documents=documents,
        max_chars=runtime.context.max_chars,
        overlap=runtime.context.overlap,
    )
    return {
        "chunks": chunks,
        "documents_count": len(documents),
        "chunks_count": len(chunks),
    }


def index_documents_node(
    state: IngestionState,
    runtime: Runtime[IngestionContext],
) -> dict:
    runtime.context.retriever.index(state["chunks"])
    return {}


def retrieve_node(
    state: QAState,
    runtime: Runtime[QAContext],
) -> dict:
    retriever = runtime.context.retriever
    if retriever is None:
        raise ValueError("A retriever is required when search results are not precomputed")

    results = retriever.search(
        query=state["question"],
        top_k=state.get("top_k", 5),
    )
    return {"results": results}


def generate_answer_node(
    state: QAState,
    runtime: Runtime[QAContext],
) -> dict:
    answer = answer_question_with_context(
        question=state["question"],
        results=state["results"],
        llm_client=runtime.context.llm_client,
    )
    return {"answer": answer}


def build_response_node(state: QAState) -> dict:
    return {
        "response": RAGResponse(
            answer=state["answer"],
            sources=state["results"],
            documents_count=state.get("documents_count", 0),
            chunks_count=state.get("chunks_count", 0),
        )
    }


def route_ingestion_input(
    state: IngestionState,
) -> Literal["load_documents", "chunk_documents"]:
    if "documents" in state:
        return "chunk_documents"
    return "load_documents"


def route_qa_input(state: QAState) -> Literal["retrieve", "generate_answer"]:
    if "results" in state:
        return "generate_answer"
    return "retrieve"


def build_ingestion_graph():
    builder = StateGraph(
        IngestionState,
        context_schema=IngestionContext,
    )
    builder.add_node("load_documents", load_documents_node)
    builder.add_node("chunk_documents", chunk_documents_node)
    builder.add_node("index_documents", index_documents_node)
    builder.add_conditional_edges(
        START,
        route_ingestion_input,
        {
            "load_documents": "load_documents",
            "chunk_documents": "chunk_documents",
        },
    )
    builder.add_edge("load_documents", "chunk_documents")
    builder.add_edge("chunk_documents", "index_documents")
    builder.add_edge("index_documents", END)
    return builder.compile(name="rag_ingestion")


def build_qa_graph():
    builder = StateGraph(
        QAState,
        context_schema=QAContext,
    )
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("generate_answer", generate_answer_node)
    builder.add_node("build_response", build_response_node)
    builder.add_conditional_edges(
        START,
        route_qa_input,
        {
            "retrieve": "retrieve",
            "generate_answer": "generate_answer",
        },
    )
    builder.add_edge("retrieve", "generate_answer")
    builder.add_edge("generate_answer", "build_response")
    builder.add_edge("build_response", END)
    return builder.compile(name="rag_qa")


ingestion_graph = build_ingestion_graph()
qa_graph = build_qa_graph()
