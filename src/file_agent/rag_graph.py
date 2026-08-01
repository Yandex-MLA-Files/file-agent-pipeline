from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime

from file_agent.chunking import Chunk
from file_agent.document import Document
from file_agent.llm.base import LLMClient
from file_agent.qa import (
    NO_CONTEXT_MESSAGE,
    QUERY_ROUTE_CLARIFY,
    answer_question_with_context,
    build_clarification_message,
    build_context_from_results,
    build_context_grading_prompt,
    build_query_analysis_prompt,
    build_query_rewrite_prompt,
    normalize_rewritten_query,
    parse_context_relevance,
    parse_query_route,
)
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
    search_query: str
    top_k: int
    results: list[SearchResult]
    answer: str
    query_route: str
    context_relevant: bool
    can_retry: bool
    retry_count: int
    max_retries: int
    search_queries: list[str]
    stop_reason: str
    documents_count: int
    chunks_count: int
    response: RAGResponse


@dataclass(frozen=True)
class QAContext:
    llm_client: LLMClient
    retriever: Retriever | None = None
    max_retries: int = 2


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

    search_query = state.get("search_query") or state["question"]
    results = retriever.search(
        query=search_query,
        top_k=state.get("top_k", 5),
    )
    search_queries = list(state.get("search_queries", []))
    search_queries.append(search_query)
    return {
        "results": results,
        "search_query": search_query,
        "search_queries": search_queries,
    }


def generate_answer_node(
    state: QAState,
    runtime: Runtime[QAContext],
) -> dict:
    answer = answer_question_with_context(
        question=state["question"],
        results=state["results"],
        llm_client=runtime.context.llm_client,
    )
    return {
        "answer": answer,
        "stop_reason": "answer_generated" if state["results"] else "no_context",
    }


def analyze_question_node(
    state: QAState,
    runtime: Runtime[QAContext],
) -> dict:
    question = state["question"].strip()
    decision = runtime.context.llm_client.generate(build_query_analysis_prompt(question))
    return {
        "query_route": parse_query_route(decision),
        "search_query": state.get("search_query") or question,
        "retry_count": state.get("retry_count", 0),
        "max_retries": state.get("max_retries", runtime.context.max_retries),
        "search_queries": list(state.get("search_queries", [])),
    }


def grade_context_node(
    state: QAState,
    runtime: Runtime[QAContext],
) -> dict:
    results = state.get("results", [])
    if not results:
        context_relevant = False
    else:
        context = build_context_from_results(results)
        decision = runtime.context.llm_client.generate(
            build_context_grading_prompt(
                question=state["question"],
                context=context,
            )
        )
        context_relevant = parse_context_relevance(decision)

    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", runtime.context.max_retries)
    return {
        "context_relevant": context_relevant,
        "can_retry": runtime.context.retriever is not None and retry_count < max_retries,
    }


def rewrite_query_node(
    state: QAState,
    runtime: Runtime[QAContext],
) -> dict:
    previous_query = state.get("search_query") or state["question"]
    rewritten = runtime.context.llm_client.generate(
        build_query_rewrite_prompt(
            original_question=state["question"],
            previous_query=previous_query,
        )
    )
    return {
        "search_query": normalize_rewritten_query(rewritten, fallback=previous_query),
        "retry_count": state.get("retry_count", 0) + 1,
    }


def clarify_question_node(state: QAState) -> dict:
    return {
        "answer": build_clarification_message(state["question"]),
        "results": [],
        "stop_reason": "clarification_needed",
    }


def no_context_node(state: QAState) -> dict:
    return {
        "answer": NO_CONTEXT_MESSAGE,
        "results": [],
        "stop_reason": "insufficient_context",
    }


def build_response_node(state: QAState) -> dict:
    return {
        "response": RAGResponse(
            answer=state["answer"],
            sources=state["results"],
            documents_count=state.get("documents_count", 0),
            chunks_count=state.get("chunks_count", 0),
            search_queries=list(state.get("search_queries", [])),
            retry_count=state.get("retry_count", 0),
            stop_reason=state.get("stop_reason", "answer_generated"),
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


def route_after_question_analysis(
    state: QAState,
) -> Literal["clarify_question", "retrieve", "grade_context"]:
    if state["query_route"] == QUERY_ROUTE_CLARIFY:
        return "clarify_question"
    if "results" in state:
        return "grade_context"
    return "retrieve"


def route_after_context_grade(
    state: QAState,
) -> Literal["generate_answer", "rewrite_query", "no_context"]:
    if state["context_relevant"]:
        return "generate_answer"
    if state["can_retry"]:
        return "rewrite_query"
    return "no_context"


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


def build_agentic_qa_graph():
    builder = StateGraph(
        QAState,
        context_schema=QAContext,
    )
    builder.add_node("analyze_question", analyze_question_node)
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("grade_context", grade_context_node)
    builder.add_node("rewrite_query", rewrite_query_node)
    builder.add_node("clarify_question", clarify_question_node)
    builder.add_node("no_context", no_context_node)
    builder.add_node("generate_answer", generate_answer_node)
    builder.add_node("build_response", build_response_node)

    builder.add_edge(START, "analyze_question")
    builder.add_conditional_edges(
        "analyze_question",
        route_after_question_analysis,
        {
            "clarify_question": "clarify_question",
            "retrieve": "retrieve",
            "grade_context": "grade_context",
        },
    )
    builder.add_edge("retrieve", "grade_context")
    builder.add_conditional_edges(
        "grade_context",
        route_after_context_grade,
        {
            "generate_answer": "generate_answer",
            "rewrite_query": "rewrite_query",
            "no_context": "no_context",
        },
    )
    builder.add_edge("rewrite_query", "retrieve")
    builder.add_edge("clarify_question", "build_response")
    builder.add_edge("no_context", "build_response")
    builder.add_edge("generate_answer", "build_response")
    builder.add_edge("build_response", END)
    return builder.compile(name="agentic_rag_qa")


ingestion_graph = build_ingestion_graph()
qa_graph = build_qa_graph()
agentic_qa_graph = build_agentic_qa_graph()
