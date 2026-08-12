from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
)
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.prebuilt import ToolNode
from langgraph.runtime import Runtime
from langgraph.types import Overwrite

from file_agent.agent_tools import (
    DOCUMENT_TOOLS,
    TOOL_AGENT_SYSTEM_PROMPT,
    TOOL_LIMIT_MESSAGE,
    ToolAgentContext,
)
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
from file_agent.telemetry import tracer


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
    search_queries: list[str]
    stop_reason: str
    documents_count: int
    chunks_count: int
    response: RAGResponse


def merge_search_results(
    current: list[SearchResult],
    new: list[SearchResult],
) -> list[SearchResult]:
    merged = list(current)
    seen_chunk_ids = {result.chunk.id for result in current}
    for result in new:
        if result.chunk.id in seen_chunk_ids:
            continue
        seen_chunk_ids.add(result.chunk.id)
        merged.append(result)
    return merged


def append_strings(current: list[str], new: list[str]) -> list[str]:
    return [*current, *new]


class ToolAgentState(MessagesState):
    question: str
    conversation_history: list[BaseMessage]
    sources: Annotated[list[SearchResult], merge_search_results]
    search_queries: Annotated[list[str], append_strings]
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
    with tracer.start_as_current_span("file_agent.index_documents") as span:
        span.set_attribute("file_agent.document_count", len(state["documents"]))
        span.set_attribute("file_agent.chunk_count", len(state["chunks"]))
        runtime.context.retriever.index(state["chunks"])
    return {}


def retrieve_node(
    state: QAState,
    runtime: Runtime[QAContext],
) -> dict:
    retriever = runtime.context.retriever
    if retriever is None:
        raise ValueError("A retriever is required when search results are not precomputed")

    search_query = state["question"]
    results = retriever.search(
        query=search_query,
        top_k=state.get("top_k", 5),
    )
    search_queries = list(state.get("search_queries", []))
    search_queries.append(search_query)
    return {
        "results": results,
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


def build_response_node(state: QAState) -> dict:
    return {
        "response": RAGResponse(
            answer=state["answer"],
            sources=state["results"],
            documents_count=state.get("documents_count", 0),
            chunks_count=state.get("chunks_count", 0),
            search_queries=list(state.get("search_queries", [])),
            retry_count=0,
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


def prepare_tool_agent_turn_node(
    state: ToolAgentState,
    runtime: Runtime[ToolAgentContext],
) -> dict:
    """Build this turn's working messages from compact persisted conversation history."""
    history = _trim_conversation_history(
        list(state.get("conversation_history", [])),
        runtime.context.max_history_turns,
    )
    return {
        "messages": [
            RemoveMessage(id=REMOVE_ALL_MESSAGES),
            SystemMessage(content=TOOL_AGENT_SYSTEM_PROMPT),
            *history,
            HumanMessage(content=state["question"]),
        ],
        "conversation_history": history,
        "sources": Overwrite([]),
        "search_queries": Overwrite([]),
    }


def tool_agent_model_node(
    state: ToolAgentState,
    runtime: Runtime[ToolAgentContext],
) -> dict:
    messages = list(state["messages"])
    completed_tool_rounds = _count_tool_rounds(messages)
    available_tools = (
        DOCUMENT_TOOLS if completed_tool_rounds < runtime.context.max_tool_rounds else []
    )
    if not available_tools:
        messages.insert(0, SystemMessage(content=TOOL_LIMIT_MESSAGE))

    response = runtime.context.llm_client.chat_with_tools(
        messages=messages,
        tools=available_tools,
    )
    if not available_tools and response.tool_calls:
        raise ValueError("LLM requested a tool after the tool-call limit was reached")
    return {"messages": [response]}


def route_tool_agent(
    state: ToolAgentState,
) -> Literal["tools", "build_response"]:
    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"
    return "build_response"


def build_tool_agent_response_node(
    state: ToolAgentState,
    runtime: Runtime[ToolAgentContext],
) -> dict:
    last_message = state["messages"][-1]
    if not isinstance(last_message, AIMessage) or last_message.tool_calls:
        raise ValueError("Tool agent did not return a final answer")

    answer = str(last_message.content).strip()
    search_queries = list(state.get("search_queries", []))
    history = _trim_conversation_history(
        [
            *state.get("conversation_history", []),
            HumanMessage(content=state["question"]),
            AIMessage(content=answer),
        ],
        runtime.context.max_history_turns,
    )
    return {
        "response": RAGResponse(
            answer=answer,
            sources=list(state.get("sources", [])),
            documents_count=state.get("documents_count", 0),
            chunks_count=state.get("chunks_count", 0),
            search_queries=search_queries,
            retry_count=max(0, len(search_queries) - 1),
            stop_reason="tool_agent_completed",
            tool_calls=_collect_tool_calls(state["messages"]),
        ),
        "conversation_history": history,
        # Tool calls and observations remain available for the full current run,
        # then are removed from the latest checkpoint. Only compact Q/A pairs live
        # into the next turn.
        "messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES)],
    }


def build_tool_agent_graph(checkpointer: BaseCheckpointSaver | None = None):
    builder = StateGraph(
        ToolAgentState,
        context_schema=ToolAgentContext,
    )
    builder.add_node("prepare_turn", prepare_tool_agent_turn_node)
    builder.add_node("agent_model", tool_agent_model_node)
    builder.add_node(
        "tools",
        ToolNode(DOCUMENT_TOOLS, handle_tool_errors=True),
    )
    builder.add_node("build_response", build_tool_agent_response_node)
    builder.add_edge(START, "prepare_turn")
    builder.add_edge("prepare_turn", "agent_model")
    builder.add_conditional_edges(
        "agent_model",
        route_tool_agent,
        {
            "tools": "tools",
            "build_response": "build_response",
        },
    )
    builder.add_edge("tools", "agent_model")
    builder.add_edge("build_response", END)
    return builder.compile(name="rag_tool_agent", checkpointer=checkpointer)


def _trim_conversation_history(
    messages: list[BaseMessage],
    max_history_turns: int,
) -> list[BaseMessage]:
    if max_history_turns < 1:
        raise ValueError("max_history_turns must be greater than zero")

    # Persist only complete user/final-assistant pairs. Tool calls, ToolMessages,
    # system instructions, and any malformed fragments are intentionally omitted.
    pairs: list[tuple[HumanMessage, AIMessage]] = []
    pending_user: HumanMessage | None = None
    for message in messages:
        if isinstance(message, HumanMessage):
            pending_user = message
        elif isinstance(message, AIMessage) and not message.tool_calls and pending_user is not None:
            pairs.append((pending_user, message))
            pending_user = None

    trimmed_pairs = pairs[-max_history_turns:]
    return [message for pair in trimmed_pairs for message in pair]


def _count_tool_rounds(messages: list[BaseMessage]) -> int:
    return sum(1 for message in messages if isinstance(message, AIMessage) and message.tool_calls)


def _collect_tool_calls(messages: list[BaseMessage]) -> list[dict]:
    collected = []
    for message in messages:
        if not isinstance(message, AIMessage):
            continue
        for tool_call in message.tool_calls:
            collected.append(
                {
                    "name": tool_call["name"],
                    "arguments": dict(tool_call["args"]),
                }
            )
    return collected


ingestion_graph = build_ingestion_graph()
qa_graph = build_qa_graph()
tool_agent_checkpointer = InMemorySaver()
tool_agent_graph = build_tool_agent_graph(checkpointer=tool_agent_checkpointer)
