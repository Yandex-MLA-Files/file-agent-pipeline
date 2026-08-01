import json
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool, tool
from langgraph.prebuilt import ToolRuntime
from langgraph.types import Command

from file_agent.document import Document
from file_agent.llm.base import ToolCallingLLMClient
from file_agent.qa import select_context_passages
from file_agent.retrieval import Retriever, SearchResult

MAX_TOOL_TOP_K = 10
MAX_TOOL_QUERY_LENGTH = 500
MAX_TOOL_PASSAGE_LENGTH = 3500

TOOL_AGENT_SYSTEM_PROMPT = """You answer questions about uploaded documents.

Use the available document tools before making factual claims about the files.
You may call search_documents more than once with different queries when the first
search is not useful. Use list_documents and get_document_outline when the question
is about the available files or their structure.

Base the final answer only on tool results. Preserve the language of the user's
question. Cite available source metadata such as source_file, page_number,
slide_number, or sheet_name. If the tools do not provide enough evidence, say so
clearly. Do not invent sources, document contents, or tool results.
"""

TOOL_LIMIT_MESSAGE = """The tool-call limit has been reached. Do not call another tool.
Provide the best final answer supported by the tool results already present in the
conversation. If they are insufficient, say that the documents do not contain enough
information.
"""


@dataclass(frozen=True)
class ToolAgentContext:
    llm_client: ToolCallingLLMClient
    retriever: Retriever
    documents: list[Document]
    max_tool_rounds: int = 4


@tool
def search_documents(
    query: str,
    runtime: ToolRuntime[Any, dict],
    top_k: int = 5,
) -> Command:
    """Search uploaded documents for passages relevant to a query.

    Args:
        query: A concise semantic search query.
        top_k: Number of retrieval results to return, from 1 to 10.
    """
    normalized_query = query.strip()
    if not normalized_query:
        raise ValueError("query must not be empty")
    if len(normalized_query) > MAX_TOOL_QUERY_LENGTH:
        raise ValueError(f"query must not exceed {MAX_TOOL_QUERY_LENGTH} characters")

    bounded_top_k = max(1, min(top_k, MAX_TOOL_TOP_K))
    results = runtime.context.retriever.search(
        query=normalized_query,
        top_k=bounded_top_k,
    )
    payload = _serialize_search_results(normalized_query, results)
    return Command(
        update={
            "messages": [
                ToolMessage(
                    content=json.dumps(payload, ensure_ascii=False, default=str),
                    tool_call_id=runtime.tool_call_id,
                )
            ],
            "sources": results,
            "search_queries": [normalized_query],
        }
    )


@tool
def list_documents(runtime: ToolRuntime[Any, dict]) -> str:
    """List the uploaded documents and their basic metadata."""
    documents = []
    for document in runtime.context.documents:
        metadata = document.metadata
        documents.append(
            {
                "source_file": document.file_name,
                "file_type": document.file_type,
                "blocks": len(document.blocks),
                "total_pages": metadata.get("total_pages"),
                "headings": len(metadata.get("table_of_contents") or []),
            }
        )
    return json.dumps({"documents": documents}, ensure_ascii=False, default=str)


@tool
def get_document_outline(
    source_file: str,
    runtime: ToolRuntime[Any, dict],
) -> str:
    """Return the table of contents for one uploaded document.

    Args:
        source_file: Exact file name returned by list_documents.
    """
    requested_name = source_file.strip().casefold()
    document = next(
        (item for item in runtime.context.documents if item.file_name.casefold() == requested_name),
        None,
    )
    if document is None:
        raise ValueError(f"document is not indexed: {source_file}")

    return json.dumps(
        {
            "source_file": document.file_name,
            "table_of_contents": document.metadata.get("table_of_contents") or [],
        },
        ensure_ascii=False,
        default=str,
    )


DOCUMENT_TOOLS: list[BaseTool] = [
    search_documents,
    list_documents,
    get_document_outline,
]


def _serialize_search_results(
    query: str,
    results: list[SearchResult],
) -> dict:
    serialized = []
    for rank, (result, passage) in enumerate(select_context_passages(results), start=1):
        metadata = dict(result.chunk.metadata)
        metadata.pop("context", None)
        serialized.append(
            {
                "rank": rank,
                "chunk_id": result.chunk.id,
                "score": result.score,
                "text": _truncate_passage(passage),
                "metadata": metadata,
            }
        )
    return {
        "query": query,
        "results_count": len(serialized),
        "results": serialized,
    }


def _truncate_passage(passage: str) -> str:
    if len(passage) <= MAX_TOOL_PASSAGE_LENGTH:
        return passage
    return passage[:MAX_TOOL_PASSAGE_LENGTH].rstrip() + "..."
