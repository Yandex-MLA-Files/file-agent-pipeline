import json
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from file_agent.agent.sandbox import run_sandboxed_code
from file_agent.chunking import Chunk
from file_agent.document import Block, Document
from file_agent.qa import build_context_from_results
from file_agent.retrieval import Retriever, SearchResult

# The full tool catalog this pipeline version registers - all three, always.
# Used as a stable pipeline-capability fingerprint in checkpoint parameters,
# not as the per-row tool list itself.
ALL_TOOL_NAMES = ("search_documents", "list_documents", "run_python")

# Tool-generated evidence carries the tool name as its document_id sentinel:
# RetrievedContext.from_dict rejects a blank one when a checkpoint reloads.


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[..., "ToolResult"]


@dataclass(frozen=True)
class ToolResult:
    content: str
    sources: list[SearchResult] = field(default_factory=list)


def search_documents(retriever: Retriever, query: str, top_k: int = 5) -> ToolResult:
    results = retriever.search(query=query, top_k=top_k)
    if not results:
        return ToolResult(content="No matching passages found.")
    return ToolResult(content=build_context_from_results(results), sources=results)


def list_documents(documents: Sequence[Document]) -> ToolResult:
    entries = []
    for document in documents:
        sheets = _unique_metadata_values(document.blocks, "sheet_name")
        slides = _integer_metadata_values(document.blocks, "slide_number")
        entries.append(
            {
                "file_name": document.file_name,
                "file_type": document.file_type,
                "total_pages": document.metadata.get("total_pages") or None,
                "total_slides": max(slides, default=None),
                "sheets": sheets,
                "headings": len(document.metadata.get("table_of_contents") or []),
            }
        )

    content = json.dumps({"documents": entries}, ensure_ascii=False, default=str)
    # Structure questions (sheet names, slide/page counts) are answered from
    # this output alone, so it has to reach the exported contexts as evidence.
    sources = (
        [_tool_evidence("list_documents", f"Document structure:\n{content}")] if entries else []
    )
    return ToolResult(content=content, sources=sources)


def _unique_metadata_values(blocks: list[Block], key: str) -> list[str]:
    values: list[str] = []
    for block in blocks:
        value = block.metadata.get(key)
        if isinstance(value, str) and value not in values:
            values.append(value)
    return values


def _integer_metadata_values(blocks: list[Block], key: str) -> list[int]:
    return sorted(
        {
            value
            for block in blocks
            if isinstance((value := block.metadata.get(key)), int) and not isinstance(value, bool)
        }
    )


def run_python(document_paths: Mapping[str, Path], code: str) -> ToolResult:
    result = run_sandboxed_code(source_paths=document_paths, code=code)
    if result.timed_out:
        return ToolResult(content="Error: execution timed out, simplify/narrow the computation.")
    if result.exit_code != 0:
        error_output = result.stderr or result.stdout
        # A failed run is still what the answer rests on when the agent reports
        # the failure (e.g. "no such sheet"), so export it rather than nothing.
        sources = (
            [_tool_evidence("run_python", f"Executed Python:\n{code}\n\nError:\n{error_output}")]
            if error_output.strip()
            else []
        )
        return ToolResult(
            content=f"Error: code raised an exception:\n{error_output}", sources=sources
        )

    stdout = result.stdout.strip()
    output = stdout or "(no output - use print() to return a result)"
    if result.truncated:
        output += "\n[output truncated]"

    # A successful run's code+output becomes an evidence "source", exported
    # to eval_pipeline's contexts alongside search_documents' sources -
    # otherwise an answer computed entirely via run_python (no search call)
    # would export empty/unrelated contexts, and RagasJudge's faithfulness/
    # context_recall would score a correct answer as ungrounded.
    sources = (
        [_tool_evidence("run_python", f"Executed Python:\n{code}\n\nOutput:\n{stdout}")]
        if stdout
        else []
    )
    return ToolResult(content=output, sources=sources)


def _tool_evidence(tool_name: str, text: str) -> SearchResult:
    return SearchResult(
        chunk=Chunk(
            id=f"{tool_name}:{uuid.uuid4().hex[:12]}",
            text=text,
            metadata={"source": tool_name, "dataset_doc_id": tool_name},
        ),
        score=1.0,
    )


def build_default_tools(
    retriever: Retriever,
    document_paths: Mapping[str, Path] | None = None,
    documents: Sequence[Document] | None = None,
    default_top_k: int = 5,
) -> list[Tool]:
    resolved_document_paths = document_paths or {}
    available_files = ", ".join(sorted(resolved_document_paths)) or "(no files uploaded)"

    return [
        Tool(
            name="search_documents",
            description=(
                "Semantic + full-text search over the indexed document chunks. "
                "Use for factual lookups, definitions, or any text-based question."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query, same language as the user's question.",
                    },
                    "top_k": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 10,
                        "default": default_top_k,
                    },
                },
                "required": ["query"],
            },
            handler=lambda query, top_k=default_top_k: search_documents(retriever, query, top_k),
        ),
        Tool(
            name="list_documents",
            description=(
                "List the uploaded documents and their structure (file type, page/"
                "slide/sheet counts, heading count). Call this first when unsure "
                "which documents or sheets are available, before a targeted search "
                "or run_python."
            ),
            parameters={"type": "object", "properties": {}, "required": []},
            handler=lambda: list_documents(documents or []),
        ),
        Tool(
            name="run_python",
            description=(
                "Execute Python (pandas/openpyxl available) in a sandboxed, "
                "network-disabled environment for calculations and structured-data "
                "questions a text search can't answer - arithmetic, aggregation, "
                "filtering, pivoting, statistics, dates. Not for looking up facts "
                "or passages in the documents' text; use search_documents for that. "
                "All uploaded files are available read-only under /data/<file name>: "
                f"{available_files}. Print the final result with print(); only "
                "stdout is returned to you."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": (
                            "Python source. Open files yourself, e.g. "
                            "pd.read_excel('/data/report.xlsx', sheet_name=None). "
                            "Print the result."
                        ),
                    },
                },
                "required": ["code"],
            },
            handler=lambda code: run_python(resolved_document_paths, code),
        ),
    ]
