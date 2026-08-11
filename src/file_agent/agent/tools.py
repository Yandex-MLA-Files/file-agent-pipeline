import ast
import json
import operator
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from file_agent.agent.sandbox import run_sandboxed_code
from file_agent.document import Block, Document
from file_agent.qa import build_context_from_results
from file_agent.retrieval import Retriever, SearchResult

# The full tool catalog this pipeline version can register (search_documents,
# calculate, and list_documents always; run_python_on_spreadsheet only when
# an XLSX is present). Used as a stable pipeline-capability fingerprint in
# checkpoint parameters, not as the per-row tool list itself.
ALL_TOOL_NAMES = ("search_documents", "calculate", "list_documents", "run_python_on_spreadsheet")

_CALC_OPERATORS: dict[type, Callable[..., float]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


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


def calculate(expression: str) -> ToolResult:
    try:
        tree = ast.parse(expression, mode="eval")
        value = _eval_calc_node(tree.body)
    except (SyntaxError, ValueError, ZeroDivisionError, TypeError) as exc:
        return ToolResult(content=f"Error: could not evaluate '{expression}': {exc}")
    return ToolResult(content=str(value))


def _eval_calc_node(node: ast.expr) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _CALC_OPERATORS:
        left = _eval_calc_node(node.left)
        right = _eval_calc_node(node.right)
        return _CALC_OPERATORS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _CALC_OPERATORS:
        return _CALC_OPERATORS[type(node.op)](_eval_calc_node(node.operand))
    raise ValueError(f"unsupported expression element: {ast.dump(node)}")


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
    return ToolResult(content=json.dumps({"documents": entries}, ensure_ascii=False, default=str))


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


def run_python_on_spreadsheet(
    document_paths: Mapping[str, Path], file_name: str, code: str
) -> ToolResult:
    source_path = document_paths.get(file_name)
    if source_path is None:
        available = ", ".join(sorted(document_paths)) or "(none)"
        return ToolResult(content=f"Error: unknown file_name '{file_name}'. Available: {available}")

    result = run_sandboxed_code(source_path=source_path, code=code)
    if result.timed_out:
        return ToolResult(content="Error: execution timed out, simplify/narrow the computation.")
    if result.exit_code != 0:
        error_output = result.stderr or result.stdout
        return ToolResult(content=f"Error: code raised an exception:\n{error_output}")

    output = result.stdout.strip() or "(no output - use print() to return a result)"
    if result.truncated:
        output += "\n[output truncated]"
    return ToolResult(content=output)


def build_default_tools(
    retriever: Retriever,
    document_paths: Mapping[str, Path] | None = None,
    documents: Sequence[Document] | None = None,
    default_top_k: int = 5,
) -> list[Tool]:
    tools = [
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
            name="calculate",
            description=(
                "Evaluate an arithmetic expression (+ - * / // % ** and parentheses) "
                "over numbers you already have. Not for text or table lookups."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "e.g. '(1200 - 950) / 950 * 100'",
                    },
                },
                "required": ["expression"],
            },
            handler=lambda expression: calculate(expression),
        ),
        Tool(
            name="list_documents",
            description=(
                "List the uploaded documents and their structure (file type, page/"
                "slide/sheet counts, heading count). Call this first when unsure "
                "which documents or sheets are available, before a targeted search "
                "or run_python_on_spreadsheet."
            ),
            parameters={"type": "object", "properties": {}, "required": []},
            handler=lambda: list_documents(documents or []),
        ),
    ]

    xlsx_paths = {
        name: path
        for name, path in (document_paths or {}).items()
        if path.suffix.lower() == ".xlsx"
    }
    if xlsx_paths:
        tools.append(
            Tool(
                name="run_python_on_spreadsheet",
                description=(
                    "Execute Python (pandas/openpyxl available) in a sandboxed, "
                    "network-disabled environment against one already-provided "
                    "spreadsheet, mounted at /data/input.xlsx. Use for "
                    "aggregation/filtering/pivoting the plain-text search can't do. "
                    "Print the final result with print(); only stdout is returned to you."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "file_name": {
                            "type": "string",
                            "description": "Exact file name from the available spreadsheet files.",
                            "enum": sorted(xlsx_paths),
                        },
                        "code": {
                            "type": "string",
                            "description": (
                                "Python source. Open the file yourself, e.g. "
                                "pd.read_excel('/data/input.xlsx', sheet_name=None). "
                                "Print the result."
                            ),
                        },
                    },
                    "required": ["file_name", "code"],
                },
                handler=lambda file_name, code: run_python_on_spreadsheet(
                    xlsx_paths, file_name, code
                ),
            )
        )

    return tools
