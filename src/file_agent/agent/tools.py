"""Tools the document agent can call while answering a question.

Each tool follows the same contract: it validates its arguments, does one
focused thing over the already-indexed documents, and returns a concise text
observation for the model (plus the raw search results when the tool touched
retrieval, so the UI can show sources). Tools never call an LLM themselves.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from file_agent.document import Block, BlockType, Document
from file_agent.qa import select_context_passages
from file_agent.retrieval import Retriever, SearchResult

logger = logging.getLogger(__name__)

DEFAULT_SEARCH_TOP_K = 4
MAX_SEARCH_TOP_K = 10
# Observations compete with the conversation for the model's context window,
# so long sections are cut instead of forwarded whole.
MAX_SECTION_CHARS = 6000
MAX_TOC_ENTRIES = 40
SECTION_TRUNCATION_NOTE = (
    "\n\n[Section truncated. Use search_documents to find specific details inside it.]"
)


class ToolError(Exception):
    """Tool arguments are invalid or the request cannot be fulfilled.

    The message is written for the model: it is fed back as an observation so
    the agent can correct itself on the next step.
    """


@dataclass
class ToolResult:
    output: str
    sources: list[SearchResult] = field(default_factory=list)


@dataclass
class Tool:
    """A named action the agent can invoke with keyword arguments."""

    name: str
    description: str
    parameters: dict[str, str]
    run: Callable[..., ToolResult]

    def describe(self) -> str:
        """One-line signature shown to the model in the system prompt."""
        if self.parameters:
            rendered = ", ".join(f"{name} ({spec})" for name, spec in self.parameters.items())
        else:
            rendered = "no arguments"
        return f"- {self.name}: {self.description} Arguments: {rendered}."


def build_default_tools(retriever: Retriever, documents: list[Document]) -> list[Tool]:
    """Standard toolset over an indexed document collection."""
    return [
        Tool(
            name="search_documents",
            description=(
                "Search the uploaded documents for passages relevant to a query. "
                "Returns the best-matching passages with their source file, "
                "section and pages."
            ),
            parameters={
                "query": "string, required - search keywords or a short question",
                "top_k": f"integer, optional, 1-{MAX_SEARCH_TOP_K}, "
                f"default {DEFAULT_SEARCH_TOP_K} - how many passages to return",
            },
            run=lambda **kwargs: _search_documents(retriever, **kwargs),
        ),
        Tool(
            name="list_documents",
            description=(
                "List the uploaded documents with their type, page count and "
                "table of contents. Useful to get an overview before searching."
            ),
            parameters={},
            run=lambda **kwargs: _list_documents(documents, **kwargs),
        ),
        Tool(
            name="read_section",
            description=(
                "Read one full section of a document by its heading, as listed "
                "in the table of contents. Useful when a whole section must be "
                "summarized rather than searched."
            ),
            parameters={
                "file_name": "string, required - document file name from list_documents",
                "section": "string, required - section heading (or a distinctive part of it)",
            },
            run=lambda **kwargs: _read_section(documents, **kwargs),
        ),
    ]


def _search_documents(
    retriever: Retriever,
    query: str = "",
    top_k: int = DEFAULT_SEARCH_TOP_K,
    **extra: Any,
) -> ToolResult:
    _reject_unknown_arguments("search_documents", extra)

    query = str(query).strip()
    if not query:
        raise ToolError("search_documents requires a non-empty 'query' argument.")

    try:
        top_k = int(top_k)
    except (TypeError, ValueError) as exc:
        raise ToolError("'top_k' must be an integer.") from exc
    top_k = max(1, min(top_k, MAX_SEARCH_TOP_K))

    results = retriever.search(query=query, top_k=top_k)
    if not results:
        return ToolResult(
            output=(
                "No matching passages were found. Try other keywords, or call "
                "list_documents to see what the documents contain."
            )
        )

    parts: list[str] = []
    for index, (result, passage) in enumerate(select_context_passages(results), start=1):
        metadata = result.chunk.metadata
        origin = [f"file={metadata.get('source_file', 'unknown')}"]
        if metadata.get("section"):
            origin.append(f"section={metadata['section']}")
        pages = metadata.get("page_numbers") or (
            [metadata["page_number"]] if metadata.get("page_number") else []
        )
        if pages:
            origin.append(f"pages={', '.join(str(page) for page in pages)}")
        header = f"[Passage {index} | score={result.score:g} | {' | '.join(origin)}]"
        parts.append(f"{header}\n{passage}")

    logger.info("search_documents(%r, top_k=%d) returned %d result(s)", query, top_k, len(results))
    return ToolResult(output="\n\n".join(parts), sources=results)


def _list_documents(documents: list[Document], **extra: Any) -> ToolResult:
    _reject_unknown_arguments("list_documents", extra)

    if not documents:
        return ToolResult(output="No documents are loaded.")

    parts: list[str] = []
    for document in documents:
        lines = [
            f"{document.file_name} ({document.file_type}, "
            f"{document.metadata.get('total_pages', 0)} page(s), "
            f"{len(document.blocks)} block(s))"
        ]
        toc = document.metadata.get("table_of_contents") or []
        if toc:
            lines.append("  Table of contents:")
            for entry in toc[:MAX_TOC_ENTRIES]:
                indent = "  " * max(int(entry.get("level") or 1), 1)
                lines.append(f"  {indent}- {entry['title']}")
            if len(toc) > MAX_TOC_ENTRIES:
                lines.append(f"  ... and {len(toc) - MAX_TOC_ENTRIES} more heading(s)")
        else:
            lines.append("  No table of contents detected; use search_documents.")
        parts.append("\n".join(lines))

    return ToolResult(output="\n\n".join(parts))


def _read_section(
    documents: list[Document],
    file_name: str = "",
    section: str = "",
    **extra: Any,
) -> ToolResult:
    _reject_unknown_arguments("read_section", extra)

    file_name = str(file_name).strip()
    section = str(section).strip()
    if not file_name or not section:
        raise ToolError("read_section requires both 'file_name' and 'section' arguments.")

    document = _find_document(documents, file_name)
    headings = [block for block in document.blocks if block.block_type == BlockType.HEADING]
    if not headings:
        raise ToolError(
            f"'{document.file_name}' has no detected sections; use search_documents instead."
        )

    heading = _find_heading(headings, section, document.file_name)
    text = _collect_section_text(document, heading)
    if len(text) > MAX_SECTION_CHARS:
        text = text[:MAX_SECTION_CHARS] + SECTION_TRUNCATION_NOTE

    logger.info("read_section(%r, %r) returned %d char(s)", document.file_name, section, len(text))
    return ToolResult(output=text)


def _find_document(documents: list[Document], file_name: str) -> Document:
    wanted = file_name.casefold()
    for document in documents:
        if document.file_name.casefold() == wanted:
            return document

    partial = [document for document in documents if wanted in document.file_name.casefold()]
    if len(partial) == 1:
        return partial[0]

    available = ", ".join(document.file_name for document in documents) or "none"
    raise ToolError(f"Document '{file_name}' not found. Available documents: {available}.")


def _find_heading(headings: list[Block], section: str, file_name: str) -> Block:
    wanted = section.casefold()
    exact = [block for block in headings if block.text.strip().casefold() == wanted]
    if exact:
        return exact[0]

    partial = [block for block in headings if wanted in block.text.casefold()]
    if partial:
        return partial[0]

    available = "; ".join(block.text.strip() for block in headings[:MAX_TOC_ENTRIES])
    raise ToolError(f"Section '{section}' not found in '{file_name}'. Sections: {available}.")


def _collect_section_text(document: Document, heading: Block) -> str:
    """Return the heading and its body, up to the next same-or-higher heading."""
    level = _heading_level(heading)
    started = False
    parts: list[str] = []

    for block in document.blocks:
        if block is heading:
            started = True
        elif started and block.block_type == BlockType.HEADING and _heading_level(block) <= level:
            break
        if started:
            rendered = block.to_markdown()
            if rendered.strip():
                parts.append(rendered)

    return "\n\n".join(parts)


def _heading_level(block: Block) -> int:
    try:
        return int(block.metadata.get("hierarchy_level", 1))
    except (TypeError, ValueError):
        return 1


def _reject_unknown_arguments(tool_name: str, extra: dict[str, Any]) -> None:
    if extra:
        unknown = ", ".join(sorted(extra))
        raise ToolError(f"{tool_name} received unknown argument(s): {unknown}.")
