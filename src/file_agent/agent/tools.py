"""Tools the document agent can call while answering a question.

Each tool follows the same contract: it validates its arguments, does one
focused thing over the already-indexed documents, and returns a concise text
observation for the model (plus the passages it showed, so the answer can cite
them and the UI can list them as sources). Tools never call an LLM themselves.

Every piece of document text a tool shows is registered in the run's
:class:`PassageRegistry` and labelled ``[P<n> | file=... | section=... |
pages=...]``; the Final Answer cites those ids.
"""

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from file_agent.agent.passages import Passage, PassageRegistry
from file_agent.agent.sandbox import SandboxError, run_code
from file_agent.agent.tables import TableView, document_tables, parse_header_row, preview_frame
from file_agent.document import Block, BlockType, Document
from file_agent.retrieval import Retriever, SearchResult

logger = logging.getLogger(__name__)

DEFAULT_SEARCH_TOP_K = 5
MAX_SEARCH_TOP_K = 10
# Extra formulations fused with the main query; more than this adds latency
# without new candidates on a small index.
MAX_SEARCH_QUERIES = 4
# Hybrid candidates fetched per formulation before fusion and document filtering.
MIN_SEARCH_POOL = 12
MAX_SEARCH_POOL = 40
RRF_K = 60
# Observations compete with the conversation for the model's context window,
# so long texts are served in parts instead of forwarded whole.
MAX_READ_CHARS = 7000
MAX_PASSAGE_CHARS = 4500
MAX_TOC_ENTRIES = 60
PREVIEW_CHARS = 400
DEFAULT_FIND_HITS = 8
MAX_FIND_HITS = 25
SNIPPET_CHARS = 350
MAX_SNIPPETS_PER_BLOCK = 3
# Hits in blocks shorter than this are shown with neighbouring blocks.
SHORT_BLOCK_CHARS = 400
NEIGHBOURHOOD_CHARS = 600

_PAGE_SPEC = re.compile(r"^\s*(\d+)\s*(?:-\s*(\d+))?\s*$")


class ToolError(Exception):
    """Tool arguments are invalid or the request cannot be fulfilled.

    The message is written for the model: it is fed back as an observation so
    the agent can correct itself on the next step.
    """


@dataclass
class ToolResult:
    output: str
    sources: list[SearchResult] = field(default_factory=list)
    passages: list[Passage] = field(default_factory=list)


@dataclass
class Tool:
    """A named action the agent can invoke with keyword arguments."""

    name: str
    description: str
    parameters: dict[str, str]
    run: Callable[..., ToolResult]

    def describe(self) -> str:
        """Signature shown to the model in the system prompt."""
        if self.parameters:
            rendered = "; ".join(f"{name}: {spec}" for name, spec in self.parameters.items())
        else:
            rendered = "no arguments"
        return f"- {self.name}: {self.description} Arguments: {rendered}."


def build_default_tools(
    retriever: Retriever,
    documents: list[Document],
    registry: PassageRegistry | None = None,
) -> list[Tool]:
    """Standard toolset over an indexed document collection."""
    registry = registry if registry is not None else PassageRegistry()
    return [
        Tool(
            name="search_documents",
            description=(
                "Hybrid (semantic + keyword) search over the documents. Give the main "
                "query plus up to three alternative formulations (synonyms, the "
                "document's own terminology, the other language) in 'queries'; results "
                "are fused. Returns the best passages with their source."
            ),
            parameters={
                "query": "string, required - the main search query",
                "queries": "list of strings, optional - alternative formulations to fuse in",
                "top_k": f"integer, optional, 1-{MAX_SEARCH_TOP_K}, default "
                f"{DEFAULT_SEARCH_TOP_K} - how many passages to return",
                "file_name": "string, optional - restrict the search to one document",
            },
            run=lambda **kwargs: _search_documents(retriever, documents, registry, **kwargs),
        ),
        Tool(
            name="find_text",
            description=(
                "Exact text search (case-insensitive substring or regular expression) "
                "over every document. Use it for numbers, codes, names, identifiers, "
                "rare terms and quotes that semantic search may miss; spreadsheet hits "
                "return the whole row with its column headers."
            ),
            parameters={
                "pattern": "string, required - text or regular expression to find",
                "file_name": "string, optional - restrict to one document",
                "max_hits": f"integer, optional, default {DEFAULT_FIND_HITS}, max {MAX_FIND_HITS}",
            },
            run=lambda **kwargs: _find_text(documents, registry, **kwargs),
        ),
        Tool(
            name="list_documents",
            description=(
                "Overview of the uploaded documents: type, pages or slides, sheets with "
                "their size and columns, table of contents, and a short preview. Call it "
                "first when the question is about a document's structure or you do not "
                "know where to look."
            ),
            parameters={},
            run=lambda **kwargs: _list_documents(documents, registry, **kwargs),
        ),
        Tool(
            name="read_section",
            description=(
                "Read one full section of a document by its heading (as shown by "
                "list_documents). Use it to summarize or enumerate everything a section "
                "says; long sections come in parts."
            ),
            parameters={
                "file_name": "string, required - document file name",
                "section": "string, required - section heading or a distinctive part of it",
                "part": "integer, optional, default 1 - which part of a long section to read",
            },
            run=lambda **kwargs: _read_section(documents, registry, **kwargs),
        ),
        Tool(
            name="read_pages",
            description=(
                "Read the full text of specific pages (PDF/DOCX), slides (PPTX) or sheets "
                "(XLSX, by number) of a document. Use it to see the neighbourhood of a "
                "passage or a page the question names."
            ),
            parameters={
                "file_name": "string, required - document file name",
                "pages": "string, required - e.g. '3', '3-5' or '2,7'",
            },
            run=lambda **kwargs: _read_pages(documents, registry, **kwargs),
        ),
        Tool(
            name="read_document",
            description=(
                "Read a whole document from the beginning, in parts. Use it for short "
                "documents, for questions about the overall structure, introduction or "
                "conclusion, and when the other tools cannot locate the answer."
            ),
            parameters={
                "file_name": "string, required - document file name",
                "part": "integer, optional, default 1 - which part to read",
            },
            run=lambda **kwargs: _read_document(documents, registry, **kwargs),
        ),
        Tool(
            name="query_table",
            description=(
                "Run Python (pandas preloaded as pd, numpy as np, math, re) over the "
                "tables of a document: spreadsheet sheets or tables inside PDF/DOCX. "
                "'df' is the selected sheet/table, 'sheets' maps every sheet or table name "
                "to its DataFrame. Use it for counts, sums, averages, maxima, filters, "
                "group-bys, joins across files and any exact lookup in large tables. Call "
                "it with empty code first to see the columns and sample rows."
            ),
            parameters={
                "file_name": "string, required - document file name",
                "code": "string, optional - Python code; print() results or end with an "
                "expression. Empty code shows the table preview",
                "sheet": "string, optional - sheet or table name (default: the first one)",
                "header_row": "integer, optional - row index holding the column names "
                "when the auto-detected header is wrong",
            },
            run=lambda **kwargs: _query_table(documents, registry, **kwargs),
        ),
        Tool(
            name="calculate",
            description=(
                "Evaluate an arithmetic or Python expression (math available), e.g. "
                "ratios, percentages, differences, unit conversions. Use it instead of "
                "doing arithmetic in your head."
            ),
            parameters={"expression": "string, required - e.g. '3.81e6 / 1.52e6'"},
            run=lambda **kwargs: _calculate(registry, **kwargs),
        ),
    ]


# ---------------------------------------------------------------------------
# search_documents
# ---------------------------------------------------------------------------


def _search_documents(
    retriever: Retriever,
    documents: list[Document],
    registry: PassageRegistry,
    query: str = "",
    queries: Any = None,
    top_k: int = DEFAULT_SEARCH_TOP_K,
    file_name: str | None = None,
    **extra: Any,
) -> ToolResult:
    _reject_unknown_arguments("search_documents", extra)

    formulations = _formulations(query, queries)
    if not formulations:
        raise ToolError("search_documents requires a non-empty 'query' argument.")
    top_k = _clamp_int(top_k, "top_k", 1, MAX_SEARCH_TOP_K, DEFAULT_SEARCH_TOP_K)

    document = _find_document(documents, file_name) if file_name else None
    pool = min(MAX_SEARCH_POOL, max(MIN_SEARCH_POOL, top_k * (4 if document else 2)))

    fused: dict[tuple[str, str], float] = {}
    first_hit: dict[tuple[str, str], SearchResult] = {}
    for formulation in formulations:
        for rank, result in enumerate(retriever.search(query=formulation, top_k=pool)):
            key = (str(result.chunk.metadata.get("source_file", "")), str(result.chunk.id))
            fused[key] = fused.get(key, 0.0) + 1.0 / (RRF_K + rank)
            first_hit.setdefault(key, result)

    ranked = sorted(fused.items(), key=lambda item: item[1], reverse=True)
    if document is not None:
        wanted = document.file_name.casefold()
        ranked = [item for item in ranked if item[0][0].casefold() == wanted]

    passages: list[Passage] = []
    seen_texts: set[str] = set()
    for key, score in ranked:
        hit = first_hit[key]
        text = hit.chunk.metadata.get("context") or hit.chunk.text
        if text in seen_texts:
            continue
        seen_texts.add(text)
        fused_hit = SearchResult(chunk=hit.chunk, score=round(score, 4))
        passages.append(registry.add_result(fused_hit, tool="search_documents"))
        if len(passages) >= top_k:
            break

    scope = f" in '{document.file_name}'" if document is not None else ""
    if not passages:
        return ToolResult(
            output=(
                f"No matching passages{scope} for {formulations}. Try other words (the "
                "document's own terminology), find_text for exact terms, or "
                "list_documents to see what the documents contain."
            )
        )

    header = f"{len(passages)} passage(s){scope} for {formulations}:"
    body = "\n\n".join(passage.render(max_chars=MAX_PASSAGE_CHARS) for passage in passages)
    logger.info(
        "search_documents(%r, top_k=%d, file=%r) returned %d passage(s)",
        formulations,
        top_k,
        file_name,
        len(passages),
    )
    return ToolResult(
        output=f"{header}\n\n{body}",
        sources=[passage.as_search_result() for passage in passages],
        passages=passages,
    )


def _formulations(query: Any, queries: Any) -> list[str]:
    candidates: list[str] = []
    if isinstance(query, list | tuple):
        candidates.extend(str(item) for item in query)
    elif query is not None:
        candidates.append(str(query))
    if isinstance(queries, str):
        candidates.append(queries)
    elif isinstance(queries, list | tuple):
        candidates.extend(str(item) for item in queries)
    result: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        text = " ".join(candidate.split())
        if text and text.casefold() not in seen:
            seen.add(text.casefold())
            result.append(text)
        if len(result) >= MAX_SEARCH_QUERIES:
            break
    return result


# ---------------------------------------------------------------------------
# find_text
# ---------------------------------------------------------------------------


def _find_text(
    documents: list[Document],
    registry: PassageRegistry,
    pattern: str = "",
    file_name: str | None = None,
    max_hits: int = DEFAULT_FIND_HITS,
    **extra: Any,
) -> ToolResult:
    _reject_unknown_arguments("find_text", extra)

    pattern = str(pattern).strip()
    if not pattern:
        raise ToolError("find_text requires a non-empty 'pattern' argument.")
    max_hits = _clamp_int(max_hits, "max_hits", 1, MAX_FIND_HITS, DEFAULT_FIND_HITS)
    regex = _compile_pattern(pattern)

    scope = [_find_document(documents, file_name)] if file_name else documents
    passages: list[Passage] = []
    total_matches = 0
    matched_files: set[str] = set()
    for document in scope:
        section: str | None = None
        for index, block in enumerate(document.blocks):
            if block.block_type == BlockType.HEADING:
                section = block.text.strip() or section
                # A heading that matches is a hit too: it locates the topic.
            matches = list(regex.finditer(block.text))
            if not matches:
                continue
            total_matches += len(matches)
            matched_files.add(document.file_name)
            if len(passages) >= max_hits:
                continue
            for snippet in _snippets(block, matches, document.blocks, index):
                metadata: dict[str, Any] = {}
                if section and block.block_type != BlockType.HEADING:
                    metadata["section"] = section
                elif block.metadata.get("section"):
                    metadata["section"] = block.metadata["section"]
                for key in ("page_number", "slide_number", "sheet_name"):
                    if block.metadata.get(key) is not None:
                        metadata[key] = block.metadata[key]
                if block.page_number is not None:
                    metadata["page_number"] = block.page_number
                metadata["block_id"] = block.id
                passages.append(
                    registry.add_text(
                        snippet,
                        document.file_name,
                        metadata=metadata,
                        tool="find_text",
                        inherit_from=block.metadata,
                    )
                )
                if len(passages) >= max_hits:
                    break

    if not passages:
        where = f" in '{scope[0].file_name}'" if file_name else ""
        return ToolResult(
            output=(
                f"No text matching {pattern!r}{where}. Try a shorter or differently "
                "spelled pattern (a word stem, a number without spaces, Latin/Cyrillic "
                "variants) or search_documents for the meaning."
            )
        )

    header = (
        f"{total_matches} match(es) for {pattern!r} in {len(matched_files)} document(s); "
        f"showing {len(passages)} snippet(s):"
    )
    body = "\n\n".join(passage.render() for passage in passages)
    logger.info("find_text(%r, file=%r) matched %d time(s)", pattern, file_name, total_matches)
    return ToolResult(
        output=f"{header}\n\n{body}",
        sources=[passage.as_search_result() for passage in passages],
        passages=passages,
    )


def _compile_pattern(pattern: str) -> re.Pattern[str]:
    """Regex when it compiles and looks like one, literal substring otherwise."""
    looks_like_regex = any(char in pattern for char in r"\[](){}|*+?^$")
    if looks_like_regex:
        try:
            return re.compile(pattern, re.IGNORECASE | re.UNICODE)
        except re.error:
            pass
    return re.compile(_literal_pattern(pattern), re.IGNORECASE | re.UNICODE)


def _literal_pattern(text: str) -> str:
    """Escape a literal and tolerate ё/е spelling and flexible whitespace."""
    parts: list[str] = []
    for char in text:
        if char.isspace():
            parts.append(r"\s+")
        elif char in "её":
            parts.append("[её]")
        elif char in "ЕЁ":
            parts.append("[ЕЁ]")
        else:
            parts.append(re.escape(char))
    # Collapse consecutive whitespace classes produced by runs of spaces.
    return re.sub(r"(?:\\s\+)+", r"\\s+", "".join(parts))


def _snippets(
    block: Block, matches: list[re.Match[str]], blocks: list[Block], index: int
) -> list[str]:
    """Context windows around the matches, merged when they overlap."""
    text = block.text
    if block.type == "xlsx_sheet" or block.metadata.get("sheet_name"):
        return _row_snippets(text, matches)
    if len(text) < SHORT_BLOCK_CHARS:
        # Word processors yield one block per paragraph or list item; a hit in
        # such a block is meaningless alone, so show its neighbourhood.
        return [_block_neighbourhood(blocks, index)]

    windows: list[tuple[int, int]] = []
    for match in matches:
        start = max(0, match.start() - SNIPPET_CHARS)
        end = min(len(text), match.end() + SNIPPET_CHARS)
        # Snap to line boundaries so tables and lists stay readable.
        newline_before = text.rfind("\n", 0, start)
        start = (
            newline_before + 1 if newline_before != -1 and start - newline_before < 120 else start
        )
        newline_after = text.find("\n", end)
        end = newline_after if newline_after != -1 and newline_after - end < 120 else end
        if windows and start <= windows[-1][1]:
            windows[-1] = (windows[-1][0], max(windows[-1][1], end))
        else:
            windows.append((start, end))
        if len(windows) >= MAX_SNIPPETS_PER_BLOCK:
            break

    snippets: list[str] = []
    for start, end in windows:
        prefix = "..." if start > 0 else ""
        suffix = "..." if end < len(text) else ""
        snippets.append(f"{prefix}{text[start:end].strip()}{suffix}")
    return snippets


def _block_neighbourhood(blocks: list[Block], index: int) -> str:
    """The hit block with enough preceding and following blocks for context."""
    before: list[str] = []
    size = 0
    for block in reversed(blocks[max(0, index - 12) : index]):
        rendered = block.to_markdown().strip()
        if not rendered or block.metadata.get("sheet_name"):
            continue
        before.insert(0, rendered)
        size += len(rendered)
        if size >= NEIGHBOURHOOD_CHARS or block.block_type == BlockType.HEADING:
            break
    after: list[str] = []
    size = 0
    for block in blocks[index + 1 : index + 13]:
        if block.block_type == BlockType.HEADING or block.metadata.get("sheet_name"):
            break
        rendered = block.to_markdown().strip()
        if not rendered:
            continue
        after.append(rendered)
        size += len(rendered)
        if size >= NEIGHBOURHOOD_CHARS:
            break
    hit = blocks[index].to_markdown().strip() or blocks[index].text.strip()
    return "\n\n".join([*before, hit, *after])


def _row_snippets(text: str, matches: list[re.Match[str]]) -> list[str]:
    """For spreadsheet text (one row per line), return header + matching rows."""
    lines = text.split("\n")
    offsets: list[int] = []
    position = 0
    for line in lines:
        offsets.append(position)
        position += len(line) + 1
    header = lines[0].strip() if lines else ""
    hit_rows: list[int] = []
    for match in matches:
        row = max(index for index, offset in enumerate(offsets) if offset <= match.start())
        if row not in hit_rows:
            hit_rows.append(row)
        if len(hit_rows) >= MAX_SNIPPETS_PER_BLOCK:
            break
    snippets: list[str] = []
    for row in hit_rows:
        body = lines[row].strip()
        if row == 0 or not header:
            snippets.append(body)
        else:
            snippets.append(f"{header}\n{body}  (row {row + 1})")
    return snippets


# ---------------------------------------------------------------------------
# list_documents
# ---------------------------------------------------------------------------


def _list_documents(
    documents: list[Document], registry: PassageRegistry, **extra: Any
) -> ToolResult:
    _reject_unknown_arguments("list_documents", extra)

    if not documents:
        return ToolResult(output="No documents are loaded.")

    parts: list[str] = []
    passages: list[Passage] = []
    for document in documents:
        overview = _document_overview(document)
        passage = registry.add_text(
            overview,
            document.file_name,
            metadata={"section": "document overview"},
            tool="list_documents",
            inherit_from=document.blocks[0].metadata if document.blocks else None,
        )
        passages.append(passage)
        parts.append(passage.render())

    return ToolResult(
        output="\n\n".join(parts),
        sources=[passage.as_search_result() for passage in passages],
        passages=passages,
    )


def _document_overview(document: Document) -> str:
    lines = [f"{document.file_name} ({document.file_type}; {_extent(document)})"]
    tables = _table_blocks(document)
    sheets = [block for block in document.blocks if block.metadata.get("sheet_name")]
    if sheets:
        lines.append("Sheets:")
        for block in sheets:
            rows = block.metadata.get("max_row")
            columns = block.metadata.get("max_column")
            size = f"{rows} rows x {columns} columns" if rows and columns else ""
            first_line = block.text.split("\n", 1)[0].replace("\t", " | ")[:200]
            lines.append(f"  - {block.metadata['sheet_name']}: {size}; first row: {first_line}")
    elif tables:
        lines.append(f"Tables: {len(tables)} (query_table can compute over them)")

    toc = document.metadata.get("table_of_contents") or []
    if toc:
        lines.append("Table of contents:")
        for entry in toc[:MAX_TOC_ENTRIES]:
            indent = "  " * max(int(entry.get("level") or 1), 1)
            page = f" (p. {entry['page']})" if entry.get("page") else ""
            lines.append(f"{indent}- {entry['title']}{page}")
        if len(toc) > MAX_TOC_ENTRIES:
            lines.append(f"  ... and {len(toc) - MAX_TOC_ENTRIES} more heading(s)")
    else:
        lines.append("No table of contents detected.")

    preview = _preview_text(document)
    if preview:
        lines.append(f"Begins with: {preview}")
    return "\n".join(lines)


def _extent(document: Document) -> str:
    pages = document.metadata.get("total_pages") or 0
    slides = [block.metadata.get("slide_number") for block in document.blocks]
    slides = [number for number in slides if number is not None]
    sheets = [block for block in document.blocks if block.metadata.get("sheet_name")]
    if sheets:
        return f"{len(sheets)} sheet(s)"
    if slides:
        return f"{max(slides)} slide(s), {len(document.blocks)} block(s)"
    if pages:
        return f"{pages} page(s), {len(document.blocks)} block(s)"
    return f"{len(document.blocks)} block(s), {sum(len(b.text) for b in document.blocks)} chars"


def _preview_text(document: Document) -> str:
    for block in document.blocks:
        if block.block_type == BlockType.HEADING:
            continue
        text = " ".join(block.text.split())
        if len(text) >= 40:
            return text[:PREVIEW_CHARS] + ("..." if len(text) > PREVIEW_CHARS else "")
    return ""


def _table_blocks(document: Document) -> list[Block]:
    return [block for block in document.blocks if block.block_type == BlockType.TABLE]


# ---------------------------------------------------------------------------
# read_section / read_pages / read_document
# ---------------------------------------------------------------------------


def _read_section(
    documents: list[Document],
    registry: PassageRegistry,
    file_name: str = "",
    section: str = "",
    part: int = 1,
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
            f"'{document.file_name}' has no detected sections; use read_document, "
            "read_pages or search_documents instead."
        )

    heading = _find_heading(headings, section, document.file_name)
    blocks = _section_blocks(document, heading)
    text = _render_blocks(blocks)
    return _paged_text(
        registry,
        document,
        text,
        part,
        label=f"section '{heading.text.strip()}'",
        metadata={"section": heading.text.strip(), **_pages_metadata(blocks)},
        tool="read_section",
        inherit_from=heading.metadata,
    )


def _read_pages(
    documents: list[Document],
    registry: PassageRegistry,
    file_name: str = "",
    pages: Any = "",
    **extra: Any,
) -> ToolResult:
    _reject_unknown_arguments("read_pages", extra)

    file_name = str(file_name).strip()
    if not file_name:
        raise ToolError("read_pages requires the 'file_name' argument.")
    wanted = _parse_pages(pages)
    if not wanted:
        raise ToolError("read_pages requires 'pages' like '3', '3-5' or '2,7'.")
    if len(wanted) > 10:
        raise ToolError("read_pages reads at most 10 pages at a time; narrow the range.")

    document = _find_document(documents, file_name)
    by_page: dict[int, list[Block]] = {}
    sheet_index = 0
    for block in document.blocks:
        number = block.page_number
        if number is None:
            number = block.metadata.get("page_number") or block.metadata.get("slide_number")
        if number is None and block.metadata.get("sheet_name"):
            sheet_index += 1
            number = sheet_index
        if number is None:
            continue
        try:
            number = int(number)
        except (TypeError, ValueError):
            continue
        if number in wanted:
            by_page.setdefault(number, []).append(block)

    if not by_page:
        available = sorted(
            {
                int(value)
                for block in document.blocks
                for value in (
                    block.page_number,
                    block.metadata.get("page_number"),
                    block.metadata.get("slide_number"),
                )
                if value is not None
            }
        )
        if not available and not any(b.metadata.get("sheet_name") for b in document.blocks):
            raise ToolError(
                f"'{document.file_name}' has no page numbers; use read_document or read_section."
            )
        span = f"{available[0]}-{available[-1]}" if available else "sheet numbers"
        raise ToolError(
            f"No content on pages {sorted(wanted)} of '{document.file_name}' ({span} available)."
        )

    passages: list[Passage] = []
    parts: list[str] = []
    budget = MAX_READ_CHARS
    for number in sorted(by_page):
        text = _render_blocks(by_page[number])
        if not text.strip():
            continue
        truncated = False
        if len(text) > budget:
            text = text[: max(budget, 500)].rstrip()
            truncated = True
        passage = registry.add_text(
            text,
            document.file_name,
            metadata={"page_number": number, "page_numbers": [number]},
            tool="read_pages",
            inherit_from=by_page[number][0].metadata,
        )
        passages.append(passage)
        rendered = passage.render()
        if truncated:
            rendered += "\n[page truncated; narrow the range to read more]"
        parts.append(rendered)
        budget -= len(text)
        if budget <= 0:
            parts.append("[Observation limit reached; ask for fewer pages to see the rest.]")
            break

    return ToolResult(
        output="\n\n".join(parts),
        sources=[passage.as_search_result() for passage in passages],
        passages=passages,
    )


def _read_document(
    documents: list[Document],
    registry: PassageRegistry,
    file_name: str = "",
    part: int = 1,
    **extra: Any,
) -> ToolResult:
    _reject_unknown_arguments("read_document", extra)

    file_name = str(file_name).strip()
    if not file_name:
        raise ToolError("read_document requires the 'file_name' argument.")
    document = _find_document(documents, file_name)
    text = _render_blocks(document.blocks)
    return _paged_text(
        registry,
        document,
        text,
        part,
        label=f"document '{document.file_name}'",
        metadata={},
        tool="read_document",
        inherit_from=document.blocks[0].metadata if document.blocks else None,
    )


def _paged_text(
    registry: PassageRegistry,
    document: Document,
    text: str,
    part: Any,
    label: str,
    metadata: dict[str, Any],
    tool: str,
    inherit_from: dict[str, Any] | None,
) -> ToolResult:
    part = _clamp_int(part, "part", 1, 10_000, 1)
    pieces = _split_parts(text, MAX_READ_CHARS)
    if not pieces:
        raise ToolError(f"The {label} is empty.")
    if part > len(pieces):
        raise ToolError(
            f"The {label} has only {len(pieces)} part(s); 'part' must be 1-{len(pieces)}."
        )

    piece = pieces[part - 1]
    passage = registry.add_text(
        piece,
        document.file_name,
        metadata={**metadata, "part": f"{part}/{len(pieces)}"},
        tool=tool,
        inherit_from=inherit_from,
    )
    output = passage.render(extra=f"part {part} of {len(pieces)}")
    if part < len(pieces):
        output += f"\n\n[Continue with part={part + 1} to read the rest of the {label}.]"
    logger.info(
        "%s(%r) served part %d/%d (%d chars)",
        tool,
        document.file_name,
        part,
        len(pieces),
        len(piece),
    )
    return ToolResult(output=output, sources=[passage.as_search_result()], passages=[passage])


def _split_parts(text: str, size: int) -> list[str]:
    text = text.strip()
    if not text:
        return []
    pieces: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + size)
        if end < len(text):
            # Prefer cutting at a paragraph, then a line, then a sentence end.
            for separator in ("\n\n", "\n", ". "):
                cut = text.rfind(separator, start + size // 2, end)
                if cut != -1:
                    end = cut + len(separator)
                    break
        pieces.append(text[start:end].strip())
        start = end
    return [piece for piece in pieces if piece]


def _render_blocks(blocks: list[Block]) -> str:
    parts: list[str] = []
    for block in blocks:
        rendered = block.to_markdown()
        if rendered.strip():
            parts.append(rendered)
    return "\n\n".join(parts)


def _section_blocks(document: Document, heading: Block) -> list[Block]:
    """The heading and its body, up to the next same-or-higher heading."""
    level = _heading_level(heading)
    started = False
    blocks: list[Block] = []
    for block in document.blocks:
        if block is heading:
            started = True
        elif started and block.block_type == BlockType.HEADING and _heading_level(block) <= level:
            break
        if started:
            blocks.append(block)
    return blocks


def _pages_metadata(blocks: list[Block]) -> dict[str, Any]:
    pages = sorted({block.page_number for block in blocks if block.page_number is not None})
    if not pages:
        return {}
    return {"page_number": pages[0], "page_numbers": pages}


def _parse_pages(value: Any) -> set[int]:
    if isinstance(value, int):
        return {value}
    if isinstance(value, list | tuple):
        value = ",".join(str(item) for item in value)
    wanted: set[int] = set()
    for piece in str(value).split(","):
        match = _PAGE_SPEC.match(piece)
        if not match:
            continue
        start = int(match.group(1))
        end = int(match.group(2)) if match.group(2) else start
        if end < start:
            start, end = end, start
        wanted.update(range(start, end + 1))
    return wanted


# ---------------------------------------------------------------------------
# query_table / calculate
# ---------------------------------------------------------------------------


def _query_table(
    documents: list[Document],
    registry: PassageRegistry,
    file_name: str = "",
    code: str = "",
    sheet: str | None = None,
    header_row: Any = 0,
    **extra: Any,
) -> ToolResult:
    _reject_unknown_arguments("query_table", extra)

    file_name = str(file_name).strip()
    if not file_name:
        raise ToolError("query_table requires the 'file_name' argument.")
    document = _find_document(documents, file_name)
    views = document_tables(document, header_row=parse_header_row(header_row))
    if not views:
        raise ToolError(
            f"'{document.file_name}' has no sheets or tables to query; use "
            "search_documents, find_text or read_document instead."
        )
    view = _select_view(views, sheet)
    names = ", ".join(f"'{item.name}'" for item in views)

    code = str(code or "").strip()
    if not code:
        output = (
            f"Tables in '{document.file_name}': {names}. Selected: '{view.name}'.\n"
            f"{preview_frame(view.frame)}\n"
            "Now call query_table with 'code' (pandas on df; sheets['<name>'] for others)."
        )
    else:
        namespace = {
            "df": view.frame,
            "sheets": {item.name: item.frame for item in views},
            "tables": [item.frame for item in views],
        }
        try:
            result = run_code(code, namespace)
        except SandboxError as exc:
            raise ToolError(
                f"query_table failed: {exc}\nAvailable: df (sheet '{view.name}': "
                f"{_columns(view)}), sheets {names}."
            ) from exc
        output = f"Python over '{document.file_name}' (sheet '{view.name}'):\n{code}\n=>\n{result}"

    passage = registry.add_text(
        output,
        document.file_name,
        metadata=view.metadata,
        tool="query_table",
        inherit_from=view.block.metadata,
    )
    return ToolResult(
        output=passage.render(),
        sources=[passage.as_search_result()],
        passages=[passage],
    )


def _select_view(views: list[TableView], sheet: str | None) -> TableView:
    if not sheet:
        return views[0]
    wanted = str(sheet).strip().casefold()
    for view in views:
        if view.name.casefold() == wanted:
            return view
    for view in views:
        if wanted in view.name.casefold():
            return view
    names = ", ".join(f"'{view.name}'" for view in views)
    raise ToolError(f"No sheet or table named '{sheet}'. Available: {names}.")


def _columns(view: TableView) -> str:
    return ", ".join(str(column) for column in view.frame.columns[:20])


def _calculate(registry: PassageRegistry, expression: str = "", **extra: Any) -> ToolResult:
    _reject_unknown_arguments("calculate", extra)
    expression = str(expression).strip()
    if not expression:
        raise ToolError("calculate requires the 'expression' argument.")
    try:
        result = run_code(expression)
    except SandboxError as exc:
        raise ToolError(f"calculate failed: {exc}") from exc
    output = f"{expression} = {result}"
    passage = registry.add_text(
        output, "calculation", metadata={"section": "calculate"}, tool="calculate"
    )
    return ToolResult(
        output=passage.render(), sources=[passage.as_search_result()], passages=[passage]
    )


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------


def _find_document(documents: list[Document], file_name: Any) -> Document:
    wanted = str(file_name or "").strip().casefold()
    if not wanted:
        raise ToolError("A 'file_name' is required.")
    for document in documents:
        if document.file_name.casefold() == wanted:
            return document

    partial = [document for document in documents if wanted in document.file_name.casefold()]
    if len(partial) == 1:
        return partial[0]
    # Tolerate a name given without its extension or with a different one.
    stem = wanted.rsplit(".", 1)[0]
    by_stem = [d for d in documents if d.file_name.casefold().rsplit(".", 1)[0] == stem]
    if len(by_stem) == 1:
        return by_stem[0]

    available = ", ".join(document.file_name for document in documents) or "none"
    raise ToolError(f"Document '{file_name}' not found. Available documents: {available}.")


def _find_heading(headings: list[Block], section: str, file_name: str) -> Block:
    wanted = " ".join(section.split()).casefold()
    normalized = [(" ".join(block.text.split()).casefold(), block) for block in headings]
    exact = [block for text, block in normalized if text == wanted]
    if exact:
        return exact[0]
    partial = [block for text, block in normalized if wanted in text]
    if partial:
        return partial[0]
    # Match on a leading number ("3.2") or on most of the words.
    words = [word for word in re.findall(r"\w+", wanted) if len(word) > 2]
    if words:
        scored = sorted(
            (
                (sum(1 for word in words if word in text) / len(words), index, block)
                for index, (text, block) in enumerate(normalized)
            ),
            key=lambda item: (-item[0], item[1]),
        )
        if scored and scored[0][0] >= 0.6:
            return scored[0][2]

    available = "; ".join(block.text.strip() for block in headings[:MAX_TOC_ENTRIES])
    raise ToolError(f"Section '{section}' not found in '{file_name}'. Sections: {available}.")


def _heading_level(block: Block) -> int:
    try:
        return int(block.metadata.get("hierarchy_level", 1))
    except (TypeError, ValueError):
        return 1


def _clamp_int(value: Any, name: str, low: int, high: int, default: int) -> int:
    if value is None or value == "":
        return default
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ToolError(f"'{name}' must be an integer.") from exc
    return max(low, min(number, high))


def _reject_unknown_arguments(tool_name: str, extra: dict[str, Any]) -> None:
    if extra:
        unknown = ", ".join(sorted(extra))
        raise ToolError(f"{tool_name} received unknown argument(s): {unknown}.")
