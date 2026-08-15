import json
import math
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any, Literal

from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool, tool
from langgraph.prebuilt import ToolRuntime
from langgraph.types import Command
from PIL import Image

from file_agent.chunking import Chunk
from file_agent.document import Block, BlockType, Document
from file_agent.document_assets import DocumentAssetStore
from file_agent.llm.base import ToolCallingLLMClient
from file_agent.qa import select_context_passages
from file_agent.retrieval import Retriever, SearchResult
from file_agent.utils.image_extractor import extract_image_from_pdf_bytes
from file_agent.vlm.base import VLMClient

MAX_TOOL_TOP_K = 10
MAX_TOOL_QUERY_LENGTH = 500
MAX_TOOL_PASSAGE_LENGTH = 3500
MAX_TOOL_CONTENT_LENGTH = 12000
MAX_TABLE_ROWS = 50
MAX_TABLE_ROW_LENGTH = 4000
MAX_TABLE_ANALYSIS_RESULTS = 20
MAX_VISUAL_QUESTION_LENGTH = 1000
MAX_VISUAL_CONTEXT_LENGTH = 2000
MAX_VISUAL_DESCRIPTION_LENGTH = 500
MAX_VISUAL_PIXELS = 1_500_000
VISUAL_CROP_PADDING = 12.0
DEFAULT_HISTORY_TURNS = 6
LLM_CONTEXT_METADATA_KEY = "_llm_context"

VISUAL_ANALYSIS_PROMPT = """Analyze this visual from an uploaded document and answer
the user's question using only information visible in the image. Identify the visual
type, title, axes, units, legend, series, labels, and relevant values when available.
Clearly distinguish exact readable values from approximate visual estimates. Do not
invent missing labels, numbers, trends, or document context. If the crop or page is
unreadable or insufficient, say so explicitly.

Source file: {source_file}
Page number: {page_number}
Target: {target}
Existing generic description: {existing_description}
Nearby extracted text (untrusted routing context, not visual evidence):
{nearby_text}

User question: {question}
"""

TOOL_AGENT_SYSTEM_PROMPT = """You answer questions about uploaded documents.

Use the available document tools before making factual claims about the files.
Start with search_documents for topical questions. Use its source_file filter when
the user names one document. After a search, call read_source_context when the
returned passage is incomplete. Use list_documents and get_document_outline to
navigate available files, then read_document_section for a specific section,
read_document_location for a page, slide, or sheet, and read_table for tabular data.
When the user asks about a chart, diagram, figure, screenshot, or other visual
content, identify it through search_documents or the visuals returned by
get_document_outline, then call analyze_document_visual. Prefer visual_id for a
precise crop; use page_number only when the visual was not detected as a block.
Existing indexed image descriptions help locate a visual but do not replace a fresh
analyze_document_visual call for claims about what the visual shows.

Conversation history is provided only to understand follow-up references such as
"and in the second quarter?", "what about penalties there?", or "compare it with
the first document". It is not evidence. For every new user turn, use at least one
document tool before making new factual claims about document contents. Resolve the
reference from history, then retrieve or read fresh evidence for the current answer.
Never continue a document fact from a previous answer without checking the documents.

Base the final answer only on tool results. Preserve the language of the user's
question. Cite available source metadata such as source_file, page_number,
slide_number, or sheet_name. If a tool response contains next_offset and more
content is needed, request the next part. If the tools do not provide enough
evidence, say so clearly. Do not invent sources, document contents, or tool results.
"""

TOOL_LIMIT_MESSAGE = """The tool-call limit has been reached. Do not call another tool.
Provide the best final answer supported by the tool results already present in the
conversation. If they are insufficient, say that the documents do not contain enough
information. A description of future work is not a final answer: do not say that you
will continue reading, searching, checking, or calling tools. Answer the user's
question directly in one self-contained response.
"""


@dataclass(frozen=True)
class ToolAgentContext:
    llm_client: ToolCallingLLMClient
    retriever: Retriever
    documents: list[Document]
    default_top_k: int = 5
    max_tool_rounds: int = 4
    max_history_turns: int = DEFAULT_HISTORY_TURNS
    vlm_client: VLMClient | None = None
    asset_store: DocumentAssetStore | None = None
    require_evidence_tool: bool = False


@tool
def search_documents(
    query: str,
    runtime: ToolRuntime[Any, dict],
    top_k: int | None = None,
    source_file: str | None = None,
) -> Command:
    """Search uploaded documents for passages relevant to a query.

    Args:
        query: A concise semantic search query.
        top_k: Number of retrieval results to return, from 1 to 10. Uses the
            active RAG top-k setting when omitted.
        source_file: Optional exact file name returned by list_documents.
    """
    normalized_query = query.strip()
    if not normalized_query:
        raise ValueError("query must not be empty")
    if len(normalized_query) > MAX_TOOL_QUERY_LENGTH:
        raise ValueError(f"query must not exceed {MAX_TOOL_QUERY_LENGTH} characters")

    resolved_source: str | None = None
    if source_file is not None:
        if not source_file.strip():
            raise ValueError("source_file must not be empty")
        resolved_source = _find_document(runtime.context.documents, source_file).file_name

    requested_top_k = runtime.context.default_top_k if top_k is None else top_k
    bounded_top_k = max(1, min(requested_top_k, MAX_TOOL_TOP_K))
    if resolved_source is None:
        results = runtime.context.retriever.search(
            query=normalized_query,
            top_k=bounded_top_k,
        )
    else:
        results = runtime.context.retriever.search(
            query=normalized_query,
            top_k=bounded_top_k,
            source_file=resolved_source,
        )
    payload = _serialize_search_results(normalized_query, results, resolved_source)
    evidence_sources = _search_evidence_sources(results)
    return _tool_command(
        payload,
        runtime,
        sources=evidence_sources,
        search_query=normalized_query,
    )


@tool
def read_source_context(
    chunk_id: str,
    runtime: ToolRuntime[Any, dict],
) -> Command:
    """Read the stored surrounding passage for a previously found search result.

    Args:
        chunk_id: Exact chunk_id returned by search_documents in this agent run.
    """
    normalized_id = chunk_id.strip()
    if not normalized_id:
        raise ValueError("chunk_id must not be empty")

    result = next(
        (item for item in runtime.state.get("sources", []) if item.chunk.id == normalized_id),
        None,
    )
    if result is None:
        raise ValueError(
            "chunk is not available; call search_documents first and use a returned chunk_id"
        )

    metadata = dict(result.chunk.metadata)
    passage = str(metadata.pop("context", None) or result.chunk.text)
    metadata.pop(LLM_CONTEXT_METADATA_KEY, None)
    payload = {
        "chunk_id": result.chunk.id,
        "text": passage,
        "metadata": metadata,
    }
    updated_metadata = dict(result.chunk.metadata)
    updated_metadata[LLM_CONTEXT_METADATA_KEY] = passage
    updated_source = SearchResult(
        chunk=Chunk(
            id=result.chunk.id,
            text=result.chunk.text,
            metadata=updated_metadata,
        ),
        score=result.score,
    )
    return _tool_command(payload, runtime, sources=[updated_source])


@tool
def list_documents(runtime: ToolRuntime[Any, dict]) -> str:
    """List uploaded documents, their structure, and useful source coordinates."""
    documents = []
    for document in runtime.context.documents:
        metadata = document.metadata
        sheets = _unique_metadata_values(document.blocks, "sheet_name")
        slides = _integer_metadata_values(document.blocks, "slide_number")
        documents.append(
            {
                "source_file": document.file_name,
                "file_type": document.file_type,
                "blocks": len(document.blocks),
                "total_pages": metadata.get("total_pages"),
                "total_slides": max(slides, default=None),
                "sheets": sheets,
                "headings": len(_document_outline(document)),
                "tables": len(_document_tables(document)),
                "visuals": len(_document_visuals(document)),
            }
        )
    return json.dumps({"documents": documents}, ensure_ascii=False, default=str)


@tool
def get_document_outline(
    source_file: str,
    runtime: ToolRuntime[Any, dict],
) -> str:
    """Return readable section, table, and visual identifiers for one document.

    Args:
        source_file: Exact file name returned by list_documents.
    """
    document = _find_document(runtime.context.documents, source_file)
    return json.dumps(
        {
            "source_file": document.file_name,
            "table_of_contents": _document_outline(document),
            "tables": _document_tables(document),
            "table_analysis_hint": (
                "Use analyze_table instead of paginating read_table for whole-table "
                "totals, distinct counts, extrema, or grouped sums."
            ),
            "visuals": _document_visuals(document),
        },
        ensure_ascii=False,
        default=str,
    )


@tool
def read_document_section(
    source_file: str,
    section_id: str,
    runtime: ToolRuntime[Any, dict],
    offset: int = 0,
) -> Command:
    """Read a section identified by get_document_outline, including subsections.

    Args:
        source_file: Exact file name returned by list_documents.
        section_id: Exact section_id returned by get_document_outline.
        offset: Character offset returned as next_offset, or 0 for the first part.
    """
    document = _find_document(runtime.context.documents, source_file)
    outline = _document_outline(document)
    normalized_id = section_id.strip()
    entry = next((item for item in outline if item.get("section_id") == normalized_id), None)
    if entry is None:
        raise ValueError(f"section is not indexed in {document.file_name}: {section_id}")

    blocks = _section_blocks(document, normalized_id, entry.get("level", 1))
    content = _render_blocks(blocks)
    page = _paginate_text(content, offset)
    metadata = _source_metadata(document, blocks)
    metadata.update(
        {
            "section": entry.get("title"),
            "section_id": normalized_id,
            "content_offset": offset,
        }
    )
    payload = {
        "source_file": document.file_name,
        "section_id": normalized_id,
        "title": entry.get("title"),
        **page,
        "metadata": metadata,
    }
    return _evidence_command(
        payload,
        runtime,
        evidence_id=f"section:{document.file_name}:{normalized_id}:{offset}",
        text=page["text"],
        metadata=metadata,
    )


@tool
def read_document_location(
    source_file: str,
    runtime: ToolRuntime[Any, dict],
    page_number: int | None = None,
    slide_number: int | None = None,
    sheet_name: str | None = None,
    offset: int = 0,
) -> Command:
    """Read one page, slide, or sheet from an uploaded document.

    Exactly one location argument must be provided.

    Args:
        source_file: Exact file name returned by list_documents.
        page_number: One-indexed PDF or DOCX page number.
        slide_number: One-indexed PPTX slide number.
        sheet_name: Exact XLSX sheet name returned by list_documents.
        offset: Character offset returned as next_offset, or 0 for the first part.
    """
    document = _find_document(runtime.context.documents, source_file)
    locator_name, locator_value = _validate_location(page_number, slide_number, sheet_name)
    blocks = [
        block
        for block in document.blocks
        if _block_matches_location(block, locator_name, locator_value)
    ]
    if not blocks:
        raise ValueError(f"{locator_name}={locator_value!r} is not indexed in {document.file_name}")
    if locator_name == "sheet_name":
        locator_value = str(blocks[0].metadata["sheet_name"])

    content = _render_blocks(blocks)
    page = _paginate_text(content, offset)
    metadata = _source_metadata(document, blocks)
    metadata.update({locator_name: locator_value, "content_offset": offset})
    payload = {
        "source_file": document.file_name,
        "location": {locator_name: locator_value},
        **page,
        "metadata": metadata,
    }
    return _evidence_command(
        payload,
        runtime,
        evidence_id=(f"location:{document.file_name}:{locator_name}:{locator_value}:{offset}"),
        text=page["text"],
        metadata=metadata,
    )


@tool
def read_table(
    source_file: str,
    table_id: str,
    runtime: ToolRuntime[Any, dict],
    offset: int = 0,
    limit: int = 25,
) -> Command:
    """Read rows from a table or XLSX sheet identified by get_document_outline.

    Args:
        source_file: Exact file name returned by list_documents.
        table_id: Exact table_id returned by get_document_outline.
        offset: Zero-based row offset returned as next_offset.
        limit: Maximum number of rows to return, from 1 to 50.
    """
    document = _find_document(runtime.context.documents, source_file)
    normalized_id = table_id.strip()
    block = next(
        (
            item
            for item in document.blocks
            if item.id == normalized_id and _is_table_block(document, item)
        ),
        None,
    )
    if block is None:
        raise ValueError(f"table is not indexed in {document.file_name}: {table_id}")

    rows = [line for line in block.text.splitlines() if line.strip()]
    row_page = _paginate_rows(rows, offset, limit)
    metadata = _source_metadata(document, [block])
    metadata.update(
        {
            "table_id": normalized_id,
            "row_offset": offset,
            "table_format": _table_format(document, block),
        }
    )
    payload = {
        "source_file": document.file_name,
        "table_id": normalized_id,
        "format": metadata["table_format"],
        **row_page,
        "metadata": metadata,
    }
    return _evidence_command(
        payload,
        runtime,
        evidence_id=f"table:{document.file_name}:{normalized_id}:{offset}",
        text="\n".join(row_page["rows"]),
        metadata=metadata,
    )


@tool
def analyze_table(
    source_file: str,
    table_id: str,
    operation: Literal["count_distinct", "sum", "min", "max", "group_sum"],
    value_column: str,
    runtime: ToolRuntime[Any, dict],
    group_by: str | None = None,
    multiply_by: str | None = None,
    top_n: int = 5,
) -> Command:
    """Calculate an exact result over every row of a table or XLSX sheet.

    Prefer this tool over read_table when the question asks for a total, unique
    count, minimum/maximum row, or grouped total. ``multiply_by`` calculates a
    row value as ``value_column * multiply_by`` (for example stock * price).

    Args:
        source_file: Exact file name returned by list_documents.
        table_id: Exact table_id returned by get_document_outline.
        operation: count_distinct, sum, min, max, or group_sum.
        value_column: Column to count or calculate.
        group_by: Required grouping column for group_sum.
        multiply_by: Optional second numeric column multiplied into each value.
        top_n: Number of ranked rows/groups to return, from 1 to 20.
    """
    document = _find_document(runtime.context.documents, source_file)
    normalized_id = table_id.strip()
    block = next(
        (
            item
            for item in document.blocks
            if item.id == normalized_id and _is_table_block(document, item)
        ),
        None,
    )
    if block is None:
        raise ValueError(f"table is not indexed in {document.file_name}: {table_id}")

    headers, rows = _parse_table_rows(document, block)
    value_key = _resolve_table_column(headers, value_column)
    group_key = _resolve_table_column(headers, group_by) if group_by is not None else None
    multiplier_key = (
        _resolve_table_column(headers, multiply_by) if multiply_by is not None else None
    )
    bounded_top_n = _bounded_table_analysis_results(top_n)

    result, rows_used, rows_skipped = _calculate_table_result(
        rows=rows,
        operation=operation,
        value_key=value_key,
        group_key=group_key,
        multiplier_key=multiplier_key,
        top_n=bounded_top_n,
    )
    metadata = _source_metadata(document, [block])
    metadata.update(
        {
            "table_id": normalized_id,
            "table_format": _table_format(document, block),
            "table_operation": operation,
        }
    )
    payload = {
        "source_file": document.file_name,
        "table_id": normalized_id,
        "operation": operation,
        "value_column": value_key,
        "multiply_by": multiplier_key,
        "group_by": group_key,
        "total_data_rows": len(rows),
        "rows_used": rows_used,
        "rows_skipped": rows_skipped,
        "result": result,
        "metadata": metadata,
    }
    evidence_text = json.dumps(payload, ensure_ascii=False, default=str)
    return _evidence_command(
        payload,
        runtime,
        evidence_id=f"table-analysis:{document.file_name}:{normalized_id}:{operation}",
        text=evidence_text,
        metadata=metadata,
    )


@tool
def analyze_document_visual(
    source_file: str,
    question: str,
    runtime: ToolRuntime[Any, dict],
    visual_id: str | None = None,
    page_number: int | None = None,
) -> Command:
    """Analyze a PDF figure or page with the configured vision-language model.

    Provide exactly one target. Prefer visual_id from get_document_outline for a
    precise crop. Use page_number as a fallback when the visual was not detected.

    Args:
        source_file: Exact PDF file name returned by list_documents.
        question: Specific question to answer about the visible content.
        visual_id: Exact visual_id returned by get_document_outline.
        page_number: One-indexed PDF page to analyze as a whole-page fallback.
    """
    document = _find_document(runtime.context.documents, source_file)
    if document.file_type.lower().lstrip(".") != "pdf":
        raise ValueError("visual analysis currently supports PDF documents only")

    normalized_question = question.strip()
    if not normalized_question:
        raise ValueError("question must not be empty")
    if len(normalized_question) > MAX_VISUAL_QUESTION_LENGTH:
        raise ValueError(f"question must not exceed {MAX_VISUAL_QUESTION_LENGTH} characters")

    normalized_visual_id = visual_id.strip() if visual_id is not None else None
    has_visual_id = bool(normalized_visual_id)
    has_page_number = page_number is not None
    if has_visual_id == has_page_number:
        raise ValueError("provide exactly one of visual_id or page_number")

    vlm_client = runtime.context.vlm_client
    if vlm_client is None:
        raise ValueError("visual analysis is unavailable: no VLM backend is configured")
    asset_store = runtime.context.asset_store
    if asset_store is None:
        raise ValueError("visual analysis is unavailable: original document asset is missing")

    selected_block: Block | None = None
    bbox: tuple[float, float, float, float] | None = None
    if has_visual_id:
        selected_block = next(
            (
                block
                for block in document.blocks
                if block.id == normalized_visual_id
                and block.block_type in (BlockType.FIGURE, BlockType.IMAGE)
            ),
            None,
        )
        if selected_block is None:
            raise ValueError(
                f"visual is not indexed in {document.file_name}: {normalized_visual_id}"
            )
        block_page_number = _block_page_number(selected_block)
        if block_page_number is None or selected_block.bbox is None:
            raise ValueError("the selected visual has no renderable PDF coordinates")
        resolved_page_number = block_page_number
        bbox = selected_block.bbox
        target = f"visual_id={selected_block.id}"
    else:
        if not isinstance(page_number, int) or isinstance(page_number, bool) or page_number <= 0:
            raise ValueError("page_number must be a positive integer")
        resolved_page_number = page_number
        target = "whole PDF page"

    image = extract_image_from_pdf_bytes(
        asset_store.get_bytes(document.file_name),
        page_number=resolved_page_number,
        bbox=bbox,
        padding=VISUAL_CROP_PADDING if bbox is not None else 0.0,
        max_pixels=MAX_VISUAL_PIXELS,
    )
    if image is None:
        raise ValueError("the selected PDF visual could not be rendered")
    image = _limit_image_pixels(image, MAX_VISUAL_PIXELS)

    relevant_blocks = (
        [selected_block]
        if selected_block is not None
        else [
            block for block in document.blocks if _block_page_number(block) == resolved_page_number
        ]
    )
    existing_description = selected_block.vlm_description if selected_block is not None else None
    prompt = VISUAL_ANALYSIS_PROMPT.format(
        source_file=document.file_name,
        page_number=resolved_page_number,
        target=target,
        existing_description=existing_description or "not available",
        nearby_text=_visual_nearby_text(
            document,
            resolved_page_number,
            selected_block,
        ),
        question=normalized_question,
    )
    analysis = vlm_client.describe_image(image, prompt).strip()
    if not analysis:
        raise ValueError("VLM returned an empty visual analysis")

    metadata = _source_metadata(document, relevant_blocks)
    metadata.update(
        {
            "page_number": resolved_page_number,
            "visual_id": selected_block.id if selected_block is not None else None,
            "bbox": bbox,
            "evidence_type": "visual_analysis",
        }
    )
    payload = {
        "source_file": document.file_name,
        "page_number": resolved_page_number,
        "visual_id": metadata["visual_id"],
        "bbox": bbox,
        "analysis": analysis,
        "metadata": metadata,
    }
    evidence_target = selected_block.id if selected_block is not None else f"page-{page_number}"
    return _evidence_command(
        payload,
        runtime,
        evidence_id=f"visual:{document.file_name}:{evidence_target}",
        text=analysis,
        metadata=metadata,
    )


DOCUMENT_TOOLS: list[BaseTool] = [
    search_documents,
    read_source_context,
    list_documents,
    get_document_outline,
    read_document_section,
    read_document_location,
    analyze_table,
    read_table,
    analyze_document_visual,
]


def _serialize_search_results(
    query: str,
    results: list[SearchResult],
    source_file: str | None = None,
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
        "source_file": source_file,
        "results_count": len(serialized),
        "results": serialized,
    }


def _search_evidence_sources(results: list[SearchResult]) -> list[SearchResult]:
    evidence_sources = []
    for result, passage in select_context_passages(results):
        metadata = dict(result.chunk.metadata)
        metadata[LLM_CONTEXT_METADATA_KEY] = _truncate_passage(passage)
        evidence_sources.append(
            SearchResult(
                chunk=Chunk(
                    id=result.chunk.id,
                    text=result.chunk.text,
                    metadata=metadata,
                ),
                score=result.score,
            )
        )
    return evidence_sources


def _find_document(documents: list[Document], source_file: str) -> Document:
    requested_name = source_file.strip().casefold()
    matches = [item for item in documents if item.file_name.casefold() == requested_name]
    if not matches:
        raise ValueError(f"document is not indexed: {source_file}")
    if len(matches) > 1:
        raise ValueError(f"multiple indexed documents share this file name: {source_file}")
    return matches[0]


def _document_outline(document: Document) -> list[dict[str, Any]]:
    raw_outline = document.metadata.get("table_of_contents") or []
    if not raw_outline:
        raw_outline = [
            {
                "title": block.text.strip(),
                "level": block.metadata.get("hierarchy_level", 1),
                "page": _block_page_number(block),
                "block_id": block.id,
            }
            for block in document.blocks
            if block.block_type == BlockType.HEADING and block.text.strip()
        ]

    outline = []
    for entry in raw_outline:
        block_id = entry.get("block_id") or entry.get("section_id")
        outline.append(
            {
                "section_id": block_id,
                "title": entry.get("title"),
                "level": entry.get("level", 1),
                "page_number": entry.get("page_number", entry.get("page")),
            }
        )
    return outline


def _document_tables(document: Document) -> list[dict[str, Any]]:
    tables = []
    current_section: str | None = None
    for block in document.blocks:
        if block.block_type == BlockType.HEADING and block.text.strip():
            current_section = block.text.strip()
        if not _is_table_block(document, block):
            continue
        tables.append(
            {
                "table_id": block.id,
                "section": current_section,
                "page_number": _block_page_number(block),
                "slide_number": block.metadata.get("slide_number"),
                "sheet_name": block.metadata.get("sheet_name"),
                "rows": len([line for line in block.text.splitlines() if line.strip()]),
                "format": _table_format(document, block),
            }
        )
    return tables


def _document_visuals(document: Document) -> list[dict[str, Any]]:
    visuals = []
    current_section: str | None = None
    for block in document.blocks:
        if block.block_type == BlockType.HEADING and block.text.strip():
            current_section = block.text.strip()
        if block.block_type not in (BlockType.FIGURE, BlockType.IMAGE):
            continue

        description = block.vlm_description or block.text.strip() or None
        if description and len(description) > MAX_VISUAL_DESCRIPTION_LENGTH:
            description = description[:MAX_VISUAL_DESCRIPTION_LENGTH].rstrip() + "..."
        visuals.append(
            {
                "visual_id": block.id,
                "block_type": block.block_type.value,
                "section": current_section,
                "page_number": _block_page_number(block),
                "bbox": block.bbox,
                "description": description,
                "renderable": _block_page_number(block) is not None and block.bbox is not None,
            }
        )
    return visuals


def _is_table_block(document: Document, block: Block) -> bool:
    file_type = document.file_type.lower().lstrip(".")
    return (
        block.block_type == BlockType.TABLE
        or block.type == BlockType.TABLE.value
        or (file_type == "xlsx" and block.type == "xlsx_sheet")
    )


def _table_format(document: Document, block: Block) -> str:
    if document.file_type.lower().lstrip(".") == "xlsx" or "\t" in block.text:
        return "tsv"
    if "|" in block.text:
        return "markdown"
    return "text"


def _parse_table_rows(
    document: Document,
    block: Block,
) -> tuple[list[str], list[dict[str, str]]]:
    lines = [line for line in block.text.splitlines() if line.strip()]
    if len(lines) < 2:
        raise ValueError("the selected table must contain a header and at least one data row")

    table_format = _table_format(document, block)
    if table_format == "tsv":
        parsed = [[cell.strip() for cell in line.split("\t")] for line in lines]
    elif table_format == "markdown":
        parsed = [[cell.strip() for cell in line.strip().strip("|").split("|")] for line in lines]
        parsed = [row for row in parsed if not _is_markdown_separator_row(row)]
    else:
        raise ValueError("table analysis currently supports TSV/XLSX and Markdown tables")

    headers = parsed[0]
    if not headers or any(not header for header in headers):
        raise ValueError("the selected table has an invalid or empty header")
    if len({header.casefold() for header in headers}) != len(headers):
        raise ValueError("the selected table contains duplicate column names")

    rows = []
    for values in parsed[1:]:
        padded = [*values[: len(headers)], *([""] * max(0, len(headers) - len(values)))]
        rows.append(dict(zip(headers, padded, strict=True)))
    if not rows:
        raise ValueError("the selected table contains no data rows")
    return headers, rows


def _is_markdown_separator_row(values: list[str]) -> bool:
    return bool(values) and all(re.fullmatch(r":?-{3,}:?", value) for value in values)


def _resolve_table_column(headers: list[str], requested: str | None) -> str:
    if requested is None or not requested.strip():
        raise ValueError("column name must not be empty")
    normalized = requested.strip().casefold()
    match = next((header for header in headers if header.casefold() == normalized), None)
    if match is None:
        raise ValueError(f"column is not available: {requested}; available columns: {headers}")
    return match


def _bounded_table_analysis_results(top_n: int) -> int:
    if not isinstance(top_n, int) or isinstance(top_n, bool):
        raise ValueError("top_n must be an integer")
    return max(1, min(top_n, MAX_TABLE_ANALYSIS_RESULTS))


def _calculate_table_result(
    rows: list[dict[str, str]],
    operation: str,
    value_key: str,
    group_key: str | None,
    multiplier_key: str | None,
    top_n: int,
) -> tuple[dict[str, Any], int, int]:
    if operation == "count_distinct":
        if group_key is not None or multiplier_key is not None:
            raise ValueError("count_distinct does not accept group_by or multiply_by")
        distinct: dict[str, str] = {}
        for row in rows:
            value = row[value_key].strip()
            if value:
                distinct.setdefault(value.casefold(), value)
        values = sorted(distinct.values(), key=str.casefold)
        return {"count": len(values), "values": values}, len(rows), 0

    if operation == "group_sum" and group_key is None:
        raise ValueError("group_by is required for group_sum")
    if operation != "group_sum" and group_key is not None:
        raise ValueError("group_by is only supported for group_sum")

    calculated: list[tuple[Decimal, dict[str, str]]] = []
    skipped = 0
    for row in rows:
        try:
            value = _decimal_value(row[value_key])
            if multiplier_key is not None:
                value *= _decimal_value(row[multiplier_key])
        except ValueError:
            skipped += 1
            continue
        calculated.append((value, row))
    if not calculated:
        raise ValueError("the selected numeric columns contain no usable values")

    if operation == "sum":
        total = sum((value for value, _ in calculated), Decimal(0))
        return {"value": _format_decimal(total)}, len(calculated), skipped

    if operation in ("min", "max"):
        reverse = operation == "max"
        ranked = sorted(calculated, key=lambda item: item[0], reverse=reverse)[:top_n]
        return (
            {"rows": [{"value": _format_decimal(value), "row": row} for value, row in ranked]},
            len(calculated),
            skipped,
        )

    if operation == "group_sum":
        grouped: dict[str, tuple[str, Decimal]] = {}
        for value, row in calculated:
            group = row[group_key].strip()  # type: ignore[index]
            if not group:
                skipped += 1
                continue
            normalized_group = group.casefold()
            label, total = grouped.get(normalized_group, (group, Decimal(0)))
            grouped[normalized_group] = (label, total + value)
        ranked_groups = sorted(grouped.values(), key=lambda item: item[1], reverse=True)[:top_n]
        return (
            {
                "groups": [
                    {"group": label, "value": _format_decimal(value)}
                    for label, value in ranked_groups
                ]
            },
            len(calculated),
            skipped,
        )

    raise ValueError(f"unsupported table operation: {operation}")


def _decimal_value(raw: str) -> Decimal:
    normalized = raw.strip().replace("\u00a0", "").replace(" ", "")
    normalized = normalized.removeprefix("$").removeprefix("€").removeprefix("£")
    normalized = normalized.removesuffix("₽").removesuffix("%").strip()
    if not normalized:
        raise ValueError("numeric value is empty")
    if "," in normalized and "." in normalized:
        if normalized.rfind(",") > normalized.rfind("."):
            normalized = normalized.replace(".", "").replace(",", ".")
        else:
            normalized = normalized.replace(",", "")
    elif "," in normalized:
        normalized = normalized.replace(",", ".")
    try:
        value = Decimal(normalized)
    except InvalidOperation as exc:
        raise ValueError(f"not a numeric value: {raw}") from exc
    if not value.is_finite():
        raise ValueError(f"not a finite numeric value: {raw}")
    return value


def _format_decimal(value: Decimal) -> str:
    rendered = format(value, "f")
    if "." in rendered:
        rendered = rendered.rstrip("0").rstrip(".")
    return rendered or "0"


def _section_blocks(document: Document, section_id: str, raw_level: Any) -> list[Block]:
    start = next(
        (index for index, block in enumerate(document.blocks) if block.id == section_id),
        None,
    )
    if start is None:
        raise ValueError(f"section block is not available in {document.file_name}: {section_id}")

    level = _heading_level(raw_level)
    end = len(document.blocks)
    for index in range(start + 1, len(document.blocks)):
        block = document.blocks[index]
        if block.block_type != BlockType.HEADING:
            continue
        if _heading_level(block.metadata.get("hierarchy_level", 1)) <= level:
            end = index
            break
    return document.blocks[start:end]


def _heading_level(value: Any) -> int:
    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        return 1


def _render_blocks(blocks: list[Block]) -> str:
    parts = []
    for block in blocks:
        rendered = block.to_markdown().strip()
        if rendered:
            parts.append(rendered)
    text = "\n\n".join(parts)
    if not text:
        raise ValueError("the selected document part contains no readable text")
    return text


def _validate_location(
    page_number: int | None,
    slide_number: int | None,
    sheet_name: str | None,
) -> tuple[str, int | str]:
    provided = [
        ("page_number", page_number) if page_number is not None else None,
        ("slide_number", slide_number) if slide_number is not None else None,
        ("sheet_name", sheet_name) if sheet_name is not None else None,
    ]
    selected = [item for item in provided if item is not None]
    if len(selected) != 1:
        raise ValueError("provide exactly one of page_number, slide_number, or sheet_name")

    name, value = selected[0]
    if name in ("page_number", "slide_number"):
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
        return name, value

    normalized_sheet = str(value).strip()
    if not normalized_sheet:
        raise ValueError("sheet_name must not be empty")
    return name, normalized_sheet


def _block_matches_location(block: Block, name: str, value: int | str) -> bool:
    if name == "page_number":
        return _block_page_number(block) == value
    if name == "slide_number":
        return block.metadata.get("slide_number") == value
    block_sheet = block.metadata.get("sheet_name")
    return isinstance(block_sheet, str) and block_sheet.casefold() == str(value).casefold()


def _block_page_number(block: Block) -> int | None:
    if block.page_number is not None:
        return block.page_number
    value = block.metadata.get("page_number")
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _source_metadata(document: Document, blocks: list[Block]) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "source_file": document.file_name,
        "file_type": document.file_type,
        "block_ids": [block.id for block in blocks],
    }
    pages = sorted({page for block in blocks if (page := _block_page_number(block)) is not None})
    slides = _integer_metadata_values(blocks, "slide_number")
    sheets = _unique_metadata_values(blocks, "sheet_name")
    dataset_doc_ids = _unique_metadata_values(blocks, "dataset_doc_id")
    dataset_record_ids = _unique_metadata_values(blocks, "dataset_record_id")
    if pages:
        metadata["page_number"] = pages[0]
        metadata["page_numbers"] = pages
    if slides:
        metadata["slide_number"] = slides[0]
        metadata["slide_numbers"] = slides
    if sheets:
        metadata["sheet_name"] = sheets[0]
        metadata["sheet_names"] = sheets
    if len(dataset_doc_ids) == 1:
        metadata["dataset_doc_id"] = dataset_doc_ids[0]
    if len(dataset_record_ids) == 1:
        metadata["dataset_record_id"] = dataset_record_ids[0]
    return metadata


def _visual_nearby_text(
    document: Document,
    page_number: int,
    selected_block: Block | None,
) -> str:
    page_blocks = [
        block
        for block in document.blocks
        if _block_page_number(block) == page_number and block is not selected_block
    ]
    parts = []
    for block in page_blocks:
        text = block.text.strip()
        if not text or block.block_type in (BlockType.FIGURE, BlockType.IMAGE):
            continue
        parts.append(text)
    nearby_text = "\n\n".join(parts).strip()
    if not nearby_text:
        return "not available"
    if len(nearby_text) > MAX_VISUAL_CONTEXT_LENGTH:
        return nearby_text[:MAX_VISUAL_CONTEXT_LENGTH].rstrip() + "..."
    return nearby_text


def _limit_image_pixels(image: Image.Image, max_pixels: int) -> Image.Image:
    pixels = image.width * image.height
    if pixels <= max_pixels:
        return image
    scale = math.sqrt(max_pixels / pixels)
    size = (
        max(1, round(image.width * scale)),
        max(1, round(image.height * scale)),
    )
    return image.resize(size, Image.Resampling.LANCZOS)


def _integer_metadata_values(blocks: list[Block], key: str) -> list[int]:
    return sorted(
        {
            value
            for block in blocks
            if isinstance((value := block.metadata.get(key)), int) and not isinstance(value, bool)
        }
    )


def _unique_metadata_values(blocks: list[Block], key: str) -> list[str]:
    values: list[str] = []
    for block in blocks:
        value = block.metadata.get(key)
        if isinstance(value, str) and value not in values:
            values.append(value)
    return values


def _paginate_text(text: str, offset: int) -> dict[str, Any]:
    if not isinstance(offset, int) or isinstance(offset, bool) or offset < 0:
        raise ValueError("offset must be a non-negative integer")
    if offset >= len(text) and text:
        raise ValueError("offset is beyond the available content")

    end = min(len(text), offset + MAX_TOOL_CONTENT_LENGTH)
    if end < len(text):
        boundary = max(text.rfind("\n", offset, end), text.rfind(" ", offset, end))
        if boundary > offset + MAX_TOOL_CONTENT_LENGTH // 2:
            end = boundary
    page = text[offset:end].strip()
    return {
        "text": page,
        "offset": offset,
        "next_offset": end if end < len(text) else None,
        "total_characters": len(text),
    }


def _paginate_rows(rows: list[str], offset: int, limit: int) -> dict[str, Any]:
    if not isinstance(offset, int) or isinstance(offset, bool) or offset < 0:
        raise ValueError("offset must be a non-negative integer")
    if not rows:
        raise ValueError("the selected table contains no readable rows")
    if offset >= len(rows):
        raise ValueError("offset is beyond the available table rows")
    if not isinstance(limit, int) or isinstance(limit, bool):
        raise ValueError("limit must be an integer")

    bounded_limit = max(1, min(limit, MAX_TABLE_ROWS))
    selected: list[str] = []
    used_characters = 0
    for row in rows[offset : offset + bounded_limit]:
        bounded_row = row
        if len(bounded_row) > MAX_TABLE_ROW_LENGTH:
            bounded_row = bounded_row[:MAX_TABLE_ROW_LENGTH].rstrip() + "..."
        addition = len(bounded_row) + (1 if selected else 0)
        if selected and used_characters + addition > MAX_TOOL_CONTENT_LENGTH:
            break
        selected.append(bounded_row)
        used_characters += addition

    next_offset = offset + len(selected)
    return {
        "rows": selected,
        "offset": offset,
        "next_offset": next_offset if next_offset < len(rows) else None,
        "total_rows": len(rows),
    }


def _tool_command(
    payload: dict[str, Any],
    runtime: ToolRuntime[Any, dict],
    sources: list[SearchResult] | None = None,
    search_query: str | None = None,
) -> Command:
    update: dict[str, Any] = {
        "messages": [
            ToolMessage(
                content=json.dumps(payload, ensure_ascii=False, default=str),
                tool_call_id=runtime.tool_call_id,
            )
        ]
    }
    if sources:
        update["sources"] = sources
    if search_query is not None:
        update["search_queries"] = [search_query]
    return Command(update=update)


def _evidence_command(
    payload: dict[str, Any],
    runtime: ToolRuntime[Any, dict],
    evidence_id: str,
    text: str,
    metadata: dict[str, Any],
) -> Command:
    source = SearchResult(
        chunk=Chunk(id=evidence_id, text=text, metadata=metadata),
        score=1.0,
    )
    return _tool_command(payload, runtime, sources=[source])


def _truncate_passage(passage: str) -> str:
    if len(passage) <= MAX_TOOL_PASSAGE_LENGTH:
        return passage
    return passage[:MAX_TOOL_PASSAGE_LENGTH].rstrip() + "..."
