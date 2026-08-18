import json
import math
import os
import tempfile
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image

from file_agent.agent.sandbox import run_sandboxed_code
from file_agent.chunking import Chunk
from file_agent.document import Block, BlockType, Document
from file_agent.qa import build_context_from_results
from file_agent.retrieval import Retriever, SearchResult
from file_agent.utils.image_extractor import extract_image_from_pdf, extract_image_from_pptx
from file_agent.vlm.base import VLMClient
from file_agent.vlm.factory import create_vlm_client

STANDALONE_IMAGE_FILE_TYPES = {"jpg", "jpeg", "png"}


ALL_TOOL_NAMES = ("search_documents", "list_documents", "read_page", "run_python", "describe_image")

MAX_PAGE_CHARS = 6000

MAX_SCHEMA_COLUMNS = 15


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


def _is_undescribed_figure(block: Block) -> bool:

    return (
        block.block_type in (BlockType.FIGURE, BlockType.IMAGE)
        and not block.text
        and not block.vlm_description
    )


def search_documents(
    retriever: Retriever,
    query: str,
    top_k: int = 5,
    source_file: str | None = None,
    documents: Sequence[Document] | None = None,
) -> ToolResult:
    if source_file:
        known_names = {document.file_name for document in documents or []}
        if known_names and source_file not in known_names:
            available = ", ".join(sorted(known_names))
            return ToolResult(
                content=f"Error: no document named '{source_file}'. Available: {available}"
            )
        results = retriever.search(query=query, top_k=top_k, source_file=source_file)
    else:
        results = retriever.search(query=query, top_k=top_k)
    if not results:
        return ToolResult(content="No matching passages found.")
    return ToolResult(content=build_context_from_results(results), sources=results)


def list_documents(documents: Sequence[Document]) -> ToolResult:
    entries = []
    for document in documents:
        sheets = _sheet_summaries(document.blocks)
        slides = _integer_metadata_values(document.blocks, "slide_number")
        figure_pages = sorted(
            {
                block.page_number
                for block in document.blocks
                if _is_undescribed_figure(block) and block.page_number is not None
            }
        )
        entries.append(
            {
                "file_name": document.file_name,
                "file_type": document.file_type,
                "total_pages": document.metadata.get("total_pages") or None,
                "total_slides": max(slides, default=None),
                "sheets": sheets,
                "headings": len(document.metadata.get("table_of_contents") or []),
                "pages_with_undescribed_images": figure_pages,
            }
        )

    content = json.dumps({"documents": entries}, ensure_ascii=False, default=str)
    sources = (
        [_tool_evidence("list_documents", f"Document structure:\n{content}")] if entries else []
    )
    return ToolResult(content=content, sources=sources)


def _sheet_summaries(blocks: list[Block]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for block in blocks:
        sheet_name = block.metadata.get("sheet_name")
        if not isinstance(sheet_name, str):
            continue

        header = block.text.partition("\n")[0]
        columns = [cell.strip() for cell in header.split("\t") if cell.strip()]
        max_row = block.metadata.get("max_row")
        total_rows = max_row if isinstance(max_row, int) and not isinstance(max_row, bool) else None
        summaries.append(
            {
                "name": sheet_name,
                "columns": columns,
                "data_rows": total_rows - 1 if total_rows and columns else total_rows,
            }
        )
    return summaries


def _spreadsheet_schema(documents: Sequence[Document]) -> str:
    lines: list[str] = []
    for document in documents:
        for sheet in _sheet_summaries(document.blocks):
            columns = sheet["columns"][:MAX_SCHEMA_COLUMNS]
            listed = ", ".join(columns) if columns else "(no header row)"
            if len(sheet["columns"]) > len(columns):
                listed += ", ..."
            rows = f" [{sheet['data_rows']} data rows]" if sheet["data_rows"] else ""
            lines.append(f"  {document.file_name} sheet '{sheet['name']}'{rows}: {listed}")
    return "\n".join(lines)


def read_page(documents: Sequence[Document], pages: Sequence[Mapping[str, Any]]) -> ToolResult:

    document_by_name = {document.file_name: document for document in documents}
    available = ", ".join(sorted(document_by_name)) or "(none)"
    labelled = len(pages) > 1  # single-page calls keep the old, header-free output

    sections: list[str] = []
    sources: list[SearchResult] = []
    for ref in pages:
        file_name = ref.get("file_name")
        page = ref.get("page")
        offset = ref.get("offset", 0)
        header = f"=== {file_name} p.{page} ===\n" if labelled else ""

        document = document_by_name.get(file_name)
        if document is None:
            sections.append(
                f"{header}Error: no document named '{file_name}'. Available: {available}"
            )
            continue

        blocks = [block for block in document.blocks if block.page_number == page]
        if not blocks:
            known_pages = _integer_metadata_values(document.blocks, "page_number") or [
                block.page_number for block in document.blocks if block.page_number
            ]
            span = f"1-{max(known_pages)}" if known_pages else "none"
            sections.append(f"{header}Error: '{file_name}' has no page {page}. Pages: {span}.")
            continue

        figure_count = sum(1 for block in blocks if _is_undescribed_figure(block))
        figure_note = (
            f"\n[This page has {figure_count} image(s)/figure(s) not shown in this text - "
            f"call describe_image(file_name='{file_name}', page={page}, question=...) if they "
            "may be relevant.]"
            if figure_count
            else ""
        )

        text = "\n\n".join(block.text.strip() for block in blocks if block.text.strip())
        if not text:
            if figure_count:
                sections.append(
                    f"{header}Page {page} of '{file_name}' has no extractable text.{figure_note}"
                )
            else:
                sections.append(f"{header}Page {page} of '{file_name}' has no extractable text.")
            continue

        if not isinstance(offset, int) or isinstance(offset, bool) or offset < 0:
            sections.append(f"{header}Error: offset must be a non-negative integer.")
            continue
        if offset >= len(text):
            sections.append(
                f"{header}Error: offset {offset} is beyond this page's {len(text)} character(s)."
            )
            continue

        page_text, next_offset = _paginate(text, offset, MAX_PAGE_CHARS)
        suffix = (
            f"\n[more available - call again with offset={next_offset} for this page]"
            if next_offset is not None
            else ""
        )
        sections.append(f"{header}{page_text}{suffix}{figure_note}")
        sources.append(_page_evidence(document, blocks, page, page_text))

    return ToolResult(content="\n\n".join(sections), sources=sources)


def _paginate(text: str, offset: int, limit: int) -> tuple[str, int | None]:

    end = min(len(text), offset + limit)
    if end < len(text):
        boundary = max(text.rfind("\n", offset, end), text.rfind(" ", offset, end))
        if boundary > offset + limit // 2:
            end = boundary
    return text[offset:end], end if end < len(text) else None


def _page_evidence(document: Document, blocks: list[Block], page: int, text: str) -> SearchResult:

    dataset_doc_id = next(
        (
            block.metadata["dataset_doc_id"]
            for block in blocks
            if block.metadata.get("dataset_doc_id")
        ),
        document.file_name,
    )
    return SearchResult(
        chunk=Chunk(
            id=f"read_page:{document.file_name}:{page}",
            text=text,
            metadata={
                "source": "read_page",
                "source_file": document.file_name,
                "page_number": page,
                "dataset_doc_id": dataset_doc_id,
            },
        ),
        score=1.0,
    )


def _integer_metadata_values(blocks: list[Block], key: str) -> list[int]:
    return sorted(
        {
            value
            for block in blocks
            if isinstance((value := block.metadata.get(key)), int) and not isinstance(value, bool)
        }
    )


DESCRIBE_IMAGE_PROMPT = (
    "Describe only what is actually visible in this image, focusing on "
    "answering the question below. State the image's type (chart, diagram, "
    "screenshot, table, photo) and transcribe the text you can read: title, "
    "axis labels, legend entries, series and node names. Clearly distinguish "
    "exact values you can read directly from approximate visual estimates. "
    "Do not guess the subject, and do not invent numbers, names, dates, or "
    "context that are not shown. If the image is decorative, unreadable, or "
    "does not answer the question, say exactly that in one short sentence.\n\n"
    "Question: {question}"
)

MAX_VLM_PIXELS = 1_500_000


def _limit_image_pixels(image: Image.Image, max_pixels: int = MAX_VLM_PIXELS) -> Image.Image:
    pixels = image.width * image.height
    if pixels <= max_pixels:
        return image
    scale = math.sqrt(max_pixels / pixels)
    size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    return image.resize(size, Image.Resampling.LANCZOS)


def describe_image(
    documents: Sequence[Document],
    document_paths: Mapping[str, Path],
    vlm_client: VLMClient,
    file_name: str,
    page: int,
    question: str,
) -> ToolResult:

    document = next((item for item in documents if item.file_name == file_name), None)
    if document is None:
        available = ", ".join(sorted(item.file_name for item in documents)) or "(none)"
        return ToolResult(content=f"Error: no document named '{file_name}'. Available: {available}")

    source_path = document_paths.get(file_name)
    if source_path is None:
        return ToolResult(content=f"Error: original file for '{file_name}' is not available.")

    figure_blocks = [
        block
        for block in document.blocks
        if block.page_number == page and block.block_type in (BlockType.FIGURE, BlockType.IMAGE)
    ]
    if not figure_blocks:
        return ToolResult(content=f"No images found on page {page} of '{file_name}'.")

    labelled = len(figure_blocks) > 1
    prompt = DESCRIBE_IMAGE_PROMPT.format(question=question)
    sections: list[str] = []
    sources: list[SearchResult] = []
    for index, block in enumerate(figure_blocks, start=1):
        header = f"Image {index} on page {page}: " if labelled else ""
        image = _extract_block_image(document, source_path, block)
        if image is None:
            sections.append(f"{header}Error: could not extract this image.")
            continue
        image = _limit_image_pixels(image)

        description = vlm_client.describe_image(image, prompt)
        if not description:
            sections.append(f"{header}(no description returned)")
            continue

        sections.append(f"{header}{description}")
        sources.append(
            SearchResult(
                chunk=Chunk(
                    id=f"describe_image:{file_name}:{page}:{index}",
                    text=description,
                    metadata={
                        "source": "describe_image",
                        "source_file": file_name,
                        "page_number": page,
                        "dataset_doc_id": block.metadata.get("dataset_doc_id", file_name),
                    },
                ),
                score=1.0,
            )
        )

    return ToolResult(content="\n\n".join(sections), sources=sources)


def _extract_block_image(document: Document, source_path: Path, block: Block):
    if document.file_type == "pdf":
        return (
            extract_image_from_pdf(source_path, block.page_number, block.bbox)
            if block.bbox
            else None
        )
    if document.file_type == "pptx":
        shape_index = block.metadata.get("shape_index")
        if shape_index is None:
            return None
        return extract_image_from_pptx(source_path, block.page_number, shape_index)
    if document.file_type in STANDALONE_IMAGE_FILE_TYPES:
        # The whole file is the image - no page/bbox to render or crop into.
        try:
            return Image.open(source_path)
        except Exception:
            return None
    return None


def run_python(
    document_paths: Mapping[str, Path],
    code: str,
    schema_hint: str = "",
    context: Sequence[SearchResult] | None = None,
    documents: Sequence[Document] | None = None,
) -> ToolResult:
    source_paths = dict(document_paths)
    context_file = _write_context_file(context) if context else None
    if context_file is not None:
        source_paths["_context.json"] = context_file

    text_file_paths = _write_document_text_files(documents or [])
    source_paths.update(text_file_paths)

    tables_file = _write_tables_file(documents or [])
    if tables_file is not None:
        source_paths["_tables.json"] = tables_file

    try:
        result = run_sandboxed_code(source_paths=source_paths, code=code)
    finally:
        if context_file is not None:
            context_file.unlink(missing_ok=True)
        for text_file_path in text_file_paths.values():
            text_file_path.unlink(missing_ok=True)
        if tables_file is not None:
            tables_file.unlink(missing_ok=True)

    dataset_doc_id = _referenced_dataset_doc_id(code, documents or [])

    if result.timed_out:
        return ToolResult(content="Error: execution timed out, simplify/narrow the computation.")
    if result.exit_code != 0:
        error_output = result.stderr or result.stdout

        sources = (
            [
                _tool_evidence(
                    "run_python",
                    f"Executed Python:\n{code}\n\nError:\n{error_output}",
                    dataset_doc_id=dataset_doc_id,
                )
            ]
            if error_output.strip()
            else []
        )
        content = f"Error: code raised an exception:\n{error_output}"

        if schema_hint:
            content += f"\n{schema_hint}"
        return ToolResult(content=content, sources=sources)

    stdout = result.stdout.strip()
    output = stdout or "(no output - use print() to return a result)"
    if result.truncated:
        output += "\n[output truncated]"

    sources = (
        [
            _tool_evidence(
                "run_python",
                f"Executed Python:\n{code}\n\nOutput:\n{stdout}",
                dataset_doc_id=dataset_doc_id,
            )
        ]
        if stdout
        else []
    )
    return ToolResult(content=output, sources=sources)


def _write_context_file(context: Sequence[SearchResult]) -> Path:

    seen: set[str] = set()
    entries: list[dict[str, Any]] = []
    for result in context:
        metadata = dict(result.chunk.metadata)
        passage = metadata.pop("context", None) or result.chunk.text
        if passage in seen:
            continue
        seen.add(passage)
        entries.append({"text": passage, "metadata": metadata})

    descriptor, raw_path = tempfile.mkstemp(prefix="file-agent-context-", suffix=".json")
    os.close(descriptor)
    path = Path(raw_path)
    path.write_text(
        json.dumps({"previous_tool_results": entries}, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    return path


def _full_document_text(document: Document) -> str:

    blocks_by_page: dict[int, list[Block]] = {}
    for block in document.blocks:
        if block.page_number is not None:
            blocks_by_page.setdefault(block.page_number, []).append(block)

    sections = []
    for page_number in sorted(blocks_by_page):
        text = "\n\n".join(
            block.text.strip() for block in blocks_by_page[page_number] if block.text.strip()
        )
        if text:
            sections.append(f"=== page {page_number} ===\n{text}")
    return "\n\n".join(sections)


def _write_document_text_files(documents: Sequence[Document]) -> dict[str, Path]:

    paths: dict[str, Path] = {}
    for document in documents:
        text = _full_document_text(document)
        if not text:
            continue
        descriptor, raw_path = tempfile.mkstemp(prefix="file-agent-doctext-", suffix=".txt")
        os.close(descriptor)
        path = Path(raw_path)
        path.write_text(text, encoding="utf-8")
        paths[f"{document.file_name}.txt"] = path
    return paths


def _document_tables(document: Document) -> list[dict[str, Any]]:

    tables = []
    index = 1
    for block in document.blocks:
        if block.block_type == BlockType.TABLE and block.text.strip():
            tables.append(
                {
                    "file_name": document.file_name,
                    "page_number": block.page_number,
                    "table_index": index,
                    "markdown": block.text.strip(),
                }
            )
            index += 1
    return tables


def _write_tables_file(documents: Sequence[Document]) -> Path | None:

    tables = [table for document in documents for table in _document_tables(document)]
    if not tables:
        return None
    descriptor, raw_path = tempfile.mkstemp(prefix="file-agent-tables-", suffix=".json")
    os.close(descriptor)
    path = Path(raw_path)
    path.write_text(json.dumps({"tables": tables}, ensure_ascii=False), encoding="utf-8")
    return path


def _tool_evidence(tool_name: str, text: str, dataset_doc_id: str | None = None) -> SearchResult:

    return SearchResult(
        chunk=Chunk(
            id=f"{tool_name}:{uuid.uuid4().hex[:12]}",
            text=text,
            metadata={"source": tool_name, "dataset_doc_id": dataset_doc_id or tool_name},
        ),
        score=1.0,
    )


def _referenced_dataset_doc_id(code: str, documents: Sequence[Document]) -> str | None:

    referenced = [document for document in documents if f"/data/{document.file_name}" in code]
    if len(referenced) != 1:
        return None
    document = referenced[0]
    return next(
        (
            block.metadata["dataset_doc_id"]
            for block in document.blocks
            if block.metadata.get("dataset_doc_id")
        ),
        document.file_name,
    )


def build_default_tools(
    retriever: Retriever,
    document_paths: Mapping[str, Path] | None = None,
    documents: Sequence[Document] | None = None,
    default_top_k: int = 5,
) -> list[Tool]:
    resolved_document_paths = document_paths or {}
    resolved_documents = documents or []
    available_files = ", ".join(sorted(resolved_document_paths)) or "(no files uploaded)"
    schema = _spreadsheet_schema(resolved_documents)
    schema_hint = f"\nSpreadsheet layout - use these exact names:\n{schema}\n" if schema else ""

    accumulated_context: list[SearchResult] = []

    def _tracked(result: ToolResult) -> ToolResult:
        accumulated_context.extend(result.sources)
        return result

    tools = [
        Tool(
            name="search_documents",
            description=(
                "Semantic + full-text search over the indexed document chunks. "
                "Use for factual lookups, definitions, or any text-based question. "
                "Pass source_file (exact name from list_documents) when the "
                "question names a specific document, to search only that one "
                "instead of letting chunks from other documents compete for "
                "the top_k slots."
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
                    "source_file": {
                        "type": "string",
                        "description": "Optional: exact file name from list_documents to "
                        "restrict the search to one document.",
                    },
                },
                "required": ["query"],
            },
            handler=lambda query, top_k=default_top_k, source_file=None: _tracked(
                search_documents(retriever, query, top_k, source_file, resolved_documents)
            ),
        ),
        Tool(
            name="list_documents",
            description=(
                "List the uploaded documents and their structure (file type, page/"
                "slide/sheet counts, heading count). Call this first when unsure "
                "which documents or sheets are available, before a targeted search "
                "or run_python. Also reports pages_with_undescribed_images per "
                "document - pages/slides holding a figure whose content never "
                "appears in search_documents or read_page output at all; if the "
                "question could depend on one of those, call describe_image on it "
                "directly instead of assuming a page with no figure-related hits "
                "has nothing to show."
            ),
            parameters={"type": "object", "properties": {}, "required": []},
            handler=lambda: _tracked(list_documents(documents or [])),
        ),
        Tool(
            name="read_page",
            description=(
                "Read one or more full pages of paginated documents (PDF, DOCX) "
                "by file name and page number. Use after search_documents when a "
                "hit looks relevant but incomplete - the search result shows its "
                "page_number, and this returns everything on that page, "
                "including the parts the search missed. For a question that "
                "needs pages from two different documents (or several pages of "
                "the same one), request them all in one call instead of one "
                "call per page - it's cheaper and leaves more of the iteration "
                "budget for actually answering. A very long page is split "
                "across calls rather than silently cut off: the response says "
                "'[more available - call again with offset=N]' when there's "
                "more - repeat the same file_name/page with that offset to "
                "continue reading it, don't assume the page ends there. If the "
                "page has a figure with nothing in the text describing it, the "
                "response says so and names the describe_image call to make - "
                "use it when the figure might matter for the question."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "pages": {
                        "type": "array",
                        "description": "Pages to read, possibly from different documents.",
                        "minItems": 1,
                        "items": {
                            "type": "object",
                            "properties": {
                                "file_name": {
                                    "type": "string",
                                    "description": "Exact file name as shown by list_documents.",
                                },
                                "page": {"type": "integer", "minimum": 1},
                                "offset": {
                                    "type": "integer",
                                    "minimum": 0,
                                    "description": "Character offset to resume from - the "
                                    "next_offset value from a previous truncated read of "
                                    "this same page, or omit for the start of the page.",
                                },
                            },
                            "required": ["file_name", "page"],
                        },
                    },
                },
                "required": ["pages"],
            },
            handler=lambda pages: _tracked(read_page(documents or [], pages)),
        ),
        Tool(
            name="run_python",
            description=(
                "Execute Python (stdlib, plus pandas, numpy, openpyxl, scipy, "
                "python-docx) in a sandboxed, network-disabled environment. Three "
                "uses, all equally common - do not treat this as an Excel-only "
                "tool: (1) exact counting or extraction over a document's full "
                "text (any file type, including PDF) that search/reading by eye "
                "can't do reliably - how many times a term appears, how many "
                "entries a list has (e.g. references cited), regex-style pattern "
                "extraction - the moment a question asks 'how many' of something "
                "in running text, come here instead of counting by eye; (2) "
                "structured-data questions a text search can't answer - "
                "arithmetic, aggregation, filtering, pivoting, statistics, dates "
                "over an Excel sheet or a DOCX table; (3) general coding/algorithm "
                "tasks - write and run a short program to work out or verify a "
                "result instead of reasoning it by hand (sorting, searching, "
                "simulation, combinatorics, etc.), even when it has nothing to do "
                "with the uploaded files. Never "
                "invent answers here - only report what the code actually computed. "
                "Print the raw computed value (a number, a row, a short list) "
                "rather than a finished sentence - state the finding, not prose. "
                "Use python-docx directly for DOCX, e.g. from docx import Document; "
                "doc = Document('/data/report.docx') - doc.tables[i].rows[j].cells "
                "for a table, doc.paragraphs for the running text; pandas has no "
                "reader for the format. For exact counting/extraction (use (2) "
                "above), every document's full extracted text - including PDF and "
                "PPTX, which nothing here can otherwise read - is available "
                "read-only as plain text at /data/<file name>.txt, one "
                "'=== page N ===' section per page, e.g. "
                "open('/data/report.pdf.txt').read(); this is not for semantic "
                "lookup of a fact or passage, use search_documents for that instead. "
                "Every table (any file type, already-parsed Markdown, no manual "
                "table-scraping needed) is also available, if any exist, as JSON at "
                '/data/_tables.json - a single object shaped {"tables": '
                "[{file_name, page_number, table_index, markdown}, ...]}. table_index "
                "is just this tool's own numbering in document order, not necessarily "
                'the paper\'s own "Table N" caption - match a specific table by its '
                "markdown content (column headers, page_number), not by assuming the "
                "number lines up. Parse a table's markdown with pandas, e.g. "
                "pd.read_csv(io.StringIO(markdown), sep='|', skipinitialspace=True). "
                "All uploaded files are available read-only under /data/<file name>: "
                f"{available_files}.{schema_hint} "
                "If earlier search_documents/read_page/list_documents calls in this "
                "conversation found relevant passages, they're also available (if any) "
                "read-only as JSON at /data/_context.json - a single object shaped "
                '{"previous_tool_results": [{text, metadata}, ...]} (not a bare list) '
                "- load it to reuse what was already found instead of re-deriving it. "
                "/tmp and /sandbox/work are writable if you need scratch files. "
                "Print the final result with print(); only stdout is returned to you."
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
            handler=lambda code: run_python(
                resolved_document_paths,
                code,
                schema_hint=schema_hint,
                context=accumulated_context,
                documents=resolved_documents,
            ),
        ),
    ]

    vlm_client = create_vlm_client()
    if vlm_client is not None:
        tools.append(
            Tool(
                name="describe_image",
                description=(
                    "Describe an image, chart, diagram, or photo on a specific page "
                    "or slide using a vision model, for figures the text extraction "
                    "can't read. Use after list_documents/search_documents/read_page "
                    "point to a page with a figure but the text alone doesn't answer "
                    "the question. Pass the specific question you need answered "
                    "about the image - a small vision model reports what's relevant "
                    "far better when told what to look for than with a generic "
                    "'describe this'. Only describes what's actually visible in the "
                    "image - never guesses or invents its subject."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "file_name": {
                            "type": "string",
                            "description": "Exact file name as shown by list_documents.",
                        },
                        "page": {"type": "integer", "minimum": 1},
                        "question": {
                            "type": "string",
                            "description": "What you need to know from this image.",
                        },
                    },
                    "required": ["file_name", "page", "question"],
                },
                handler=lambda file_name, page, question: _tracked(
                    describe_image(
                        resolved_documents,
                        resolved_document_paths,
                        vlm_client,
                        file_name,
                        page,
                        question,
                    )
                ),
            )
        )

    return tools
