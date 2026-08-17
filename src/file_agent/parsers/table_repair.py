"""Second opinion on PDF tables the layout model failed to reconstruct.

TableFormer recovers the grid of most tables well — an audit of the project's
corpus found 65 tables of which 56 were sound — but it fails in recognisable
ways: it returns a single column when the vertical rules are missing, or a
header with no rows when the body is a bitmap. Such a table is worse than
useless in retrieval: it looks like data and carries none.

Those tables (and only those) are re-read from the page image by the
multimodal model, which sees the ruling lines and the alignment a text-based
extractor cannot. The repair is accepted only when it comes back with more
structure than the original, so a failed second opinion can never make a table
worse. Because the trigger is a *defect*, the cost is proportional to how badly
extraction went, not to the size of the document: on this corpus it fires on
three tables.

Controlled by ``PDF_TABLE_VLM``: ``auto`` (default, repair degenerate tables),
``off``, or ``always`` (re-read every table — for auditing an extraction).
"""

import logging
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from file_agent.document import Block, BlockType, Document
from file_agent.telemetry import tracer
from file_agent.utils.image_extractor import extract_image_from_pdf
from file_agent.vlm.base import VLMClient

logger = logging.getLogger(__name__)

DEFAULT_MODE = "auto"
# Never spend more than this many requests per document on table repair.
DEFAULT_MAX_TABLES = 6
DEFAULT_MAX_TOKENS = 1500
# A table region needs to be big enough to be readable when cropped.
MIN_TABLE_AREA = 8000.0
# Share of empty cells above which a grid is considered broken.
MAX_EMPTY_CELL_SHARE = 0.6

TABLE_PROMPT = (
    "This image is a table from a document. Reproduce it as a Markdown table.\n"
    "Rules:\n"
    "- Keep every row and column, including the row labels in the first column "
    "and any units or footnote markers.\n"
    "- Copy the values exactly, in the original language; do not compute, "
    "reorder, translate or summarize anything.\n"
    "- Merge the header cells that span several columns into the column names.\n"
    "- If the image is not a table, output exactly [not a table].\n"
    "Output only the Markdown table."
)

NOT_A_TABLE = "[not a table]"


def resolve_mode() -> str:
    mode = (os.getenv("PDF_TABLE_VLM") or DEFAULT_MODE).strip().lower()
    if mode not in ("auto", "off", "always"):
        raise ValueError(f"PDF_TABLE_VLM must be 'auto', 'off' or 'always', got {mode!r}")
    return mode


def repair_tables(
    document: Document,
    file_path: Path,
    vlm_client: VLMClient,
    max_tables: int | None = None,
) -> int:
    """Re-read broken table blocks from the page image; return how many changed."""
    mode = resolve_mode()
    if mode == "off":
        return 0

    limit = max_tables or int(os.getenv("PDF_TABLE_VLM_MAX", DEFAULT_MAX_TABLES))
    candidates = [
        block
        for block in document.blocks
        if block.block_type == BlockType.TABLE
        and block.page_number is not None
        and block.bbox is not None
        and _bbox_area(block.bbox) >= MIN_TABLE_AREA
        and (mode == "always" or is_degenerate(block.text))
    ][:limit]
    if not candidates:
        return 0

    with tracer.start_as_current_span("file_agent.repair_tables") as span:
        span.set_attribute("file_agent.file_name", Path(file_path).name)
        span.set_attribute("file_agent.table_candidates", len(candidates))
        logger.info(
            "Re-reading %s degenerate table(s) of %s with the VLM",
            len(candidates),
            Path(file_path).name,
        )
        with ThreadPoolExecutor(max_workers=min(4, len(candidates))) as pool:
            results = list(pool.map(lambda b: _repair_one(b, file_path, vlm_client), candidates))
        repaired = sum(1 for changed in results if changed)
        span.set_attribute("file_agent.tables_repaired", repaired)

    if repaired:
        document.metadata["vlm_repaired_tables"] = repaired
    return repaired


def _repair_one(block: Block, file_path: Path, vlm_client: VLMClient) -> bool:
    try:
        image = extract_image_from_pdf(file_path, block.page_number, block.bbox)
        if image is None:
            return False
        markdown = (
            vlm_client.describe_image(image, TABLE_PROMPT, max_tokens=DEFAULT_MAX_TOKENS) or ""
        ).strip()
    except Exception as exc:  # noqa: BLE001 - a failed repair keeps the original
        logger.warning("Table repair failed on page %s: %s", block.page_number, exc)
        return False

    if not markdown or markdown.lower().startswith(NOT_A_TABLE):
        return False
    markdown = _strip_fences(markdown)
    if not is_improvement(block.text, markdown):
        return False

    caption = str(block.metadata.get("caption") or "")
    block.text = f"{caption}\n{markdown}".strip() if caption else markdown
    block.metadata["table_source"] = "vlm"
    return True


def is_degenerate(text: str) -> bool:
    """Whether an extracted table carries no usable grid."""
    rows = _data_rows(text)
    if len(rows) < 2:
        return True
    columns = max(len(row) for row in rows)
    if columns < 2:
        return True
    # The header is always filled; counting it would hide a body of empty cells.
    body = rows[1:] if _has_header(text) else rows
    cells = [cell for row in body for cell in row]
    empty = sum(1 for cell in cells if not cell)
    return bool(cells) and empty / len(cells) > MAX_EMPTY_CELL_SHARE


def _has_header(text: str) -> bool:
    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    return len(lines) >= 2 and "|" in lines[0] and set(lines[1]) <= set("|-: ")


def is_improvement(original: str, candidate: str) -> bool:
    """Accept the re-read table only when it has more structure than the original."""
    if is_degenerate(candidate):
        return False
    original_rows, candidate_rows = _data_rows(original), _data_rows(candidate)
    original_columns = max((len(row) for row in original_rows), default=0)
    candidate_columns = max((len(row) for row in candidate_rows), default=0)
    return (candidate_columns, len(candidate_rows)) > (original_columns, len(original_rows))


def _data_rows(text: str) -> list[list[str]]:
    rows: list[list[str]] = []
    for line in (text or "").splitlines():
        stripped = line.strip()
        if "|" not in stripped or set(stripped) <= set("|-: "):
            continue
        rows.append([cell.strip() for cell in stripped.strip("|").split("|")])
    return rows


def _strip_fences(text: str) -> str:
    lines = text.strip().splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _bbox_area(bbox) -> float:
    x0, y0, x1, y1 = bbox
    return abs((x1 - x0) * (y1 - y0))
