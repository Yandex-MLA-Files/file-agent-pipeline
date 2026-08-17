"""OCR of scanned PDF pages with a vision-language model.

Classic OCR engines (EasyOCR/RapidOCR) read characters; a multimodal LLM reads
the *page*: it copes with skew, low contrast and mixed Cyrillic/Latin, keeps
headings, lists and tables, and writes formulas as LaTeX. Since the answering
model served for the project (Qwen3.5) is multimodal, the same endpoint can do
this with no extra deployment.

Only the pages that :mod:`file_agent.parsers.routing` flags as lacking a text
layer are transcribed; born-digital pages keep Docling's exact text. Each page
is rendered once, sent with a strict "transcribe, do not interpret" prompt and
the returned Markdown is parsed into typed blocks with the page number attached.
"""

import io
import logging
from pathlib import Path

import fitz  # PyMuPDF
from PIL import Image

from file_agent.document import Block, BlockType
from file_agent.parsers.md_parser import parse_markdown_blocks
from file_agent.telemetry import tracer
from file_agent.vlm.base import VLMClient

logger = logging.getLogger(__name__)

PAGE_RENDER_DPI = 150
NO_TEXT_MARKER = "[no text]"

PAGE_TRANSCRIPTION_PROMPT = (
    "Transcribe this scanned document page into Markdown.\n"
    "Rules:\n"
    "- Reproduce the text exactly as written, in its original language. Do not "
    "translate, summarize, correct or add anything.\n"
    "- Keep the reading order. Use '#'/'##' for headings, '-' for bullet items, "
    "Markdown tables for tables, and LaTeX ($...$) for formulas.\n"
    "- Skip page numbers, running headers/footers, watermarks and stamps.\n"
    "- If the page contains no readable text (blank page or picture only), "
    f"output exactly {NO_TEXT_MARKER}.\n"
    "Output only the Markdown, without explanations or code fences."
)


class VLMPageOCR:
    """Transcribes selected PDF pages with a VLM and returns typed blocks."""

    def __init__(self, vlm_client: VLMClient, dpi: int = PAGE_RENDER_DPI) -> None:
        self.vlm_client = vlm_client
        self.dpi = dpi

    def transcribe(self, pdf_path: Path, page_numbers: list[int]) -> dict[int, list[Block]]:
        results: dict[int, list[Block]] = {}
        if not page_numbers:
            return results

        with tracer.start_as_current_span("file_agent.vlm_ocr") as span:
            span.set_attribute("file_agent.file_name", Path(pdf_path).name)
            span.set_attribute("file_agent.page_count", len(page_numbers))

            with fitz.open(str(pdf_path)) as pdf:
                for page_number in page_numbers:
                    if page_number < 1 or page_number > len(pdf):
                        continue
                    image = self._render(pdf[page_number - 1])
                    try:
                        markdown = self.vlm_client.describe_image(image, PAGE_TRANSCRIPTION_PROMPT)
                    except Exception:
                        logger.warning(
                            "VLM OCR failed on page %s of %s; aborting VLM OCR for this file",
                            page_number,
                            Path(pdf_path).name,
                            exc_info=True,
                        )
                        raise
                    results[page_number] = self._to_blocks(
                        markdown, Path(pdf_path).name, page_number
                    )
            span.set_attribute("file_agent.transcribed_pages", len(results))
        return results

    def _render(self, page: fitz.Page) -> Image.Image:
        zoom = self.dpi / 72.0
        pixmap = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
        return Image.open(io.BytesIO(pixmap.tobytes("png"))).convert("RGB")

    @staticmethod
    def _to_blocks(markdown: str, source_file: str, page_number: int) -> list[Block]:
        text = _strip_fences(markdown or "").strip()
        if not text or text.lower().startswith(NO_TEXT_MARKER):
            return []
        blocks = parse_markdown_blocks(
            text, source_file=source_file, page_number=page_number, id_prefix=f"ocr-p{page_number}"
        )
        for block in blocks:
            block.metadata["ocr_engine"] = "vlm"
            block.metadata["ocr_page"] = True
        return [
            block for block in blocks if block.text.strip() or block.block_type != BlockType.TEXT
        ]


def _strip_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines)
    return stripped


def merge_ocr_blocks(blocks: list[Block], transcribed: dict[int, list[Block]]) -> list[Block]:
    """Replace the blocks of transcribed pages with the VLM transcript, in page order.

    Docling emits little for a scanned page (usually one full-page picture);
    those placeholders are dropped and the transcript is inserted where the
    page belongs so the reading order of the whole document stays intact.
    """
    if not transcribed:
        return blocks
    replaced_pages = set(transcribed)
    kept = [block for block in blocks if block.page_number not in replaced_pages]

    result: list[Block] = []
    pending = sorted(transcribed.items())
    for block in kept:
        while pending and block.page_number is not None and pending[0][0] < block.page_number:
            result.extend(pending.pop(0)[1])
        result.append(block)
    for _, page_blocks in pending:
        result.extend(page_blocks)
    return result
