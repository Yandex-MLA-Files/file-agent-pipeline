"""OCR of scanned PDF pages with a vision-language model.

Classic OCR engines (EasyOCR/RapidOCR) read characters; a multimodal LLM reads
the *page*: it copes with skew, low contrast and mixed Cyrillic/Latin, keeps
headings, lists and tables, rejoins words broken by hyphenation and writes
formulas as LaTeX. Since the answering model served for the project (Qwen3.5)
is multimodal, the same endpoint can do this with no extra deployment.

Measured on the project's own documents (pages with a text layer rendered to
images, then degraded; the text layer is the ground truth) the difference is
large — character error rate, VLM vs EasyOCR:

===============  ==============  ==============  ==============
page             clean render    scanner-like    photo-like
===============  ==============  ==============  ==============
Russian prose    0.08  / 0.54     0.07 / 0.54     0.08 / 0.69
financial table  0.06  / 0.32     0.06 / 0.39     0.06 / 0.56
medical prose    0.00  / 0.10     0.01 / 0.11     0.01 / 0.37
===============  ==============  ==============  ==============

The VLM is also *stable* under degradation (skew, JPEG artefacts, uneven
light) where the classic engine collapses, and it is the only one of the two
that preserves table structure. Its cost is latency: 10-45 s per page against
1-3 s, which is why pages are transcribed concurrently here.

Robustness is the other half of the story: a generative model can also fail in
ways an OCR engine cannot — loop on a repeated line, stop early on a dense
page, or return nothing for a page full of text. Every transcript is therefore
validated (:func:`validate_transcript`); a page that fails validation is
retried with a larger budget or handed to the local engine, so the result is
never worse than classic OCR.

Only the pages that :mod:`file_agent.parsers.routing` flags as lacking a text
layer are transcribed; born-digital pages keep Docling's exact text.
"""

import logging
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import fitz  # PyMuPDF
from PIL import Image

from file_agent.document import Block, BlockType
from file_agent.parsers.local_ocr import LocalPageOCR, render_page
from file_agent.parsers.md_parser import parse_markdown_blocks
from file_agent.telemetry import tracer
from file_agent.vlm.base import VLMClient

logger = logging.getLogger(__name__)

PAGE_RENDER_DPI = 150
NO_TEXT_MARKER = "[no text]"
# A dense A4 page is ~1000-1500 tokens of Markdown; leave headroom for tables.
DEFAULT_PAGE_MAX_TOKENS = 2500
# Pages are independent requests, so they are sent together: vLLM batches them
# and a 30-page scan finishes in the time of a few pages instead of thirty.
DEFAULT_OCR_CONCURRENCY = 4
# Scanned pages are the one case where extra resolution pays for itself, so the
# OCR path overrides the client's default downscale.
DEFAULT_OCR_IMAGE_SIDE = 1800

PAGE_TRANSCRIPTION_PROMPT = (
    "Transcribe this scanned document page into Markdown.\n"
    "Rules:\n"
    "- Reproduce the text exactly as written, in its original language. Do not "
    "translate, summarize, correct or add anything.\n"
    "- Keep the reading order. Use '#'/'##' for headings, '-' for bullet items, "
    "Markdown tables for tables, and LaTeX ($...$) for formulas.\n"
    "- Join words that are split by a hyphen at a line break.\n"
    "- Skip page numbers, running headers/footers, watermarks and stamps.\n"
    "- If the page contains a chart, diagram or photo, transcribe every label "
    "and value in it and add one line starting with 'Рисунок:' (or 'Figure:' "
    "for an English page) describing what it shows.\n"
    "- If the page contains no readable text and no figure (blank page), "
    f"output exactly {NO_TEXT_MARKER}.\n"
    "Output only the Markdown, without explanations or code fences."
)


@dataclass
class TranscriptCheck:
    """Outcome of validating one page transcript."""

    ok: bool
    reason: str = ""
    #: True when a larger token budget is likely to fix the problem.
    retryable: bool = False


class VLMPageOCR:
    """Transcribes selected PDF pages with a VLM and returns typed blocks.

    :param fallback: local engine used for pages whose transcript fails
        validation. ``None`` disables the fallback (the VLM text is kept).
    """

    def __init__(
        self,
        vlm_client: VLMClient,
        dpi: int | None = None,
        max_tokens: int | None = None,
        concurrency: int | None = None,
        fallback: LocalPageOCR | None = None,
    ) -> None:
        self.vlm_client = vlm_client
        self.dpi = dpi or int(os.getenv("OCR_DPI", PAGE_RENDER_DPI))
        self.max_tokens = max_tokens or int(
            os.getenv("VLM_OCR_MAX_TOKENS", DEFAULT_PAGE_MAX_TOKENS)
        )
        self.concurrency = max(
            1, concurrency or int(os.getenv("VLM_OCR_CONCURRENCY", DEFAULT_OCR_CONCURRENCY))
        )
        self.fallback = fallback
        self.stats: dict[str, int] = {}
        self._render_lock = threading.Lock()

    # -- public API ---------------------------------------------------------

    def transcribe(self, pdf_path: Path, page_numbers: list[int]) -> dict[int, list[Block]]:
        results: dict[int, list[Block]] = {}
        if not page_numbers:
            return results

        path = Path(pdf_path)
        with tracer.start_as_current_span("file_agent.vlm_ocr") as span:
            span.set_attribute("file_agent.file_name", path.name)
            span.set_attribute("file_agent.page_count", len(page_numbers))

            with fitz.open(str(path)) as pdf:
                pages = [number for number in page_numbers if 1 <= number <= len(pdf)]
                if not pages:
                    return results

                # The first page runs alone: if the endpoint is down or the model
                # is text-only, the whole file falls back to the classic engine
                # instead of paying one timeout per page.
                first, rest = pages[0], pages[1:]
                results[first] = self._page_blocks(pdf, path, first)

                if rest:
                    workers = min(self.concurrency, len(rest))
                    with ThreadPoolExecutor(max_workers=workers) as pool:
                        for page_number, blocks in zip(
                            rest,
                            pool.map(lambda n: self._safe_page_blocks(pdf, path, n), rest),
                            strict=True,
                        ):
                            results[page_number] = blocks

            results = {number: blocks for number, blocks in results.items() if blocks}
            span.set_attribute("file_agent.transcribed_pages", len(results))
            if self.stats:
                logger.info("VLM OCR of %s: %s", path.name, dict(sorted(self.stats.items())))
        return results

    # -- per page -----------------------------------------------------------

    def _safe_page_blocks(self, pdf, path: Path, page_number: int) -> list[Block]:
        """Never let one failed page abort the rest: fall back or return nothing."""
        try:
            return self._page_blocks(pdf, path, page_number)
        except Exception:
            logger.warning(
                "VLM OCR failed on page %s of %s; using the local engine for it.",
                page_number,
                path.name,
                exc_info=True,
            )
            self._count("vlm_error")
            return self._fallback_blocks(self._render(pdf, page_number), path, page_number)

    def _page_blocks(self, pdf, path: Path, page_number: int) -> list[Block]:
        image = self._render(pdf, page_number)

        ink = ink_ratio(image)
        if ink < BLANK_INK_RATIO:
            # A blank or near-blank page: no engine will find text on it, and a
            # generative model is exactly where an empty page invites invention.
            self._count("blank_skipped")
            return []

        markdown, finish_reason = self._describe(image, self.max_tokens)
        check = validate_transcript(markdown, ink=ink, image=image)

        if not check.ok and check.retryable:
            self._count(f"retry_{check.reason}")
            markdown, finish_reason = self._describe(image, self.max_tokens * 2)
            check = validate_transcript(markdown, ink=ink, image=image)
        elif finish_reason == "length":
            self._count("retry_truncated")
            markdown, finish_reason = self._describe(image, self.max_tokens * 2)
            check = validate_transcript(markdown, ink=ink, image=image)

        if not check.ok:
            logger.info(
                "VLM transcript of page %s of %s rejected (%s); using the local engine.",
                page_number,
                path.name,
                check.reason,
            )
            self._count(f"fallback_{check.reason}")
            fallback_blocks = self._fallback_blocks(image, path, page_number)
            if fallback_blocks:
                return fallback_blocks
            # No local engine, or it found nothing either: a suspect transcript
            # is still better than dropping the page.
            self._count("kept_suspect_transcript")
        else:
            self._count("vlm_ok")
        return self._to_blocks(markdown, path.name, page_number)

    def _describe(self, image: Image.Image, max_tokens: int) -> tuple[str, str | None]:
        verbose = getattr(self.vlm_client, "describe_image_verbose", None)
        if callable(verbose):
            return verbose(
                image,
                PAGE_TRANSCRIPTION_PROMPT,
                max_tokens=max_tokens,
                max_image_side=DEFAULT_OCR_IMAGE_SIDE,
            )
        return self.vlm_client.describe_image(
            image, PAGE_TRANSCRIPTION_PROMPT, max_tokens=max_tokens
        ), None

    def _render(self, pdf, page_number: int) -> Image.Image:
        # PyMuPDF documents are not thread-safe; rendering is ~50 ms, so a lock
        # costs nothing next to the network call it feeds.
        with self._render_lock:
            return render_page(pdf[page_number - 1], self.dpi)

    def _fallback_blocks(self, image: Image.Image, path: Path, page_number: int) -> list[Block]:
        if self.fallback is None:
            return []
        try:
            text = self.fallback.transcribe_image(image)
        except Exception:
            logger.warning("Local OCR fallback failed on page %s.", page_number, exc_info=True)
            return []
        return self.fallback.text_to_blocks(text, path.name, page_number)

    def _count(self, key: str) -> None:
        self.stats[key] = self.stats.get(key, 0) + 1

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


# -- transcript validation ---------------------------------------------------

# Below this share of dark pixels a page carries no content worth transcribing.
# Calibrated deliberately low against real renders: one short line of 11 pt text
# on A4 at 150 dpi already measures ~1.8e-4, a dense page ~5e-2, and a blank
# page 0. Skipping is only meant to spare the model pages with nothing on them.
BLANK_INK_RATIO = 0.00005
# Ink coverage above which a page is considered to carry text (not just a logo).
TEXT_INK_RATIO = 0.015
# Expected content is estimated from the number of *text lines* on the page,
# not from raw ink: a slide with one chart covers a third of the page in ink
# while carrying three lines of text, and an ink-based estimate flagged almost
# every such page as truncated. A conservative characters-per-line figure and a
# low ratio keep this check aimed at gross omissions only.
CHARS_PER_TEXT_LINE = 30
SHORT_TRANSCRIPT_RATIO = 0.3
MIN_LINES_FOR_LENGTH_CHECK = 8
# A line repeated this many times in a row is a decoding loop, not a document.
MAX_REPEATED_LINES = 4
_CJK = re.compile(r"[぀-ヿ一-鿿]")
# Refusals and meta-answers ("I cannot read this image") are not transcripts.
_REFUSAL = re.compile(
    r"^(i (?:cannot|can't|am unable)|sorry|unable to|as an ai|извин|я не мог|не могу)",
    re.IGNORECASE,
)


def validate_transcript(
    markdown: str, ink: float, image: Image.Image | None = None
) -> TranscriptCheck:
    """Decide whether a VLM page transcript can be trusted.

    Rejects the failure modes a generative transcriber has and a classic OCR
    engine does not: empty output for a page full of ink, decoding loops,
    refusals, foreign-script hallucination and answers cut off mid-page.
    """
    text = _strip_fences(markdown or "").strip()
    stripped = text.lower()

    if not text or stripped.startswith(NO_TEXT_MARKER):
        # "[no text]" is legitimate for a picture-only page, but not for a page
        # covered in ink - there the model simply refused to read it.
        if ink >= TEXT_INK_RATIO:
            return TranscriptCheck(False, "empty_on_inked_page")
        return TranscriptCheck(True)

    if _REFUSAL.match(text):
        return TranscriptCheck(False, "refusal")

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if _max_consecutive_repeats(lines) > MAX_REPEATED_LINES:
        return TranscriptCheck(False, "repetition_loop")

    letters = sum(character.isalpha() for character in text)
    if letters and len(_CJK.findall(text)) / max(letters, 1) > 0.05:
        return TranscriptCheck(False, "foreign_script")

    if image is not None:
        lines = text_line_count(image)
        expected = lines * CHARS_PER_TEXT_LINE
        if lines >= MIN_LINES_FOR_LENGTH_CHECK and len(text) < SHORT_TRANSCRIPT_RATIO * expected:
            # Either the model stopped early (retry with a bigger budget helps)
            # or it summarized instead of transcribing (the local engine wins).
            return TranscriptCheck(False, "short_transcript", retryable=True)

    return TranscriptCheck(True)


def ink_ratio(image: Image.Image) -> float:
    """Share of dark pixels — a cheap, model-free measure of page content.

    Sampled on a stride rather than a resize: downscaling blends thin glyphs
    into the background and would report a page of small print as blank.
    """
    try:
        import numpy as np

        grayscale = np.asarray(image.convert("L"))[::4, ::4]
        return float((grayscale < 160).mean())
    except Exception:  # pragma: no cover - numpy always available in practice
        return 1.0


def text_line_count(image: Image.Image) -> int:
    """Number of text lines on the page, from the horizontal ink profile.

    Rows of a rendered page that are partly inked are text; rows that are
    almost fully inked belong to a figure, a photo or a filled table header and
    are ignored. Counting the groups of consecutive text rows gives a estimate
    of how much writing the page holds that a chart cannot inflate.
    """
    try:
        import numpy as np

        dark = np.asarray(image.convert("L").resize((256, 360))) < 160
        row_ink = dark.mean(axis=1)
        is_text_row = (row_ink > 0.02) & (row_ink < 0.6)
        # Count rising edges: each group of consecutive text rows is one line.
        padded = np.concatenate(([False], is_text_row))
        return int(np.sum(padded[1:] & ~padded[:-1]))
    except Exception:  # pragma: no cover - numpy always available in practice
        return 0


def _max_consecutive_repeats(lines: list[str]) -> int:
    best = current = 0
    previous: str | None = None
    for line in lines:
        if line == previous and len(line) > 3:
            current += 1
            best = max(best, current)
        else:
            current = 1
            previous = line
    return best


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
