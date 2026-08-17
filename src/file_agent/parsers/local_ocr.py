"""Classic, fully offline OCR of PDF pages (EasyOCR).

This is the counterpart of :mod:`file_agent.parsers.vlm_ocr`: it reads
*characters* with a local detector/recognizer instead of asking a multimodal
model to read the *page*. It needs no endpoint and no tokens and is an order of
magnitude faster per page, but it loses the layout (headings, lists, table
grids) and degrades sharply on skewed or noisy scans.

The pipeline uses it in two roles:

* as the per-page **fallback** when a VLM transcript looks broken (empty output
  on a page full of ink, a repetition loop, a truncated answer);
* as the **only** engine when no VLM endpoint is configured or
  ``OCR_ENGINE=easyocr`` is set explicitly.

The reader is expensive to construct (model download plus GPU/CPU init), so it
is built once per language set and shared process-wide.
"""

import io
import logging
import os
import threading
from functools import lru_cache
from pathlib import Path

import fitz  # PyMuPDF
from PIL import Image

from file_agent.document import Block, BlockType
from file_agent.telemetry import tracer

logger = logging.getLogger(__name__)

DEFAULT_LANGUAGES = ("ru", "en")
# EasyOCR reads best around 150-200 dpi; higher resolutions mostly cost time.
DEFAULT_DPI = 150


@lru_cache(maxsize=4)
def _load_reader(languages: tuple[str, ...], gpu: bool):
    import easyocr  # imported lazily: the package pulls in torch

    logger.info("Loading EasyOCR reader (languages=%s, gpu=%s)", ",".join(languages), gpu)
    return easyocr.Reader(list(languages), gpu=gpu, verbose=False)


def _gpu_available() -> bool:
    override = (os.getenv("OCR_USE_GPU") or "").strip().lower()
    if override in {"1", "true", "yes", "on"}:
        return True
    if override in {"0", "false", "no", "off"}:
        return False
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:  # pragma: no cover - torch always present in practice
        return False


def ocr_languages() -> tuple[str, ...]:
    raw = os.getenv("OCR_LANGS", ",".join(DEFAULT_LANGUAGES))
    langs = tuple(code.strip() for code in raw.split(",") if code.strip())
    return langs or DEFAULT_LANGUAGES


class LocalPageOCR:
    """Transcribes PDF pages with EasyOCR into plain-text blocks."""

    def __init__(
        self,
        languages: tuple[str, ...] | None = None,
        dpi: int | None = None,
        gpu: bool | None = None,
    ) -> None:
        self.languages = languages or ocr_languages()
        self.dpi = dpi or int(os.getenv("OCR_DPI", DEFAULT_DPI))
        self._gpu = _gpu_available() if gpu is None else gpu
        self._lock = threading.Lock()

    @property
    def available(self) -> bool:
        try:
            import easyocr  # noqa: F401
        except Exception:
            return False
        return True

    def transcribe_image(self, image: Image.Image) -> str:
        """Return the text of one page image in reading order."""
        import numpy as np

        reader = _load_reader(tuple(self.languages), self._gpu)
        array = np.asarray(image.convert("RGB"))
        # EasyOCR is not thread-safe: the pipeline calls this from the worker
        # threads that also drive the VLM, so serialise the recognizer.
        with self._lock:
            detections = reader.readtext(array, detail=1, paragraph=True)
        return _detections_to_text(detections)

    def transcribe(self, pdf_path: Path, page_numbers: list[int]) -> dict[int, list[Block]]:
        results: dict[int, list[Block]] = {}
        if not page_numbers:
            return results
        with tracer.start_as_current_span("file_agent.local_ocr") as span:
            span.set_attribute("file_agent.file_name", Path(pdf_path).name)
            span.set_attribute("file_agent.page_count", len(page_numbers))
            with fitz.open(str(pdf_path)) as pdf:
                for page_number in page_numbers:
                    if page_number < 1 or page_number > len(pdf):
                        continue
                    image = render_page(pdf[page_number - 1], self.dpi)
                    text = self.transcribe_image(image)
                    blocks = self.text_to_blocks(text, Path(pdf_path).name, page_number)
                    if blocks:
                        results[page_number] = blocks
        return results

    @staticmethod
    def text_to_blocks(text: str, source_file: str, page_number: int) -> list[Block]:
        """One block per paragraph; classic OCR carries no reliable structure."""
        blocks: list[Block] = []
        for index, paragraph in enumerate(p.strip() for p in (text or "").split("\n\n")):
            if not paragraph:
                continue
            blocks.append(
                Block(
                    id=f"ocr-p{page_number}-{index}",
                    text=paragraph,
                    type=BlockType.TEXT.value,
                    metadata={
                        "source_file": source_file,
                        "ocr_engine": "easyocr",
                        "ocr_page": True,
                    },
                    block_type=BlockType.TEXT,
                    page_number=page_number,
                )
            )
        return blocks


def _detections_to_text(detections) -> str:
    """Order EasyOCR paragraph detections top-to-bottom, left-to-right."""
    entries = []
    for detection in detections:
        if len(detection) < 2:
            continue
        box, text = detection[0], detection[1]
        if not str(text).strip():
            continue
        try:
            ys = [point[1] for point in box]
            xs = [point[0] for point in box]
            top, left = min(ys), min(xs)
        except (TypeError, IndexError):  # pragma: no cover - unexpected shape
            top, left = 0.0, 0.0
        entries.append((float(top), float(left), str(text).strip()))
    entries.sort(key=lambda item: (round(item[0] / 20.0), item[1]))
    return "\n".join(text for _, _, text in entries)


def render_page(page: fitz.Page, dpi: int) -> Image.Image:
    zoom = dpi / 72.0
    pixmap = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
    return Image.open(io.BytesIO(pixmap.tobytes("png"))).convert("RGB")
