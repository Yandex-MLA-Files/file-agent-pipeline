import io
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

from file_agent.document import Block, BlockType, Document
from file_agent.telemetry import tracer
from file_agent.utils.image_extractor import extract_image_from_pdf
from file_agent.vlm.base import VLMClient

logger = logging.getLogger(__name__)

# Deliberately extraction-oriented and restrictive: small local VLMs happily
# invent a plausible topic for a chart they cannot read, and a hallucinated
# description is worse than none once it is indexed as document content.
FIGURE_PROMPT = (
    "Describe only what is actually visible in this image. State its type "
    "(chart, diagram, screenshot, table, photo) and transcribe the text you can "
    "read: title, axis labels, legend entries, series and node names, and the "
    "values shown. If it is a diagram or flowchart, list the nodes and the "
    "connections between them. Do not guess the subject, and do not invent "
    "numbers, names, dates or context that are not shown. If the image is "
    "decorative or unreadable, say exactly that in one short sentence. "
    "Answer in the language of the text in the image (Russian if the image "
    "contains Russian text)."
)

# Figures smaller than this (in PDF points squared, ~1 pt = 1/72") are almost
# always icons, logos or bullets — not worth a VLM call.
DEFAULT_MIN_FIGURE_AREA = 5000.0
# Embedded images (DOCX/PPTX) have no page geometry; use pixel area instead.
DEFAULT_MIN_IMAGE_PIXELS = 120 * 120
# Upper bound on VLM calls per document, so a 100-figure deck cannot silently
# turn into a 100-request bill. The largest figures are described first.
DEFAULT_MAX_FIGURES = 8
# Figures of one document are described concurrently: vLLM batches the
# requests, so wall-clock time is close to that of a single call.
DEFAULT_VLM_CONCURRENCY = 4


class DocumentEnhancer:
    """Enriches figure/image blocks with a VLM-generated textual description.

    Figures come from two sources: PDF pages (cropped by bounding box) and
    container formats whose parsers attach the embedded image bytes to the
    block (DOCX, PPTX, Docling pictures). To keep the cost predictable, tiny
    decorative images are skipped and at most ``max_figures`` figures per
    document are described (largest first — the big diagram matters more than
    a footer icon).
    """

    def __init__(
        self,
        vlm_client: VLMClient,
        min_figure_area: float | None = None,
        max_figures: int | None = None,
    ) -> None:
        self.vlm_client = vlm_client
        self.min_figure_area = (
            float(os.getenv("VLM_MIN_FIGURE_AREA", DEFAULT_MIN_FIGURE_AREA))
            if min_figure_area is None
            else min_figure_area
        )
        self.max_figures = (
            int(os.getenv("VLM_MAX_FIGURES", DEFAULT_MAX_FIGURES))
            if max_figures is None
            else max_figures
        )

    def enhance(self, doc: Document, file_path: Path) -> Document:
        is_pdf = Path(file_path).suffix.lower() == ".pdf"
        candidates = [block for block in doc.blocks if self._should_describe(block, is_pdf)]
        candidates.sort(key=self._figure_area, reverse=True)
        selected = candidates[: self.max_figures]
        skipped = len(candidates) - len(selected)
        if skipped > 0:
            logger.info(
                "VLM: describing %s largest figures, skipping %s (max_figures=%s)",
                len(selected),
                skipped,
                self.max_figures,
            )

        with tracer.start_as_current_span("file_agent.enhance_document") as span:
            span.set_attribute("file_agent.figure_count", len(selected))
            span.set_attribute("file_agent.figure_candidates", len(candidates))

            described = 0
            if selected:
                # The first figure runs alone: if the endpoint is down or the model
                # is text-only it fails fast and the rest is skipped instead of
                # paying a timeout per figure. The remaining figures run concurrently.
                first, rest = selected[0], selected[1:]
                outcome = self._describe(first, file_path, is_pdf)
                if isinstance(outcome, Exception):
                    self._record_failure(doc, first, outcome)
                    rest = []
                else:
                    described += int(outcome)
                if rest:
                    workers = max(1, int(os.getenv("VLM_CONCURRENCY", DEFAULT_VLM_CONCURRENCY)))
                    with ThreadPoolExecutor(max_workers=workers) as pool:
                        outcomes = list(
                            pool.map(lambda b: self._describe(b, file_path, is_pdf), rest)
                        )
                    for block, outcome in zip(rest, outcomes, strict=True):
                        if isinstance(outcome, Exception):
                            self._record_failure(doc, block, outcome)
                        else:
                            described += int(outcome)

            span.set_attribute("file_agent.described_count", described)

        if described:
            doc.metadata["vlm_described_figures"] = described
        return doc

    def _describe(self, block: Block, file_path: Path, is_pdf: bool) -> bool | Exception:
        try:
            image = self._load_image(block, file_path, is_pdf)
            if image is None:
                return False
            description = self.vlm_client.describe_image(image, FIGURE_PROMPT)
            if not description:
                return False
            block.vlm_description = description
            # Fold the description into the block text so it becomes part of
            # the indexed/searchable content and the Markdown export.
            addition = f"[Image description]: {description}"
            block.text = f"{block.text}\n\n{addition}".strip() if block.text else addition
            logger.debug("Described block %s on page %s", block.id, block.page_number)
            return True
        except Exception as exc:  # noqa: BLE001 - reported per figure by the caller
            return exc

    @staticmethod
    def _record_failure(doc: Document, block: Block, exc: Exception) -> None:
        logger.warning("VLM description failed for block %s: %s", block.id, exc)
        block.metadata["vlm_error"] = str(exc)
        doc.metadata["vlm_error"] = str(exc)

    def _should_describe(self, block: Block, is_pdf: bool) -> bool:
        if block.block_type not in (BlockType.FIGURE, BlockType.IMAGE):
            return False
        if block.vlm_description:
            return False
        if block.image_bytes is not None:
            return self._image_pixels(block) >= DEFAULT_MIN_IMAGE_PIXELS
        return (
            is_pdf
            and block.bbox is not None
            and block.page_number is not None
            and self._bbox_area(block) >= self.min_figure_area
        )

    def _figure_area(self, block: Block) -> float:
        if block.image_bytes is not None:
            return float(self._image_pixels(block))
        return self._bbox_area(block)

    @staticmethod
    def _load_image(block: Block, file_path: Path, is_pdf: bool) -> Image.Image | None:
        if block.image_bytes is not None:
            try:
                image = Image.open(io.BytesIO(block.image_bytes))
                image.load()
                return image.convert("RGB")
            except Exception as exc:
                logger.debug("Unreadable embedded image in block %s: %s", block.id, exc)
                return None
        if is_pdf and block.bbox is not None and block.page_number is not None:
            return extract_image_from_pdf(file_path, block.page_number, block.bbox)
        return None

    @staticmethod
    def _image_pixels(block: Block) -> int:
        cached = block.metadata.get("_image_pixels")
        if isinstance(cached, int):
            return cached
        try:
            with Image.open(io.BytesIO(block.image_bytes or b"")) as image:
                pixels = image.width * image.height
        except Exception:
            pixels = 0
        block.metadata["_image_pixels"] = pixels
        return pixels

    @staticmethod
    def _bbox_area(block: Block) -> float:
        if not block.bbox:
            return 0.0
        x0, y0, x1, y1 = block.bbox
        return abs((x1 - x0) * (y1 - y0))
