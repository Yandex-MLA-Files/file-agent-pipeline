import logging
import os
from pathlib import Path

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
    "read: title, axis labels, legend entries, series and node names. Do not "
    "guess the subject, and do not invent numbers, names, dates or context that "
    "are not shown. If the image is decorative or unreadable, say exactly that "
    "in one short sentence."
)

# Figures smaller than this (in PDF points squared, ~1 pt = 1/72") are almost
# always icons, logos or bullets — not worth a VLM call.
DEFAULT_MIN_FIGURE_AREA = 5000.0
# Upper bound on VLM calls per document, so a 100-figure deck cannot silently
# turn into a 100-request bill. The largest figures are described first.
DEFAULT_MAX_FIGURES = 8


class DocumentEnhancer:
    """Enriches figure/image blocks with a VLM-generated textual description.

    Only blocks that a parser classified as figures and located with a bounding
    box on a PDF page can be cropped and sent to the VLM. To keep the cost
    predictable, tiny decorative images are skipped and at most
    ``max_figures`` figures per document are described (largest first —
    the big diagram matters more than a footer icon).
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
        if Path(file_path).suffix.lower() != ".pdf":
            # Cropping figures requires rendering PDF page regions; other formats
            # are not supported by the VLM enhancer yet.
            return doc

        candidates = [block for block in doc.blocks if self._should_describe(block)]
        candidates.sort(key=self._bbox_area, reverse=True)
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
            for block in selected:
                try:
                    image = extract_image_from_pdf(file_path, block.page_number, block.bbox)
                    if image is None:
                        continue

                    description = self.vlm_client.describe_image(image, FIGURE_PROMPT)
                    if not description:
                        continue
                    block.vlm_description = description
                    # Fold the description into the block text so it becomes part of
                    # the indexed/searchable content and the Markdown export.
                    addition = f"[Image description]: {description}"
                    block.text = f"{block.text}\n\n{addition}".strip() if block.text else addition
                    described += 1
                    logger.debug("Described block %s on page %s", block.id, block.page_number)
                except Exception as exc:
                    logger.warning("VLM description failed for block %s: %s", block.id, exc)
                    block.metadata["vlm_error"] = str(exc)

            span.set_attribute("file_agent.described_count", described)

        if described:
            doc.metadata["vlm_described_figures"] = described
        return doc

    def _should_describe(self, block: Block) -> bool:
        return (
            block.block_type in (BlockType.FIGURE, BlockType.IMAGE)
            and block.bbox is not None
            and block.page_number is not None
            and self._bbox_area(block) >= self.min_figure_area
        )

    @staticmethod
    def _bbox_area(block: Block) -> float:
        if not block.bbox:
            return 0.0
        x0, y0, x1, y1 = block.bbox
        return abs((x1 - x0) * (y1 - y0))
