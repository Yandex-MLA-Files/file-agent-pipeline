import logging
from pathlib import Path

from file_agent.document import Block, BlockType, Document
from file_agent.utils.image_extractor import extract_image_from_pdf
from file_agent.vlm.base import VLMClient

logger = logging.getLogger(__name__)

FIGURE_PROMPT = (
    "Describe the structure, key elements, text and meaning of this diagram, "
    "chart or figure in detail. This description is used for retrieval-augmented "
    "generation, so be specific and factual. If it is a purely decorative image "
    "or a logo, say so briefly."
)


class DocumentEnhancer:
    """Enriches figure/image blocks with a VLM-generated textual description.

    Whether a block is worth describing is decided structurally: only blocks
    that a parser classified as a figure/image and that carry a bounding box on
    a PDF page can be cropped and sent to the VLM. Everything else is left
    untouched, so a document with no figures makes no VLM calls at all.
    """

    def __init__(self, vlm_client: VLMClient) -> None:
        self.vlm_client = vlm_client

    def enhance(self, doc: Document, file_path: Path) -> Document:
        if Path(file_path).suffix.lower() != ".pdf":
            # Cropping figures requires rendering PDF page regions; other formats
            # are not supported by the VLM enhancer yet.
            return doc

        for block in doc.blocks:
            if not self._should_describe(block):
                continue
            try:
                image = extract_image_from_pdf(file_path, block.page_number, block.bbox)
                if image is None:
                    continue

                description = self.vlm_client.describe_image(image, FIGURE_PROMPT)
                block.vlm_description = description
                # Fold the description into the block text so it becomes part of
                # the indexed/searchable content and the Markdown export.
                addition = f"[Image description]: {description}"
                block.text = f"{block.text}\n\n{addition}".strip() if block.text else addition
                logger.debug("Described block %s on page %s", block.id, block.page_number)
            except Exception as exc:
                logger.warning("VLM description failed for block %s: %s", block.id, exc)
                block.metadata["vlm_error"] = str(exc)

        return doc

    @staticmethod
    def _should_describe(block: Block) -> bool:
        return (
            block.block_type in (BlockType.FIGURE, BlockType.IMAGE)
            and block.bbox is not None
            and block.page_number is not None
        )
