import logging
from pathlib import Path
from file_agent.document import Document, BlockType
from file_agent.vlm.base import VLMClient
from file_agent.utils.image_extractor import extract_image_from_pdf

logger = logging.getLogger(__name__)

class DocumentEnhancer:
    def __init__(self, vlm_client: VLMClient, force_ocr: bool = False):
        self.vlm_client = vlm_client
        self.force_ocr = force_ocr

    def enhance(self, doc: Document, file_path: Path) -> Document:
        """Применяет умные эвристики для обогащения документа"""
        
        if self.force_ocr or self._is_likely_scanned(doc):
            logger.info(f"Обнаружен потенциальный скан: {file_path.name}. Переход в режим OCR.")
            doc.metadata["parsing_method"] = "docling_ocr_forced"

        for block in doc.blocks:
            if block.block_type in (BlockType.FIGURE, BlockType.IMAGE) and block.bbox:
                if file_path.suffix.lower() == ".pdf":
                    try:
                        img = extract_image_from_pdf(file_path, block.page_number, block.bbox)
                        if img:
                            prompt = (
                                "Describe the structure, key elements, text, and meaning of this diagram or flowchart in detail. "
                                "This information will be used for RAG (Retrieval-Augmented Generation). "
                                "If it is a decorative image or a logo, state that briefly."
                            )
                            desc = self.vlm_client.describe_image(img, prompt)
                            block.vlm_description = desc
                            
                            block.content += f"\n\n[AI_VISION_DESC]: {desc}"
                            logger.debug(f"Успешно описан блок {block.id} на стр. {block.page_number}")
                    except Exception as e:
                        logger.warning(f"Сбой VLM для блока {block.id}: {e}")
                        block.metadata["vlm_error"] = str(e)

        return doc

    def _is_likely_scanned(self, doc: Document) -> bool:
        """Агентная эвристика: если на страницу приходится в среднем менее 50 символов, это скорее всего скан"""
        total_pages = max(doc.metadata.get("total_pages", 1), 1)
        total_text_len = sum(len(b.content) for b in doc.blocks if b.block_type == BlockType.TEXT)
        avg_chars_per_page = total_text_len / total_pages
        
        return avg_chars_per_page < 50