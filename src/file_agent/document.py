import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

class BlockType(str, Enum):
    TEXT = "text"
    HEADING = "heading"
    TABLE = "table"
    FIGURE = "figure"       
    IMAGE = "image"         
    FORMULA = "formula"
    PDF_PAGE = "pdf_page"   

@dataclass
class Block:
    id: str
    block_type: BlockType
    page_number: int
    content: str  # Markdown-представление (текст, MD-таблица и т.д.)
    bbox: Optional[Tuple[float, float, float, float]] = None  # (x0, y0, x1, y1)
    vlm_description: Optional[str] = None  # Семантическое описание от VLM (для figure/image)
    metadata: Dict[str, Any] = field(default_factory=dict) # Доп. метаданные (напр., confidence score OCR)

    @property
    def text(self) -> str:
        """Свойство для обратной совместимости со старым кодом, который ожидает block.text"""
        if self.vlm_description and self.block_type in (BlockType.FIGURE, BlockType.IMAGE):
            return f"[Описание изображения]: {self.vlm_description}\n\n{self.content}"
        return self.content

    def to_dict(self) -> Dict[str, Any]:
        """Сериализация для экспорта в JSON/CSV"""
        return {
            "id": self.id,
            "block_type": self.block_type.value,
            "page_number": self.page_number,
            "bbox": self.bbox,
            "content": self.content,
            "vlm_description": self.vlm_description,
            "metadata": self.metadata
        }

@dataclass
class Document:
    file_name: str
    file_type: str
    blocks: List[Block]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Инициализация метаданных по умолчанию, если они не переданы"""
        if "table_of_contents" not in self.metadata:
            self.metadata["table_of_contents"] = []
        if "total_pages" not in self.metadata:
            self.metadata["total_pages"] = max((b.page_number for b in self.blocks), default=0)
        if "parsing_method" not in self.metadata:
            self.metadata["parsing_method"] = "unknown"

    def extract_toc_from_headings(self):
        """Автоматическое построение оглавления из блоков типа HEADING"""
        toc = []
        for block in self.blocks:
            if block.block_type == BlockType.HEADING:
                toc.append({
                    "title": block.content.strip(),
                    "page": block.page_number,
                    "block_id": block.id
                })
        self.metadata["table_of_contents"] = toc

    def to_dict(self) -> Dict[str, Any]:
        """Сериализация документа для экспорта"""
        return {
            "file_name": self.file_name,
            "file_type": self.file_type,
            "metadata": self.metadata,
            "blocks": [b.to_dict() for b in self.blocks]
        }