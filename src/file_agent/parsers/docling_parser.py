import uuid
from pathlib import Path
from typing import Any

from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat

from file_agent.document import Block, BlockType, Document
from file_agent.parsers.base import BaseParser


class DoclingParser(BaseParser):
    def __init__(self):
        self.converter = DocumentConverter(
            allowed_formats=[InputFormat.PDF, InputFormat.DOCX]
        )

    def parse(self, file_path: Path) -> Document:
        path = Path(file_path)
        
        conv_result = self.converter.convert(path)
        docling_doc = conv_result.document
        
        blocks = []
        total_pages = len(conv_result.pages) if conv_result.pages else 0
        
        # итерация по логическим элементам документа (сохраняет порядок чтения)
        for item, level in docling_doc.iterate_items():
            block_type = self._map_docling_type_to_block_type(item.label)
            try:
                if hasattr(item, 'text') and item.text:
                    content = item.text
                elif hasattr(item, 'export_to_markdown'):
                    content = item.export_to_markdown()
                else:
                    content = str(item)
            except Exception:
                content = str(item.text) if hasattr(item, 'text') else ""
            
            page_number = 1
            bbox = None
            
            if hasattr(item, 'prov') and item.prov:
                prov = item.prov[0]
                page_number = getattr(prov, 'page_no', 1)
                if hasattr(prov, 'bbox') and prov.bbox:
                    # bbox имеет атрибуты l, t, r, b (left, top, right, bottom)
                    bbox = (float(prov.bbox.l), float(prov.bbox.t), float(prov.bbox.r), float(prov.bbox.b))
            
            block = Block(
                id=f"block_{uuid.uuid4().hex[:8]}",
                block_type=block_type,
                page_number=page_number,
                content=content,
                bbox=bbox,
                vlm_description=None,  
                metadata={
                    "source_file": path.name,
                    "docling_label": str(item.label),
                    "hierarchy_level": level
                }
            )
            blocks.append(block)
            
        if not blocks:
            blocks.append(Block(
                id="block_empty",
                block_type=BlockType.TEXT,
                page_number=1,
                content="",
                metadata={"source_file": path.name, "warning": "Empty document or parsing failed"}
            ))

        return Document(
            file_name=path.name,
            file_type=path.suffix.lower().replace(".", ""),
            blocks=blocks,
            metadata={
                "total_pages": total_pages,
                "parsing_method": "docling_native",
            }
        )

    def _map_docling_type_to_block_type(self, label: str) -> BlockType:
        """Маппинг внутренних меток Docling на наш Enum BlockType"""
        label_lower = str(label).lower()
        if "title" in label_lower or "heading" in label_lower:
            return BlockType.HEADING
        if "table" in label_lower:
            return BlockType.TABLE
        if "picture" in label_lower or "figure" in label_lower:
            return BlockType.FIGURE
        if "formula" in label_lower:
            return BlockType.FORMULA
        return BlockType.TEXT