import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List
from .document import Document, Block, BlockType

@dataclass
class Chunk:
    id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Сериализация чанка для экспорта в CSV/JSON или сохранения в Vector DB"""
        return {
            "id": self.id,
            "text": self.text,
            "metadata": self.metadata
        }

class DocumentChunker:
    def __init__(self, max_chunk_size: int = 1000, chunk_overlap: int = 100):
        self.max_chunk_size = max_chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_document(self, doc: Document) -> List[Chunk]:
        """
        Разбиение документа на чанки, сохраняя сквозные метаданные 
        для последующей фильтрации в RAG.
        """
        chunks = []
        
        doc_level_metadata = {
            "file_name": doc.file_name,
            "file_type": doc.file_type,
            "parsing_method": doc.metadata.get("parsing_method", "unknown"),
            "total_pages": doc.metadata.get("total_pages", 0)
        }

        # Разбиение длинных блоков
        for block in doc.blocks:
            block_text = block.text
            
            if len(block_text) <= self.max_chunk_size:
                chunk = self._create_chunk(block, block_text, doc_level_metadata)
                chunks.append(chunk)
            else:
                if block.block_type == BlockType.TABLE:
                    # таблицы не разбиваются или разбиваются только по строкам 
                    chunk = self._create_chunk(block, block_text, doc_level_metadata)
                    chunks.append(chunk)
                else:
                    # разбиение текста с overlap
                    start_idx = 0
                    chunk_idx = 0
                    while start_idx < len(block_text):
                        end_idx = start_idx + self.max_chunk_size
                        chunk_text = block_text[start_idx:end_idx]
                        
                        sub_metadata = doc_level_metadata.copy()
                        sub_metadata["chunk_index_in_block"] = chunk_idx
                        sub_metadata["is_truncated"] = end_idx < len(block_text)
                        
                        chunk = self._create_chunk(block, chunk_text, sub_metadata)
                        chunks.append(chunk)
                        
                        start_idx = end_idx - self.chunk_overlap
                        chunk_idx += 1

        return chunks

    def _create_chunk(self, block: Block, text: str, doc_metadata: Dict[str, Any]) -> Chunk:
        """Создает чанк, объединяя метаданные документа и конкретного блока"""
        chunk_id = f"chunk_{uuid.uuid4().hex[:8]}"
        
        combined_metadata = doc_metadata.copy()
        combined_metadata.update({
            "block_id": block.id,
            "block_type": block.block_type.value,
            "page_number": block.page_number,
            "bbox": block.bbox,
            "has_vlm_description": bool(block.vlm_description)
        })
        
        if block.vlm_description:
            combined_metadata["vlm_description"] = block.vlm_description

        return Chunk(
            id=chunk_id,
            text=text,
            metadata=combined_metadata
        )