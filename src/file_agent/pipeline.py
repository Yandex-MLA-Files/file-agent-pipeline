import logging
import os
from pathlib import Path

from file_agent.document import Document
from file_agent.parsers.html_parser import HTMLParser
from file_agent.parsers.md_parser import MarkdownParser
from file_agent.parsers.pdf_parser import PDFParser
from file_agent.parsers.pptx_parser import PPTXParser
from file_agent.parsers.xlsx_parser import XLSXParser
from file_agent.parsers.docling_parser import DoclingParser
from file_agent.parsers.enhancer import DocumentEnhancer
from file_agent.vlm.openai_compatible import OpenAICompatibleVLMClient
from file_agent.vlm.base import MockVLMClient

from dotenv import load_dotenv
load_dotenv()


logger = logging.getLogger(__name__)

def parse_file(
    file_path: str | Path, 
    enable_vlm: bool = True, 
    enable_ocr_fallback: bool = True
) -> Document:
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix in {".pdf", ".docx"}:
        doc = DoclingParser().parse(path)
        
        if enable_vlm or enable_ocr_fallback:
            # инициализация VLM клиента (здесь можно читать URL из .env)
            # пример для локального Ollama: base_url="http://localhost:11434/v1", model="qwen2.5-vl:7b"
            vlm_client = OpenAICompatibleVLMClient(
                base_url=os.getenv("VLM_BASE_URL", "http://localhost:11434/v1"),
                model=os.getenv("VLM_MODEL", "qwen2.5-vl:7b"),
                api_key=os.getenv("VLM_API_KEY", "dummy")
            )
            
            # Graceful Degradation: проверка доступности VLM
            try:
                from PIL import Image
                vlm_client.describe_image(Image.new('RGB', (10, 10)), "test")
            except Exception:
                logger.warning("VLM-сервер недоступен. Используется MockVLMClient (описания будут заглушками).")
                vlm_client = MockVLMClient()

            enhancer = DocumentEnhancer(vlm_client=vlm_client, force_ocr=enable_ocr_fallback)
            doc = enhancer.enhance(doc, path)
            
        return doc
    
    if suffix == ".md":
        return MarkdownParser().parse(path)
    
    if suffix in {".html", ".htm"}:
        return HTMLParser().parse(path)
    
    if suffix == ".xlsx":
        return XLSXParser().parse(path)
    
    if suffix == ".pptx":
        return PPTXParser().parse(path)

    raise ValueError(f"Unsupported file type: {suffix or '<no extension>'}")