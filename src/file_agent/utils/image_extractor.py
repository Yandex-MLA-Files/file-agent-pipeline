import io
import logging
from pathlib import Path
from typing import Optional, Tuple

import fitz  # PyMuPDF
from PIL import Image

logger = logging.getLogger(__name__)

def extract_image_from_pdf(
    pdf_path: Path, 
    page_number: int, 
    bbox: Tuple[float, float, float, float],
    zoom: float = 2.0
) -> Optional[Image.Image]:
    """
    Вырезает изображение из pdf по координатам bounding box.
    
    :param pdf_path: Путь к pdf файлу
    :param page_number: Номер страницы (1-indexed)
    :param bbox: Кортеж (x0, y0, x1, y1)
    :param zoom: Коэффициент масштабирования для улучшения качества (default 2.0)
    :return: PIL.Image или None в случае ошибки
    """
    try:
        doc = fitz.open(str(pdf_path))
        page = doc[page_number - 1]
        rect = fitz.Rect(bbox)
        
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, clip=rect)
        
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        doc.close()
        return img
    except Exception as e:
        logger.warning(f"Не удалось извлечь изображение со стр. {page_number}, bbox={bbox}: {e}")
        return None