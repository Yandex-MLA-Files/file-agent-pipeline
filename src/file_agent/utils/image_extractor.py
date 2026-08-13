import io
import logging
import math
from pathlib import Path

import fitz  # PyMuPDF
from PIL import Image

logger = logging.getLogger(__name__)


def extract_image_from_pdf(
    pdf_path: Path,
    page_number: int,
    bbox: tuple[float, float, float, float],
    zoom: float = 2.0,
) -> Image.Image | None:
    """Render a region of a PDF page to a PIL image.

    :param pdf_path: path to the PDF file.
    :param page_number: 1-indexed page number.
    :param bbox: region to crop as (x0, y0, x1, y1) in top-left PDF coordinates.
    :param zoom: scale factor to improve resolution of the crop (default 2.0).
    :return: a PIL.Image, or None if the region could not be rendered.
    """
    return _render_pdf_image(
        pdf_path=pdf_path,
        pdf_bytes=None,
        page_number=page_number,
        bbox=bbox,
        zoom=zoom,
    )


def extract_image_from_pdf_bytes(
    pdf_bytes: bytes,
    page_number: int,
    bbox: tuple[float, float, float, float] | None = None,
    zoom: float = 2.0,
    padding: float = 0.0,
    max_pixels: int | None = None,
) -> Image.Image | None:
    """Render a PDF page or a bounded region from in-memory document bytes."""
    padded_bbox = None
    if bbox is not None:
        x0, y0, x1, y1 = bbox
        padded_bbox = (x0 - padding, y0 - padding, x1 + padding, y1 + padding)
    return _render_pdf_image(
        pdf_path=None,
        pdf_bytes=pdf_bytes,
        page_number=page_number,
        bbox=padded_bbox,
        zoom=zoom,
        max_pixels=max_pixels,
    )


def _render_pdf_image(
    pdf_path: Path | None,
    pdf_bytes: bytes | None,
    page_number: int,
    bbox: tuple[float, float, float, float] | None,
    zoom: float,
    max_pixels: int | None = None,
) -> Image.Image | None:
    try:
        if page_number < 1:
            raise ValueError("page_number must be a positive integer")
        if zoom <= 0:
            raise ValueError("zoom must be greater than zero")

        document = (
            fitz.open(stream=pdf_bytes, filetype="pdf")
            if pdf_bytes is not None
            else fitz.open(str(pdf_path))
        )
        with document as doc:
            if page_number > len(doc):
                raise ValueError(f"page_number exceeds PDF page count: {page_number}")
            page = doc[page_number - 1]
            clip = page.rect
            if bbox is not None:
                clip = fitz.Rect(bbox) & page.rect
                if clip.is_empty or clip.width <= 0 or clip.height <= 0:
                    raise ValueError("bbox does not intersect the selected PDF page")
            render_zoom = zoom
            if max_pixels is not None:
                if max_pixels < 1:
                    raise ValueError("max_pixels must be greater than zero")
                estimated_pixels = clip.width * clip.height * render_zoom**2
                if estimated_pixels > max_pixels:
                    render_zoom = math.sqrt(max_pixels / (clip.width * clip.height))
            matrix = fitz.Matrix(render_zoom, render_zoom)
            pixmap = page.get_pixmap(matrix=matrix, clip=clip, alpha=False)
            image = Image.open(io.BytesIO(pixmap.tobytes("png")))
            image.load()
            return image
    except Exception as exc:
        logger.warning(
            "Could not extract image from page %s, bbox=%s: %s",
            page_number,
            bbox,
            exc,
        )
        return None
