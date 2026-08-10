from pathlib import Path
from typing import Any

from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE_TYPE

from file_agent.document import Block, BlockType, Document
from file_agent.parsers.base import BaseParser


class PPTXParser(BaseParser):
    def parse(self, file_path: Path) -> Document:
        path = Path(file_path)
        presentation = Presentation(path)
        blocks: list[Block] = []

        for slide_index, slide in enumerate(presentation.slides, start=1):
            blocks.append(
                Block(
                    id=f"slide-{slide_index}",
                    text=_slide_to_text(slide, slide_index),
                    type="pptx_slide",
                    metadata={
                        "source_file": path.name,
                        "slide_number": slide_index,
                        "shapes_count": len(slide.shapes),
                    },
                )
            )
            blocks.extend(_picture_blocks(slide, slide_index, path.name))

        return Document(
            file_name=path.name,
            file_type="pptx",
            blocks=blocks,
        )


def _picture_blocks(slide: Any, slide_index: int, source_file: str) -> list[Block]:
    """One empty BlockType.IMAGE block per picture shape, ready for VLM description.

    Text stays empty here (mirrors the PDF FIGURE/IMAGE convention) - the
    description is filled in later by DocumentEnhancer, which turns the
    picture's own bytes (via extract_image_from_pptx) into readable text.
    """
    blocks: list[Block] = []
    for shape_index, shape in enumerate(slide.shapes):
        if shape.shape_type != MSO_SHAPE_TYPE.PICTURE:
            continue
        left, top = shape.left.pt, shape.top.pt
        blocks.append(
            Block(
                id=f"slide-{slide_index}-image-{shape_index}",
                text="",
                type=BlockType.IMAGE.value,
                block_type=BlockType.IMAGE,
                page_number=slide_index,
                bbox=(left, top, left + shape.width.pt, top + shape.height.pt),
                metadata={
                    "source_file": source_file,
                    "slide_number": slide_index,
                    "shape_index": shape_index,
                },
            )
        )
    return blocks


def _slide_to_text(slide: Any, slide_number: int) -> str:
    parts = [f"Slide {slide_number}"]

    title = _get_slide_title(slide)
    if title:
        parts.append(f"Title: {title}")

    text_lines = _get_text_lines(slide, title)
    if text_lines:
        parts.append("Text:")
        parts.extend(text_lines)

    table_lines = _get_table_lines(slide)
    if table_lines:
        parts.append("Table:")
        parts.extend(table_lines)

    return "\n".join(parts)


def _get_slide_title(slide: Any) -> str:
    if slide.shapes.title is None:
        return ""
    return slide.shapes.title.text.strip()


def _get_text_lines(slide: Any, title: str) -> list[str]:
    lines: list[str] = []

    for shape in slide.shapes:
        if not getattr(shape, "has_text_frame", False):
            continue
        text = shape.text.strip()
        if not text or text == title:
            continue
        lines.append(text)

    return lines


def _get_table_lines(slide: Any) -> list[str]:
    lines: list[str] = []

    for shape in slide.shapes:
        if not getattr(shape, "has_table", False):
            continue
        for row in shape.table.rows:
            values = [cell.text.strip() for cell in row.cells]
            if any(values):
                lines.append("\t".join(values))

    return lines
