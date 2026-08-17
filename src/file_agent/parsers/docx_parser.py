"""Structured DOCX parser built on python-docx.

Word documents carry their structure explicitly (paragraph styles, outline
levels, list numbering, tables, embedded pictures), so reading it directly is
both faster and more faithful than running a layout model over a rendering.
Docling's Word backend splits paragraphs into formatting runs, which produces
sentence fragments as blocks; this parser keeps paragraphs whole and emits:

- ``HEADING`` blocks with a hierarchy level from ``Heading N`` / ``Заголовок N``
  styles, outline levels, or (for unstyled documents) a conservative
  bold-and-larger-than-body heuristic;
- ``TEXT`` paragraphs, ``LIST`` blocks (consecutive list paragraphs joined),
  ``TABLE`` blocks rendered as Markdown (merged cells de-duplicated),
  ``FIGURE`` blocks carrying the embedded image bytes and any caption, and
  ``CODE`` blocks for monospaced paragraphs.

Body order is preserved (paragraphs and tables interleaved as in the file).
"""

import logging
import re
import statistics
from pathlib import Path
from typing import Any

from docx import Document as load_docx
from docx.oxml.ns import qn
from docx.table import Table
from docx.text.paragraph import Paragraph

from file_agent.document import BlockType, Document
from file_agent.parsers.base import BaseParser
from file_agent.parsers.common import (
    BlockFactory,
    clean_text,
    infer_heading_level,
    list_to_markdown,
    looks_like_heading,
    strip_bullet,
    table_to_markdown,
)
from file_agent.telemetry import tracer

logger = logging.getLogger(__name__)

_HEADING_STYLE = re.compile(r"^(heading|заголовок|título|titre|überschrift)\s*(\d)", re.IGNORECASE)
_TITLE_STYLE = re.compile(r"^(title|название|заголовок$|subtitle|подзаголовок)", re.IGNORECASE)
_LIST_STYLE = re.compile(r"(list|список|bullet|маркир|нумер|number)", re.IGNORECASE)
_CAPTION_STYLE = re.compile(r"(caption|название объекта|подпись)", re.IGNORECASE)
_CAPTION_TEXT = re.compile(
    r"^\s*(рис(унок|\.)?|figure|fig\.|схема|диаграмма|табл(ица|\.)?|table)\b",
    re.IGNORECASE,
)
_CODE_FONTS = ("courier", "consolas", "mono", "menlo", "lucida console", "source code")
# Legacy VML images (``<v:imagedata r:id=...>``); python-docx does not register
# the VML namespace, so it is spelled out here.
_VML_IMAGEDATA = "{urn:schemas-microsoft-com:vml}imagedata"

# A heading found by the formatting heuristic (no explicit style) is trusted
# only when the paragraph is short and visibly larger/bolder than body text.
_HEURISTIC_MAX_CHARS = 120
_MAX_TABLE_ROWS = 2000


class DOCXParser(BaseParser):
    def parse(self, file_path: Path) -> Document:
        path = Path(file_path)
        with tracer.start_as_current_span("file_agent.docx_parse") as span:
            span.set_attribute("file_agent.file_name", path.name)

            docx = load_docx(str(path))
            factory = BlockFactory(source_file=path.name)
            walker = _BodyWalker(docx, factory)
            walker.run()

            if not factory.blocks:
                factory.add(
                    "",
                    BlockType.TEXT,
                    {"warning": "empty document or no extractable content"},
                    skip_empty=False,
                )

            document = Document(
                file_name=path.name,
                file_type="docx",
                blocks=factory.blocks,
                metadata={
                    "parsing_method": "python-docx",
                    "title": walker.title,
                    "figure_count": walker.figure_count,
                    "table_count": walker.table_count,
                },
            )
            document.build_table_of_contents()
            span.set_attribute("file_agent.block_count", len(document.blocks))
            logger.info("Parsed %s into %d block(s)", path.name, len(document.blocks))
            return document


class _BodyWalker:
    def __init__(self, docx: Any, factory: BlockFactory) -> None:
        self.docx = docx
        self.factory = factory
        self.title: str | None = None
        self.figure_count = 0
        self.table_count = 0
        self._body_size = self._estimate_body_font_size()
        self._pending_list: list[tuple[int, str]] = []
        self._pending_list_ordered = False
        self._last_figure = None

    # -- driver ---------------------------------------------------------------

    def run(self) -> None:
        core = getattr(self.docx, "core_properties", None)
        if core is not None and isinstance(core.title, str) and core.title.strip():
            self.title = core.title.strip()

        for element in self.docx.element.body.iterchildren():
            tag = element.tag
            if tag == qn("w:p"):
                self._handle_paragraph(Paragraph(element, self.docx))
            elif tag == qn("w:tbl"):
                self._flush_list()
                self._handle_table(Table(element, self.docx))
            elif tag == qn("w:sdt"):
                # Content controls (structured document tags) wrap paragraphs
                # and tables; unwrap them in document order.
                for inner in element.iter():
                    if inner.tag == qn("w:p") and inner.getparent().tag == qn("w:sdtContent"):
                        self._handle_paragraph(Paragraph(inner, self.docx))
                    elif inner.tag == qn("w:tbl") and inner.getparent().tag == qn("w:sdtContent"):
                        self._flush_list()
                        self._handle_table(Table(inner, self.docx))
        self._flush_list()

    # -- paragraphs -------------------------------------------------------------

    def _handle_paragraph(self, paragraph: Paragraph) -> None:
        images = self._paragraph_images(paragraph)
        text = clean_text(self._paragraph_text(paragraph))

        if images:
            self._flush_list()
            for blob in images:
                self.figure_count += 1
                self._last_figure = self.factory.add(
                    text if text and _CAPTION_TEXT.match(text) else "",
                    BlockType.FIGURE,
                    {"figure_index": self.figure_count},
                    image_bytes=blob,
                    skip_empty=False,
                )
            if text and not _CAPTION_TEXT.match(text):
                self.factory.add(text, BlockType.TEXT)
            return

        if not text:
            # Blank paragraphs terminate lists but carry nothing themselves.
            self._flush_list()
            return

        style_name = self._style_name(paragraph)
        level = self._heading_level(paragraph, style_name, text)
        if level is not None:
            self._flush_list()
            self._last_figure = None
            if self.title is None and level == 1:
                self.title = text
            self.factory.heading(text, level, metadata={"style": style_name})
            return

        if self._is_caption(paragraph, style_name, text):
            self._flush_list()
            self._attach_caption(text)
            return

        list_indent = self._list_indent(paragraph, style_name)
        if list_indent is not None:
            ordered = self._list_is_ordered(paragraph)
            if not self._pending_list:
                self._pending_list_ordered = ordered
            self._pending_list.append((list_indent, strip_bullet(text)))
            return

        self._flush_list()
        self._last_figure = None
        if self._is_code(paragraph):
            self.factory.add(text, BlockType.CODE)
        else:
            self.factory.add(text, BlockType.TEXT)

    def _flush_list(self) -> None:
        if not self._pending_list:
            return
        markdown = list_to_markdown(self._pending_list, ordered=self._pending_list_ordered)
        self.factory.add(markdown, BlockType.LIST, {"item_count": len(self._pending_list)})
        self._pending_list = []
        self._pending_list_ordered = False

    def _attach_caption(self, text: str) -> None:
        target = self._last_figure
        if target is not None and target.block_type == BlockType.FIGURE:
            target.text = f"{target.text}\n{text}".strip() if target.text else text
            target.metadata["caption"] = text
            self._last_figure = None
            return
        # A caption preceding its figure/table or belonging to a table: keep as text.
        self.factory.add(text, BlockType.TEXT, {"caption": True})

    @staticmethod
    def _paragraph_text(paragraph: Paragraph) -> str:
        # ``paragraph.text`` skips text inside hyperlinks/fields in older
        # python-docx builds; walk every text run in document order instead.
        parts: list[str] = []
        for node in paragraph._p.iter():
            tag = node.tag
            if tag == qn("w:t"):
                parts.append(node.text or "")
            elif tag == qn("w:tab"):
                parts.append("\t")
            elif tag in (qn("w:br"), qn("w:cr")):
                parts.append("\n")
        text = "".join(parts)
        return text if text.strip() else paragraph.text

    def _paragraph_images(self, paragraph: Paragraph) -> list[bytes]:
        blobs: list[bytes] = []
        for blip in paragraph._p.iter(qn("a:blip")):
            rid = blip.get(qn("r:embed")) or blip.get(qn("r:link"))
            blob = self._related_blob(rid)
            if blob:
                blobs.append(blob)
        for image_data in paragraph._p.iter(_VML_IMAGEDATA):
            rid = image_data.get(qn("r:id"))
            blob = self._related_blob(rid)
            if blob:
                blobs.append(blob)
        return blobs

    def _related_blob(self, rid: str | None) -> bytes | None:
        if not rid:
            return None
        try:
            part = self.docx.part.related_parts[rid]
        except (KeyError, AttributeError):
            return None
        blob = getattr(part, "blob", None)
        if not blob or len(blob) < 512:
            return None
        return blob

    # -- classification ---------------------------------------------------------

    @staticmethod
    def _style_name(paragraph: Paragraph) -> str:
        try:
            return (paragraph.style.name or "").strip()
        except Exception:  # pragma: no cover - broken style references
            return ""

    def _heading_level(self, paragraph: Paragraph, style_name: str, text: str) -> int | None:
        match = _HEADING_STYLE.match(style_name)
        if match:
            return int(match.group(2))
        if _TITLE_STYLE.match(style_name):
            return 1

        outline = self._outline_level(paragraph)
        if outline is not None:
            return outline + 1

        # Unstyled documents: short, bold, larger-than-body paragraphs.
        if not looks_like_heading(text, _HEURISTIC_MAX_CHARS):
            return None
        size = self._font_size(paragraph)
        bold = self._is_bold(paragraph)
        if size and self._body_size and size >= self._body_size + 2 and (bold or size >= 16):
            numbered = infer_heading_level(text)
            if numbered is not None:
                return numbered
            return 1 if size >= self._body_size + 6 else 2
        if bold:
            # "1.2 Постановка задачи" in bold is a heading even at body size.
            return infer_heading_level(text)
        return None

    @staticmethod
    def _outline_level(paragraph: Paragraph) -> int | None:
        for source in (paragraph._p.pPr, getattr(paragraph.style, "element", None)):
            if source is None:
                continue
            node = source.find(f".//{qn('w:outlineLvl')}")
            if node is not None:
                try:
                    level = int(node.get(qn("w:val")))
                except (TypeError, ValueError):
                    continue
                if 0 <= level < 9:
                    return level
        return None

    def _list_indent(self, paragraph: Paragraph, style_name: str) -> int | None:
        num_pr = paragraph._p.pPr.numPr if paragraph._p.pPr is not None else None
        if num_pr is not None:
            ilvl = num_pr.ilvl
            return int(ilvl.val) if ilvl is not None and ilvl.val is not None else 0
        if _LIST_STYLE.search(style_name):
            return 0
        return None

    @staticmethod
    def _list_is_ordered(paragraph: Paragraph) -> bool:
        # Without resolving numbering.xml we cannot know the list format; visible
        # numbering in the text is the practical signal.
        return bool(re.match(r"^\s*\d+[.)]", paragraph.text))

    @staticmethod
    def _is_caption(paragraph: Paragraph, style_name: str, text: str) -> bool:
        return bool(_CAPTION_STYLE.search(style_name)) or (
            bool(_CAPTION_TEXT.match(text)) and len(text) <= 200
        )

    @staticmethod
    def _is_code(paragraph: Paragraph) -> bool:
        fonts = {
            (run.font.name or "").lower()
            for run in paragraph.runs
            if run.text.strip() and run.font is not None
        }
        fonts.discard("")
        return bool(fonts) and all(any(code in font for code in _CODE_FONTS) for font in fonts)

    @staticmethod
    def _font_size(paragraph: Paragraph) -> float | None:
        sizes = [run.font.size.pt for run in paragraph.runs if run.text.strip() and run.font.size]
        if sizes:
            return max(sizes)
        style_font = getattr(paragraph.style, "font", None)
        if style_font is not None and style_font.size:
            return style_font.size.pt
        return None

    @staticmethod
    def _is_bold(paragraph: Paragraph) -> bool:
        runs = [run for run in paragraph.runs if run.text.strip()]
        if not runs:
            return False
        if all(run.bold for run in runs):
            return True
        style_font = getattr(paragraph.style, "font", None)
        inherited = style_font is not None and bool(style_font.bold)
        return inherited and all(run.bold is None for run in runs)

    def _estimate_body_font_size(self) -> float | None:
        sizes: list[float] = []
        for paragraph in self.docx.paragraphs[:400]:
            if len(paragraph.text.strip()) < 40:
                continue
            size = self._font_size(paragraph)
            if size:
                sizes.append(size)
        if sizes:
            return statistics.median(sizes)
        try:
            return self.docx.styles["Normal"].font.size.pt
        except Exception:  # pragma: no cover
            return None

    # -- tables -------------------------------------------------------------------

    def _handle_table(self, table: Table) -> None:
        rows: list[list[str]] = []
        for row in table.rows[:_MAX_TABLE_ROWS]:
            cells: list[str] = []
            previous = None
            for cell in row.cells:
                # Horizontally merged cells are reported once per grid column with
                # the same underlying element; show the value only once.
                if previous is not None and cell._tc is previous:
                    cells.append("")
                    continue
                previous = cell._tc
                cells.append(clean_text(cell.text).replace("\n", " "))
            rows.append(cells)
        markdown = table_to_markdown(rows)
        if not markdown:
            return
        self.table_count += 1
        self.factory.add(
            markdown,
            BlockType.TABLE,
            {"table_index": self.table_count, "row_count": len(rows)},
        )
        self._last_figure = None
