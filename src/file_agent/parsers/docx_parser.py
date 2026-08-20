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
from docx.opc.constants import RELATIONSHIP_TYPE as RT
from docx.oxml.ns import qn
from docx.table import Table
from docx.text.paragraph import Paragraph
from lxml import etree

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
from file_agent.parsers.docx_math import (
    display_equation,
    is_math,
    omml_to_latex,
    outermost_math,
)
from file_agent.parsers.docx_numbering import DocxNumbering
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
# Text frames: a shape's text lives in ``w:txbxContent``, which sits *inside* a
# paragraph of the body. Its runs must be pulled out of the host paragraph
# (they belong to a different reading order) and emitted as their own blocks.
_TEXTBOX_CONTENT = qn("w:txbxContent")

# Titles left in the document properties by Word, its templates and common
# converters; they name the file format, not the document.
_PLACEHOLDER_TITLES = frozenset(
    {
        "word document",
        "microsoft word document",
        "документ microsoft word",
        "документ",
        "document",
        "untitled",
        "без имени",
        "новый документ",
        "normal",
        "normal.dotm",
        "заголовок",
        "title",
    }
)

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
                    "footnote_count": walker.footnote_count,
                    "formula_count": walker.formula_count,
                    "text_box_count": walker.text_box_count,
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
        self.footnote_count = 0
        self.formula_count = 0
        self.text_box_count = 0
        self._body_size = self._estimate_body_font_size()
        self._pending_list: list[tuple[int, str]] = []
        self._pending_list_ordered = False
        self._last_figure = None
        self._numbering = DocxNumbering.from_document(docx)
        self._notes = _collect_notes(docx)

    # -- driver ---------------------------------------------------------------

    def run(self) -> None:
        core = getattr(self.docx, "core_properties", None)
        if core is not None and isinstance(core.title, str):
            title = core.title.strip()
            # Word and its templates leave placeholder titles behind; taking one
            # would put "Word Document" in front of every chunk's breadcrumb.
            if title and title.strip(" .").lower() not in _PLACEHOLDER_TITLES:
                self.title = title

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
        # Word counts *every* paragraph that carries a numId, including the
        # numbered headings a list is nested under: the sub-items of the
        # second section read "2.1, 2.2" precisely because the heading
        # advanced level 0. Counting only the paragraphs that end up in a list
        # block would renumber them "1.1, 1.2". The counter is therefore
        # advanced here, in document order, before any branch can return.
        marker = self._advance_numbering(paragraph)
        images = self._paragraph_images(paragraph)
        text = clean_text(self._paragraph_text(paragraph))
        text = self._append_link_targets(paragraph, text)
        text = self._append_note_markers(paragraph, text)
        text_boxes = self._text_boxes(paragraph)

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
            self._emit_text_boxes(text_boxes)
            self._emit_notes(paragraph)
            return

        if not text:
            # Blank paragraphs terminate lists but carry nothing themselves —
            # except when the shape anchored to them holds the text.
            self._flush_list()
            self._emit_text_boxes(text_boxes)
            return

        equation = display_equation(paragraph._p)
        if equation:
            # A paragraph that is nothing but an equation is the statement
            # itself, not a phrase inside one.
            self._flush_list()
            self._last_figure = None
            self.formula_count += 1
            self.factory.add(equation, BlockType.FORMULA, {"latex": True})
            self._emit_text_boxes(text_boxes)
            self._emit_notes(paragraph)
            return

        style_name = self._style_name(paragraph)
        level = self._heading_level(paragraph, style_name, text)
        if level is not None:
            self._flush_list()
            self._last_figure = None
            if marker and not _repeats_marker(text, marker):
                # "Цель и содержание" is displayed as "2. Цель и содержание";
                # without the number the heading cannot be found by it, and the
                # breadcrumb of every chunk below it loses the section number.
                text = f"{marker} {text}"
            if self.title is None and level == 1:
                self.title = text
            self.factory.heading(text, level, metadata={"style": style_name})
            self._emit_text_boxes(text_boxes)
            self._emit_notes(paragraph)
            return

        if self._is_caption(paragraph, style_name, text):
            self._flush_list()
            self._attach_caption(text)
            self._emit_text_boxes(text_boxes)
            return

        list_indent = self._list_indent(paragraph, style_name)
        if list_indent is not None:
            item = strip_bullet(text)
            if marker:
                # Word computes "8." from numbering.xml and never stores it in
                # the text; without it the item reads as an anonymous bullet.
                if not _repeats_marker(item, marker):
                    item = f"{marker} {item}".strip()
                if not self._pending_list:
                    self._pending_list_ordered = False
            elif not self._pending_list:
                self._pending_list_ordered = self._list_is_ordered(paragraph)
            self._pending_list.append((list_indent, item))
            self._emit_text_boxes(text_boxes)
            self._emit_notes(paragraph)
            return

        self._flush_list()
        self._last_figure = None
        if self._is_code(paragraph):
            self.factory.add(text, BlockType.CODE)
        else:
            self.factory.add(text, BlockType.TEXT)
        self._emit_text_boxes(text_boxes)
        self._emit_notes(paragraph)

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
        # Text frames are anchored *inside* a paragraph but belong to a
        # different reading order, so their runs are excluded here and emitted
        # as their own blocks (see :meth:`_text_boxes`).
        framed = {node for box in paragraph._p.iter(_TEXTBOX_CONTENT) for node in box.iter()}
        # An equation is an ``m:oMath`` tree, not ``w:t`` runs: it is converted
        # to LaTeX in place, so "Дисперсия равна $\sigma^{2}=…$ для выборки"
        # keeps both its sentence and its formula.
        equations = outermost_math(paragraph._p)
        inside_equation = {node for math in equations for node in math.iter() if node is not math}
        parts: list[str] = []
        for node in paragraph._p.iter():
            if node in framed or node in inside_equation:
                continue
            if is_math(node):
                latex = omml_to_latex(node)
                if latex:
                    parts.append(f" ${latex}$ ")
                continue
            tag = node.tag
            if tag == qn("w:t"):
                parts.append(node.text or "")
            elif tag == qn("w:tab"):
                parts.append("\t")
            elif tag in (qn("w:br"), qn("w:cr")):
                parts.append("\n")
        text = "".join(parts)
        if text.strip():
            return text
        return "" if framed else paragraph.text

    # -- text frames, footnotes, endnotes ----------------------------------------

    @staticmethod
    def _text_boxes(paragraph: Paragraph) -> list[list[str]]:
        """Paragraphs of every text frame anchored to this paragraph.

        Word writes a shape twice: the DrawingML version and a VML fallback
        wrapped in ``mc:AlternateContent``. Both carry the same text, so
        identical frames are reported once.
        """
        boxes: list[list[str]] = []
        seen: set[tuple[str, ...]] = set()
        for box in paragraph._p.iter(_TEXTBOX_CONTENT):
            lines: list[str] = []
            for inner in box.iter(qn("w:p")):
                text = clean_text("".join(node.text or "" for node in inner.iter(qn("w:t"))))
                if text:
                    lines.append(text)
            key = tuple(lines)
            if lines and key not in seen:
                seen.add(key)
                boxes.append(lines)
        return boxes

    def _emit_text_boxes(self, boxes: list[list[str]]) -> None:
        for lines in boxes:
            self._flush_list()
            self.text_box_count += 1
            self.factory.add(
                "\n".join(lines),
                BlockType.TEXT,
                {"text_box": True, "text_box_index": self.text_box_count},
            )

    def _append_link_targets(self, paragraph: Paragraph, text: str) -> str:
        """Keep the address of a hyperlink, not only the words it hides behind.

        A manual reads "скачайте установщик" with the URL only in the
        relationship; asked where to download from, the pipeline had nothing to
        answer with. External targets are appended once, and only when the text
        does not already spell them out.
        """
        if not text:
            return text
        targets: list[str] = []
        for link in paragraph._p.iter(qn("w:hyperlink")):
            rid = link.get(qn("r:id"))
            if not rid:
                continue
            try:
                relationship = self.docx.part.rels[rid]
            except (KeyError, AttributeError):
                continue
            if not getattr(relationship, "is_external", False):
                continue
            target = str(getattr(relationship, "target_ref", "") or "").strip()
            if target and target not in text and target not in targets:
                targets.append(target)
        if not targets:
            return text
        return f"{text} ({', '.join(targets)})"

    def _note_references(self, paragraph: Paragraph) -> list[tuple[str, str]]:
        references: list[tuple[str, str]] = []
        for kind, tag in (("footnote", "w:footnoteReference"), ("endnote", "w:endnoteReference")):
            for node in paragraph._p.iter(qn(tag)):
                note_id = node.get(qn("w:id"))
                if note_id and (kind, note_id) in self._notes:
                    references.append((kind, note_id))
        return references

    def _append_note_markers(self, paragraph: Paragraph, text: str) -> str:
        """Mark the places a footnote was attached, the way a reader sees them."""
        if not text or not self._notes:
            return text
        markers = "".join(f" [{note_id}]" for _, note_id in self._note_references(paragraph))
        return f"{text}{markers}" if markers else text

    def _emit_notes(self, paragraph: Paragraph) -> None:
        """Emit the text of the footnotes this paragraph refers to.

        Footnotes live in a separate part of the package and are invisible to a
        parser that only walks the body — yet they carry definitions, sources
        and caveats that questions are asked about. They are emitted right
        after the paragraph that references them, so retrieval keeps them
        together with the sentence they belong to.
        """
        for kind, note_id in self._note_references(paragraph):
            text = self._notes.get((kind, note_id))
            if not text:
                continue
            self.footnote_count += 1
            self.factory.add(
                f"[{note_id}] {text}",
                BlockType.TEXT,
                {"note_type": kind, "note_id": note_id},
            )

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

    def _advance_numbering(self, paragraph: Paragraph) -> str:
        """Advance this paragraph's list counter and return its marker.

        Empty for a paragraph that carries no numbering, for bullets, and for
        definitions the numbering module cannot resolve.
        """
        if not self._numbering.available:
            return ""
        num_pr = paragraph._p.pPr.numPr if paragraph._p.pPr is not None else None
        if num_pr is None or num_pr.numId is None or num_pr.numId.val is None:
            return ""
        ilvl = num_pr.ilvl
        level = int(ilvl.val) if ilvl is not None and ilvl.val is not None else 0
        return self._numbering.marker(str(num_pr.numId.val), level)

    @staticmethod
    def _list_is_ordered(paragraph: Paragraph) -> bool:
        # Fallback for documents whose numbering.xml is missing or unreadable:
        # visible numbering in the text is the only remaining signal.
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

    @staticmethod
    def _cell_text(cell: Any) -> str:
        """Cell text with its equations, which ``cell.text`` does not include."""
        equations = outermost_math(cell._tc)
        if not equations:
            return cell.text
        inside = {node for math in equations for node in math.iter() if node is not math}
        parts: list[str] = []
        for node in cell._tc.iter():
            if node in inside:
                continue
            if is_math(node):
                latex = omml_to_latex(node)
                if latex:
                    parts.append(f" ${latex}$ ")
            elif node.tag == qn("w:t"):
                parts.append(node.text or "")
            elif node.tag in (qn("w:br"), qn("w:cr"), qn("w:p")):
                parts.append(" ")
        return "".join(parts)

    def _handle_table(self, table: Table, nested_in: int | None = None) -> None:
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
                cells.append(clean_text(self._cell_text(cell)).replace("\n", " "))
            rows.append(cells)
        markdown = table_to_markdown(rows)
        if markdown:
            self.table_count += 1
            metadata: dict[str, Any] = {
                "table_index": self.table_count,
                "row_count": len(rows),
            }
            if nested_in is not None:
                metadata["nested_in_table"] = nested_in
            self.factory.add(markdown, BlockType.TABLE, metadata)
            self._last_figure = None
            parent_index = self.table_count
        else:
            parent_index = nested_in

        # A table inside a cell is invisible in the parent grid (``cell.text``
        # only reads the cell's own paragraphs), so its rows would be lost.
        # Each one is emitted as its own table right after its parent.
        for nested in self._nested_tables(table):
            self._handle_table(nested, nested_in=parent_index)

    def _nested_tables(self, table: Table) -> list[Table]:
        nested: list[Table] = []
        for cell in table._tbl.iter(qn("w:tc")):
            for child in cell.iterchildren(qn("w:tbl")):
                nested.append(Table(child, self.docx))
        return nested


def _repeats_marker(text: str, marker: str) -> bool:
    """Whether the text already opens with the number Word would render.

    Authors regularly type "1.2" into a paragraph that *also* carries automatic
    numbering (usually after pasting from another document). Prefixing the
    computed marker there would produce "1.2 1.2 Установка".
    """
    digits = marker.strip(" .)")
    if not digits:
        return False
    head = text.lstrip()[: len(digits) + 1]
    if not head.startswith(digits):
        return False
    # "1" must not match the start of "1996 год": a real marker is followed by
    # its punctuation or by the space before the text.
    return len(head) == len(digits) or head[len(digits)] in " .)"


def _collect_notes(docx: Any) -> dict[tuple[str, str], str]:
    """Text of every footnote and endnote in the package, keyed by kind and id.

    Both live in their own parts (``footnotes.xml`` / ``endnotes.xml``) that a
    body walk never reaches. Word's own bookkeeping notes (the separator and
    continuation marks) carry no content and are skipped.
    """
    notes: dict[tuple[str, str], str] = {}
    for kind, relationship, container in (
        ("footnote", RT.FOOTNOTES, "w:footnote"),
        ("endnote", RT.ENDNOTES, "w:endnote"),
    ):
        try:
            part = docx.part.part_related_by(relationship)
            root = etree.fromstring(part.blob)
        except Exception:  # noqa: BLE001 - no such part, or unreadable XML
            continue
        for node in root.iter(qn(container)):
            note_id = node.get(qn("w:id"))
            if note_id is None or node.get(qn("w:type")) in ("separator", "continuationSeparator"):
                continue
            text = clean_text(" ".join(t.text or "" for t in node.iter(qn("w:t"))))
            if text:
                notes[(kind, note_id)] = text
    return notes
