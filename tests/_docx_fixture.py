"""Builds a DOCX exercising footnotes, text frames, nested tables and numbering.

Written as raw OOXML because python-docx cannot create any of these parts.
"""

import zipfile
from pathlib import Path

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
CT = "http://schemas.openxmlformats.org/package/2006/content-types"
PR = "http://schemas.openxmlformats.org/package/2006/relationships"

DOCUMENT = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="{W}" xmlns:r="{R}">
  <w:body>
    <w:p><w:pPr><w:pStyle w:val="Heading1"/></w:pPr><w:r><w:t>Регламент</w:t></w:r></w:p>
    <w:p>
      <w:r><w:t>Основной абзац со сноской</w:t></w:r>
      <w:r><w:footnoteReference w:id="2"/></w:r>
      <w:r><w:t>.</w:t></w:r>
    </w:p>
    <w:p><w:pPr><w:numPr><w:ilvl w:val="0"/><w:numId w:val="1"/></w:numPr></w:pPr>
      <w:r><w:t>Первый пункт</w:t></w:r></w:p>
    <w:p><w:pPr><w:numPr><w:ilvl w:val="1"/><w:numId w:val="1"/></w:numPr></w:pPr>
      <w:r><w:t>Подпункт один</w:t></w:r></w:p>
    <w:p><w:pPr><w:numPr><w:ilvl w:val="1"/><w:numId w:val="1"/></w:numPr></w:pPr>
      <w:r><w:t>Подпункт два</w:t></w:r></w:p>
    <w:p><w:pPr><w:numPr><w:ilvl w:val="0"/><w:numId w:val="1"/></w:numPr></w:pPr>
      <w:r><w:t>Второй пункт</w:t></w:r></w:p>
    <w:p>
      <w:r><w:pict><v:shape xmlns:v="urn:schemas-microsoft-com:vml"><v:textbox><w:txbxContent>
        <w:p><w:r><w:t>Важно: срок хранения — 5 лет</w:t></w:r></w:p>
      </w:txbxContent></v:textbox></v:shape></w:pict></w:r>
    </w:p>
    <w:tbl>
      <w:tr>
        <w:tc><w:p><w:r><w:t>Раздел</w:t></w:r></w:p></w:tc>
        <w:tc><w:p><w:r><w:t>Содержание</w:t></w:r></w:p></w:tc>
      </w:tr>
      <w:tr>
        <w:tc><w:p><w:r><w:t>Приложение</w:t></w:r></w:p></w:tc>
        <w:tc>
          <w:p><w:r><w:t>См. вложенную таблицу</w:t></w:r></w:p>
          <w:tbl>
            <w:tr><w:tc><w:p><w:r><w:t>Код</w:t></w:r></w:p></w:tc>
                  <w:tc><w:p><w:r><w:t>Срок</w:t></w:r></w:p></w:tc></w:tr>
            <w:tr><w:tc><w:p><w:r><w:t>А-1</w:t></w:r></w:p></w:tc>
                  <w:tc><w:p><w:r><w:t>3 года</w:t></w:r></w:p></w:tc></w:tr>
          </w:tbl>
        </w:tc>
      </w:tr>
    </w:tbl>
  </w:body>
</w:document>"""

FOOTNOTES = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:footnotes xmlns:w="{W}">
  <w:footnote w:id="0" w:type="separator"><w:p><w:r><w:separator/></w:r></w:p></w:footnote>
  <w:footnote w:id="2"><w:p><w:r>
    <w:t>Утверждён приказом № 35 от 30.12.2025.</w:t></w:r></w:p></w:footnote>
</w:footnotes>"""

NUMBERING = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:numbering xmlns:w="{W}">
  <w:abstractNum w:abstractNumId="0">
    <w:lvl w:ilvl="0"><w:start w:val="1"/><w:numFmt w:val="decimal"/>
      <w:lvlText w:val="%1."/></w:lvl>
    <w:lvl w:ilvl="1"><w:start w:val="1"/><w:numFmt w:val="decimal"/>
      <w:lvlText w:val="%1.%2."/></w:lvl>
  </w:abstractNum>
  <w:num w:numId="1"><w:abstractNumId w:val="0"/></w:num>
</w:numbering>"""

STYLES = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="{W}">
  <w:style w:type="paragraph" w:styleId="Heading1"><w:name w:val="heading 1"/></w:style>
</w:styles>"""

OOXML = "application/vnd.openxmlformats-officedocument.wordprocessingml"
PKG = "application/vnd.openxmlformats-package"

CONTENT_TYPES = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="{CT}">
  <Default Extension="rels" ContentType="{PKG}.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="{OOXML}.document.main+xml"/>
  <Override PartName="/word/styles.xml" ContentType="{OOXML}.styles+xml"/>
  <Override PartName="/word/numbering.xml" ContentType="{OOXML}.numbering+xml"/>
  <Override PartName="/word/footnotes.xml" ContentType="{OOXML}.footnotes+xml"/>
</Types>"""

ROOT_RELS = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="{PR}">
  <Relationship Id="rId1" Type="{R}/officeDocument" Target="word/document.xml"/>
</Relationships>"""

DOC_RELS = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="{PR}">
  <Relationship Id="rId1" Type="{R}/styles" Target="styles.xml"/>
  <Relationship Id="rId2" Type="{R}/numbering" Target="numbering.xml"/>
  <Relationship Id="rId3" Type="{R}/footnotes" Target="footnotes.xml"/>
</Relationships>"""


# A manual (lab-report style) document: numbered *headings* with numbered
# sub-items under them, which is how Word documents nest section numbering.
NUMBERED_SECTION = """
    <w:p><w:pPr><w:pStyle w:val="Heading1"/>
      <w:numPr><w:ilvl w:val="0"/><w:numId w:val="1"/></w:numPr></w:pPr>
      <w:r><w:t>{title}</w:t></w:r></w:p>
    <w:p><w:pPr><w:numPr><w:ilvl w:val="1"/><w:numId w:val="1"/></w:numPr></w:pPr>
      <w:r><w:t>{first}</w:t></w:r></w:p>
    <w:p><w:pPr><w:numPr><w:ilvl w:val="1"/><w:numId w:val="1"/></w:numPr></w:pPr>
      <w:r><w:t>{second}</w:t></w:r></w:p>
"""

_SECTIONS = NUMBERED_SECTION.format(
    title="Цель и содержание", first="Изучить MongoDB", second="Установить Compass"
) + NUMBERED_SECTION.format(
    title="Порядок выполнения", first="Создать базу", second="Проверить запросы"
)

NUMBERED_HEADINGS = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="{W}" xmlns:r="{R}">
  <w:body>{_SECTIONS}</w:body>
</w:document>"""


def _write(out: Path, document: str) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as package:
        package.writestr("[Content_Types].xml", CONTENT_TYPES)
        package.writestr("_rels/.rels", ROOT_RELS)
        package.writestr("word/document.xml", document)
        package.writestr("word/_rels/document.xml.rels", DOC_RELS)
        package.writestr("word/styles.xml", STYLES)
        package.writestr("word/numbering.xml", NUMBERING)
        package.writestr("word/footnotes.xml", FOOTNOTES)
    return out


def write_fixture(out: Path) -> Path:
    """Write the fixture document to ``out`` and return the path."""
    return _write(out, DOCUMENT)


def write_numbered_headings(out: Path) -> Path:
    """Write a document whose sections are numbered by ``numbering.xml``."""
    return _write(out, NUMBERED_HEADINGS)
