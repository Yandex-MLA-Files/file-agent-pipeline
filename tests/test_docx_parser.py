import io

import docx
from docx.shared import Pt
from PIL import Image

from file_agent.document import BlockType
from file_agent.parsers.docx_parser import DOCXParser
from file_agent.pipeline import parse_file


def _png_bytes(size=(200, 150)) -> bytes:
    buffer = io.BytesIO()
    image = Image.new("RGB", size, "white")
    # A gradient keeps the PNG above the "decorative 1-pixel image" size floor.
    for x in range(size[0]):
        for y in range(0, size[1], 3):
            image.putpixel((x, y), (x % 256, y % 256, (x * y) % 256))
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def create_docx(file_path):
    document = docx.Document()
    document.add_heading("Программа экзамена", level=1)
    paragraph = document.add_paragraph()
    paragraph.add_run("Целью ").bold = True
    paragraph.add_run("испытания является оценка уровня освоения ")
    paragraph.add_run("компетенций.").italic = True
    document.add_heading("1.2 Раздел второй", level=2)
    document.add_paragraph("Первый пункт", style="List Bullet")
    document.add_paragraph("Второй пункт", style="List Bullet")
    table = document.add_table(rows=2, cols=2)
    table.cell(0, 0).text = "Метрика"
    table.cell(0, 1).text = "Значение"
    table.cell(1, 0).text = "Точность"
    table.cell(1, 1).text = "0.9"
    document.add_picture(io.BytesIO(_png_bytes()))
    document.add_paragraph("Рисунок 1 – Схема процесса")
    code = document.add_paragraph()
    run = code.add_run("print('hi')")
    run.font.name = "Courier New"
    run.font.size = Pt(10)
    document.save(file_path)


def test_docx_parser_keeps_paragraphs_whole_and_types_blocks(tmp_path):
    file_path = tmp_path / "sample.docx"
    create_docx(file_path)

    document = DOCXParser().parse(file_path)

    assert document.file_type == "docx"
    assert document.metadata["parsing_method"] == "python-docx"
    assert document.metadata["title"] == "Программа экзамена"
    types = [b.block_type for b in document.blocks]
    assert types == [
        BlockType.HEADING,
        BlockType.TEXT,
        BlockType.HEADING,
        BlockType.LIST,
        BlockType.TABLE,
        BlockType.FIGURE,
        BlockType.CODE,
    ]
    # Runs with different formatting are one paragraph, not three fragments.
    assert document.blocks[1].text == "Целью испытания является оценка уровня освоения компетенций."
    assert document.blocks[0].metadata["hierarchy_level"] == 1
    assert document.blocks[2].metadata["hierarchy_level"] == 2
    assert document.blocks[3].text == "- Первый пункт\n- Второй пункт"
    assert document.blocks[4].text.splitlines() == [
        "| Метрика | Значение |",
        "| --- | --- |",
        "| Точность | 0.9 |",
    ]
    figure = document.blocks[5]
    assert figure.image_bytes is not None
    assert figure.metadata["caption"] == "Рисунок 1 – Схема процесса"
    assert figure.text == "Рисунок 1 – Схема процесса"
    assert document.blocks[6].text == "print('hi')"


def test_pipeline_routes_docx_to_python_docx_parser(tmp_path, monkeypatch):
    file_path = tmp_path / "sample.docx"
    create_docx(file_path)
    monkeypatch.setenv("VLM_BACKEND", "off")

    document = parse_file(file_path)

    assert document.metadata["parsing_method"] == "python-docx"
    assert document.blocks[0].block_type == BlockType.HEADING
