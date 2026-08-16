from pptx import Presentation
from pptx.util import Inches

from file_agent.parsers.pptx_parser import PPTXParser


def create_presentation(file_path):
    presentation = Presentation()

    title_slide = presentation.slides.add_slide(presentation.slide_layouts[1])
    title_slide.shapes.title.text = "Project Overview"
    title_slide.placeholders[1].text = "This project is about document processing."

    table_slide = presentation.slides.add_slide(presentation.slide_layouts[5])
    table_slide.shapes.title.text = "Metrics"
    table = table_slide.shapes.add_table(
        rows=2,
        cols=2,
        left=Inches(1),
        top=Inches(1.5),
        width=Inches(4),
        height=Inches(1),
    ).table
    table.cell(0, 0).text = "Name"
    table.cell(0, 1).text = "Value"
    table.cell(1, 0).text = "Accuracy"
    table.cell(1, 1).text = "0.85"

    grouped_slide = presentation.slides.add_slide(presentation.slide_layouts[5])
    grouped_slide.shapes.title.text = "Grouped content"
    group = grouped_slide.shapes.add_group_shape()
    grouped_text = group.shapes.add_textbox(
        left=Inches(1),
        top=Inches(1.5),
        width=Inches(4),
        height=Inches(1),
    )
    grouped_text.text = "Text nested inside a PowerPoint group."

    presentation.save(file_path)


def test_pptx_parser_creates_block_for_each_slide(tmp_path):
    file_path = tmp_path / "sample.pptx"
    create_presentation(file_path)

    document = PPTXParser().parse(file_path)

    assert document.file_name == "sample.pptx"
    assert document.file_type == "pptx"
    assert len(document.blocks) == 3
    assert document.blocks[0].type == "pptx_slide"
    assert document.blocks[1].type == "pptx_slide"


def test_pptx_parser_extracts_slide_text_and_tables(tmp_path):
    file_path = tmp_path / "sample.pptx"
    create_presentation(file_path)

    document = PPTXParser().parse(file_path)

    assert "Slide 1" in document.blocks[0].text
    assert "Title: Project Overview" in document.blocks[0].text
    assert "This project is about document processing." in document.blocks[0].text
    assert "Slide 2" in document.blocks[1].text
    assert "Title: Metrics" in document.blocks[1].text
    assert "Name\tValue" in document.blocks[1].text
    assert "Accuracy\t0.85" in document.blocks[1].text
    assert "Text nested inside a PowerPoint group." in document.blocks[2].text


def test_pptx_parser_preserves_slide_metadata(tmp_path):
    file_path = tmp_path / "sample.pptx"
    create_presentation(file_path)

    document = PPTXParser().parse(file_path)

    assert document.blocks[0].metadata["source_file"] == "sample.pptx"
    assert document.blocks[0].metadata["slide_number"] == 1
    assert document.blocks[0].metadata["shapes_count"] > 0
    assert document.blocks[1].metadata["source_file"] == "sample.pptx"
    assert document.blocks[1].metadata["slide_number"] == 2
    assert document.blocks[1].metadata["shapes_count"] > 0
