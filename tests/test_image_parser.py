from file_agent.document import BlockType
from file_agent.parsers.image_parser import ImageParser


def test_image_parser_returns_a_single_figure_block(tmp_path):
    file_path = tmp_path / "diagram.png"
    file_path.write_bytes(b"not a real png, parser never opens the file")

    document = ImageParser().parse(file_path)

    assert document.file_name == "diagram.png"
    assert document.file_type == "png"
    assert len(document.blocks) == 1
    block = document.blocks[0]
    assert block.id == "block-1"
    assert block.type == "figure"
    assert block.block_type == BlockType.FIGURE
    assert block.page_number == 1
    assert "diagram.png" in block.text
