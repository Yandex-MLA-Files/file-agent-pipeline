from file_agent.parsers.md_parser import MarkdownParser


def test_markdown_parser_returns_document(tmp_path):
    file_path = tmp_path / "notes.md"
    file_path.write_text("# Title\n\nHello, Markdown!", encoding="utf-8")

    document = MarkdownParser().parse(file_path)

    assert document.file_name == "notes.md"
    assert document.file_type == "md"
    assert len(document.blocks) == 1
    assert document.blocks[0].id == "block-1"
    assert document.blocks[0].text == "# Title\n\nHello, Markdown!"
    assert document.blocks[0].type == "markdown"
    assert document.blocks[0].metadata == {"block_type": "markdown"}
