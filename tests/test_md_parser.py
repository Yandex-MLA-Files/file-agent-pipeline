from file_agent.parsers.md_parser import MarkdownParser


def test_markdown_parser_returns_document(tmp_path):
    file_path = tmp_path / "notes.md"
    file_path.write_text("# Title\n\nHello, Markdown!", encoding="utf-8")

    document = MarkdownParser().parse(file_path)

    assert document.file_name == "notes.md"
    assert document.file_type == "md"
    assert [block.block_type for block in document.blocks] == [
        BlockType.HEADING,
        BlockType.TEXT,
        BlockType.HEADING,
        BlockType.TEXT,
    ]
    assert document.blocks[0].text == "Title"
    assert document.blocks[0].metadata["hierarchy_level"] == 1
    assert document.blocks[1].text == "Intro text."
    assert document.blocks[2].text == "Details"
    assert document.blocks[2].metadata["hierarchy_level"] == 2
    assert document.blocks[3].text == "More text."
    assert [block.id for block in document.blocks] == [
        "block-1",
        "block-2",
        "block-3",
        "block-4",
    ]


def test_markdown_parser_builds_table_of_contents(tmp_path):
    file_path = tmp_path / "notes.md"
    file_path.write_text("# One\n\ntext\n\n### Deep\n\nmore", encoding="utf-8")

    document = MarkdownParser().parse(file_path)

    toc = document.metadata["table_of_contents"]
    assert [(entry["title"], entry["level"]) for entry in toc] == [("One", 1), ("Deep", 3)]
    assert document.metadata["parsing_method"] == "markdown"


def test_markdown_parser_ignores_hashes_inside_code_fences(tmp_path):
    file_path = tmp_path / "notes.md"
    file_path.write_text(
        "# Real heading\n\n```bash\n# just a comment\ng++ -v test.cpp\n```\n\ntail text",
        encoding="utf-8",
    )

    document = MarkdownParser().parse(file_path)

    headings = [block for block in document.blocks if block.block_type == BlockType.HEADING]
    assert [block.text for block in headings] == ["Real heading"]
    code = [block for block in document.blocks if block.block_type == BlockType.CODE]
    assert len(code) == 1
    assert "# just a comment" in code[0].text
    assert code[0].metadata["language"] == "bash"
    assert document.blocks[-1].text == "tail text"


def test_markdown_parser_strips_html_tags_from_heading_titles(tmp_path):
    file_path = tmp_path / "notes.md"
    file_path.write_text(
        '## <span style="background:red">1.</span> Sorting\n\ntext',
        encoding="utf-8",
    )

    document = MarkdownParser().parse(file_path)

    assert document.blocks[0].text == "1. Sorting"


def test_markdown_parser_without_headings_returns_single_block(tmp_path):
    file_path = tmp_path / "notes.md"
    file_path.write_text("Just a paragraph without any headings.", encoding="utf-8")

    document = MarkdownParser().parse(file_path)

    assert len(document.blocks) == 1
    assert document.blocks[0].id == "block-1"
    assert document.blocks[0].text == "Just a paragraph without any headings."
    assert document.metadata["table_of_contents"] == []


def test_markdown_parser_emits_tables_lists_and_setext_headings(tmp_path):
    file_path = tmp_path / "notes.md"
    file_path.write_text(
        "---\ntitle: Front matter title\n---\n"
        "Setext title\n===\n\n"
        "| a | b |\n| --- | --- |\n| 1 | 2 |\n\n"
        "- one\n- two\n  continued\n\n"
        "![alt text](img.png)\n",
        encoding="utf-8",
    )

    document = MarkdownParser().parse(file_path)

    assert document.metadata["title"] == "Front matter title"
    types = [block.block_type for block in document.blocks]
    assert types == [BlockType.HEADING, BlockType.TABLE, BlockType.LIST, BlockType.FIGURE]
    assert document.blocks[0].text == "Setext title"
    assert document.blocks[1].text.splitlines()[0] == "| a | b |"
    assert document.blocks[2].text == "- one\n- two continued"
    assert document.blocks[3].text == "alt text"
