from file_agent.document import Block, BlockType, Document


def test_block_legacy_constructor_still_works():
    # The interface used by every legacy parser and by main's tests.
    block = Block(id="b1", text="hello", type="markdown", metadata={"source_file": "a.md"})

    assert block.block_type is None
    assert block.page_number is None
    assert block.to_markdown() == "hello"


def test_document_defaults_are_populated():
    document = Document(file_name="a.md", file_type="md", blocks=[])

    assert document.metadata["table_of_contents"] == []
    assert document.metadata["total_pages"] == 0
    assert document.metadata["parsing_method"] == "unknown"


def test_heading_block_renders_with_level():
    block = Block(
        id="h1",
        text="Chapter 1",
        type="heading",
        metadata={"hierarchy_level": 2},
        block_type=BlockType.HEADING,
    )

    assert block.to_markdown() == "## Chapter 1"


def test_figure_block_renders_with_description():
    block = Block(
        id="f1",
        text="",
        type="figure",
        block_type=BlockType.FIGURE,
        vlm_description="A flowchart of the login process",
    )

    assert block.to_markdown() == "![A flowchart of the login process]()"


def test_build_table_of_contents_from_headings():
    blocks = [
        Block(id="h1", text="Intro", type="heading", block_type=BlockType.HEADING, page_number=1),
        Block(id="p1", text="body text", type="text", block_type=BlockType.TEXT, page_number=1),
        Block(id="h2", text="Details", type="heading", block_type=BlockType.HEADING, page_number=2),
    ]
    document = Document(file_name="doc.pdf", file_type="pdf", blocks=blocks)

    toc = document.build_table_of_contents()

    assert [entry["title"] for entry in toc] == ["Intro", "Details"]
    assert toc[1]["page"] == 2
    assert document.metadata["table_of_contents"] == toc


def test_document_to_markdown_assembles_blocks():
    blocks = [
        Block(id="h1", text="Title", type="heading", block_type=BlockType.HEADING),
        Block(id="p1", text="Some paragraph.", type="text", block_type=BlockType.TEXT),
    ]
    document = Document(file_name="doc.pdf", file_type="pdf", blocks=blocks)

    markdown = document.to_markdown()

    assert "# Title" in markdown
    assert "Some paragraph." in markdown
