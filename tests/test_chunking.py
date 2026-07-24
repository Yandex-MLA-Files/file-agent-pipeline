import pytest

from file_agent.chunking import chunk_document
from file_agent.document import Block, Document


def test_short_block_becomes_single_chunk():
    document = Document(
        file_name="notes.md",
        file_type="md",
        blocks=[
            Block(
                id="block-1",
                text="Short text",
                type="markdown",
                metadata={},
            )
        ],
    )

    chunks = chunk_document(document, max_chars=100, overlap=10)

    assert len(chunks) == 1
    assert chunks[0].id == "block-1-chunk-1"
    assert chunks[0].text == "Short text"


def test_long_block_splits_into_multiple_chunks():
    document = Document(
        file_name="long.md",
        file_type="md",
        blocks=[
            Block(
                id="block-1",
                text="abcdefghij",
                type="markdown",
                metadata={},
            )
        ],
    )

    chunks = chunk_document(document, max_chars=4, overlap=1)

    assert [chunk.text for chunk in chunks] == ["abcd", "defg", "ghij", "j"]


def test_chunk_metadata_is_preserved():
    document = Document(
        file_name="sample.pdf",
        file_type="pdf",
        blocks=[
            Block(
                id="page-1",
                text="PDF text",
                type="pdf_page",
                metadata={
                    "source_file": "sample.pdf",
                    "page_number": 1,
                },
            )
        ],
    )

    chunks = chunk_document(document, max_chars=100, overlap=10)

    assert chunks[0].metadata == {
        "source_file": "sample.pdf",
        "page_number": 1,
        "file_type": "pdf",
        "block_ids": ["page-1"],
        "block_type": "pdf_page",
        "block_types": ["pdf_page"],
    }


def test_chunk_metadata_preserves_format_specific_coordinates():
    document = Document(
        file_name="slides.pptx",
        file_type="pptx",
        blocks=[
            Block(
                id="slide-2",
                text="Slide text",
                type="slide",
                metadata={
                    "slide_number": 2,
                    "sheet_name": "Summary",
                },
            )
        ],
    )

    chunks = chunk_document(document, max_chars=100, overlap=10)

    assert chunks[0].metadata == {
        "slide_number": 2,
        "sheet_name": "Summary",
        "source_file": "slides.pptx",
        "file_type": "pptx",
        "block_ids": ["slide-2"],
        "block_type": "slide",
        "block_types": ["slide"],
    }


def test_small_blocks_are_packed_into_fewer_chunks():
    # The core regression: many tiny Docling-style blocks must not become many
    # tiny chunks — they should be packed up to the size budget.
    blocks = [
        Block(id=f"b{i}", text=f"Sentence number {i}.", type="text", metadata={}) for i in range(20)
    ]
    document = Document(file_name="doc.pdf", file_type="pdf", blocks=blocks)

    chunks = chunk_document(document, max_chars=100, overlap=20)

    # 20 blocks of ~19 chars pack into a handful of ~100-char chunks, not 20.
    assert len(chunks) < len(blocks)
    assert all(len(chunk.text) <= 100 for chunk in chunks)
    assert max(len(chunk.text) for chunk in chunks) > 50
    # Every source block is represented across the chunks' metadata.
    covered = {bid for chunk in chunks for bid in chunk.metadata["block_ids"]}
    assert covered == {f"b{i}" for i in range(20)}


def test_heading_is_recorded_as_section():
    from file_agent.document import BlockType

    blocks = [
        Block(id="h1", text="Overview", type="heading", block_type=BlockType.HEADING),
        Block(
            id="p1", text="Body of the overview section.", type="text", block_type=BlockType.TEXT
        ),
    ]
    document = Document(file_name="doc.pdf", file_type="pdf", blocks=blocks)

    chunks = chunk_document(document, max_chars=1000, overlap=100)

    assert len(chunks) == 1
    assert "Overview" in chunks[0].text
    assert "Body of the overview section." in chunks[0].text
    assert chunks[0].metadata["section"] == "Overview"


def test_overlap_must_be_smaller_than_max_chars():
    document = Document(file_name="notes.md", file_type="md", blocks=[])

    with pytest.raises(ValueError, match="overlap must be smaller"):
        chunk_document(document, max_chars=10, overlap=10)
