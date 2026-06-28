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
        "block_id": "page-1",
        "block_type": "pdf_page",
        "page_number": 1,
    }


def test_overlap_must_be_smaller_than_max_chars():
    document = Document(file_name="notes.md", file_type="md", blocks=[])

    with pytest.raises(ValueError, match="overlap must be smaller"):
        chunk_document(document, max_chars=10, overlap=10)
