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


def test_long_section_repeats_heading_in_each_chunk():
    from file_agent.document import BlockType

    heading = Block(id="h1", text="Deep Section", type="heading", block_type=BlockType.HEADING)
    body = [
        Block(
            id=f"p{i}",
            text=f"Paragraph {i} with enough words to fill up some space here. ",
            type="text",
            block_type=BlockType.TEXT,
        )
        for i in range(20)
    ]
    document = Document(file_name="doc.pdf", file_type="pdf", blocks=[heading, *body])

    chunks = chunk_document(document, max_chars=300, overlap=50)

    assert len(chunks) > 1
    # Every continuation chunk carries the heading as a breadcrumb for retrieval.
    assert all("Deep Section" in chunk.text for chunk in chunks)
    assert all(chunk.metadata["section"] == "Deep Section" for chunk in chunks)


def test_overlap_must_be_smaller_than_max_chars():
    document = Document(file_name="notes.md", file_type="md", blocks=[])

    with pytest.raises(ValueError, match="overlap must be smaller"):
        chunk_document(document, max_chars=10, overlap=10)


class _CharTokenizer:
    """Test double: one "token" per character, so a small max_tokens is easy to
    reason about without pulling in a real HF tokenizer."""

    def encode(self, text: str, **kwargs) -> list[int]:
        return list(range(len(text)))


def test_oversized_heading_is_dropped_instead_of_starving_the_budget():
    # Regression for a real corpus bug: Docling misclassified a long sentence
    # as a HEADING, and that "heading" alone nearly filled the token budget,
    # leaving ~0 room for content. _fit_windows then degenerated into a
    # near-empty chunk per character of the remaining text (a 527-char
    # heading turned ~30 ordinary blocks into 3855 chunks). A heading this
    # large relative to the budget must be dropped as a breadcrumb, not
    # crowd out the content it's supposed to introduce.
    #
    # max_tokens=40 with a 1-char-per-token tokenizer: a 36-char heading
    # clears the HEADER_REPEAT_MAX_RATIO (0.25) drop threshold of 10 tokens
    # but still fits in a single chunk on its own (36 <= 40) - it's only
    # pathological once reserved as a breadcrumb, which leaves just
    # max(1, 40 - 36 - separator) = ~2 tokens for every paragraph's content.
    from file_agent.document import BlockType

    oversized_heading = Block(id="h1", text="H" * 36, type="heading", block_type=BlockType.HEADING)
    body = [
        Block(id=f"p{i}", text=f"Paragraph {i} here.", type="text", block_type=BlockType.TEXT)
        for i in range(20)
    ]
    document = Document(file_name="doc.docx", file_type="docx", blocks=[oversized_heading, *body])

    chunks = chunk_document(
        document, max_chars=1000, overlap=5, max_tokens=40, tokenizer=_CharTokenizer()
    )

    # Sane output (roughly one chunk per couple of paragraphs), not a chunk
    # explosion (the same shape of bug produced 3855 chunks from ~30 blocks).
    assert len(chunks) < 15
    # The heading still shows up once as ordinary packed content, but it is
    # not repeated as a breadcrumb into every other chunk.
    heading_text = "H" * 36
    occurrences = sum(chunk.text.count(heading_text) for chunk in chunks)
    assert occurrences <= 1
    assert all(chunk.metadata.get("section") is None for chunk in chunks)
