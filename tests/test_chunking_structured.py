from file_agent.chunking import chunk_document
from file_agent.document import Block, BlockType, Document


def test_small_table_is_kept_whole():
    table = "| a | b |\n| - | - |\n| 1 | 2 |\n| 3 | 4 |"
    document = Document(
        file_name="doc.pdf",
        file_type="pdf",
        blocks=[
            Block(id="t1", text=table, type="table", block_type=BlockType.TABLE, page_number=3)
        ],
    )

    chunks = chunk_document(document, max_chars=1000, overlap=100)

    assert len(chunks) == 1
    assert chunks[0].text == table
    assert chunks[0].metadata["block_type"] == "table"
    assert chunks[0].metadata["page_number"] == 3


def test_large_table_is_split_by_rows_with_header():
    header = "| a | b |\n| - | - |"
    rows = [f"| {i} | {i * i} |" for i in range(500)]
    long_table = header + "\n" + "\n".join(rows)
    document = Document(
        file_name="doc.pdf",
        file_type="pdf",
        blocks=[
            Block(id="t1", text=long_table, type="table", block_type=BlockType.TABLE, page_number=3)
        ],
    )

    chunks = chunk_document(document, max_chars=200, overlap=20)

    # A huge table is split so each piece fits an embedding window ...
    assert len(chunks) > 1
    assert all(len(chunk.text) <= 260 for chunk in chunks)
    # ... every piece repeats the header row, and all rows are preserved.
    assert all("| a | b |" in chunk.text for chunk in chunks)
    joined = "\n".join(chunk.text for chunk in chunks)
    assert "| 0 | 0 |" in joined
    assert "| 499 | 249001 |" in joined
    assert chunks[0].metadata["block_type"] == "table"
    assert chunks[0].metadata["page_number"] == 3


def test_structural_metadata_is_propagated():
    document = Document(
        file_name="doc.pdf",
        file_type="pdf",
        blocks=[
            Block(
                id="f1",
                text="[Image description]: a bar chart",
                type="figure",
                block_type=BlockType.FIGURE,
                page_number=2,
                bbox=(10.0, 20.0, 110.0, 120.0),
                vlm_description="a bar chart",
            )
        ],
    )

    chunks = chunk_document(document, max_chars=1000, overlap=100)

    metadata = chunks[0].metadata
    assert metadata["page_number"] == 2
    assert metadata["bbox"] == (10.0, 20.0, 110.0, 120.0)
    assert metadata["vlm_description"] == "a bar chart"
    assert metadata["block_type"] == "figure"


def test_untyped_blocks_keep_legacy_behavior():
    # A block with no structural type must chunk exactly like before.
    document = Document(
        file_name="notes.md",
        file_type="md",
        blocks=[Block(id="b1", text="abcdefghij", type="markdown", metadata={})],
    )

    chunks = chunk_document(document, max_chars=4, overlap=1)

    assert [chunk.text for chunk in chunks] == ["abcd", "defg", "ghij", "j"]
