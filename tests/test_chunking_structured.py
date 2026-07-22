from file_agent.chunking import chunk_document
from file_agent.document import Block, BlockType, Document


def test_table_block_is_not_split():
    long_table = "| a | b |\n| - | - |\n" + "\n".join(f"| {i} | {i} |" for i in range(500))
    document = Document(
        file_name="doc.pdf",
        file_type="pdf",
        blocks=[
            Block(
                id="t1",
                text=long_table,
                type="table",
                block_type=BlockType.TABLE,
                page_number=3,
            )
        ],
    )

    chunks = chunk_document(document, max_chars=100, overlap=10)

    assert len(chunks) == 1
    assert chunks[0].text == long_table
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
