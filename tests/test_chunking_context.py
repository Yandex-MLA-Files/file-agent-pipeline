from file_agent.chunking import Chunk, chunk_document
from file_agent.document import Block, BlockType, Document
from file_agent.qa import build_context_from_results
from file_agent.retrieval import SearchResult


def _long_section_document():
    heading = Block(id="h1", text="Benchmark", type="heading", block_type=BlockType.HEADING)
    body = [
        Block(
            id=f"p{i}",
            text=f"Fact number {i}: the benchmark uses public datasets as its source. ",
            type="text",
            block_type=BlockType.TEXT,
        )
        for i in range(30)
    ]
    return Document(file_name="doc.pdf", file_type="pdf", blocks=[heading, *body])


def test_split_section_chunks_carry_parent_context():
    document = _long_section_document()

    chunks = chunk_document(document, max_chars=200, overlap=20)

    assert len(chunks) > 1
    for chunk in chunks:
        parent = chunk.metadata.get("context")
        assert parent, "every piece of a split section must link to its parent passage"
        # The parent holds the surrounding section, so it is strictly larger
        # than the piece and contains its heading.
        assert len(parent) > len(chunk.text.replace("Benchmark\n\n", ""))
        assert "Benchmark" in parent


def test_single_chunk_sections_have_no_parent_context():
    document = Document(
        file_name="doc.pdf",
        file_type="pdf",
        blocks=[
            Block(id="h1", text="Small", type="heading", block_type=BlockType.HEADING),
            Block(id="p1", text="Tiny body.", type="text", block_type=BlockType.TEXT),
        ],
    )

    chunks = chunk_document(document, max_chars=1000, overlap=100)

    assert len(chunks) == 1
    assert "context" not in chunks[0].metadata


def test_split_table_pieces_carry_full_table_as_context():
    header = "| metric | value |\n| - | - |"
    rows = [f"| metric-{i} | {i * 3} |" for i in range(120)]
    table = header + "\n" + "\n".join(rows)
    document = Document(
        file_name="doc.pdf",
        file_type="pdf",
        blocks=[Block(id="t1", text=table, type="table", block_type=BlockType.TABLE)],
    )

    chunks = chunk_document(document, max_chars=300, overlap=30)

    assert len(chunks) > 1
    for chunk in chunks:
        parent = chunk.metadata.get("context")
        assert parent and parent.startswith("| metric | value |")


def test_empty_table_rows_are_dropped():
    table = "| a | b |\n| - | - |\n|  |  |\n| 1 | 2 |\n|   |   |"
    document = Document(
        file_name="doc.pdf",
        file_type="pdf",
        blocks=[
            Block(id="t1", text=table * 30, type="table", block_type=BlockType.TABLE),
        ],
    )

    chunks = chunk_document(document, max_chars=200, overlap=20)

    for chunk in chunks:
        for line in chunk.text.splitlines():
            if "|" in line and set(line.strip()) <= set("|-: "):
                continue  # separator row
            if "|" in line:
                assert any(cell.strip() for cell in line.split("|"))


def _data_rows(chunk_text: str) -> list[str]:
    return [
        line
        for line in chunk_text.splitlines()
        if "|" in line
        and not set(line.strip()) <= set("|-: ")
        and any(cell.strip() for cell in line.split("|"))
    ]


def test_degenerate_table_pieces_stay_informative():
    # Docling sometimes exports a wide table as a giant header row and no data
    # rows. Such a table cannot be kept whole (it would overflow the encoder),
    # but every piece must still carry content and link back to the full table.
    giant_header = "| " + " ".join(f"col{i}" for i in range(120)) + " |"
    table = giant_header + "\n|" + "-" * 200 + "|"
    document = Document(
        file_name="doc.pdf",
        file_type="pdf",
        blocks=[Block(id="t1", text=table, type="table", block_type=BlockType.TABLE)],
    )

    chunks = chunk_document(document, max_chars=200, overlap=20)

    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.text.strip(), "no empty pieces"
        assert not set(chunk.text.strip()) <= set("|-: "), "no separator-only pieces"
        assert chunk.metadata.get("context"), "every piece links to the full table"


def test_huge_header_is_not_repeated_so_rows_survive():
    # Header nearly as large as the budget: repeating it would leave no room for
    # data, which produced "header-only" chunks in real papers.
    header = "| " + " | ".join(f"column-name-{i}" for i in range(20)) + " |"
    separator = "|" + "|".join("---" for _ in range(20)) + "|"
    rows = [f"| row-{i} | " + " | ".join(str(i * j) for j in range(19)) + " |" for i in range(40)]
    table = header + "\n" + separator + "\n" + "\n".join(rows)
    document = Document(
        file_name="doc.pdf",
        file_type="pdf",
        blocks=[Block(id="t1", text=table, type="table", block_type=BlockType.TABLE)],
    )

    chunks = chunk_document(document, max_chars=300, overlap=30)

    assert len(chunks) > 1
    # Every emitted piece must carry actual data, never just a header.
    for chunk in chunks:
        assert _data_rows(chunk.text), f"header-only piece: {chunk.text[:120]!r}"
    # And the full table remains available to the LLM.
    assert all(chunk.metadata.get("context") for chunk in chunks)


def test_qa_context_prefers_parent_and_deduplicates():
    parent = "Benchmark section: datasets come from public sources and PG19."
    results = [
        SearchResult(
            chunk=Chunk(id="c1", text="piece one", metadata={"context": parent, "page_number": 3}),
            score=0.9,
        ),
        SearchResult(
            chunk=Chunk(id="c2", text="piece two", metadata={"context": parent, "page_number": 3}),
            score=0.8,
        ),
        SearchResult(
            chunk=Chunk(id="c3", text="unrelated chunk", metadata={"page_number": 7}),
            score=0.5,
        ),
    ]

    context = build_context_from_results(results)

    # The parent passage appears exactly once, and raw pieces are not shown.
    assert context.count(parent) == 1
    assert "piece one" not in context
    assert "unrelated chunk" in context
    # The context payload itself is not echoed into the metadata line.
    assert "context=" not in context
