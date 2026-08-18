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
    windows = [c for c in chunks if not c.metadata.get("representation")]

    # A huge table is split so each piece fits an embedding window ...
    assert len(windows) > 1
    assert all(len(chunk.text) <= 260 for chunk in windows)
    # ... every piece repeats the header row, and all rows are preserved.
    assert all("| a | b |" in chunk.text for chunk in windows)
    joined = "\n".join(chunk.text for chunk in windows)
    assert "| 0 | 0 |" in joined
    assert "| 499 | 249001 |" in joined
    assert windows[0].metadata["block_type"] == "table"
    assert windows[0].metadata["page_number"] == 3


def test_table_rows_are_also_indexed_as_records(monkeypatch):
    """Every row is additionally indexed as "column: value" for lookup queries."""
    header = "| Chess.com Bullet | FIDE Regular |\n| --- | --- |"
    rows = [f"| {1000 + i * 10} | {1260 + i * 10} |" for i in range(20)]
    document = Document(
        file_name="ratings.xlsx",
        file_type="xlsx",
        blocks=[
            Block(
                id="t1",
                text=header + "\n" + "\n".join(rows),
                type="table",
                block_type=BlockType.TABLE,
                page_number=1,
            )
        ],
    )

    chunks = chunk_document(document, max_chars=400, overlap=40)
    records = [c for c in chunks if c.metadata.get("representation") == "row"]

    assert len(records) == 20
    # The query words and the answer sit side by side in one short passage.
    assert any("Chess.com Bullet: 1000; FIDE Regular: 1260" in c.text for c in records)
    assert [c.metadata["row_index"] for c in records] == list(range(1, 21))
    # ... and each record still hands the LLM the surrounding table.
    for record in records:
        assert "| Chess.com Bullet | FIDE Regular |" in record.metadata["context"]
        assert record.metadata["block_type"] == "table"

    monkeypatch.setenv("TABLE_ROW_RECORDS", "off")
    assert not [
        c
        for c in chunk_document(document, max_chars=400, overlap=40)
        if c.metadata.get("representation") == "row"
    ]


def test_row_records_skip_tiny_and_prose_tables():
    prose = "| Термин | Определение |\n| --- | --- |\n" + "\n".join(
        f"| термин {i} | {'очень длинное определение ' * 20} |" for i in range(6)
    )
    tiny = "| a | b |\n| --- | --- |\n| 1 | 2 |\n| 3 | 4 |"
    document = Document(
        file_name="doc.docx",
        file_type="docx",
        blocks=[
            Block(id="t1", text=prose, type="table", block_type=BlockType.TABLE),
            Block(id="t2", text=tiny, type="table", block_type=BlockType.TABLE),
        ],
    )

    chunks = chunk_document(document, max_chars=1000, overlap=100)

    assert not [c for c in chunks if c.metadata.get("representation") == "row"]


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


def test_semantic_split_cuts_at_topic_boundaries(monkeypatch):
    """With CHUNK_SEMANTIC_SPLIT=on, long prose is cut where the subject changes."""
    from file_agent import chunking

    topic_a = " ".join(f"Первое предложение про базы данных номер {i}." for i in range(14))
    topic_b = " ".join(f"Второе предложение про кулинарию номер {i}." for i in range(14))
    document = Document(
        file_name="doc.txt",
        file_type="txt",
        blocks=[
            Block(id="b1", text=topic_a + " " + topic_b, type="text", block_type=BlockType.TEXT)
        ],
    )

    monkeypatch.setenv("CHUNK_SEMANTIC_SPLIT", "on")
    # The boundary between the two topics is sentence 14.
    monkeypatch.setattr(chunking, "topic_boundaries", lambda sentences: {14})

    chunks = chunk_document(document, max_chars=400, overlap=40)
    texts = [c.text for c in chunks]

    assert len(texts) > 1
    # No chunk mixes the two topics: the cut lands exactly on the boundary.
    assert not any("баз" in text and "кулинари" in text for text in texts)


def test_semantic_split_is_off_by_default():
    from file_agent import chunking

    assert chunking._semantic_split_enabled() is False


def test_row_records_of_one_table_share_their_parent_passage():
    """Records of neighbouring rows must not print five slices of one table."""
    header = "| Регион | Выручка |\n| --- | --- |"
    rows = [f"| Регион {i} | {i * 100} |" for i in range(30)]
    document = Document(
        file_name="sales.xlsx",
        file_type="xlsx",
        blocks=[
            Block(
                id="t1",
                text=header + "\n" + "\n".join(rows),
                type="table",
                block_type=BlockType.TABLE,
            )
        ],
    )

    records = [
        c
        for c in chunk_document(document, max_chars=400, overlap=40)
        if c.metadata.get("representation") == "row"
    ]

    contexts = {c.metadata["context"] for c in records}
    assert len(records) == 30
    # The whole table fits the parent budget, so every record points at it once.
    assert len(contexts) == 1
    assert "| Регион 29 | 2900 |" in contexts.pop()


def test_pdf_tables_get_an_automatic_profile(monkeypatch):
    """ "Which revenue was the largest" needs every row at once, so it is precomputed."""
    table = "| Вид выручки | 2026 | 2025 |\n| --- | --- | --- |\n" + "\n".join(
        f"| Вид {i} | {i * 1000} | {i * 900} |" for i in range(1, 8)
    )
    document = Document(
        file_name="report.pdf",
        file_type="pdf",
        blocks=[
            Block(id="t1", text=table, type="table", block_type=BlockType.TABLE, page_number=4)
        ],
    )

    profiles = [
        c
        for c in chunk_document(document, max_chars=800, overlap=80)
        if c.metadata.get("representation") == "profile"
    ]

    assert len(profiles) == 1
    text = profiles[0].text
    assert "максимум 7 000 (Вид 7)" in text
    assert "строк: 7" in text
    assert profiles[0].metadata["page_number"] == 4

    monkeypatch.setenv("TABLE_PROFILES", "off")
    assert not [
        c
        for c in chunk_document(document, max_chars=800, overlap=80)
        if c.metadata.get("representation") == "profile"
    ]


def test_spreadsheet_tables_are_not_profiled_twice():
    """The XLSX parser already emits a profile block for its own tables."""
    table = "| Регион | Выручка |\n| --- | --- |\n" + "\n".join(
        f"| Регион {i} | {i * 100} |" for i in range(1, 9)
    )
    document = Document(
        file_name="sales.xlsx",
        file_type="xlsx",
        blocks=[
            Block(
                id="t1",
                text=table,
                type="table",
                block_type=BlockType.TABLE,
                metadata={"sheet_name": "Sales"},
            )
        ],
    )

    assert not [
        c
        for c in chunk_document(document, max_chars=800, overlap=80)
        if c.metadata.get("representation") == "profile"
    ]


def _big_table(rows: int, columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    separator = "|" + "|".join(["---"] * len(columns)) + "|"
    body = "\n".join(
        "| " + " | ".join(f"{column[:4]}{index}" for column in columns) + " |"
        for index in range(rows)
    )
    return f"{header}\n{separator}\n{body}"


def test_a_header_too_wide_to_repeat_is_abbreviated_not_dropped():
    """Continuation pieces of a wide table must still name their columns."""
    columns = [
        "Резерв переоценки инструментов хеджирования",
        "Нераспределенная прибыль и прочие резервы",
        "Неконтролирующие доли участия",
        "Итого капитал по группе на конец периода",
    ]
    document = Document(
        file_name="report.pdf",
        file_type="pdf",
        blocks=[
            Block(
                id="b1",
                text=_big_table(40, columns),
                type=BlockType.TABLE.value,
                metadata={"source_file": "report.pdf"},
                block_type=BlockType.TABLE,
            )
        ],
        metadata={},
    )

    chunks = chunk_document(document=document, max_chars=400, overlap=0)
    windows = [c for c in chunks if (c.metadata or {}).get("representation") != "row"]

    assert len(windows) > 1
    for chunk in windows:
        assert "Резерв пере" in chunk.text
        assert "|-|" in chunk.text  # still a Markdown table, just a terse one
    # Abbreviated, not verbatim: the full column name would not fit twice.
    assert "Резерв переоценки инструментов хеджирования" not in windows[-1].text


def test_a_heading_with_nothing_under_it_does_not_become_a_chunk():
    """The running header of a financial report repeats on every page."""
    blocks = []
    for page in (1, 2, 3):
        blocks.append(
            Block(
                id=f"h{page}",
                text="ОАО «Российские железные дороги»",
                type=BlockType.HEADING.value,
                metadata={"source_file": "report.pdf", "hierarchy_level": 1},
                block_type=BlockType.HEADING,
                page_number=page,
            )
        )
    blocks.append(
        Block(
            id="body",
            text="Выручка за период составила 1 234 млн рублей.",
            type=BlockType.TEXT.value,
            metadata={"source_file": "report.pdf"},
            block_type=BlockType.TEXT,
            page_number=3,
        )
    )
    document = Document(file_name="report.pdf", file_type="pdf", blocks=blocks, metadata={})

    chunks = chunk_document(document=document, max_chars=200, overlap=0)

    assert len(chunks) == 1
    assert "Выручка за период" in chunks[0].text


def test_a_document_of_headings_only_is_still_indexed():
    document = Document(
        file_name="outline.md",
        file_type="md",
        blocks=[
            Block(
                id=f"h{index}",
                text=f"Глава {index}",
                type=BlockType.HEADING.value,
                metadata={"source_file": "outline.md", "hierarchy_level": 1},
                block_type=BlockType.HEADING,
            )
            for index in range(1, 4)
        ],
        metadata={},
    )

    chunks = chunk_document(document=document, max_chars=200, overlap=0)

    assert chunks
    assert "Глава 1" in chunks[0].text
