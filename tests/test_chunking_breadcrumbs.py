from file_agent.chunking import chunk_document
from file_agent.document import Block, BlockType, Document


def _heading(id_, text, level):
    return Block(
        id=id_,
        text=text,
        type="heading",
        metadata={"hierarchy_level": level},
        block_type=BlockType.HEADING,
    )


def _text(id_, text):
    return Block(id=id_, text=text, type="text", block_type=BlockType.TEXT)


def test_chunks_carry_heading_path_breadcrumbs():
    body = "Предложение о выручке за квартал. " * 12
    document = Document(
        file_name="report.pdf",
        file_type="pdf",
        blocks=[
            _heading("h1", "Отчет РЖД", 1),
            _heading("h2", "1. Финансовые результаты", 1),
            _heading("h3", "1.1 Выручка", 2),
            _text("p1", body),
            _text("p2", body),
            _heading("h4", "1.2 Расходы", 2),
            _text("p3", "Расходы выросли."),
        ],
    )

    chunks = chunk_document(document, max_chars=400, overlap=40)

    revenue = [c for c in chunks if "1.1 Выручка" in c.metadata.get("heading_path", [])]
    assert revenue, "revenue chunks must know their heading path"
    for chunk in revenue:
        assert chunk.metadata["heading_path"] == ["1. Финансовые результаты", "1.1 Выручка"]
        assert chunk.metadata["doc_title"] == "Отчет РЖД"
        assert chunk.text.startswith("Отчет РЖД > 1. Финансовые результаты > 1.1 Выручка")
        # Continuation pieces of a long section never consist of the heading alone.
        assert chunk.text.replace(chunk.text.split("\n\n", 1)[0], "").strip()
    # The parent passage of a split section opens with the same heading path.
    parents = {c.metadata.get("context") for c in revenue if c.metadata.get("context")}
    assert parents and all(p.startswith("1. Финансовые результаты > 1.1 Выручка") for p in parents)


def test_breadcrumb_is_not_duplicated_when_chunk_opens_with_the_heading():
    document = Document(
        file_name="notes.md",
        file_type="md",
        blocks=[_heading("h1", "Введение", 1), _text("p1", "Короткий абзац.")],
    )

    chunks = chunk_document(document, max_chars=1000, overlap=100)

    assert len(chunks) == 1
    assert chunks[0].text == "Введение\n\nКороткий абзац."


def test_lists_split_on_items_and_code_on_lines():
    items = "\n".join(f"- пункт номер {i} со словами" for i in range(40))
    code = "\n".join(f"line_{i} = compute({i})" for i in range(60))
    document = Document(
        file_name="doc.md",
        file_type="md",
        blocks=[
            _heading("h1", "Список", 1),
            Block(id="l1", text=items, type="list", block_type=BlockType.LIST),
            _heading("h2", "Код", 1),
            Block(id="c1", text=code, type="code", block_type=BlockType.CODE),
        ],
    )

    chunks = chunk_document(document, max_chars=300, overlap=30)

    list_chunks = [c for c in chunks if c.metadata["block_type"] == "list"]
    code_chunks = [c for c in chunks if c.metadata["block_type"] == "code"]
    assert len(list_chunks) > 1 and len(code_chunks) > 1
    for chunk in list_chunks:
        body = chunk.text.split("\n\n", 1)[-1]
        assert all(line.startswith("- пункт") for line in body.splitlines())
    for chunk in code_chunks:
        body = chunk.text.split("\n\n", 1)[-1]
        assert all(line.startswith("line_") for line in body.splitlines())


def test_legacy_strategy_is_selectable(monkeypatch):
    document = Document(
        file_name="notes.md",
        file_type="md",
        blocks=[_heading("h1", "Введение", 1), _text("p1", "Короткий абзац.")],
    )
    monkeypatch.setenv("CHUNKING_STRATEGY", "legacy")

    chunks = chunk_document(document, max_chars=1000, overlap=100)

    assert len(chunks) == 1
    assert "heading_path" not in chunks[0].metadata
