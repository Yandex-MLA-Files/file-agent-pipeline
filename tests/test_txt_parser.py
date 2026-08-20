from file_agent.document import BlockType
from file_agent.parsers.legacy import TXTParser as LegacyTXTParser
from file_agent.parsers.txt_parser import TXTParser


def test_txt_parser_returns_single_paragraph_for_short_note(tmp_path):
    file_path = tmp_path / "notes.txt"
    file_path.write_text("Привет из TXT!", encoding="utf-8-sig")

    document = TXTParser().parse(file_path)

    assert document.file_name == "notes.txt"
    assert document.file_type == "txt"
    assert len(document.blocks) == 1
    assert document.blocks[0].text == "Привет из TXT!"
    assert document.blocks[0].block_type == BlockType.TEXT
    assert document.blocks[0].metadata["source_file"] == "notes.txt"


def test_txt_parser_reflows_hard_wrapped_prose_and_detects_titles(tmp_path):
    file_path = tmp_path / "book.txt"
    file_path.write_text(
        "А.А.Нейхардт. Легенды и сказания древнего Рима\n"
        "---------------------------------------------\n"
        " * БОГИ. ДРЕВНИЕ ИТАЛИЙСКИЕ БОЖЕСТВА * \n"
        "Юпитер\n"
        "   Могущественный властитель неба, олицетворение солнечного света,\n"
        "бури, в гневе метавший молнии, - таков был верховный владыка богов.\n"
        "   Свою волю Юпитер выражал раскатами грома, блеском молнии,\n"
        "полетом птиц; иногда он посылал вещие сны.\n"
        "Юнона\n"
        "   Царица неба, супруга Юпитера, покровительница брака.\n",
        encoding="utf-8",
    )

    document = TXTParser().parse(file_path)

    headings = [
        (b.text, b.metadata["hierarchy_level"])
        for b in document.blocks
        if b.block_type == BlockType.HEADING
    ]
    assert ("БОГИ. ДРЕВНИЕ ИТАЛИЙСКИЕ БОЖЕСТВА", 1) in headings
    assert ("Юпитер", 2) in headings
    assert ("Юнона", 2) in headings
    paragraphs = [b.text for b in document.blocks if b.block_type == BlockType.TEXT]
    # Wrapped lines are joined; indented lines open new paragraphs.
    assert any(
        p.startswith("Могущественный властитель неба") and "владыка богов." in p for p in paragraphs
    )
    assert any(p.startswith("Свою волю Юпитер") for p in paragraphs)
    assert document.metadata["parsing_method"] == "text-prose"


def test_txt_parser_reflows_transcripts_and_drops_timestamps(tmp_path):
    file_path = tmp_path / "talk.txt"
    lines = ["📺 Title: Intro to LLMs", "", "🎙️ Channel: Some Channel", ""]
    for index in range(60):
        lines.append(f"[{index // 60:02d}:{index % 60:02d}]  word{index} spoken here.")
    file_path.write_text("\n".join(lines), encoding="utf-8")

    document = TXTParser().parse(file_path)

    assert document.metadata["parsing_method"] == "text-transcript"
    assert document.blocks[0].block_type == BlockType.HEADING
    assert document.blocks[0].text == "Intro to LLMs"
    body = [
        b
        for b in document.blocks
        if b.block_type == BlockType.TEXT and b.metadata.get("time_start")
    ]
    assert body, "transcript captions must become text blocks"
    assert body[0].metadata["time_start"] == "00:00"
    assert "[00:" not in body[0].text
    assert "word0 spoken here." in body[0].text


def test_txt_parser_decodes_cp1251(tmp_path):
    file_path = tmp_path / "legacy.txt"
    file_path.write_bytes("Текст в кодировке Windows-1251.".encode("cp1251"))

    document = TXTParser().parse(file_path)

    assert document.blocks[0].text == "Текст в кодировке Windows-1251."


def test_legacy_txt_parser_returns_one_plain_block(tmp_path):
    file_path = tmp_path / "notes.txt"
    file_path.write_text("Привет из TXT!", encoding="utf-8-sig")

    document = LegacyTXTParser().parse(file_path)

    assert document.blocks[0].type == "plain_text"
    assert document.blocks[0].text == "Привет из TXT!"
