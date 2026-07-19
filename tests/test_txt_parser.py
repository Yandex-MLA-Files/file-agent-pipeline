from file_agent.parsers.txt_parser import TXTParser


def test_txt_parser_returns_plain_text_document(tmp_path):
    file_path = tmp_path / "notes.txt"
    file_path.write_text("Привет из TXT!", encoding="utf-8-sig")

    document = TXTParser().parse(file_path)

    assert document.file_name == "notes.txt"
    assert document.file_type == "txt"
    assert len(document.blocks) == 1
    assert document.blocks[0].id == "block-1"
    assert document.blocks[0].text == "Привет из TXT!"
    assert document.blocks[0].type == "plain_text"
    assert document.blocks[0].metadata == {
        "source_file": "notes.txt",
        "block_type": "plain_text",
    }
