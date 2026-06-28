import pytest

from file_agent.pipeline import parse_file


def test_parse_file_uses_markdown_parser(tmp_path):
    file_path = tmp_path / "example.md"
    file_path.write_text("Hello from pipeline", encoding="utf-8")

    document = parse_file(file_path)

    assert document.file_name == "example.md"
    assert document.file_type == "md"
    assert document.blocks[0].text == "Hello from pipeline"


def test_parse_file_rejects_unsupported_extension(tmp_path):
    file_path = tmp_path / "example.txt"
    file_path.write_text("Unsupported", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported file type"):
        parse_file(file_path)
