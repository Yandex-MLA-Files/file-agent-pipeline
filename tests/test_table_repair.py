from pathlib import Path

from file_agent.document import Block, BlockType, Document
from file_agent.parsers.table_repair import (
    is_degenerate,
    is_improvement,
    repair_tables,
)
from file_agent.vlm.base import VLMClient

GOOD = (
    "| Показатель | 2025 | 2026 |\n| --- | --- | --- |\n| Выручка | 10 | 12 |\n| Расходы | 5 | 6 |"
)
SINGLE_COLUMN = "| Показатель |\n| --- |\n| Выручка |\n| Расходы |"
HEADER_ONLY = "| Показатель | 2025 |\n| --- | --- |"
MOSTLY_EMPTY = "| a | b | c |\n| --- | --- | --- |\n|  |  | 1 |\n|  |  |  |\n| 2 |  |  |"


class StubVLM(VLMClient):
    def __init__(self, reply):
        self.reply = reply
        self.calls = 0

    def describe_image(self, image, prompt, max_tokens=None):
        self.calls += 1
        return self.reply


def _document(text, bbox=(0.0, 0.0, 300.0, 200.0)):
    return Document(
        file_name="report.pdf",
        file_type="pdf",
        blocks=[
            Block(
                id="t1",
                text=text,
                type="table",
                block_type=BlockType.TABLE,
                page_number=2,
                bbox=bbox,
            )
        ],
    )


def test_degenerate_detection():
    assert not is_degenerate(GOOD)
    assert is_degenerate(SINGLE_COLUMN)
    assert is_degenerate(HEADER_ONLY)
    assert is_degenerate(MOSTLY_EMPTY)


def test_repair_replaces_a_broken_table(monkeypatch):
    document = _document(SINGLE_COLUMN)
    vlm = StubVLM(f"```markdown\n{GOOD}\n```")
    monkeypatch.setattr(
        "file_agent.parsers.table_repair.extract_image_from_pdf",
        lambda *args, **kwargs: object(),
    )

    assert repair_tables(document, Path("report.pdf"), vlm) == 1
    assert document.blocks[0].text == GOOD
    assert document.blocks[0].metadata["table_source"] == "vlm"
    assert document.metadata["vlm_repaired_tables"] == 1


def test_sound_tables_are_never_sent(monkeypatch):
    document = _document(GOOD)
    vlm = StubVLM(GOOD)
    monkeypatch.setattr(
        "file_agent.parsers.table_repair.extract_image_from_pdf",
        lambda *args, **kwargs: object(),
    )

    assert repair_tables(document, Path("report.pdf"), vlm) == 0
    assert vlm.calls == 0


def test_a_worse_reading_is_rejected(monkeypatch):
    document = _document(SINGLE_COLUMN)
    vlm = StubVLM("| Показатель |\n| --- |\n| Выручка |")
    monkeypatch.setattr(
        "file_agent.parsers.table_repair.extract_image_from_pdf",
        lambda *args, **kwargs: object(),
    )

    assert repair_tables(document, Path("report.pdf"), vlm) == 0
    assert document.blocks[0].text == SINGLE_COLUMN


def test_mode_off_disables_repair(monkeypatch):
    monkeypatch.setenv("PDF_TABLE_VLM", "off")
    vlm = StubVLM(GOOD)
    assert repair_tables(_document(SINGLE_COLUMN), Path("report.pdf"), vlm) == 0
    assert vlm.calls == 0


def test_improvement_requires_more_structure():
    assert is_improvement(SINGLE_COLUMN, GOOD)
    assert not is_improvement(GOOD, SINGLE_COLUMN)
