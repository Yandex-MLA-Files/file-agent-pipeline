import fitz
from PIL import Image

from file_agent.document import Block, BlockType
from file_agent.parsers.vlm_ocr import (
    NO_TEXT_MARKER,
    PAGE_TRANSCRIPTION_PROMPT,
    VLMPageOCR,
    merge_ocr_blocks,
)
from file_agent.vlm.base import VLMClient


class ScriptedVLM(VLMClient):
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def describe_image(self, image: Image.Image, prompt: str) -> str:
        self.calls.append((image.size, prompt))
        return self.replies.pop(0)


def _make_pdf(path, pages=3):
    document = fitz.open()
    for index in range(pages):
        page = document.new_page()
        page.insert_text((72, 72), f"page {index + 1}")
    document.save(path)
    document.close()


def test_vlm_page_ocr_transcribes_selected_pages_into_typed_blocks(tmp_path):
    pdf = tmp_path / "scan.pdf"
    _make_pdf(pdf)
    vlm = ScriptedVLM(
        [
            "```markdown\n# Раздел 1\n\nАбзац текста.\n\n| a | b |\n| --- | --- |\n| 1 | 2 |\n```",
            NO_TEXT_MARKER,
        ]
    )

    result = VLMPageOCR(vlm).transcribe(pdf, [2, 3])

    assert set(result) == {2, 3}
    assert result[3] == []
    types = [b.block_type for b in result[2]]
    assert types == [BlockType.HEADING, BlockType.TEXT, BlockType.TABLE]
    assert all(b.page_number == 2 for b in result[2])
    assert all(b.metadata["ocr_engine"] == "vlm" for b in result[2])
    assert result[2][0].id.startswith("ocr-p2-")
    assert vlm.calls[0][1] == PAGE_TRANSCRIPTION_PROMPT
    assert vlm.calls[0][0][0] > 500  # rendered at 150 dpi


def test_merge_ocr_blocks_replaces_pages_in_reading_order():
    docling = [
        Block(id="a", text="page one text", type="text", block_type=BlockType.TEXT, page_number=1),
        Block(id="b", text="", type="figure", block_type=BlockType.FIGURE, page_number=2),
        Block(id="c", text="page three", type="text", block_type=BlockType.TEXT, page_number=3),
    ]
    transcript = {
        2: [Block(id="o2", text="scan two", type="text", block_type=BlockType.TEXT, page_number=2)],
        4: [
            Block(id="o4", text="scan four", type="text", block_type=BlockType.TEXT, page_number=4)
        ],
    }

    merged = merge_ocr_blocks(docling, transcript)

    assert [b.id for b in merged] == ["a", "o2", "c", "o4"]


def test_merge_ocr_blocks_without_transcripts_is_identity():
    blocks = [Block(id="a", text="x", type="text", block_type=BlockType.TEXT, page_number=1)]
    assert merge_ocr_blocks(blocks, {}) is blocks
