import fitz
from PIL import Image

from file_agent.document import Block, BlockType
from file_agent.parsers.vlm_ocr import (
    NO_TEXT_MARKER,
    PAGE_TRANSCRIPTION_PROMPT,
    VLMPageOCR,
    merge_ocr_blocks,
    validate_transcript,
)
from file_agent.vlm.base import VLMClient


class ScriptedVLM(VLMClient):
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def describe_image(self, image: Image.Image, prompt: str, max_tokens=None) -> str:
        self.calls.append((image.size, prompt, max_tokens))
        return self.replies.pop(0)


def _make_pdf(path, pages=3, lines=12):
    document = fitz.open()
    for index in range(pages):
        page = document.new_page()
        for line in range(lines):
            page.insert_text((72, 72 + line * 18), f"page {index + 1} line {line} с текстом")
    document.save(path)
    document.close()


def test_vlm_page_ocr_transcribes_selected_pages_into_typed_blocks(tmp_path):
    pdf = tmp_path / "scan.pdf"
    _make_pdf(pdf)
    body = "Абзац текста страницы. " * 30
    vlm = ScriptedVLM(
        [
            f"```markdown\n# Раздел 1\n\n{body}\n\n| a | b |\n| --- | --- |\n| 1 | 2 |\n```",
            NO_TEXT_MARKER,
        ]
    )

    result = VLMPageOCR(vlm).transcribe(pdf, [2, 3])

    # A page the model reports as empty is left to Docling (it may still hold a
    # picture worth describing), so only page 2 is replaced by a transcript.
    assert set(result) == {2}
    types = [b.block_type for b in result[2]]
    assert types == [BlockType.HEADING, BlockType.TEXT, BlockType.TABLE]
    assert all(b.page_number == 2 for b in result[2])
    assert all(b.metadata["ocr_engine"] == "vlm" for b in result[2])
    assert result[2][0].id.startswith("ocr-p2-")
    assert vlm.calls[0][1] == PAGE_TRANSCRIPTION_PROMPT
    assert vlm.calls[0][0][0] > 500  # rendered at 150 dpi


class StubLocalOCR:
    """Stand-in for EasyOCR: records the pages it was asked to read."""

    available = True

    def __init__(self, text="локальный ocr текст"):
        self.text = text
        self.calls = 0

    def transcribe_image(self, image):
        self.calls += 1
        return self.text

    @staticmethod
    def text_to_blocks(text, source_file, page_number):
        return [
            Block(
                id=f"ocr-p{page_number}-0",
                text=text,
                type="text",
                metadata={"source_file": source_file, "ocr_engine": "easyocr"},
                block_type=BlockType.TEXT,
                page_number=page_number,
            )
        ]


def test_repetition_loop_falls_back_to_the_local_engine(tmp_path):
    pdf = tmp_path / "scan.pdf"
    _make_pdf(pdf, pages=1)
    looping = "\n".join(["Повторяющаяся строка отчёта"] * 30)
    fallback = StubLocalOCR()

    result = VLMPageOCR(ScriptedVLM([looping]), fallback=fallback).transcribe(pdf, [1])

    assert fallback.calls == 1
    assert [b.metadata["ocr_engine"] for b in result[1]] == ["easyocr"]


def test_short_transcript_is_retried_with_a_larger_budget(tmp_path):
    """A page read only halfway is re-asked with more room, not indexed as is."""
    pdf = tmp_path / "scan.pdf"
    _make_pdf(pdf, pages=1)
    full = "Полный текст страницы. " * 40
    vlm = ScriptedVLM(["обрывок", full])

    result = VLMPageOCR(vlm, max_tokens=1000, fallback=None).transcribe(pdf, [1])

    assert "Полный текст страницы." in result[1][0].text
    assert [call[2] for call in vlm.calls] == [1000, 2000]


def test_failed_page_does_not_abort_the_rest(tmp_path):
    pdf = tmp_path / "scan.pdf"
    _make_pdf(pdf, pages=3)
    body = "Абзац текста страницы. " * 30

    class FlakyVLM(ScriptedVLM):
        def describe_image(self, image, prompt, max_tokens=None):
            self.calls.append((image.size, prompt))
            if len(self.calls) == 2:
                raise RuntimeError("endpoint hiccup")
            return body

    fallback = StubLocalOCR()
    result = VLMPageOCR(FlakyVLM([]), fallback=fallback, concurrency=1).transcribe(pdf, [1, 2, 3])

    assert set(result) == {1, 2, 3}
    assert result[2][0].metadata["ocr_engine"] == "easyocr"
    assert result[3][0].metadata["ocr_engine"] == "vlm"


def test_blank_page_is_never_sent_to_the_model(tmp_path):
    pdf = tmp_path / "blank.pdf"
    document = fitz.open()
    document.new_page()
    document.save(pdf)
    document.close()
    vlm = ScriptedVLM([])

    assert VLMPageOCR(vlm).transcribe(pdf, [1]) == {}
    assert vlm.calls == []


def test_validate_transcript_rejects_empty_output_on_a_page_full_of_ink():
    assert not validate_transcript(NO_TEXT_MARKER, ink=0.06).ok
    assert validate_transcript(NO_TEXT_MARKER, ink=0.001).ok
    assert not validate_transcript("I cannot read this image", ink=0.06).ok
    assert validate_transcript("Обычный текст страницы", ink=0.06).ok


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


def test_page_transcripts_are_cached_by_their_pixels(tmp_path):
    """A scanned deck costs 16 minutes once, and the same text on every re-run."""
    import fitz

    from file_agent.parsers.formula_enrichment import TranscriptCache
    from file_agent.parsers.vlm_ocr import VLMPageOCR

    pdf_path = tmp_path / "scan.pdf"
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 100), "Отчет о продажах за квартал", fontsize=18)
    document.save(pdf_path)
    document.close()

    class CountingClient:
        def __init__(self):
            self.calls = 0

        def describe_image(self, image, prompt, max_tokens=None):
            return self.describe_image_verbose(image, prompt, max_tokens)[0]

        def describe_image_verbose(self, image, prompt, max_tokens=None, max_image_side=None):
            self.calls += 1
            return "# Отчет о продажах\n\nВыручка выросла на 12 %.", None

    cache = TranscriptCache(tmp_path / "cache")
    client = CountingClient()

    first = VLMPageOCR(client, fallback=None, cache=cache).transcribe(pdf_path, [1])
    second_ocr = VLMPageOCR(client, fallback=None, cache=cache)
    second = second_ocr.transcribe(pdf_path, [1])

    assert client.calls == 1
    assert [b.text for b in first[1]] == [b.text for b in second[1]]
    assert second_ocr.stats["cached"] == 1
