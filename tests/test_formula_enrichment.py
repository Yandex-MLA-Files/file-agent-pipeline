"""Formula and code regions read from the page image by the serving model."""

import fitz
import pytest
from PIL import Image

from file_agent.document import Block, BlockType, Document
from file_agent.parsers.formula_enrichment import (
    ENRICHMENT_PENDING,
    FormulaEnricher,
    TranscriptCache,
    clean_transcript,
    resolve_cache,
    resolve_engine,
    validate_transcript,
)

FORMULA = r"P(A \mid B) = \frac{P(A \cap B)}{P(B)}"


class FakeVLM:
    """Answers with a canned transcript and records every call."""

    def __init__(self, answers=None, error: Exception | None = None):
        self.answers = answers if answers is not None else [FORMULA]
        self.error = error
        self.calls: list[tuple[str, int]] = []

    def describe_image(self, image, prompt, max_tokens=None):
        return self.describe_image_verbose(image, prompt, max_tokens)[0]

    def describe_image_verbose(self, image, prompt, max_tokens=None, max_image_side=None):
        self.calls.append((prompt[:20], max_tokens or 0))
        if self.error is not None:
            raise self.error
        index = min(len(self.calls) - 1, len(self.answers) - 1)
        return self.answers[index], None


def make_pdf(path, text=r"P(A|B) = P(AB)/P(B)"):
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 100), text, fontsize=14)
    document.save(path)
    document.close()
    return path


def pending_document(name="lecture.pdf", boxes=((60.0, 80.0, 300.0, 120.0),)):
    blocks = [
        Block(
            id=f"block-{index}",
            text="",
            type=BlockType.FORMULA.value,
            metadata={"source_file": name, ENRICHMENT_PENDING: True},
            block_type=BlockType.FORMULA,
            page_number=1,
            bbox=box,
        )
        for index, box in enumerate(boxes)
    ]
    return Document(file_name=name, file_type="pdf", blocks=blocks, metadata={})


# -- engine choice -----------------------------------------------------------


def test_engine_prefers_the_serving_model_and_falls_back_to_docling(monkeypatch):
    monkeypatch.delenv("PDF_ENRICHMENT", raising=False)
    monkeypatch.delenv("PDF_ENRICHMENT_ENGINE", raising=False)

    assert resolve_engine(vlm_available=True, gpu_available=False) == "vlm"
    assert resolve_engine(vlm_available=True, gpu_available=True) == "vlm"
    assert resolve_engine(vlm_available=False, gpu_available=True) == "docling"
    # No endpoint and no GPU: enrichment would take tens of minutes on a CPU.
    assert resolve_engine(vlm_available=False, gpu_available=False) == "off"


def test_engine_respects_the_two_switches(monkeypatch):
    monkeypatch.setenv("PDF_ENRICHMENT", "off")
    assert resolve_engine(True, True) == "off"

    monkeypatch.setenv("PDF_ENRICHMENT", "on")
    monkeypatch.setenv("PDF_ENRICHMENT_ENGINE", "docling")
    assert resolve_engine(True, False) == "docling"  # forced without a GPU

    monkeypatch.setenv("PDF_ENRICHMENT_ENGINE", "vlm")
    assert resolve_engine(False, False) == "docling"  # asked for vlm, none there

    monkeypatch.setenv("PDF_ENRICHMENT_ENGINE", "nonsense")
    with pytest.raises(ValueError, match="PDF_ENRICHMENT_ENGINE"):
        resolve_engine(True, True)


# -- validation --------------------------------------------------------------


def test_transcript_wrappers_are_stripped():
    assert clean_transcript("$$x^2$$") == "x^2"
    assert clean_transcript("```latex\nx^2\n```") == "x^2"
    assert clean_transcript(r"\[x^2\]") == "x^2"
    assert clean_transcript("$x$") == "x"


def test_a_run_of_spacing_macros_is_collapsed():
    """Models pad a right-aligned equation number with a wall of \\qquad."""
    padded = r"(A \cap P). \qquad \qquad \qquad \qquad \qquad (3.14)"

    assert clean_transcript(padded) == r"(A \cap P). \qquad (3.14)"
    # Two of them are ordinary spacing and stay as they are.
    assert clean_transcript(r"x \quad \quad y") == r"x \quad \quad y"


def test_a_description_of_the_image_is_not_a_transcription():
    for answer in (
        "The image shows the formula for conditional probability.",
        "На изображении представлена формула Байеса.",
        "I cannot read this image.",
        "",
    ):
        assert validate_transcript(answer).ok is False


def test_a_transcript_cut_off_mid_formula_is_retried_not_dropped():
    """One unclosed group is worth another try, but not worth losing outright."""
    check = validate_transcript(r"P(A) = \frac{ \sum_{i=1}^{n} { x_i }")

    assert check.ok is True
    assert check.retryable is True

    # Nothing but unclosed groups is not a formula at all.
    broken = validate_transcript(r"\frac{ { { { { { { {")
    assert broken.ok is False
    assert broken.retryable is True


def test_loops_and_runaway_output_are_rejected():
    assert validate_transcript("x = 1\n" * 9).ok is False
    assert validate_transcript("a" * 3000).ok is False


def test_real_transcriptions_pass():
    assert validate_transcript(FORMULA).ok
    # A region the layout model mislabelled: plain text is transcribed as text.
    assert validate_transcript("Теорема 3. Пусть A и B независимы.").ok
    assert validate_transcript("for i in range(10):\n    print(i)", kind="code").ok


# -- the enricher ------------------------------------------------------------


def test_regions_are_filled_from_the_page_image(tmp_path):
    pdf = make_pdf(tmp_path / "lecture.pdf")
    document = pending_document()
    client = FakeVLM()

    stats = FormulaEnricher(client, cache=None).enrich(document, pdf)

    assert len(document.blocks) == 1
    assert document.blocks[0].text == FORMULA
    assert document.blocks[0].metadata["enrichment"] == "vlm"
    assert ENRICHMENT_PENDING not in document.blocks[0].metadata
    assert stats["read"] == 1


def test_a_region_nothing_could_read_is_dropped_not_indexed_empty(tmp_path):
    pdf = make_pdf(tmp_path / "lecture.pdf")
    document = pending_document()
    client = FakeVLM(answers=["The image shows a formula.", "The image shows a formula."])

    stats = FormulaEnricher(client, cache=None).enrich(document, pdf)

    assert document.blocks == []
    assert stats["rejected_prose"] == 1


def test_an_unreachable_endpoint_costs_one_request_not_one_per_region(tmp_path):
    pdf = make_pdf(tmp_path / "lecture.pdf")
    boxes = tuple((60.0, 80.0 + 30 * i, 300.0, 100.0 + 30 * i) for i in range(5))
    document = pending_document(boxes=boxes)
    client = FakeVLM(error=RuntimeError("connection refused"))

    stats = FormulaEnricher(client, cache=None).enrich(document, pdf)

    assert len(client.calls) == 1
    assert stats["endpoint_unavailable"] == 1
    assert document.blocks == []


def test_the_same_crop_is_transcribed_once(tmp_path):
    pdf = make_pdf(tmp_path / "lecture.pdf")
    box = (60.0, 80.0, 300.0, 120.0)
    document = pending_document(boxes=(box, box, box))
    client = FakeVLM()

    stats = FormulaEnricher(client, cache=None).enrich(document, pdf)

    assert len(client.calls) == 1
    assert [block.text for block in document.blocks] == [FORMULA] * 3
    assert stats["unique"] == 1


def test_transcripts_are_reused_across_runs(tmp_path):
    pdf = make_pdf(tmp_path / "lecture.pdf")
    cache = TranscriptCache(tmp_path / "cache")

    first = pending_document()
    FormulaEnricher(FakeVLM(), cache=cache).enrich(first, pdf)

    second = pending_document()
    offline = FakeVLM(error=AssertionError("the model must not be called again"))
    stats = FormulaEnricher(offline, cache=cache).enrich(second, pdf)

    assert second.blocks[0].text == FORMULA
    assert second.blocks[0].metadata["enrichment"] == "cache"
    assert stats["cached"] == 1
    assert offline.calls == []


def test_the_cache_can_be_switched_off(monkeypatch, tmp_path):
    monkeypatch.setenv("PDF_ENRICHMENT_CACHE", "off")
    assert resolve_cache() is None

    monkeypatch.setenv("PDF_ENRICHMENT_CACHE", str(tmp_path / "here"))
    cache = resolve_cache()
    cache.put("abcdef", "x^2")
    assert cache.get("abcdef") == "x^2"
    assert cache.get("missing") is None


def test_a_truncated_answer_is_retried_with_a_larger_budget(tmp_path):
    pdf = make_pdf(tmp_path / "lecture.pdf")
    document = pending_document()
    client = FakeVLM(answers=[r"\frac{ { {", FORMULA])

    FormulaEnricher(client, max_tokens=100, cache=None).enrich(document, pdf)

    assert [tokens for _, tokens in client.calls] == [100, 200]
    assert document.blocks[0].text == FORMULA


def test_a_region_too_small_to_read_is_never_sent(tmp_path):
    pdf = make_pdf(tmp_path / "lecture.pdf")
    document = pending_document(boxes=((60.0, 80.0, 63.0, 82.0),))
    client = FakeVLM()

    stats = FormulaEnricher(client, cache=None).enrich(document, pdf)

    assert client.calls == []
    assert stats["region_too_small"] == 1
    assert document.blocks == []


def test_crops_are_rendered_from_the_pdf(tmp_path):
    """The model must receive the region, not the whole page."""
    pdf = make_pdf(tmp_path / "lecture.pdf")
    document = pending_document(boxes=((60.0, 80.0, 300.0, 120.0),))
    seen: list[Image.Image] = []

    class Recorder(FakeVLM):
        def describe_image_verbose(self, image, prompt, max_tokens=None, max_image_side=None):
            seen.append(image)
            return super().describe_image_verbose(image, prompt, max_tokens, max_image_side)

    FormulaEnricher(Recorder(), dpi=144, cache=None).enrich(document, pdf)

    width, height = seen[0].size
    # 240 x 40 points plus 18 % padding, rendered at 2x.
    assert 600 < width < 750
    assert 80 < height < 160
