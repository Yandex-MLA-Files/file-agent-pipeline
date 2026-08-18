from file_agent.document import Block, BlockType
from file_agent.parsers.docling_parser import (
    DoclingParser,
    _postprocess_blocks,
    repair_column_order,
)


def _block(id_, text, block_type, parent="", parent_label="", page=1, level=1):
    return Block(
        id=id_,
        text=text,
        type=block_type.value,
        metadata={
            "docling_label": block_type.value,
            "docling_parent": parent,
            "docling_parent_label": parent_label,
            "hierarchy_level": level,
        },
        block_type=block_type,
        page_number=page,
    )


def test_consecutive_list_items_become_one_list_block():
    blocks = [
        _block("h", "1.5 Классификация", BlockType.HEADING),
        _block("l1", " локализованная;", BlockType.LIST, parent="#/groups/2"),
        _block("l2", " распространенная;", BlockType.LIST, parent="#/groups/2"),
        _block("l3", "o другая", BlockType.LIST, parent="#/groups/3"),
    ]

    merged = _postprocess_blocks(blocks)

    assert [b.block_type for b in merged] == [BlockType.HEADING, BlockType.LIST, BlockType.LIST]
    assert merged[1].text == "- локализованная;\n- распространенная;"
    assert merged[1].metadata["item_count"] == 2
    assert merged[2].text == "- другая"
    assert "_items" not in merged[1].metadata


def test_words_broken_by_line_hyphenation_are_rejoined():
    blocks = [
        _block("t1", "Дивиденды по обыкновен- ным и привилеги- рованным акциям", BlockType.TEXT),
        _block("t2", "врача-офтальмолога и из-за чего", BlockType.TEXT),
        _block("t3", "| Показатель | 2024-2025 |", BlockType.TABLE),
    ]

    merged = _postprocess_blocks(blocks)

    assert merged[0].text == "Дивиденды по обыкновенным и привилегированным акциям"
    # Real compounds and ranges have no space after the hyphen and stay intact.
    assert merged[1].text == "врача-офтальмолога и из-за чего"
    assert merged[2].text == "| Показатель | 2024-2025 |"


def test_inline_group_fragments_are_merged_into_one_paragraph():
    blocks = [
        _block("t1", "Целью", BlockType.TEXT, parent="#/groups/1", parent_label="inline"),
        _block(
            "t2", "испытания является", BlockType.TEXT, parent="#/groups/1", parent_label="inline"
        ),
        _block("t3", "1.2. «Науки»", BlockType.TEXT, parent="#/groups/1", parent_label="inline"),
        _block("t4", "Другой абзац", BlockType.TEXT, parent="#/groups/2", parent_label="inline"),
    ]

    merged = _postprocess_blocks(blocks)

    assert [b.text for b in merged] == ["Целью испытания является 1.2. «Науки»", "Другой абзац"]


def test_headings_split_across_lines_are_stitched():
    blocks = [
        _block("h1", "1. Краткая информация (группе заболеваний или", BlockType.HEADING),
        _block("h2", "состояний)", BlockType.HEADING),
        _block("h3", "1.1 Определение", BlockType.HEADING),
    ]

    merged = _postprocess_blocks(blocks)

    assert [b.text for b in merged] == [
        "1. Краткая информация (группе заболеваний или состояний)",
        "1.1 Определение",
    ]


def test_map_label_covers_new_block_types():
    assert DoclingParser._map_label("list_item") == BlockType.LIST
    assert DoclingParser._map_label("code") == BlockType.CODE
    assert DoclingParser._map_label("section_header") == BlockType.HEADING
    assert DoclingParser._map_label("picture") == BlockType.FIGURE


def _positioned(block_id: str, x0: float, y0: float, x1: float, y1: float, page: int = 1):
    return Block(
        id=block_id,
        text=block_id,
        type="text",
        block_type=BlockType.TEXT,
        page_number=page,
        bbox=(x0, y0, x1, y1),
    )


def test_interleaved_columns_are_re_sorted_column_by_column():
    """Left/right/left/right reading order is damage, not a layout."""
    interleaved = [
        _positioned("L1", 50, 100, 290, 140),
        _positioned("R1", 320, 100, 560, 140),
        _positioned("L2", 50, 150, 290, 190),
        _positioned("R2", 320, 150, 560, 190),
        _positioned("L3", 50, 200, 290, 240),
        _positioned("R3", 320, 200, 560, 240),
    ]

    assert [b.id for b in repair_column_order(interleaved)] == [
        "L1",
        "L2",
        "L3",
        "R1",
        "R2",
        "R3",
    ]


def test_correct_orders_are_left_untouched():
    single_column = [_positioned(f"P{i}", 50, 100 + 30 * i, 560, 130 + 30 * i) for i in range(8)]
    two_columns = [
        _positioned("L1", 50, 100, 290, 140),
        _positioned("L2", 50, 150, 290, 190),
        _positioned("L3", 50, 200, 290, 240),
        _positioned("R1", 320, 100, 560, 140),
        _positioned("R2", 320, 150, 560, 190),
        _positioned("R3", 320, 200, 560, 240),
    ]

    assert repair_column_order(single_column) == single_column
    assert repair_column_order(two_columns) == two_columns


def test_a_full_width_element_rules_out_the_column_repair():
    # A page with a title spanning both columns is not a clean two-column page:
    # re-sorting it would move the title away from the text it introduces.
    with_title = [
        _positioned("TITLE", 50, 60, 560, 90),
        _positioned("L1", 50, 100, 290, 140),
        _positioned("R1", 320, 100, 560, 140),
        _positioned("L2", 50, 150, 290, 190),
        _positioned("R2", 320, 150, 560, 190),
        _positioned("L3", 50, 200, 290, 240),
        _positioned("R3", 320, 200, 560, 240),
    ]

    assert repair_column_order(with_title) == with_title


def test_blocks_without_geometry_are_never_reordered():
    stream = [
        Block(id="a", text="a", type="text", block_type=BlockType.TEXT, page_number=1),
        Block(id="b", text="b", type="text", block_type=BlockType.TEXT, page_number=1),
    ]

    assert repair_column_order(stream) == stream


def test_enrichment_batch_size_is_opt_in_and_survives_a_bad_value(monkeypatch, caplog):
    from file_agent.parsers import docling_parser
    from file_agent.parsers.docling_parser import apply_enrichment_batch_size

    monkeypatch.delenv("PDF_ENRICHMENT_BATCH", raising=False)
    monkeypatch.setattr(docling_parser, "free_gpu_memory_gb", lambda: None)
    assert apply_enrichment_batch_size() is None

    monkeypatch.setenv("PDF_ENRICHMENT_BATCH", "not-a-number")
    assert apply_enrichment_batch_size() is None

    monkeypatch.setenv("PDF_ENRICHMENT_BATCH", "16")
    assert apply_enrichment_batch_size() == 16

    from docling.models.stages.code_formula.code_formula_vlm_model import CodeFormulaVlmModel

    assert CodeFormulaVlmModel.elements_batch_size == 16
    CodeFormulaVlmModel.elements_batch_size = 5


def test_enrichment_that_returns_nothing_is_reported(caplog):
    """Docling swallows a CUDA OOM inside the stage and returns empty text."""
    from pathlib import Path

    from file_agent.parsers.docling_parser import _warn_on_lost_enrichment

    with caplog.at_level("WARNING"):
        _warn_on_lost_enrichment(Path("lecture.pdf"), enriched=378, empty=378)
    assert "no text for 378 of 378" in caplog.text

    caplog.clear()
    with caplog.at_level("WARNING"):
        _warn_on_lost_enrichment(Path("lecture.pdf"), enriched=378, empty=4)
        _warn_on_lost_enrichment(Path("paper.pdf"), enriched=0, empty=0)
    assert caplog.text == ""


def test_the_enrichment_batch_is_sized_from_the_free_gpu_memory():
    """Docling hides an out-of-memory and returns *no* formulas, so plan for it."""
    from file_agent.parsers.docling_parser import resolve_enrichment_batch_size

    # No GPU, or a card with nothing to spare: Docling's own default.
    assert resolve_enrichment_batch_size(None) == 5
    assert resolve_enrichment_batch_size(2.0) == 5
    # The shared A100 next to vLLM (~8 GB free) — the size measured as safe.
    assert resolve_enrichment_batch_size(8.0) == 18
    # A free card is capped, not extrapolated.
    assert resolve_enrichment_batch_size(70.0) == 32


def test_an_explicit_batch_size_still_wins(monkeypatch):
    from docling.models.stages.code_formula.code_formula_vlm_model import CodeFormulaVlmModel

    from file_agent.parsers import docling_parser
    from file_agent.parsers.docling_parser import apply_enrichment_batch_size

    monkeypatch.setattr(docling_parser, "free_gpu_memory_gb", lambda: 70.0)
    monkeypatch.setenv("PDF_ENRICHMENT_BATCH", "7")
    assert apply_enrichment_batch_size() == 7
    assert CodeFormulaVlmModel.elements_batch_size == 7

    monkeypatch.setenv("PDF_ENRICHMENT_BATCH", "auto")
    assert apply_enrichment_batch_size() == 32
    CodeFormulaVlmModel.elements_batch_size = 5
