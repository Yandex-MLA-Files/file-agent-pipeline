from file_agent.document import Block, BlockType
from file_agent.parsers.docling_parser import DoclingParser, _postprocess_blocks


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
