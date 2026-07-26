from file_agent.parsers.docling_parser import DoclingParser

normalize = DoclingParser._normalize_table_markdown


def test_alignment_padding_is_removed():
    padded = (
        "| Model Name                      | Context   |\n"
        "|---------------------------------|-----------|\n"
        "| Llama2-7B-Chat                  | 4k        |\n"
    )

    result = normalize(padded)

    assert result.splitlines() == [
        "| Model Name | Context |",
        "| --- | --- |",
        "| Llama2-7B-Chat | 4k |",
    ]
    # The point of the exercise: far fewer characters for the same content.
    assert len(result) < len(padded) * 0.75


def test_content_and_empty_cells_are_preserved():
    table = "| a |  | c |\n| --- | --- | --- |\n| 1 |  | 3 |"

    result = normalize(table)

    assert result.splitlines()[0] == "| a |  | c |"
    assert result.splitlines()[2] == "| 1 |  | 3 |"


def test_non_table_lines_are_untouched():
    text = "Table 3: results\n\n| a | b |\n| --- | --- |\n| 1 | 2 |"

    result = normalize(text)

    assert result.startswith("Table 3: results\n\n")
    assert "| a | b |" in result


def test_multi_space_runs_inside_cells_collapse():
    table = "| Llama2-7B    Qwen-7B     Vicuna-7B | 4k    8k |"

    result = normalize(table)

    assert result == "| Llama2-7B Qwen-7B Vicuna-7B | 4k 8k |"
