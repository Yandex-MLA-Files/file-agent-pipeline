import json
from pathlib import Path

import pytest

from file_agent.docbench_dataset import (
    docbench_domain,
    load_docbench_records,
    select_docbench_records,
)


def _write_folder(root: Path, folder_id: int, rows: list[dict]) -> None:
    folder = root / str(folder_id)
    folder.mkdir(parents=True)
    (folder / f"document-{folder_id}.pdf").write_bytes(b"%PDF-test")
    (folder / f"{folder_id}_qa.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )


def test_load_docbench_records_uses_numeric_folder_order_and_stable_ids(tmp_path: Path) -> None:
    _write_folder(
        tmp_path,
        49,
        [
            {
                "question": "Finance question?",
                "answer": "Finance answer.",
                "type": "text-only",
                "evidence": "Evidence.",
            }
        ],
    )
    _write_folder(
        tmp_path,
        0,
        [
            {
                "question": "First?",
                "answer": "First answer.",
                "type": "multimodal-t",
                "evidence": "Table.",
            },
            {
                "question": "Second?",
                "answer": "",
                "type": "unanswerable",
                "evidence": "",
            },
        ],
    )

    records = load_docbench_records(tmp_path)

    assert [record.id for record in records] == [
        "docbench-000-q0001",
        "docbench-000-q0002",
        "docbench-049-q0001",
    ]
    assert records[0].domain == "academia"
    assert records[-1].domain == "finance"
    assert records[1].reference_answer == ""
    assert records[1].evidence == ""


def test_select_docbench_records_combines_filters(tmp_path: Path) -> None:
    _write_folder(
        tmp_path,
        0,
        [
            {"question": "A?", "answer": "A", "type": "text-only", "evidence": "A"},
            {"question": "B?", "answer": "B", "type": "multimodal-t", "evidence": "B"},
        ],
    )
    _write_folder(
        tmp_path,
        1,
        [{"question": "C?", "answer": "C", "type": "text-only", "evidence": "C"}],
    )
    records = load_docbench_records(tmp_path)

    selected = select_docbench_records(
        records,
        folder_start=0,
        folder_end=1,
        folder_ids=(0,),
        question_types=("text-only",),
        limit=1,
    )

    assert [record.id for record in selected] == ["docbench-000-q0001"]


def test_select_docbench_records_rejects_unknown_exact_id(tmp_path: Path) -> None:
    _write_folder(
        tmp_path,
        0,
        [{"question": "A?", "answer": "A", "type": "text-only", "evidence": "A"}],
    )
    records = load_docbench_records(tmp_path)

    with pytest.raises(ValueError, match="Unknown DocBench record IDs"):
        select_docbench_records(records, record_ids=("missing",))


@pytest.mark.parametrize(
    ("folder_id", "expected"),
    [
        (0, "academia"),
        (48, "academia"),
        (49, "finance"),
        (89, "government"),
        (133, "law"),
        (179, "news"),
        (228, "news"),
        (229, "unknown"),
    ],
)
def test_docbench_domain_boundaries(folder_id: int, expected: str) -> None:
    assert docbench_domain(folder_id) == expected


def test_load_docbench_records_requires_exactly_one_pdf(tmp_path: Path) -> None:
    _write_folder(
        tmp_path,
        0,
        [{"question": "A?", "answer": "A", "type": "text-only", "evidence": "A"}],
    )
    (tmp_path / "0" / "another.pdf").write_bytes(b"%PDF-more")

    with pytest.raises(ValueError, match="exactly one PDF"):
        load_docbench_records(tmp_path)
