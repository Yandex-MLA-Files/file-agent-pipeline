from pathlib import Path

import pytest

from file_agent.docbench_cli import (
    DocBenchGenerationConfig,
    _validate_or_write_run_config,
    create_argument_parser,
)
from file_agent.docbench_dataset import DocBenchRecord


def test_docbench_cli_parses_repeatable_selection_options() -> None:
    args = create_argument_parser().parse_args(
        [
            "--data-dir",
            "data",
            "--output-dir",
            "run",
            "--folder-id",
            "1",
            "--folder-id",
            "3",
            "--domain",
            "academia",
            "--question-type",
            "unanswerable",
            "--resume",
        ]
    )

    assert args.folder_ids == [1, 3]
    assert args.domains == ["academia"]
    assert args.question_types == ["unanswerable"]
    assert args.resume is True


def test_docbench_config_validates_chunking_and_folder_range() -> None:
    with pytest.raises(ValueError, match="overlap"):
        DocBenchGenerationConfig(
            data_dir=Path("data"),
            output_dir=Path("run"),
            max_chars=100,
            overlap=100,
        )

    with pytest.raises(ValueError, match="folder_start"):
        DocBenchGenerationConfig(
            data_dir=Path("data"),
            output_dir=Path("run"),
            folder_start=3,
            folder_end=1,
        )


def test_run_config_prevents_resume_with_changed_parameters(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    pdf_path = data_dir / "doc.pdf"
    pdf_path.write_bytes(b"%PDF-test")
    record = DocBenchRecord(
        id="docbench-000-q0001",
        folder_id=0,
        question_index=0,
        question="Question?",
        reference_answer="Gold",
        question_type="text-only",
        evidence="Evidence",
        domain="academia",
        pdf_path=pdf_path,
        qa_path=data_dir / "0_qa.jsonl",
    )
    output_dir = tmp_path / "run"
    fresh = DocBenchGenerationConfig(data_dir=data_dir, output_dir=output_dir)
    _validate_or_write_run_config(
        config=fresh,
        selected_records=[record],
        parameters={"top_k": 5},
    )

    resumed = DocBenchGenerationConfig(data_dir=data_dir, output_dir=output_dir, resume=True)
    _validate_or_write_run_config(
        config=resumed,
        selected_records=[record],
        parameters={"top_k": 5},
    )
    with pytest.raises(ValueError, match="resume configuration"):
        _validate_or_write_run_config(
            config=resumed,
            selected_records=[record],
            parameters={"top_k": 8},
        )
