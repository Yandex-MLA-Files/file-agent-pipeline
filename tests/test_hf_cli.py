import json

import pytest
from datasets import Dataset

from file_agent.hf_batch import BatchGenerationResult
from file_agent.hf_cli import (
    HFGenerationConfig,
    HFGenerationRunResult,
    main,
    run_hf_dataset_generation,
)
from file_agent.hf_output import GeneratedDatasetArtifacts
from file_agent.hf_rag import GeneratedQARecord


class DummyLLM:
    model = "fake/model"
    temperature = 0.0
    max_tokens = 128

    def generate(self, prompt: str) -> str:
        return "unused"


def make_source_dataset() -> Dataset:
    return Dataset.from_dict(
        {
            "id": ["q0001", "q0002"],
            "question": ["First question?", "Second question?"],
            "answer": ["First gold answer", "Second gold answer"],
            "doc_ids": [["q0001/first.txt"], ["q0002/second.txt"]],
        },
        split="train",
    )


def generated_record_from_row(row) -> GeneratedQARecord:
    return GeneratedQARecord(
        id=row["id"],
        question=row["question"],
        doc_ids=tuple(row["doc_ids"]),
        answer_model=f"Generated answer for {row['id']}",
        contexts=(),
        answer=row["answer"],
    )


def test_run_hf_dataset_generation_orchestrates_limited_run_and_writes_manifest(
    monkeypatch,
    tmp_path,
):
    source_dataset = make_source_dataset()
    llm_client = DummyLLM()
    calls = {}

    def fake_create_llm_client(**kwargs):
        calls["llm"] = kwargs
        return llm_client

    def fake_load_qa_dataset(**kwargs):
        calls["load"] = kwargs
        return source_dataset

    def fake_generate_hf_qa_records(**kwargs):
        calls["generate"] = kwargs
        selected_dataset = kwargs["dataset"]
        records = tuple(generated_record_from_row(row) for row in selected_dataset)
        return BatchGenerationResult(records=records, processed_count=1, resumed_count=0)

    def fake_save_generated_qa_dataset(**kwargs):
        calls["save"] = kwargs
        output_dir = tmp_path / "run"
        parquet_path = output_dir / "answers.parquet"
        hf_dataset_path = output_dir / "hf_dataset"
        output_dir.mkdir(parents=True)
        parquet_path.write_text("parquet", encoding="utf-8")
        hf_dataset_path.mkdir()
        return GeneratedDatasetArtifacts(
            parquet_path=parquet_path,
            hf_dataset_path=hf_dataset_path,
            row_count=1,
        )

    monkeypatch.setattr("file_agent.hf_cli.create_llm_client", fake_create_llm_client)
    monkeypatch.setattr("file_agent.hf_cli.load_qa_dataset", fake_load_qa_dataset)
    monkeypatch.setattr(
        "file_agent.hf_cli.generate_hf_qa_records",
        fake_generate_hf_qa_records,
    )
    monkeypatch.setattr(
        "file_agent.hf_cli.save_generated_qa_dataset",
        fake_save_generated_qa_dataset,
    )
    monkeypatch.setattr("file_agent.hf_cli._utc_timestamp", lambda: "2026-07-19T10:00:00Z")

    config = HFGenerationConfig(
        dataset_id="owner/rag-qa",
        output_dir=tmp_path / "run",
        config_name="default",
        split="train",
        revision="commit-sha",
        cache_dir=tmp_path / "cache",
        env_file=tmp_path / ".env",
        top_k=3,
        max_chars=800,
        overlap=80,
        limit=1,
        resume=True,
    )
    result = run_hf_dataset_generation(config)

    assert calls["llm"] == {"env_file": tmp_path / ".env"}
    assert calls["load"] == {
        "dataset_id": "owner/rag-qa",
        "config_name": "default",
        "split": "train",
        "revision": "commit-sha",
        "cache_dir": tmp_path / "cache",
    }
    assert len(calls["generate"]["dataset"]) == 1
    assert calls["generate"]["dataset"][0]["id"] == "q0001"
    assert calls["generate"]["llm_client"] is llm_client
    assert calls["generate"]["top_k"] == 3
    assert calls["generate"]["max_chars"] == 800
    assert calls["generate"]["overlap"] == 80
    assert calls["generate"]["resume"] is True
    assert calls["save"]["records"] == result.batch.records

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 2
    assert manifest["created_at_utc"] == "2026-07-19T10:00:00Z"
    assert manifest["dataset"] == {
        "id": "owner/rag-qa",
        "config_name": "default",
        "split": "train",
        "revision": "commit-sha",
        "available_rows": 2,
        "selected_rows": 1,
        "limit": 1,
        "records_sha256": manifest["dataset"]["records_sha256"],
    }
    assert len(manifest["dataset"]["records_sha256"]) == 64
    assert manifest["generation"]["model_id"] == "fake/model"
    assert manifest["generation"]["rag_pipeline_version"] == "structured-parsers-row-records-v3"
    assert manifest["generation"]["top_k"] == 3
    assert manifest["generation"]["prompt_sha256"]
    assert manifest["generation"]["resume_requested"] is True
    assert manifest["result"] == {
        "total_count": 1,
        "processed_count": 1,
        "resumed_count": 0,
        "failed_count": 0,
    }
    assert manifest["artifacts"] == {
        "parquet": "answers.parquet",
        "hf_dataset": "hf_dataset",
    }


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"top_k": 0}, "top_k"),
        ({"max_chars": 0}, "max_chars"),
        ({"max_chars": 100, "overlap": 100}, "overlap"),
        ({"overlap": -1}, "overlap"),
        ({"limit": 0}, "limit"),
    ],
)
def test_generation_config_rejects_invalid_numeric_values(tmp_path, overrides, message):
    values = {
        "dataset_id": "owner/rag-qa",
        "output_dir": tmp_path / "run",
        **overrides,
    }

    with pytest.raises(ValueError, match=message):
        HFGenerationConfig(**values)


def test_run_hf_dataset_generation_rejects_final_outputs_before_loading(
    monkeypatch,
    tmp_path,
):
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    (output_dir / "answers.parquet").write_text("keep", encoding="utf-8")

    def fail_if_called(**kwargs):
        raise AssertionError("Dataset loader must not run when final output exists")

    monkeypatch.setattr("file_agent.hf_cli.load_qa_dataset", fail_if_called)

    with pytest.raises(FileExistsError, match="answers.parquet"):
        run_hf_dataset_generation(
            HFGenerationConfig(dataset_id="owner/rag-qa", output_dir=output_dir),
            llm_client=DummyLLM(),
        )

    assert (output_dir / "answers.parquet").read_text(encoding="utf-8") == "keep"


def test_main_maps_cli_arguments_and_prints_artifact_paths(monkeypatch, tmp_path, capsys):
    calls = []
    output_dir = tmp_path / "run"
    expected_result = HFGenerationRunResult(
        batch=BatchGenerationResult(
            records=(generated_record_from_row(make_source_dataset()[0]),) * 3,
            processed_count=1,
            resumed_count=2,
        ),
        artifacts=GeneratedDatasetArtifacts(
            parquet_path=output_dir / "answers.parquet",
            hf_dataset_path=output_dir / "hf_dataset",
            row_count=3,
        ),
        manifest_path=output_dir / "run_manifest.json",
    )

    def fake_run(config):
        calls.append(config)
        return expected_result

    monkeypatch.setattr("file_agent.hf_cli.run_hf_dataset_generation", fake_run)

    exit_code = main(
        [
            "--dataset-id",
            "owner/rag-qa",
            "--output-dir",
            str(output_dir),
            "--config-name",
            "default",
            "--revision",
            "commit-sha",
            "--top-k",
            "3",
            "--max-chars",
            "800",
            "--overlap",
            "80",
            "--limit",
            "2",
            "--resume",
        ]
    )

    assert exit_code == 0
    assert calls == [
        HFGenerationConfig(
            dataset_id="owner/rag-qa",
            output_dir=output_dir,
            config_name="default",
            revision="commit-sha",
            top_k=3,
            max_chars=800,
            overlap=80,
            limit=2,
            resume=True,
        )
    ]
    output = capsys.readouterr().out
    assert "Completed 3 rows (processed: 1, resumed: 2)" in output
    assert str(output_dir / "answers.parquet") in output
    assert str(output_dir / "hf_dataset") in output
    assert str(output_dir / "run_manifest.json") in output
