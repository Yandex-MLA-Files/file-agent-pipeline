import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from datasets import Dataset, Features, List, Value

from file_agent.hf_dataset import QADatasetRecord, validate_qa_dataset
from file_agent.hf_rag import GeneratedQARecord

GENERATED_QA_FEATURES = Features(
    {
        "id": Value("string"),
        "question": Value("string"),
        "doc_ids": List(Value("string")),
        "answer_model": Value("string"),
        "contexts": List(
            {
                "rank": Value("int32"),
                "chunk_id": Value("string"),
                "document_id": Value("string"),
                "text": Value("string"),
                "score": Value("float64"),
                "metadata_json": Value("string"),
            }
        ),
        "answer": Value("string"),
    }
)


@dataclass(frozen=True)
class GeneratedDatasetArtifacts:
    parquet_path: Path
    hf_dataset_path: Path
    row_count: int


def build_generated_qa_dataset(
    source_dataset: Dataset,
    records: Sequence[GeneratedQARecord],
) -> Dataset:
    """Build a typed output dataset after matching every result to its source row."""
    validate_qa_dataset(source_dataset)

    if len(records) != len(source_dataset):
        raise ValueError(
            "Generated records count does not match source dataset: "
            f"{len(records)} != {len(source_dataset)}"
        )

    rows: list[dict] = []
    for row_index, (source_row, generated_record) in enumerate(
        zip(source_dataset, records, strict=True)
    ):
        if not isinstance(generated_record, GeneratedQARecord):
            raise TypeError(f"Generated record at row {row_index} must be a GeneratedQARecord")

        source_record = QADatasetRecord.from_row(source_row, row_index=row_index)
        validated_record = GeneratedQARecord.from_dict(generated_record.to_dict())
        _validate_record_matches_source(
            source_record=source_record,
            generated_record=validated_record,
            row_index=row_index,
        )
        rows.append(validated_record.to_dict())

    return Dataset.from_list(
        rows,
        features=GENERATED_QA_FEATURES,
        split=source_dataset.split,
    )


def save_generated_qa_dataset(
    source_dataset: Dataset,
    records: Sequence[GeneratedQARecord],
    output_dir: str | Path,
) -> GeneratedDatasetArtifacts:
    """Save generated rows as Parquet and a reloadable Hugging Face dataset."""
    dataset = build_generated_qa_dataset(source_dataset, records)
    output_path = Path(output_dir)
    parquet_path = output_path / "answers.parquet"
    hf_dataset_path = output_path / "hf_dataset"

    existing_paths = [path for path in (parquet_path, hf_dataset_path) if path.exists()]
    if existing_paths:
        names = ", ".join(path.name for path in existing_paths)
        raise FileExistsError(
            f"Output artifacts already exist ({names}); use a new output directory"
        )

    output_path.mkdir(parents=True, exist_ok=True)
    suffix = uuid4().hex
    temporary_parquet_path = output_path / f".answers.{suffix}.parquet.tmp"
    temporary_hf_dataset_path = output_path / f".hf_dataset.{suffix}.tmp"
    parquet_published = False

    try:
        dataset.to_parquet(temporary_parquet_path)
        dataset.save_to_disk(temporary_hf_dataset_path)
        temporary_parquet_path.replace(parquet_path)
        parquet_published = True
        temporary_hf_dataset_path.replace(hf_dataset_path)
    except Exception:
        _remove_path(temporary_parquet_path)
        _remove_path(temporary_hf_dataset_path)
        if parquet_published:
            _remove_path(parquet_path)
        raise

    return GeneratedDatasetArtifacts(
        parquet_path=parquet_path,
        hf_dataset_path=hf_dataset_path,
        row_count=dataset.num_rows,
    )


def _validate_record_matches_source(
    source_record: QADatasetRecord,
    generated_record: GeneratedQARecord,
    row_index: int,
) -> None:
    fields = ("id", "question", "doc_ids", "answer")
    mismatches = [
        field_name
        for field_name in fields
        if getattr(source_record, field_name) != getattr(generated_record, field_name)
    ]
    if mismatches:
        raise ValueError(
            f"Generated record does not match source row {row_index}: {', '.join(mismatches)}"
        )


def _remove_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()
