import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from datasets import Dataset, load_dataset
from huggingface_hub import hf_hub_download, list_repo_files
from huggingface_hub.errors import EntryNotFoundError

REQUIRED_QA_COLUMNS = ("id", "question", "answer", "doc_ids")


@dataclass(frozen=True)
class QADatasetRecord:
    id: str
    question: str
    answer: str
    doc_ids: tuple[str, ...]

    @classmethod
    def from_row(
        cls,
        row: Mapping[str, Any],
        row_index: int | None = None,
    ) -> "QADatasetRecord":
        location = f" at row {row_index}" if row_index is not None else ""
        missing = [column for column in REQUIRED_QA_COLUMNS if column not in row]
        if missing:
            raise ValueError(f"Missing required columns{location}: {', '.join(missing)}")

        record_id = _require_non_empty_string(row["id"], "id", location)
        question = _require_non_empty_string(row["question"], "question", location)
        answer = _require_non_empty_string(row["answer"], "answer", location)
        doc_ids = _validate_doc_ids(row["doc_ids"], location)

        return cls(
            id=record_id,
            question=question,
            answer=answer,
            doc_ids=doc_ids,
        )


def load_qa_dataset(
    dataset_id: str,
    config_name: str | None = None,
    split: str = "train",
    revision: str | None = None,
    cache_dir: str | Path | None = None,
    token: str | bool | None = None,
) -> Dataset:
    dataset_id = _require_non_empty_string(dataset_id, "dataset_id")
    split = _require_non_empty_string(split, "split")

    load_kwargs: dict[str, Any] = {
        "path": dataset_id,
        "split": split,
    }
    if config_name is not None:
        load_kwargs["name"] = _require_non_empty_string(config_name, "config_name")
    if revision is not None:
        load_kwargs["revision"] = _require_non_empty_string(revision, "revision")
    if cache_dir is not None:
        load_kwargs["cache_dir"] = str(cache_dir)
    if token is not None:
        load_kwargs["token"] = token

    dataset = load_dataset(**load_kwargs)
    if not isinstance(dataset, Dataset):
        raise TypeError("Expected a single Dataset; provide a concrete split")

    validate_qa_dataset(dataset)
    return dataset


def validate_qa_dataset(dataset: Dataset) -> None:
    missing = [column for column in REQUIRED_QA_COLUMNS if column not in dataset.column_names]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(missing)}")

    seen_ids: set[str] = set()
    for row_index, row in enumerate(dataset):
        record = QADatasetRecord.from_row(row, row_index=row_index)
        if record.id in seen_ids:
            raise ValueError(f"Duplicate id at row {row_index}: {record.id}")
        seen_ids.add(record.id)


def download_record_documents(
    record: QADatasetRecord,
    dataset_id: str,
    revision: str | None = None,
    cache_dir: str | Path | None = None,
    token: str | bool | None = None,
) -> list[Path]:
    dataset_id = _require_non_empty_string(dataset_id, "dataset_id")
    download_kwargs: dict[str, Any] = {
        "repo_id": dataset_id,
        "repo_type": "dataset",
    }
    if revision is not None:
        download_kwargs["revision"] = _require_non_empty_string(revision, "revision")
    if cache_dir is not None:
        download_kwargs["cache_dir"] = str(cache_dir)
    if token is not None:
        download_kwargs["token"] = token

    local_paths: list[Path] = []
    repo_files: list[str] | None = None
    for doc_id in record.doc_ids:
        _validate_repo_relative_path(doc_id)
        try:
            downloaded_path = hf_hub_download(
                filename=doc_id,
                **download_kwargs,
            )
        except EntryNotFoundError:
            if repo_files is None:
                repo_files = list_repo_files(
                    repo_id=dataset_id,
                    repo_type="dataset",
                    revision=revision,
                    token=token,
                )

            resolved_doc_id = _resolve_unicode_normalized_path(doc_id, repo_files)
            if resolved_doc_id is None:
                raise

            _validate_repo_relative_path(resolved_doc_id)
            downloaded_path = hf_hub_download(
                filename=resolved_doc_id,
                **download_kwargs,
            )
        local_paths.append(Path(downloaded_path))

    return local_paths


def _resolve_unicode_normalized_path(
    requested_path: str,
    repo_files: Sequence[str],
) -> str | None:
    normalized_requested_path = unicodedata.normalize("NFC", requested_path)
    matches = [
        repo_path
        for repo_path in repo_files
        if unicodedata.normalize("NFC", repo_path) == normalized_requested_path
    ]

    if not matches:
        return None
    if len(matches) > 1:
        raise ValueError(
            f"Multiple repository files match the Unicode-normalized doc_id: {requested_path}"
        )
    return matches[0]


def _require_non_empty_string(
    value: Any,
    field_name: str,
    location: str = "",
) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string{location}")
    return value


def _validate_doc_ids(value: Any, location: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"doc_ids must be a non-empty list of strings{location}")

    doc_ids = tuple(value)
    if not doc_ids:
        raise ValueError(f"doc_ids must be a non-empty list of strings{location}")

    for doc_id in doc_ids:
        if not isinstance(doc_id, str) or not doc_id.strip():
            raise ValueError(f"doc_ids must contain only non-empty strings{location}")
        _validate_repo_relative_path(doc_id, location)

    return doc_ids


def _validate_repo_relative_path(path: str, location: str = "") -> None:
    if path.startswith("/") or "\\" in path:
        raise ValueError(f"doc_id must be a repository-relative POSIX path{location}: {path}")

    parts = path.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise ValueError(f"doc_id contains an unsafe path component{location}: {path}")
