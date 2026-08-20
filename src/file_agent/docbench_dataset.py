import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

DOCBENCH_DOMAINS = ("academia", "finance", "government", "law", "news")
DOCBENCH_QUESTION_TYPES = (
    "text-only",
    "multimodal-f",
    "multimodal-t",
    "multimodal",
    "meta-data",
    "una",
    "unanswerable",
    "una-web",
)


@dataclass(frozen=True)
class DocBenchRecord:
    """One question and its local DocBench source document."""

    id: str
    folder_id: int
    question_index: int
    question: str
    reference_answer: str
    question_type: str
    evidence: str
    domain: str
    pdf_path: Path
    qa_path: Path

    @property
    def question_number(self) -> int:
        return self.question_index + 1

    @property
    def source_file(self) -> str:
        return self.pdf_path.name

    def source_payload(self) -> dict[str, str | int]:
        """Return the immutable benchmark fields used to validate checkpoints."""
        return {
            "id": self.id,
            "folder_id": self.folder_id,
            "question_index": self.question_index,
            "question": self.question,
            "reference_answer": self.reference_answer,
            "question_type": self.question_type,
            "evidence": self.evidence,
            "domain": self.domain,
            "source_file": self.source_file,
        }


def load_docbench_records(data_dir: str | Path) -> tuple[DocBenchRecord, ...]:
    """Load the official local ``DocBench/data`` directory in stable order."""
    root = Path(data_dir)
    if not root.exists():
        raise FileNotFoundError(f"DocBench data directory does not exist: {root}")
    if not root.is_dir():
        raise ValueError(f"DocBench data path is not a directory: {root}")

    folders = sorted(
        (path for path in root.iterdir() if path.is_dir() and path.name.isdigit()),
        key=lambda path: int(path.name),
    )
    if not folders:
        raise ValueError(f"No numeric DocBench document folders found in {root}")

    records: list[DocBenchRecord] = []
    ids: set[str] = set()
    for folder in folders:
        folder_id = int(folder.name)
        pdf_path = _find_single_pdf(folder)
        qa_path = _find_qa_file(folder, folder_id)
        for question_index, row in enumerate(_read_jsonl(qa_path)):
            record = _record_from_row(
                row=row,
                folder_id=folder_id,
                question_index=question_index,
                pdf_path=pdf_path,
                qa_path=qa_path,
            )
            if record.id in ids:
                raise ValueError(f"Duplicate DocBench record ID: {record.id}")
            ids.add(record.id)
            records.append(record)

    if not records:
        raise ValueError(f"DocBench contains no questions: {root}")
    return tuple(records)


def select_docbench_records(
    records: Sequence[DocBenchRecord],
    *,
    folder_start: int | None = None,
    folder_end: int | None = None,
    folder_ids: Iterable[int] = (),
    record_ids: Iterable[str] = (),
    domains: Iterable[str] = (),
    question_types: Iterable[str] = (),
    limit: int | None = None,
) -> tuple[DocBenchRecord, ...]:
    """Filter records without changing their canonical benchmark order."""
    selected = tuple(records)
    folder_id_set = set(folder_ids)
    record_id_values = tuple(record_ids)
    record_id_set = set(record_id_values)
    domain_set = set(domains)
    question_type_set = set(question_types)

    if folder_start is not None:
        selected = tuple(record for record in selected if record.folder_id >= folder_start)
    if folder_end is not None:
        selected = tuple(record for record in selected if record.folder_id <= folder_end)
    if folder_id_set:
        selected = tuple(record for record in selected if record.folder_id in folder_id_set)
    if record_id_set:
        available_ids = {record.id for record in records}
        missing_ids = [
            record_id for record_id in record_id_values if record_id not in available_ids
        ]
        if missing_ids:
            raise ValueError("Unknown DocBench record IDs: " + ", ".join(missing_ids))
        selected = tuple(record for record in selected if record.id in record_id_set)
    if domain_set:
        selected = tuple(record for record in selected if record.domain in domain_set)
    if question_type_set:
        selected = tuple(record for record in selected if record.question_type in question_type_set)
    if limit is not None:
        selected = selected[:limit]

    if not selected:
        raise ValueError("The selected DocBench subset contains no questions")
    return selected


def docbench_domain(folder_id: int) -> str:
    if 0 <= folder_id <= 48:
        return "academia"
    if folder_id <= 88:
        return "finance"
    if folder_id <= 132:
        return "government"
    if folder_id <= 178:
        return "law"
    if folder_id <= 228:
        return "news"
    return "unknown"


def _record_from_row(
    row: object,
    folder_id: int,
    question_index: int,
    pdf_path: Path,
    qa_path: Path,
) -> DocBenchRecord:
    if not isinstance(row, dict):
        raise ValueError(f"DocBench row must be an object: {qa_path}:{question_index + 1}")

    question = _required_string(row, "question", qa_path, question_index)
    reference_answer = _string_field(row, "answer", qa_path, question_index)
    question_type = _required_string(row, "type", qa_path, question_index)
    evidence = _string_field(row, "evidence", qa_path, question_index)
    return DocBenchRecord(
        id=f"docbench-{folder_id:03d}-q{question_index + 1:04d}",
        folder_id=folder_id,
        question_index=question_index,
        question=question,
        reference_answer=reference_answer,
        question_type=question_type,
        evidence=evidence,
        domain=docbench_domain(folder_id),
        pdf_path=pdf_path,
        qa_path=qa_path,
    )


def _find_single_pdf(folder: Path) -> Path:
    paths = sorted(
        (path for path in folder.iterdir() if path.is_file() and path.suffix.casefold() == ".pdf"),
        key=lambda path: path.name.casefold(),
    )
    if len(paths) != 1:
        raise ValueError(f"Expected exactly one PDF in {folder}, found {len(paths)}")
    return paths[0]


def _find_qa_file(folder: Path, folder_id: int) -> Path:
    expected_name = f"{folder_id}_qa.jsonl".casefold()
    paths = sorted(
        (
            path
            for path in folder.iterdir()
            if path.is_file() and path.name.casefold().endswith("_qa.jsonl")
        ),
        key=lambda path: path.name.casefold(),
    )
    exact = [path for path in paths if path.name.casefold() == expected_name]
    if len(exact) == 1:
        return exact[0]
    if len(paths) == 1:
        return paths[0]
    raise ValueError(f"Expected exactly one QA JSONL file in {folder}, found {len(paths)}")


def _read_jsonl(path: Path) -> list[object]:
    rows: list[object] = []
    try:
        with path.open("r", encoding="utf-8-sig") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON in {path}:{line_number}") from exc
    except OSError as exc:
        raise ValueError(f"Could not read DocBench QA file: {path}") from exc
    if not rows:
        raise ValueError(f"DocBench QA file is empty: {path}")
    return rows


def _required_string(row: dict, key: str, path: Path, row_index: int) -> str:
    value = _string_field(row, key, path, row_index)
    if not value:
        raise ValueError(f"DocBench field {key!r} is empty: {path}:{row_index + 1}")
    return value


def _string_field(row: dict, key: str, path: Path, row_index: int) -> str:
    value = row.get(key)
    if not isinstance(value, str):
        raise ValueError(f"DocBench field {key!r} must be a string: {path}:{row_index + 1}")
    return value.strip()
