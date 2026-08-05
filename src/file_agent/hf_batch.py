import hashlib
import json
import logging
import os
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from datasets import Dataset

from file_agent.document import Document
from file_agent.hf_dataset import QADatasetRecord, validate_qa_dataset
from file_agent.hf_rag import DocumentLoader, GeneratedQARecord, process_hf_qa_record
from file_agent.lancedb_retriever import DEFAULT_SEMANTIC_MODEL_NAME
from file_agent.llm.base import LLMClient
from file_agent.qa import build_qa_prompt
from file_agent.rag import load_documents
from file_agent.retrieval import Retriever

CHECKPOINT_SCHEMA_VERSION = 2
CHECKPOINTS_DIRECTORY_NAME = "checkpoints"
RAG_PIPELINE_VERSION = "section-token-small-to-big-v1"
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class BatchGenerationResult:
    records: tuple[GeneratedQARecord, ...]
    processed_count: int
    resumed_count: int

    @property
    def total_count(self) -> int:
        return len(self.records)


def generate_hf_qa_records(
    dataset: Dataset,
    dataset_id: str,
    llm_client: LLMClient,
    output_dir: str | Path,
    revision: str | None = None,
    cache_dir: str | Path | None = None,
    token: str | bool | None = None,
    top_k: int = 5,
    max_chars: int = 1000,
    overlap: int = 100,
    retriever: Retriever | None = None,
    resume: bool = False,
    use_router: bool = False,
    router_llm_client: LLMClient | None = None,
) -> BatchGenerationResult:
    validate_qa_dataset(dataset)
    if not isinstance(dataset_id, str) or not dataset_id.strip():
        raise ValueError("dataset_id must be a non-empty string")

    checkpoints_dir = Path(output_dir) / CHECKPOINTS_DIRECTORY_NAME
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    _validate_existing_checkpoint_files(
        checkpoints_dir=checkpoints_dir,
        dataset_size=len(dataset),
        resume=resume,
    )

    parameters = build_generation_parameters(
        dataset_id=dataset_id,
        revision=revision,
        llm_client=llm_client,
        retriever=retriever,
        top_k=top_k,
        max_chars=max_chars,
        overlap=overlap,
        use_router=use_router,
        router_llm_client=router_llm_client,
    )
    records: list[GeneratedQARecord] = []
    processed_count = 0
    resumed_count = 0
    document_loader = _create_cached_document_loader()

    for row_index, row in enumerate(dataset):
        record = QADatasetRecord.from_row(row, row_index=row_index)
        checkpoint_path = _checkpoint_path(checkpoints_dir, row_index)

        if checkpoint_path.exists():
            generated_record = _load_checkpoint(
                checkpoint_path=checkpoint_path,
                source_record=record,
                expected_parameters=parameters,
            )
            resumed_count += 1
            LOGGER.info(
                "Loaded checkpoint for row %s/%s (%s)",
                row_index + 1,
                len(dataset),
                record.id,
            )
        else:
            generated_record = process_hf_qa_record(
                record=record,
                dataset_id=dataset_id,
                llm_client=llm_client,
                revision=revision,
                cache_dir=cache_dir,
                token=token,
                top_k=top_k,
                max_chars=max_chars,
                overlap=overlap,
                retriever=retriever,
                document_loader=document_loader,
                use_router=use_router,
                router_llm_client=router_llm_client,
            )
            _validate_generated_record(generated_record, record)
            _write_checkpoint(
                checkpoint_path=checkpoint_path,
                parameters=parameters,
                generated_record=generated_record,
            )
            processed_count += 1
            LOGGER.info(
                "Processed row %s/%s (%s)",
                row_index + 1,
                len(dataset),
                record.id,
            )

        records.append(generated_record)

    return BatchGenerationResult(
        records=tuple(records),
        processed_count=processed_count,
        resumed_count=resumed_count,
    )


def _create_cached_document_loader() -> DocumentLoader:
    cache: dict[Path, Document] = {}

    def load_cached_documents(file_paths: list[str | Path]) -> list[Document]:
        documents: list[Document] = []

        for file_path in file_paths:
            source_path = Path(file_path)
            cache_key = source_path.resolve()
            cached_document = cache.get(cache_key)
            if cached_document is None:
                # Hugging Face snapshot files are symlinks to extensionless blob
                # paths. Use the resolved path only as the cache identity and
                # keep the original filename so parser selection still sees
                # extensions such as .pdf and .docx.
                cached_document = load_documents([source_path])[0]
                cache[cache_key] = cached_document
            else:
                LOGGER.info("Reusing parsed document from batch cache: %s", cache_key)

            # HF processing adds row-specific dataset metadata to every block.
            # Return an isolated copy so one question cannot mutate the cached
            # document or leak its metadata into another question.
            documents.append(deepcopy(cached_document))

        return documents

    return load_cached_documents


def build_generation_parameters(
    dataset_id: str,
    revision: str | None,
    llm_client: LLMClient,
    retriever: Retriever | None,
    top_k: int,
    max_chars: int,
    overlap: int,
    use_router: bool = False,
    router_llm_client: LLMClient | None = None,
) -> dict[str, Any]:
    prompt_template = build_qa_prompt(
        question="{question}",
        context="{context}",
    )
    return {
        "dataset_id": dataset_id,
        "revision": revision,
        "model_id": _model_identifier(llm_client),
        "temperature": _optional_scalar_attribute(llm_client, "temperature"),
        "max_tokens": _optional_scalar_attribute(llm_client, "max_tokens"),
        "router_model_id": (
            _model_identifier(router_llm_client) if router_llm_client is not None else None
        ),
        "retriever": _component_identifier(retriever) if retriever is not None else "default",
        "rag_pipeline_version": RAG_PIPELINE_VERSION,
        "use_router": use_router,
        "embedding_model": os.getenv("EMBEDDING_MODEL") or DEFAULT_SEMANTIC_MODEL_NAME,
        "ocr_engine": os.getenv("OCR_ENGINE", "easyocr").strip().lower(),
        "ocr_langs": os.getenv("OCR_LANGS", "ru,en").strip(),
        "vlm_backend": os.getenv("VLM_BACKEND", "off").strip().lower(),
        "vlm_model": _vlm_model_identifier(),
        "top_k": top_k,
        "max_chars": max_chars,
        "overlap": overlap,
        "prompt_sha256": hashlib.sha256(prompt_template.encode("utf-8")).hexdigest(),
    }


def _vlm_model_identifier() -> str | None:
    backend = os.getenv("VLM_BACKEND", "off").strip().lower()
    if backend == "smolvlm":
        return os.getenv(
            "VLM_LOCAL_MODEL",
            "HuggingFaceTB/SmolVLM-256M-Instruct",
        )
    if backend == "openai":
        return os.getenv("VLM_MODEL")
    return None


def _model_identifier(llm_client: LLMClient) -> str:
    model = getattr(llm_client, "model", None)
    if isinstance(model, str) and model.strip():
        return model
    return _component_identifier(llm_client)


def _component_identifier(component: object) -> str:
    component_type = type(component)
    return f"{component_type.__module__}.{component_type.__qualname__}"


def _optional_scalar_attribute(component: object, name: str) -> str | int | float | bool | None:
    value = getattr(component, name, None)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _validate_existing_checkpoint_files(
    checkpoints_dir: Path,
    dataset_size: int,
    resume: bool,
) -> None:
    existing_paths = sorted(checkpoints_dir.glob("*.json"))
    if existing_paths and not resume:
        raise FileExistsError(
            f"Checkpoints already exist in {checkpoints_dir}; use resume=True or a new output_dir"
        )

    expected_names = {_checkpoint_filename(row_index) for row_index in range(dataset_size)}
    unexpected_names = [path.name for path in existing_paths if path.name not in expected_names]
    if unexpected_names:
        raise ValueError(
            "Checkpoint directory contains rows outside the current dataset: "
            + ", ".join(unexpected_names)
        )


def _checkpoint_filename(row_index: int) -> str:
    return f"{row_index:06d}.json"


def _checkpoint_path(checkpoints_dir: Path, row_index: int) -> Path:
    return checkpoints_dir / _checkpoint_filename(row_index)


def _write_checkpoint(
    checkpoint_path: Path,
    parameters: dict[str, Any],
    generated_record: GeneratedQARecord,
) -> None:
    payload = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "parameters": parameters,
        "result": generated_record.to_dict(),
    }
    temporary_path = checkpoint_path.with_suffix(".json.tmp")
    temporary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary_path.replace(checkpoint_path)


def _load_checkpoint(
    checkpoint_path: Path,
    source_record: QADatasetRecord,
    expected_parameters: dict[str, Any],
) -> GeneratedQARecord:
    try:
        payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read checkpoint: {checkpoint_path}") from exc

    if not isinstance(payload, dict):
        raise ValueError(f"Checkpoint must contain a JSON object: {checkpoint_path}")
    if payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(f"Unsupported checkpoint schema: {checkpoint_path}")
    if payload.get("parameters") != expected_parameters:
        raise ValueError(f"Checkpoint parameters do not match the current run: {checkpoint_path}")

    result_value = payload.get("result")
    if not isinstance(result_value, dict):
        raise ValueError(f"Checkpoint result must be a JSON object: {checkpoint_path}")

    generated_record = GeneratedQARecord.from_dict(result_value)
    _validate_generated_record(generated_record, source_record)
    return generated_record


def _validate_generated_record(
    generated_record: GeneratedQARecord,
    source_record: QADatasetRecord,
) -> None:
    if generated_record.id != source_record.id:
        raise ValueError(f"Generated id does not match source record: {source_record.id}")
    if generated_record.question != source_record.question:
        raise ValueError(f"Generated question does not match source record: {source_record.id}")
    if generated_record.doc_ids != source_record.doc_ids:
        raise ValueError(f"Generated doc_ids do not match source record: {source_record.id}")
    if generated_record.answer != source_record.answer:
        raise ValueError(f"Generated gold answer does not match source record: {source_record.id}")
