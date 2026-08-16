from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd

CHECKPOINT_VERSION = 1
CHECKPOINT_FILENAME = "checkpoint.parquet"
MANIFEST_FILENAME = "checkpoint_manifest.json"


class CheckpointError(RuntimeError):
    """Raised when a checkpoint cannot safely be used."""


class Judge(Protocol):
    metric_names: tuple[str, ...]

    def evaluate(self, run_df: pd.DataFrame) -> pd.DataFrame: ...


@dataclass(frozen=True)
class ResumableEvaluationResult:
    scored_df: pd.DataFrame
    processed_rows: int
    resumed_rows: int
    batches: int


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _manifest(
    run_path: Path,
    run_df: pd.DataFrame,
    metric_names: tuple[str, ...],
    judge_model: str,
    embedding_model: str,
) -> dict:
    return {
        "version": CHECKPOINT_VERSION,
        "run_path": str(run_path.resolve()),
        "run_sha256": _sha256(run_path),
        "n_rows": len(run_df),
        "metric_names": list(metric_names),
        "judge_model": judge_model,
        "embedding_model": embedding_model,
    }


def _atomic_write_json(value: dict, path: Path) -> None:
    temporary_path = path.with_name(f"{path.name}.tmp")
    with temporary_path.open("w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
    os.replace(temporary_path, path)


def _atomic_write_parquet(df: pd.DataFrame, path: Path) -> None:
    temporary_path = path.with_name(f"{path.stem}.tmp{path.suffix}")
    df.to_parquet(temporary_path, index=False)
    os.replace(temporary_path, path)


def _validate_manifest(saved: dict, expected: dict) -> None:
    checked_fields = (
        "version",
        "run_sha256",
        "n_rows",
        "metric_names",
        "judge_model",
        "embedding_model",
    )
    mismatches = [field for field in checked_fields if saved.get(field) != expected.get(field)]
    if mismatches:
        details = ", ".join(
            f"{field}: saved={saved.get(field)!r}, current={expected.get(field)!r}"
            for field in mismatches
        )
        raise CheckpointError(
            "checkpoint does not match this evaluation configuration; "
            f"start with a new --out directory or omit --resume ({details})"
        )


def _validate_checkpoint(
    checkpoint_df: pd.DataFrame,
    run_df: pd.DataFrame,
    metric_names: tuple[str, ...],
) -> None:
    required_columns = set(run_df.columns) | set(metric_names)
    missing = sorted(required_columns - set(checkpoint_df.columns))
    if missing:
        raise CheckpointError(f"checkpoint is missing required columns: {missing}")
    if checkpoint_df["id"].isnull().any():
        raise CheckpointError("checkpoint contains an empty id")
    if checkpoint_df["id"].duplicated().any():
        raise CheckpointError("checkpoint contains duplicate ids")

    unexpected_ids = checkpoint_df.loc[~checkpoint_df["id"].isin(run_df["id"]), "id"].tolist()
    if unexpected_ids:
        raise CheckpointError(f"checkpoint contains ids absent from the run: {unexpected_ids[:5]}")

    metric_values = checkpoint_df.loc[:, metric_names].apply(pd.to_numeric, errors="coerce")
    complete = metric_values.notna().all(axis=1) & np.isfinite(metric_values).all(axis=1)
    if not complete.all():
        bad_ids = checkpoint_df.loc[~complete, "id"].tolist()
        raise CheckpointError(
            f"checkpoint contains incomplete metric scores for ids: {bad_ids[:5]}"
        )


def _sort_like_input(scored_df: pd.DataFrame, run_df: pd.DataFrame) -> pd.DataFrame:
    order = {row_id: position for position, row_id in enumerate(run_df["id"])}
    result = scored_df.copy()
    result["_input_order"] = result["id"].map(order)
    result = result.sort_values("_input_order").drop(columns="_input_order")
    return result.reset_index(drop=True)


def _prepare_state(
    out_dir: Path,
    expected_manifest: dict,
    run_df: pd.DataFrame,
    metric_names: tuple[str, ...],
    resume: bool,
) -> pd.DataFrame | None:
    checkpoint_path = out_dir / CHECKPOINT_FILENAME
    manifest_path = out_dir / MANIFEST_FILENAME

    if not resume:
        for path in (
            checkpoint_path,
            manifest_path,
            out_dir / "report.json",
            out_dir / "scored.parquet",
        ):
            path.unlink(missing_ok=True)
        _atomic_write_json(expected_manifest, manifest_path)
        return None

    if checkpoint_path.exists() and not manifest_path.exists():
        raise CheckpointError(
            f"checkpoint exists without its manifest: {checkpoint_path}; "
            "use a new --out directory or omit --resume to restart"
        )

    if manifest_path.exists():
        try:
            saved_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise CheckpointError(f"cannot read checkpoint manifest: {manifest_path}") from exc
        _validate_manifest(saved_manifest, expected_manifest)
    else:
        _atomic_write_json(expected_manifest, manifest_path)

    if not checkpoint_path.exists():
        return None

    try:
        checkpoint_df = pd.read_parquet(checkpoint_path)
    except Exception as exc:
        raise CheckpointError(f"cannot read checkpoint: {checkpoint_path}") from exc
    _validate_checkpoint(checkpoint_df, run_df, metric_names)
    return _sort_like_input(checkpoint_df, run_df)


def evaluate_in_batches(
    run_df: pd.DataFrame,
    judge: Judge,
    *,
    run_path: str | Path,
    out_dir: str | Path,
    batch_size: int,
    resume: bool,
    judge_model: str,
    embedding_model: str,
    on_checkpoint: Callable[[int, int], None] | None = None,
) -> ResumableEvaluationResult:
    """Evaluate a run in checkpointed batches and optionally resume it.

    Only rows for which every metric has a finite score are checkpointed. A
    failure can therefore require repeating at most the currently running
    batch, while completed batches are reused by ``resume=True``.
    """

    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero")

    run_path = Path(run_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metric_names = tuple(judge.metric_names)
    expected_manifest = _manifest(run_path, run_df, metric_names, judge_model, embedding_model)
    checkpoint_df = _prepare_state(out_dir, expected_manifest, run_df, metric_names, resume)
    resumed_rows = 0 if checkpoint_df is None else len(checkpoint_df)
    completed_ids = set() if checkpoint_df is None else set(checkpoint_df["id"])
    pending_df = run_df.loc[~run_df["id"].isin(completed_ids)].reset_index(drop=True)

    processed_rows = 0
    batches = 0
    checkpoint_path = out_dir / CHECKPOINT_FILENAME

    for start in range(0, len(pending_df), batch_size):
        batch_df = pending_df.iloc[start : start + batch_size].copy()
        scored_batch = judge.evaluate(batch_df)
        batches += 1

        missing_metrics = sorted(set(metric_names) - set(scored_batch.columns))
        if missing_metrics:
            raise RuntimeError(f"judge result is missing metric columns: {missing_metrics}")
        if list(scored_batch["id"]) != list(batch_df["id"]):
            raise RuntimeError("judge result ids or row order do not match the input batch")

        metric_values = scored_batch.loc[:, metric_names].apply(pd.to_numeric, errors="coerce")
        complete = metric_values.notna().all(axis=1) & np.isfinite(metric_values).all(axis=1)
        completed_batch = scored_batch.loc[complete].copy()

        if not completed_batch.empty:
            checkpoint_df = (
                completed_batch
                if checkpoint_df is None
                else pd.concat([checkpoint_df, completed_batch], ignore_index=True)
            )
            checkpoint_df = _sort_like_input(checkpoint_df, run_df)
            _atomic_write_parquet(checkpoint_df, checkpoint_path)
            processed_rows += len(completed_batch)
            if on_checkpoint is not None:
                on_checkpoint(len(checkpoint_df), len(run_df))

        if not complete.all():
            incomplete_ids = scored_batch.loc[~complete, "id"].tolist()
            raise RuntimeError(
                "judge returned incomplete metric scores; completed rows were checkpointed, "
                f"but these ids must be retried with --resume: {incomplete_ids}"
            )

    if checkpoint_df is None:
        checkpoint_df = run_df.copy()
        for metric in metric_names:
            checkpoint_df[metric] = pd.Series(dtype="float64")

    if len(checkpoint_df) != len(run_df):
        raise RuntimeError(
            f"evaluation stopped with {len(checkpoint_df)}/{len(run_df)} completed rows"
        )

    return ResumableEvaluationResult(
        scored_df=_sort_like_input(checkpoint_df, run_df),
        processed_rows=processed_rows,
        resumed_rows=resumed_rows,
        batches=batches,
    )
