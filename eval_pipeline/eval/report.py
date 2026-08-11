from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd


def build_report(
    scored_df: pd.DataFrame,
    metric_names: tuple[str, ...],
    negative_example_pattern: str | None = None,
) -> dict:

    report: dict = {"n_examples": len(scored_df)}

    is_negative = None
    if negative_example_pattern is not None:
        is_negative = scored_df["answer"].str.contains(
            negative_example_pattern, case=False, na=False, regex=True
        )
        report["n_negative_examples"] = int(is_negative.sum())

    for metric in metric_names:
        report[metric] = {
            "mean": round(scored_df[metric].mean(), 4),
            "median": round(scored_df[metric].median(), 4),
            "min": round(scored_df[metric].min(), 4),
            "max": round(scored_df[metric].max(), 4),
        }
        if is_negative is not None and is_negative.any():
            report[metric]["mean_on_negative_examples"] = round(
                scored_df.loc[is_negative, metric].mean(), 4
            )

    return report


def _json_default(obj):

    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def save_report(report: dict, scored_df: pd.DataFrame, out_dir: str | Path) -> Path:
    """Save the summary (report.json) and per-row scores (scored.parquet)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=_json_default)

    scored_df.to_parquet(out_dir / "scored.parquet", index=False)
    return out_dir / "report.json"


def append_run_log(
    report: dict,
    metric_names: tuple[str, ...],
    run_path: str | Path,
    out_dir: str | Path,
    judge_name: str,
    log_path: str | Path = "runs_log.jsonl",
) -> None:

    entry = {
        "timestamp": datetime.now(UTC).isoformat(),
        "run_file": str(run_path),
        "out_dir": str(out_dir),
        "judge": judge_name,
        "n_examples": report["n_examples"],
    }
    for metric in metric_names:
        entry[f"{metric}_mean"] = report[metric]["mean"]

    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def report_to_mlflow_metrics(report: dict, metric_names: tuple[str, ...]) -> dict[str, float]:
    """Flatten a report's per-metric mean/median/min/max into MLflow metric names."""
    metrics = {"n_examples": float(report["n_examples"])}
    for metric in metric_names:
        for stat in ("mean", "median", "min", "max"):
            metrics[f"{metric}_{stat}"] = float(report[metric][stat])
    return metrics


def log_mlflow_run(
    run_name: str,
    params: dict[str, Any],
    metrics: dict[str, float],
    tags: dict[str, str] | None = None,
    artifact_paths: list[str | Path] | None = None,
) -> None:
    """Log one MLflow run for comparing eval runs (params + RagasJudge metrics).

    A silent no-op when MLFLOW_TRACKING_URI isn't set, so a missing/unreachable
    MLflow server never breaks a normal `run_eval.py`/`compare_runs.py` call —
    same posture as Langfuse tracing in the main app.

    artifact_paths are attached as downloadable files on the run (report.json,
    scored.parquet, the judge's per-row reasoning trace, ...) - skips any path
    that doesn't exist rather than failing the whole run, since none of these
    are more essential than the metrics themselves.
    """
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        return

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(os.environ.get("MLFLOW_EXPERIMENT_NAME", "file-agent-eval"))
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        if tags:
            mlflow.set_tags(tags)
        for artifact_path in artifact_paths or []:
            if Path(artifact_path).is_file():
                mlflow.log_artifact(str(artifact_path))
