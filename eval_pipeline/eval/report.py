
from __future__ import annotations

from pathlib import Path

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

    import numpy as np

    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def save_report(report: dict, scored_df: pd.DataFrame, out_dir: str | Path) -> Path:
    """Save the summary (report.json) and per-row scores (scored.parquet)."""
    import json

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

    import json
    from datetime import UTC, datetime

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
