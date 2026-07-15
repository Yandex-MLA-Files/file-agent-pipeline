"""Aggregate judge scores into a report."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def build_report(
    scored_df: pd.DataFrame,
    metric_names: tuple[str, ...],
    negative_example_pattern: str | None = None,
) -> dict:
    """Compute mean/median/min/max per metric.

    `negative_example_pattern` is an optional regex matched against
    `ground_truth` to additionally break metrics down over "negative"
    examples (typically: questions with no answer in the source document,
    where the model should admit it doesn't know rather than confabulate).
    This is left as a caller-supplied pattern rather than hardcoded, since
    how a dataset marks "no answer" — and in what language — is specific to
    that dataset, not something this package should assume.
    """
    report: dict = {"n_examples": len(scored_df)}

    is_negative = None
    if negative_example_pattern is not None:
        is_negative = scored_df["ground_truth"].str.contains(
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
    """Fallback for json.dump: convert numpy scalar types to native Python.

    pandas aggregations (.mean()/.min()/.max()/...) return numpy scalar
    types (np.float64, np.int64, np.bool_), and their exact type can differ
    across platforms/numpy versions depending on column dtype inference.
    np.float64 happens to subclass Python's float so it serializes fine on
    its own, but np.int64 and np.bool_ do not — rather than chase down
    which specific computation produced one on a given platform, handle
    the whole class of numpy scalar types at the serialization boundary.
    """
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
