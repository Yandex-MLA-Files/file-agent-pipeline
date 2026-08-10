"""Compare two scored runs (e.g. v0 baseline vs v1 router), segmented by query_type.

query_type isn't part of the run schema (RunValidationError would reject it),
so it isn't in scored.parquet either -- it comes from a separate CSV built by
../scripts/classify_dataset_queries.py in the main repo (id, question,
query_type), joined in here by `id`.

Usage:
    uv run --env-file .env python scripts/compare_runs.py \
        --baseline reports/v0-qwen-002/scored.parquet \
        --candidate reports/v1-router-002/scored.parquet \
        --query-types query_types.csv
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.report import log_mlflow_run  # noqa: E402

METRICS = (
    "faithfulness",
    "answer_correctness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path, help="baseline scored.parquet")
    parser.add_argument("--candidate", required=True, type=Path, help="candidate scored.parquet")
    parser.add_argument("--query-types", required=True, type=Path, help="id,question,query_type")
    parser.add_argument("--baseline-name", default="baseline")
    parser.add_argument("--candidate-name", default="candidate")
    args = parser.parse_args()

    query_types = pd.read_csv(args.query_types)[["id", "query_type"]]

    baseline = pd.read_parquet(args.baseline).merge(query_types, on="id", how="left")
    candidate = pd.read_parquet(args.candidate).merge(query_types, on="id", how="left")

    missing_baseline = baseline["query_type"].isna().sum()
    missing_candidate = candidate["query_type"].isna().sum()
    if missing_baseline or missing_candidate:
        print(
            f"Warning: {missing_baseline} baseline / {missing_candidate} candidate rows "
            "have no query_type match (id mismatch with the CSV) -- excluded below."
        )

    for query_type in ("simple", "complex", "tool"):
        base_group = baseline[baseline["query_type"] == query_type]
        cand_group = candidate[candidate["query_type"] == query_type]
        if base_group.empty and cand_group.empty:
            continue

        print(
            f"\n== query_type={query_type} (n_baseline={len(base_group)}, "
            f"n_candidate={len(cand_group)}) =="
        )
        print(f"{'metric':<20}{args.baseline_name:>12}{args.candidate_name:>12}{'delta':>10}")
        segment_metrics = {}
        for metric in METRICS:
            base_mean = base_group[metric].mean()
            cand_mean = cand_group[metric].mean()
            delta = cand_mean - base_mean
            print(f"{metric:<20}{base_mean:>12.4f}{cand_mean:>12.4f}{delta:>+10.4f}")
            segment_metrics[f"{args.baseline_name}_{metric}_mean"] = float(base_mean)
            segment_metrics[f"{args.candidate_name}_{metric}_mean"] = float(cand_mean)
            segment_metrics[f"delta_{metric}_mean"] = float(delta)

        log_mlflow_run(
            run_name=f"{args.candidate_name}_vs_{args.baseline_name}_{query_type}",
            params={
                "baseline_name": args.baseline_name,
                "candidate_name": args.candidate_name,
                "query_type": query_type,
                "n_baseline": len(base_group),
                "n_candidate": len(cand_group),
            },
            metrics=segment_metrics,
            tags={"query_type": query_type},
        )


if __name__ == "__main__":
    main()
