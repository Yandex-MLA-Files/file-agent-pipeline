from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.judge.ragas_judge import RagasJudge
from eval.report import (
    append_run_log,
    build_report,
    log_mlflow_run,
    report_to_mlflow_metrics,
    save_report,
)
from eval.run_loader import RunValidationError, load_run

JUDGE_NAME = "ragas"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="path to run_*.parquet")
    parser.add_argument("--out", required=True, help="output directory for the report")
    parser.add_argument(
        "--negative-example-pattern",
        default=None,
        help="regex matched against answer (the reference/ground-truth column) "
        "to break metrics down over 'no answer in source' examples; "
        "dataset-specific, off by default",
    )
    parser.add_argument(
        "--runs-log",
        default="logs/runs_log.jsonl",
        help="append a one-line summary (timestamp, run file, out dir, judge, "
        "per-metric means) here after every run, so separate runs (e.g. "
        "different RAG versions) can be compared -- unlike --out, this is "
        "never overwritten",
    )
    args = parser.parse_args()

    try:
        run_df = load_run(args.run)
    except RunValidationError as e:
        print(f"Run file failed validation: {e}", file=sys.stderr)
        sys.exit(1)

    judge = RagasJudge()
    scored_df = judge.evaluate(run_df)
    report = build_report(scored_df, judge.metric_names, args.negative_example_pattern)
    report_path = save_report(report, scored_df, args.out)
    append_run_log(report, judge.metric_names, args.run, args.out, JUDGE_NAME, args.runs_log)

    metrics = report_to_mlflow_metrics(report, judge.metric_names)
    if judge.last_usage is not None:
        metrics["cost_rub"] = judge.last_usage["cost_rub"]
        metrics["input_tokens"] = judge.last_usage["input_tokens"]
        metrics["output_tokens"] = judge.last_usage["output_tokens"]
        metrics["cached_tokens"] = judge.last_usage["cached_tokens"]

    log_mlflow_run(
        run_name=Path(args.out).name,
        params={
            "run_file": str(args.run),
            "judge": JUDGE_NAME,
            "judge_model": os.environ.get("JUDGE_MODEL", ""),
        },
        metrics=metrics,
        artifact_paths=[
            path
            for path in (report_path, Path(args.out) / "scored.parquet", judge.last_trace_path)
            if path is not None
        ],
    )

    print(f"Done: {len(run_df)} examples, report saved to {report_path}")
    for metric in judge.metric_names:
        print(f"  {metric}: mean={report[metric]['mean']}")


if __name__ == "__main__":
    main()
