from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.judge.ragas_judge import RagasJudge
from eval.report import append_run_log, build_report, save_report
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

    print(f"Done: {len(run_df)} examples, report saved to {report_path}")
    for metric in judge.metric_names:
        print(f"  {metric}: mean={report[metric]['mean']}")


if __name__ == "__main__":
    main()
