from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.judge.ragas_judge import RagasJudge
from eval.report import append_run_log, build_report, save_report
from eval.resumable import CheckpointError, IncompleteEvaluationError, evaluate_in_batches
from eval.run_loader import RunValidationError, load_run

JUDGE_NAME = "ragas"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="path to run_*.parquet")
    parser.add_argument("--out", required=True, help="output directory for the report")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5,
        help="rows evaluated before writing a checkpoint (default: 5)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="reuse compatible completed batches from --out",
    )
    parser.add_argument(
        "--negative-example-pattern",
        default=os.environ.get("JUDGE_NEGATIVE_EXAMPLE_PATTERN") or None,
        help="regex matched against answer (the reference/ground-truth column) "
        "to break metrics down over answerable and 'no answer in source' examples; "
        "default: JUDGE_NEGATIVE_EXAMPLE_PATTERN or off",
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

    out_dir = Path(args.out)
    judge = RagasJudge(
        usage_log_path=os.environ.get("JUDGE_USAGE_LOG_PATH") or out_dir / "usage_log.jsonl",
        trace_log_path=os.environ.get("JUDGE_TRACE_LOG_PATH") or out_dir / "judge_trace_log.jsonl",
    )
    try:
        evaluation = evaluate_in_batches(
            run_df,
            judge,
            run_path=args.run,
            out_dir=args.out,
            batch_size=args.batch_size,
            resume=args.resume,
            judge_model=os.environ.get("JUDGE_MODEL", ""),
            embedding_model=os.environ.get(
                "JUDGE_EMBEDDING_MODEL", "intfloat/multilingual-e5-small"
            ),
            judge_config=judge.checkpoint_config,
            on_checkpoint=lambda completed, total: print(
                f"Checkpoint saved: {completed}/{total} rows"
            ),
            on_incomplete=lambda ids, total: print(
                f"Incomplete row(s) deferred: {ids}; continuing with the remaining {total} rows"
            ),
        )
    except (CheckpointError, IncompleteEvaluationError, ValueError) as e:
        print(f"Cannot continue evaluation: {e}", file=sys.stderr)
        sys.exit(1)

    scored_df = evaluation.scored_df
    report = build_report(scored_df, judge.metric_names, args.negative_example_pattern)
    report_path = save_report(report, scored_df, args.out)
    append_run_log(report, judge.metric_names, args.run, args.out, JUDGE_NAME, args.runs_log)

    print(
        f"Done: {len(run_df)} examples "
        f"(processed: {evaluation.processed_rows}, resumed: {evaluation.resumed_rows}), "
        f"report saved to {report_path}"
    )
    for metric in judge.metric_names:
        print(f"  {metric}: mean={report[metric]['mean']}")


if __name__ == "__main__":
    main()
