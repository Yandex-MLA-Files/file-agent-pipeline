"""Sanity-check the eval pipeline.

Run:
    uv run python verify_setup.py

Always checks: imports, loading and validating a run file (no credentials
needed). If JUDGE_BASE_URL/JUDGE_API_KEY/JUDGE_MODEL are set, additionally
runs one real RagasJudge call to confirm the model responds and its output
parses correctly.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

print("1/3 Checking imports...")
try:
    from eval.report import build_report, save_report
    from eval.run_loader import RunValidationError, load_run
except ImportError as e:
    print(f"IMPORT ERROR: {e}")
    print("Run: uv sync")
    sys.exit(1)
print("    OK")

print("2/3 Loading and validating sample_run.parquet...")
sample_path = Path(__file__).resolve().parent / "sample_run.parquet"
if not sample_path.exists():
    print(f"ERROR: {sample_path} not found")
    sys.exit(1)
try:
    run_df = load_run(sample_path)
except RunValidationError as e:
    print(f"VALIDATION ERROR: {e}")
    sys.exit(1)
print(f"    OK, {len(run_df)} rows, schema is valid")

print("3/3 Checking RagasJudge...")
required_env = ("JUDGE_BASE_URL", "JUDGE_API_KEY", "JUDGE_MODEL")
if not all(os.environ.get(v) for v in required_env):
    print("    Skipped: JUDGE_BASE_URL / JUDGE_API_KEY / JUDGE_MODEL are not set.")
    print("    The base pipeline (loading + validation) works.")
    print("    To exercise a real judge call, set those 3 environment")
    print("    variables and run this script again.")
else:
    from eval.judge.ragas_judge import RagasJudge

    judge = RagasJudge()
    scored_df = judge.evaluate(run_df.head(1))  # 1 row — don't burn extra calls
    report = build_report(scored_df, judge.metric_names)
    save_report(report, scored_df, "verify_output")
    print("    OK, the judge responded and its output parsed correctly:")
    for metric in judge.metric_names:
        print(f"      {metric}: {report[metric]['mean']}")

print()
print("Done.")
