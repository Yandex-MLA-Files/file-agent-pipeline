import os

import pytest

# Set before any test module can import ragas, so telemetry is off for the
# whole test run regardless of import order within a given test file.
os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")


@pytest.fixture(autouse=True)
def _isolate_usage_log(tmp_path, monkeypatch):
    # RagasJudge.evaluate() appends a cost-tracking line on every call; point
    # it at a per-test tmp file so tests never write into the project's real
    # usage_log.jsonl.
    monkeypatch.setenv("JUDGE_USAGE_LOG_PATH", str(tmp_path / "usage_log.jsonl"))
