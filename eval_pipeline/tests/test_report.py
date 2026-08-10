import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.report import build_report


def _scored_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"answer": "regular reference answer", "faithfulness": 0.9},
            {"answer": "regular reference answer", "faithfulness": 0.8},
            {"answer": "no information available in the document", "faithfulness": 0.2},
        ]
    )


def test_report_without_pattern_has_no_negative_breakdown():
    report = build_report(_scored_df(), ("faithfulness",))
    assert "n_negative_examples" not in report
    assert "mean_on_negative_examples" not in report["faithfulness"]
    assert report["n_examples"] == 3


def test_report_with_pattern_adds_negative_breakdown():
    report = build_report(
        _scored_df(), ("faithfulness",), negative_example_pattern="no information"
    )
    assert report["n_negative_examples"] == 1
    assert report["faithfulness"]["mean_on_negative_examples"] == 0.2


def test_report_pattern_matching_nothing_is_not_an_error():
    report = build_report(_scored_df(), ("faithfulness",), negative_example_pattern="xyz-no-match")
    assert report["n_negative_examples"] == 0
    assert "mean_on_negative_examples" not in report["faithfulness"]


def test_save_report_serializes_numpy_int64(tmp_path):

    import json

    import numpy as np

    from eval.report import save_report

    scored_df = pd.DataFrame({"answer": ["a", "b"], "faithfulness": [1.0, 0.0]})
    report = {
        "n_examples": np.int64(2),
        "faithfulness": {"mean": np.float64(0.5), "min": np.int64(0), "max": np.int64(1)},
        "flagged": np.bool_(True),
    }

    report_path = save_report(report, scored_df, tmp_path)

    with open(report_path, encoding="utf-8") as f:
        loaded = json.load(f)
    assert loaded["n_examples"] == 2
    assert loaded["faithfulness"]["min"] == 0
    assert loaded["flagged"] is True


def test_save_report_still_raises_on_genuinely_unserializable_object(tmp_path):
    from eval.report import save_report

    scored_df = pd.DataFrame({"answer": ["a"], "faithfulness": [1.0]})

    with pytest.raises(TypeError, match="not JSON serializable"):
        save_report({"bad": object()}, scored_df, tmp_path)


def test_append_run_log_writes_one_line_per_call(tmp_path):
    import json

    from eval.report import append_run_log

    log_path = tmp_path / "runs_log.jsonl"
    report = {"n_examples": 2, "faithfulness": {"mean": 0.9}}

    append_run_log(report, ("faithfulness",), "run.parquet", "reports/v1", "ragas", log_path)
    append_run_log(report, ("faithfulness",), "run.parquet", "reports/v2", "ragas", log_path)

    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    entry = json.loads(lines[0])
    assert entry["run_file"] == "run.parquet"
    assert entry["out_dir"] == "reports/v1"
    assert entry["judge"] == "ragas"
    assert entry["n_examples"] == 2
    assert entry["faithfulness_mean"] == 0.9
    assert "timestamp" in entry


def test_report_to_mlflow_metrics_flattens_per_metric_stats():
    from eval.report import report_to_mlflow_metrics

    report = {
        "n_examples": 3,
        "faithfulness": {"mean": 0.9, "median": 0.9, "min": 0.8, "max": 1.0},
    }

    metrics = report_to_mlflow_metrics(report, ("faithfulness",))

    assert metrics == {
        "n_examples": 3.0,
        "faithfulness_mean": 0.9,
        "faithfulness_median": 0.9,
        "faithfulness_min": 0.8,
        "faithfulness_max": 1.0,
    }


def test_log_mlflow_run_is_a_noop_without_tracking_uri(monkeypatch):
    import mlflow

    from eval.report import log_mlflow_run

    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.setattr(
        mlflow, "start_run", lambda **kwargs: (_ for _ in ()).throw(AssertionError("must not run"))
    )

    log_mlflow_run(run_name="v0", params={"a": 1}, metrics={"m": 1.0})


def test_log_mlflow_run_logs_params_metrics_and_tags_when_configured(monkeypatch):
    import mlflow

    from eval.report import log_mlflow_run

    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://localhost:5050")
    monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", "test-experiment")
    calls = {}

    class FakeRunContext:
        def __enter__(self):
            return self

        def __exit__(self, *exc_info):
            return False

    def fake_start_run(**kwargs):
        calls["run_name"] = kwargs.get("run_name")
        return FakeRunContext()

    monkeypatch.setattr(mlflow, "set_tracking_uri", lambda uri: calls.__setitem__("uri", uri))
    monkeypatch.setattr(
        mlflow, "set_experiment", lambda name: calls.__setitem__("experiment", name)
    )
    monkeypatch.setattr(mlflow, "start_run", fake_start_run)
    monkeypatch.setattr(mlflow, "log_params", lambda params: calls.__setitem__("params", params))
    monkeypatch.setattr(
        mlflow, "log_metrics", lambda metrics: calls.__setitem__("metrics", metrics)
    )
    monkeypatch.setattr(mlflow, "set_tags", lambda tags: calls.__setitem__("tags", tags))

    log_mlflow_run(
        run_name="v0-qwen-002",
        params={"judge": "ragas"},
        metrics={"faithfulness_mean": 0.9},
        tags={"query_type": "complex"},
    )

    assert calls["uri"] == "http://localhost:5050"
    assert calls["experiment"] == "test-experiment"
    assert calls["run_name"] == "v0-qwen-002"
    assert calls["params"] == {"judge": "ragas"}
    assert calls["metrics"] == {"faithfulness_mean": 0.9}
    assert calls["tags"] == {"query_type": "complex"}
