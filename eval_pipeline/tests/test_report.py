import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.report import build_report


def _scored_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"ground_truth": "regular reference answer", "faithfulness": 0.9},
            {"ground_truth": "regular reference answer", "faithfulness": 0.8},
            {"ground_truth": "no information available in the document", "faithfulness": 0.2},
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
    # reproduces the exact bug: a metric column that ends up int64-dtype
    # (platform/numpy-version dependent) must still serialize to JSON fine.
    import json

    import numpy as np

    from eval.report import save_report

    scored_df = pd.DataFrame({"ground_truth": ["a", "b"], "faithfulness": [1.0, 0.0]})
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

    scored_df = pd.DataFrame({"ground_truth": ["a"], "faithfulness": [1.0]})

    with pytest.raises(TypeError, match="not JSON serializable"):
        save_report({"bad": object()}, scored_df, tmp_path)
