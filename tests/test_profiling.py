import json

import numpy as np
import pandas as pd
import pytest

from unknown_dataset.profiling import build_profile


def test_profile_raw_missingness_constants_duplicates_and_summary():
    frame = pd.DataFrame({"x": [1.0, np.nan, 3.0, 5.0], "copy": [1.0, np.nan, 3.0, 5.0],
                          "empty": [np.nan] * 4, "fixed": [7.0] * 4,
                          "kind": ["a", "b", None, "a"]})
    types = {name: "categorical" if name == "kind" else "numeric" for name in frame}
    original = frame.copy(deep=True)
    target = pd.Series(["yes", "no", "yes", "yes"], name="label")
    profile = build_profile(frame, types, target)
    assert profile["row_count"] == 4
    assert profile["feature_count"] == 5
    assert profile["missing_count"] == 7
    assert profile["feature_type_counts"] == {"numeric": 4, "categorical": 1}
    assert set(profile["constant_columns"]) == {"empty", "fixed"}
    assert {"column": "copy", "duplicate_of": "x"} in profile["duplicate_columns"]
    assert profile["columns"]["x"]["missing_fraction"] == 0.25
    assert profile["columns"]["x"]["numeric_summary"]["mean"] == 3
    assert profile["columns"]["x"]["numeric_summary"]["std"] == pytest.approx(np.sqrt(8 / 3))
    assert profile["columns"]["empty"]["all_missing"]
    assert all(value is None for value in profile["columns"]["empty"]["numeric_summary"].values())
    assert {item["value"]: item["count"] for item in profile["target"]["distribution"]} == {"yes": 3, "no": 1}
    json.dumps(profile, allow_nan=False)
    pd.testing.assert_frame_equal(frame, original)


def test_different_missing_positions_are_not_duplicate_columns():
    frame = pd.DataFrame({"a": [1.0, np.nan], "b": [np.nan, 1.0]})
    assert build_profile(frame, {"a": "numeric", "b": "numeric"})["duplicate_columns"] == []


def test_profile_rejects_target_length_mismatch():
    with pytest.raises(ValueError):
        build_profile(pd.DataFrame({"x": [1, 2]}), {"x": "numeric"}, pd.Series([0]))
