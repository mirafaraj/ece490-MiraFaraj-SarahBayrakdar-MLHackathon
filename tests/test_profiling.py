import json
from collections import Counter

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
    assert profile["columns"]["kind"]["categorical_summary"] == {
        "top_values": [{"value": "a", "count": 2}, {"value": "b", "count": 1}],
        "omitted_category_count": 0, "omitted_value_count": 0,
    }
    json.dumps(profile, allow_nan=False)
    pd.testing.assert_frame_equal(frame, original)


def test_different_missing_positions_are_not_duplicate_columns():
    frame = pd.DataFrame({"a": [1.0, np.nan], "b": [np.nan, 1.0]})
    assert build_profile(frame, {"a": "numeric", "b": "numeric"})["duplicate_columns"] == []


def test_profile_rejects_target_length_mismatch():
    with pytest.raises(ValueError):
        build_profile(pd.DataFrame({"x": [1, 2]}), {"x": "numeric"}, pd.Series([0]))


def test_categorical_counts_use_full_training_data_with_bounded_output():
    values = [f"{i:05d}" for i in range(5000)] * 6 + ["late"] * 10000 + [None] * 5000
    frame = pd.DataFrame({"category": values})
    original = frame.copy(deep=True)
    profile = build_profile(frame, {"category": "categorical"})
    entry = profile["columns"]["category"]
    summary = entry["categorical_summary"]
    # Independent full-data Python counter, not production value_counts/nlargest.
    expected = Counter(value for value in values if value is not None)
    expected_top = expected.most_common(20)
    assert summary["top_values"] == [{"value": v, "count": n} for v, n in expected_top]
    assert summary["top_values"][0] == {"value": "late", "count": 10000}
    assert summary["omitted_category_count"] == len(expected) - len(expected_top)
    assert summary["omitted_value_count"] == sum(expected.values()) - sum(n for _, n in expected_top)
    assert entry["missing_count"] == 5000
    assert sum(v["count"] for v in summary["top_values"]) + summary["omitted_value_count"] + entry["missing_count"] == len(values)
    assert "missing_count" not in summary  # Already reported explicitly in the column entry.
    assert len(json.dumps(summary, allow_nan=False)) < 2000
    pd.testing.assert_frame_equal(frame, original)


def test_categorical_lexical_values_and_true_missing_counts_are_distinct():
    frame = pd.DataFrame({"kind": ["01", "1", "NA", "nan", "", None, np.nan, pd.NA]})
    entry = build_profile(frame, {"kind": "categorical"})["columns"]["kind"]
    assert entry["missing_count"] == 3 and entry["unique_count"] == 5
    assert entry["categorical_summary"]["top_values"] == [
        {"value": value, "count": 1} for value in ["01", "1", "NA", "nan", ""]]
    json.dumps(entry, allow_nan=False)


def test_categorical_unused_levels_do_not_become_observations():
    frame = pd.DataFrame({"kind": pd.Categorical(["b", "a", None], categories=["a", "b", "unused"])})
    entry = build_profile(frame, {"kind": "categorical"})["columns"]["kind"]
    assert entry["unique_count"] == 2 and entry["missing_count"] == 1
    assert entry["categorical_summary"] == {
        "top_values": [{"value": "b", "count": 1}, {"value": "a", "count": 1}],
        "omitted_category_count": 0, "omitted_value_count": 0,
    }


@pytest.mark.parametrize('values', [[None, None, None], []])
def test_all_missing_or_empty_categorical_columns(values):
    frame = pd.DataFrame({"kind": pd.Series(values, dtype=object)})
    entry = build_profile(frame, {"kind": "categorical"})["columns"]["kind"]
    assert entry["all_missing"] and entry["missing_count"] == len(values)
    assert entry["categorical_summary"] == {
        "top_values": [], "omitted_category_count": 0, "omitted_value_count": 0}
    json.dumps(entry, allow_nan=False)


def test_requested_clusters_are_metadata_not_observed_groups():
    frame = pd.DataFrame({"x": [0.] * 8})
    profile = build_profile(frame, {"x": "numeric"}, n_clusters=np.int64(3))
    assert profile["requested_cluster_count"] == 3
    assert "observed_cluster_count" not in profile
    assert profile["columns"]["x"]["unique_count"] == 1
    assert "requested_cluster_count" not in build_profile(frame, {"x": "numeric"})


@pytest.mark.parametrize('count', [True, False, 0, -1, 1.5, '3', 4])
def test_requested_cluster_count_rejects_invalid_metadata(count):
    with pytest.raises(ValueError):
        build_profile(pd.DataFrame({"x": [1, 2, 3]}), {"x": "numeric"}, n_clusters=count)
