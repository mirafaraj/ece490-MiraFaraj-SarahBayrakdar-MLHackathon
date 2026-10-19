"""Independent harness checks: generated cases, metrics, and semantic validation."""
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.datasets import make_broad_cases
from scripts.evaluate import classification_metrics, evaluate_case, make_cases, validate_profile
from unknown_dataset.profiling import build_profile


@pytest.mark.parametrize("labels", [[.25, 1.75, .25], ["a", "b", "a"]])
def test_paired_classification_encoding(labels):
    result = classification_metrics(np.array(labels), np.array(labels), np.array(labels))
    assert result["accuracy"] == 1
    assert result["balanced_accuracy"] == 1
    assert result["majority_accuracy"] == pytest.approx(2 / 3)


def test_broad_suite_is_balanced_offline_and_reproducible(tmp_path):
    first = make_broad_cases(tmp_path / "first", 42)
    second = make_broad_cases(tmp_path / "second", 42)
    assert len(first) == 12
    assert len({case["name"] for case in first}) == 12
    assert {task: sum(case["task"] == task for case in first) for task in ("classification", "regression", "clustering")} == {
        "classification": 4, "regression": 4, "clustering": 4}
    assert sum("real" in case["source"] for case in first) == 3
    for left, right in zip(first, second):
        np.testing.assert_array_equal(left["reference"], right["reference"])
        assert (left["input"] / "train.csv").read_bytes() == (right["input"] / "train.csv").read_bytes()
        assert len(pd.read_csv(left["input"] / "train.csv")) <= 1500
        assert "target" not in pd.read_csv(left["input"] / "test.csv").columns if left["task"] != "clustering" else left["training_target"] is None


def test_independent_profile_validator_catches_wrong_counts(tmp_path):
    case = make_cases(tmp_path, 42)[0]
    metadata = json.loads((case["input"] / "metadata.json").read_text())
    train = pd.read_csv(case["input"] / "train.csv", dtype={"target": str})
    features = train.drop(columns=["target", "row_id"])
    profile = build_profile(features, metadata["feature_types"], train["target"])
    validate_profile(case, profile)
    profile["columns"]["x"]["missing_count"] += 1
    with pytest.raises(ValueError, match="Profile mismatch"):
        validate_profile(case, profile)


def test_actual_evaluator_preserves_lexical_classes(tmp_path):
    folder = tmp_path / "input"
    folder.mkdir()
    (folder / "metadata.json").write_text(json.dumps({"task": "classification", "target": "target", "feature_types": {"x": "numeric"}}))
    (folder / "train.csv").write_text("x,target\n0,01\n1,01\n2,01\n8,1\n9,1\n10,1\n")
    (folder / "test.csv").write_text("x\n0\n10\n")
    case = {"task": "classification", "input": folder, "reference": np.array(["01", "1"]),
            "training_target": np.array(["01", "01", "01", "1", "1", "1"])}
    result = evaluate_case(case, tmp_path / "output", 42, 30, 20)
    assert result["valid"], result["error"]
    assert result["metrics"]["accuracy"] == 1
