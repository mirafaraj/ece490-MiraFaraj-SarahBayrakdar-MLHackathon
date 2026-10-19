import json

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression

import unknown_dataset.supervised as supervised
from unknown_dataset.preprocessing import TableCleaner


def test_linear_regression_candidate_recovers_linear_relationship():
    x, y = make_regression(n_samples=240, n_features=6, noise=0.1, random_state=42)
    frame = pd.DataFrame(x, columns=[f"x{i}" for i in range(x.shape[1])])
    prediction, info = supervised.fit_predict_supervised("regression", frame.iloc[:200], y[:200], frame.iloc[200:])
    assert info["selected_candidate"] == "linear"
    assert np.sqrt(np.mean((prediction - y[200:]) ** 2)) < 2
    assert set(info["validation_scores"]) == {"extra_trees", "linear", "histogram"}
    json.dumps(info, allow_nan=False)


def test_candidate_preprocessing_fits_training_splits_only(monkeypatch):
    frame = pd.DataFrame({"x": np.arange(100), "category": ["a", "b"] * 50})
    seen = []
    original = TableCleaner.fit

    def record(self, data, y=None):
        seen.append(set(data.index))
        return original(self, data, y)

    monkeypatch.setattr(TableCleaner, "fit", record)
    test = frame.iloc[:3].copy()
    test.index = [1001, 1002, 1003]
    supervised.fit_predict_supervised("regression", frame, np.arange(100) ** 2, test)
    assert len(seen) == 4
    assert all(len(indices) == 75 for indices in seen[:-1])
    assert all(indices == seen[0] for indices in seen[:-1])
    assert seen[-1] == set(frame.index)
    assert not set(test.index).intersection(set.union(*seen))


@pytest.mark.parametrize("budget,reason", [(5, "low_time_budget"), (50, "tiny_data_or_rare_classes")])
def test_small_or_low_budget_inputs_skip_selection(budget, reason):
    frame = pd.DataFrame({"x": np.arange(20)})
    prediction, info = supervised.fit_predict_supervised("classification", frame,
        np.array([0.25, 1.75] * 10), frame.iloc[:3], time_budget=budget)
    assert info["selection_skipped"] == reason
    assert info["validation_scores"] == {}
    assert set(prediction).issubset({0.25, 1.75})
    assert info["n_estimators"] == (32 if budget < 15 else 96)


def test_mixed_missing_categorical_input_and_unseen_test_categories():
    x, y = make_classification(n_samples=150, n_features=5, random_state=42)
    frame = pd.DataFrame(x, columns=[f"x{i}" for i in range(x.shape[1])])
    frame["category"] = np.where(y, "yes", "no")
    frame.iloc[::10, 0] = np.nan
    frame.loc[::11, "category"] = None
    test = frame.iloc[:5].copy()
    test["category"] = "unseen"
    prediction, info = supervised.fit_predict_supervised("classification", frame, y, test)
    assert len(prediction) == 5
    assert set(prediction).issubset({0, 1})
    assert all(np.isfinite(list(info["validation_scores"].values())))


def test_dense_candidate_is_skipped_without_failing_other_candidates(monkeypatch):
    monkeypatch.setattr(supervised, "_DENSE_CELLS", 1)
    frame = pd.DataFrame({"x": np.arange(100), "z": np.arange(100) ** 2})
    _, info = supervised.fit_predict_supervised("regression", frame, np.arange(100), frame.iloc[:2])
    assert "histogram" in info["candidate_errors"]
    assert info["selected_candidate"] in {"linear", "extra_trees"}


def test_constant_inputs_and_single_class_stay_valid():
    frame = pd.DataFrame({"fixed": [1] * 80})
    prediction, info = supervised.fit_predict_supervised("classification", frame, ["only"] * 80, frame.iloc[:3])
    assert prediction.tolist() == ["only"] * 3
    assert info["model"] == "DummyClassifier"
