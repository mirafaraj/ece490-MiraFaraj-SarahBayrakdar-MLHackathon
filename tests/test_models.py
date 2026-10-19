import json

import numpy as np
import pandas as pd
import pytest

from unknown_dataset.models import fallback_predictions, fit_predict


@pytest.mark.parametrize("task", ["classification", "regression", "clustering"])
def test_mixed_missing_columns_and_unseen_categories_produce_valid_predictions(task):
    rng = np.random.default_rng(7)
    groups = np.repeat([0, 1, 2], 30)
    x = groups * 8 + rng.normal(0, 0.4, 90)
    frame = pd.DataFrame({"x": x, "scale": x * 1000, "category": np.array(["a", "b", "c"])[groups],
                          "constant": 1, "missing": np.nan})
    frame.loc[[2, 9, 40], "x"] = np.nan
    frame.loc[[3, 11], "category"] = None
    frame["duplicate"] = frame["x"]
    original = frame.copy(deep=True)
    test = frame.iloc[[75, 5, 40, 31]].copy()
    test["category"] = "unseen"
    types = {name: "categorical" if name == "category" else "numeric" for name in frame}
    target = None if task == "clustering" else pd.Series(np.array(["low", "medium", "high"])[groups] if task == "classification" else x ** 2)
    predictions, info = fit_predict(task, frame, target, None if task == "clustering" else test,
                                    types, 3 if task == "clustering" else None, time_budget=15, seed=7)
    predictions = np.asarray(predictions)
    assert predictions.shape == ((90,) if task == "clustering" else (4,))
    assert not pd.isna(predictions).any()
    if task == "classification":
        assert set(predictions).issubset({"low", "medium", "high"})
    elif task == "regression":
        assert np.isfinite(predictions).all()
    else:
        assert len(np.unique(predictions)) == 3
    json.dumps(info, allow_nan=False)
    pd.testing.assert_frame_equal(frame, original)


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_no_informative_features_uses_valid_fallback(task):
    frame = pd.DataFrame({"empty": [np.nan] * 6, "fixed": [1] * 6})
    target = pd.Series(["only"] * 6) if task == "classification" else pd.Series([2.0] * 6)
    predictions, _ = fit_predict(task, frame, target, frame.iloc[:2],
                                 {"empty": "numeric", "fixed": "numeric"}, time_budget=5, seed=2)
    assert list(predictions) == (["only", "only"] if task == "classification" else [2.0, 2.0])


def test_identical_rows_still_receive_required_number_of_groups():
    frame = pd.DataFrame({"fixed": [1] * 9, "empty": [np.nan] * 9})
    predictions, _ = fit_predict("clustering", frame, feature_types={"fixed": "numeric", "empty": "numeric"},
                                 n_clusters=3, time_budget=5, seed=2)
    assert len(predictions) == 9
    assert len(np.unique(predictions)) == 3


@pytest.mark.parametrize("task", ["classification", "regression", "clustering"])
def test_emergency_fallback_produces_valid_outputs_and_records_reason(task):
    frame = pd.DataFrame({"x": [1, 2, 3, 4, 5, 6]})
    target = None if task == "clustering" else pd.Series(["a", "a", "b", "a", "b", "a"] if task == "classification" else [1e308] * 6)
    predictions, info = fallback_predictions(
        task, frame, target, frame.iloc[:2], n_clusters=3, seed=17, reason="test worker deadline"
    )
    assert len(predictions) == (6 if task == "clustering" else 2)
    assert not pd.isna(predictions).any()
    if task == "classification":
        assert set(predictions).issubset(set(target))
    elif task == "regression":
        assert np.isfinite(predictions).all()
    else:
        assert len(np.unique(predictions)) == 3
    assert info["fallback"] is True
    assert info["fallback_reason"] == "test worker deadline"
    assert info["seed"] == 17
    json.dumps(info, allow_nan=False)


@pytest.mark.parametrize("labels", [[0.25] * 6, [0.25] * 3 + [1.75] * 3])
def test_fractional_class_labels_are_preserved(labels):
    frame = pd.DataFrame({"x": [0, 1, 2, 8, 9, 10]})
    predictions, _ = fit_predict("classification", frame, pd.Series(labels), frame.iloc[[5, 0]],
                                 {"x": "numeric"}, time_budget=5, seed=3)
    assert len(predictions) == 2
    assert set(predictions).issubset(set(labels))
