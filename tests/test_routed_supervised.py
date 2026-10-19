import json

import numpy as np
import pandas as pd
import pytest

import unknown_dataset.supervised as supervised
from unknown_dataset.routing import make_route_plan, summarize_training
from unknown_dataset.preprocessing import TableCleaner


def case(task="classification", config=None):
    X = pd.DataFrame({"x": np.arange(120), "category": ["001", "1"] * 60})
    y = np.array(["001", "1"] * 60) if task == "classification" else np.arange(120) * 2.0
    return X, y, make_route_plan(summarize_training(task, X, y), config=config)


@pytest.mark.parametrize("metric,weight", [("accuracy", None), ("balanced_accuracy", "balanced")])
def test_routed_metric_weights_and_lexical_labels(metric, weight):
    X, y, plan = case(config={"metrics": {"classification": metric}})
    prediction, info = supervised.fit_predict_supervised("classification", X, y, X.iloc[:5], route_plan=plan)
    assert set(prediction).issubset({"001", "1"})
    assert info["selection_metric"] == metric
    assert info["class_weight"] == weight
    assert info["strategy"] == "routed"
    assert set(info["validation_scores"]) == set(plan.candidates)
    json.dumps(info, allow_nan=False)


def test_routed_preprocessing_uses_identical_fitting_fold(monkeypatch):
    X, y, plan = case("regression")
    calls = []
    original = TableCleaner.fit
    def record(self, frame, y=None):
        calls.append(tuple(frame.index))
        return original(self, frame, y)
    monkeypatch.setattr(TableCleaner, "fit", record)
    supervised.fit_predict_supervised("regression", X, y, X.iloc[:2], route_plan=plan)
    assert len(calls) == 4
    assert calls[0] == calls[1] == calls[2]
    assert len(calls[0]) == 90
    assert calls[-1] == tuple(X.index)


def test_supervised_test_mutation_cannot_change_selection():
    X, y, plan = case("regression")
    _, first = supervised.fit_predict_supervised("regression", X, y, X.iloc[:5], route_plan=plan)
    changed = X.iloc[:5].copy()
    changed["x"] = 100000
    changed["category"] = "unseen"
    _, second = supervised.fit_predict_supervised("regression", X, y, changed, route_plan=plan)
    assert first["validation_scores"] == second["validation_scores"]
    assert first["selected_candidate"] == second["selected_candidate"]
    assert first["routing"] == second["routing"]


def test_baseline_checkpoint_api_uses_plan_and_preserves_fractional_labels():
    X, _, plan = case(config={"metrics": {"classification": "accuracy"}, "limits": {"final_tree_count": 7}})
    y = np.array([0.25, 1.75] * 60)
    prediction, info = supervised.fit_baseline_supervised("classification", X, y, X.iloc[:4], route_plan=plan)
    assert set(prediction).issubset({0.25, 1.75})
    assert info["n_estimators"] == 7
    assert info["class_weight"] is None


def test_loss_metrics_and_near_tie_selection():
    assert supervised.metric_score("balanced_accuracy", [0.25, 1.75], [0.25, 1.75]) == 1
    assert supervised.metric_score("mae", [1, 4], [0, 2]) == 1.5
    assert supervised.select_candidate({"extra_trees": 0.9, "linear": 0.89}, "accuracy", 0) == "extra_trees"
    assert supervised.select_candidate({"extra_trees": 0.9, "linear": 0.89}, "accuracy", 0.02) == "linear"
    assert supervised.select_candidate({"extra_trees": 1, "linear": 1.1}, "rmse", 0) == "extra_trees"
    # Tolerance is relative to global best: no chained near-tie drift.
    assert supervised.select_candidate({"histogram": 1, "extra_trees": 1.1, "linear": 1.2}, "mae", 0.11) == "extra_trees"


def test_categorical_search_fits_fold_only_and_refits_selected_parameters(monkeypatch):
    from unknown_dataset.categorical import SmoothedCategoricalClassifier
    X, y, plan = case(config={"experimental": {"categorical_enabled": True}, "limits": {"max_candidates": 4}})
    fitted_rows = []
    original = SmoothedCategoricalClassifier.fit
    def record(self, frame, target):
        fitted_rows.append(tuple(frame.index))
        return original(self, frame, target)
    monkeypatch.setattr(SmoothedCategoricalClassifier, "fit", record)
    _, info = supervised.fit_predict_supervised("classification", X, y, X.iloc[:3], route_plan=plan)
    assert info["selected_candidate"] == "categorical"
    assert len(fitted_rows) == 3  # two alpha values then final refit
    assert fitted_rows[0] == fitted_rows[1]
    assert len(fitted_rows[0]) == 90
    assert fitted_rows[-1] == tuple(X.index)
    assert info["candidate_parameters"]["feature"] == "category"


def test_routed_fractional_labels_are_classes_during_selection():
    X, _, plan = case()
    prediction, info = supervised.fit_predict_supervised("classification", X, [0.25, 1.75] * 60,
        X.iloc[:4], route_plan=plan)
    assert set(prediction).issubset({0.25, 1.75})
    assert len(info["validation_scores"]) == 3


@pytest.mark.parametrize("refit_failure", [False, True])
def test_completed_baseline_reused_after_selection_or_optional_refit_failure(monkeypatch, refit_failure):
    X, y, plan = case("regression")
    completed = supervised.fit_baseline_supervised("regression", X, y, X.iloc[:3], route_plan=plan)
    original = supervised.fit_predict_fixed_candidate
    final_fits = []
    def recorded(name, *args, **kwargs):
        if kwargs.get("final", True):
            final_fits.append(name)
            if refit_failure:
                raise ValueError("forced optional refit failure")
        return original(name, *args, **kwargs)
    monkeypatch.setattr(supervised, "fit_predict_fixed_candidate", recorded)
    monkeypatch.setattr(supervised, "select_candidate", lambda *args: "linear" if refit_failure else "extra_trees")
    prediction, info = supervised.fit_predict_supervised("regression", X, y, X.iloc[:3],
        route_plan=plan, completed_baseline=completed)
    assert np.array_equal(prediction, completed[0])
    assert info["baseline_reused"] is True
    assert len(info["validation_scores"]) == 3
    assert final_fits == (["linear"] if refit_failure else [])
    if refit_failure:
        assert info["refit_fallback_reason"] == "forced optional refit failure"
    assert "baseline_reused" not in completed[1]  # no caller metadata mutation
