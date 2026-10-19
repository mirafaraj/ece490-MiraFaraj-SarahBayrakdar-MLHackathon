from dataclasses import replace
import numpy as np
import pandas as pd
import pytest

from unknown_dataset.routing import make_route_plan, summarize_training
from unknown_dataset import supervised as s


def fixture():
    frame = pd.DataFrame({"x": np.linspace(-2, 2, 160), "category": ["a", "b"] * 80})
    labels = np.where(frame.x > .8, "001", "1")
    return frame, labels, make_route_plan(summarize_training("classification", frame, labels))


@pytest.mark.parametrize("kwargs", [{"holdouts": 3}, {"holdouts": True}, {"objective": "roc_auc"}, {"weighting": "bad"}])
def test_control_validation(kwargs):
    with pytest.raises(ValueError):
        s.ClassificationControls(**kwargs)


def test_default_dispatch_and_fixed_override():
    frame, labels, plan = fixture()
    a, ai = s.fit_predict_supervised("classification", frame, labels, frame.iloc[:10], route_plan=plan)
    b, bi = s.fit_predict_supervised("classification", frame, labels, frame.iloc[:10], route_plan=plan,
                                    classification_controls=None)
    assert np.array_equal(a, b)
    assert ai["validation_scores"] == bi["validation_scores"]
    assert "experimental_classification" not in ai
    # Fitting weights no longer require changing the selection objective.
    _, info = s.fit_predict_fixed_candidate("linear", "classification", frame, labels, frame.iloc[:10],
                                            route_plan=plan, class_weight=None)
    assert info["selection_metric"] == "balanced_accuracy"
    assert info["class_weight"] is None


def test_two_holdouts_metrics_and_test_feature_invariance():
    frame, labels, plan = fixture()
    controls = s.ClassificationControls("macro_f1", "both", 2)
    _, first = s.fit_predict_supervised("classification", frame, labels, frame.iloc[:10],
                                       route_plan=plan, classification_controls=controls)
    changed = frame.iloc[:10].copy()
    changed["x"] = 9999
    changed["category"] = "unseen"
    _, second = s.fit_predict_supervised("classification", frame, labels, changed,
                                        route_plan=plan, classification_controls=controls)
    assert first["validation_scores"] == second["validation_scores"]
    assert first["selected_variant"] == second["selected_variant"]
    assert len(first["validation_scores"]) == 6
    for records in first["validation_fold_metrics"].values():
        assert len(records) == 2
        assert all({"accuracy", "balanced_accuracy", "macro_f1", "class_recall"} <= record.keys() for record in records)
    assert first["selection_time_limit_seconds"] == 20


def test_config_controls_and_completed_baseline(monkeypatch):
    frame, labels, plan = fixture()
    plan = replace(plan, candidates=["extra_trees"], validation_mode="none",
                   experimental={**plan.experimental, "classification_weighting": "unweighted"})
    metadata = {"selected_candidate": "extra_trees", "class_weight": None, "fit_seconds": .2, "prediction_seconds": .1}
    values = labels[:10].copy()
    def forbidden(*args, **kwargs):
        raise AssertionError("Compatible completed baseline must be reused")
    monkeypatch.setattr(s, "fit_predict_fixed_candidate", forbidden)
    prediction, info = s.fit_predict_supervised("classification", frame, labels, frame.iloc[:10],
        route_plan=plan, completed_baseline=(values, metadata))
    assert np.array_equal(prediction, values)
    assert info["baseline_reused"]
    assert info["selection_skipped"] == "route_validation_disabled"


def test_incomplete_folds_are_excluded(monkeypatch):
    frame, labels, plan = fixture()
    # Fake clock trips the shared selection cap between the first and second
    # folds. One-fold averages must never compete with complete two-fold scores.
    clock = [0.0]
    def fake_fit(name, task, train, target, test, *args, **kwargs):
        clock[0] += 21
        return np.repeat(target[0], len(test)), {"selected_candidate": name,
            "fit_seconds": 0.0, "prediction_seconds": 0.0}
    monkeypatch.setattr(s.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(s, "fit_predict_fixed_candidate", fake_fit)
    _, info = s.fit_predict_supervised("classification", frame, labels, frame.iloc[:10], route_plan=plan,
        classification_controls=s.ClassificationControls(holdouts=2))
    assert info["validation_scores"] == {}
    assert len(info["validation_fold_metrics"]["extra_trees:balanced"]) == 1
    assert all("budget exhausted" in err for err in info["candidate_errors"].values())


def test_controls_reject_regression_and_unrouted():
    frame, labels, plan = fixture()
    for task, route in [("regression", plan), ("classification", None)]:
        with pytest.raises(ValueError):
            s.fit_predict_supervised(task, frame, labels, frame.iloc[:10], route_plan=route,
                                    classification_controls=s.ClassificationControls())


def test_offline_evaluation_truth_cannot_change_selection():
    from copy import deepcopy
    from scripts.batch_classification import select_offline
    candidates = {
        "linear:balanced": {"family": "linear", "inner": [{"metrics": {"accuracy": .8}}], "outer": {"accuracy": .1}},
        "extra_trees:balanced": {"family": "extra_trees", "inner": [{"metrics": {"accuracy": .7}}], "outer": {"accuracy": 1.0}},
    }
    changed = deepcopy(candidates)
    changed["linear:balanced"]["outer"] = {"accuracy": 1.0, "class_recall": [1, 1]}
    changed["extra_trees:balanced"]["outer"] = {"accuracy": 0.0, "class_recall": [0, 0]}
    assert select_offline(candidates, list(candidates), "accuracy", 1) == select_offline(changed, list(changed), "accuracy", 1)


@pytest.mark.parametrize("selected_family", ["linear", "extra_trees"])
def test_failed_refit_recovers_checkpoint_even_with_other_weight(monkeypatch, selected_family):
    frame, labels, plan = fixture()
    plan = replace(plan, candidates=["extra_trees", "linear"] if selected_family == "linear" else ["extra_trees"])
    calls = []
    def fake_fit(name, task, train, target, test, *args, **kwargs):
        calls.append((name, kwargs.get("final", True)))
        if kwargs.get("final", True):
            raise ValueError("forced final failure")
        # Perfect scores create a tie resolved toward linear.
        prediction = labels[test.index]
        return prediction, {"fit_seconds": 0.0, "prediction_seconds": 0.0}
    monkeypatch.setattr(s, "fit_predict_fixed_candidate", fake_fit)
    values = labels[:10].copy()
    metadata = {"selected_candidate": "extra_trees", "class_weight": "balanced", "fit_seconds": .2}
    prediction, info = s.fit_predict_supervised("classification", frame, labels, frame.iloc[:10],
        route_plan=plan, classification_controls=s.ClassificationControls(weighting="unweighted"),
        completed_baseline=(values, metadata))
    assert np.array_equal(prediction, values)
    assert info["baseline_reused"]
    assert info["selected_variant"] == "extra_trees:balanced"
    assert info["class_weight"] == "balanced"
    expected = [("extra_trees", False), ("linear", False), ("linear", True)] if selected_family == "linear" else [("extra_trees", False), ("extra_trees", True)]
    assert calls == expected
