import numpy as np
import pytest
from scripts.third_regression import case_data, split_indices, metrics, choose, digest, MECHANISMS, SEEDS


def test_frozen_design_and_splits():
    assert len(MECHANISMS)*len(SEEDS) == 18
    frame, y, types = case_data("mixed_category", SEEDS[0])
    again, yy, _ = case_data("mixed_category", SEEDS[0])
    assert frame.equals(again)
    assert np.array_equal(y, yy)
    assert frame.shape == (1200, 13) and types["category"] == "categorical"
    split = split_indices(len(frame), SEEDS[0])
    assert len(split["test"]) == 300 and len(split["train"]) == 900
    assert set(split["inner_train"]) | set(split["inner_validation"]) == set(split["train"])
    assert not set(split["test"]) & set(split["train"])
    assert not set(split["inner_train"]) & set(split["inner_validation"])


def test_metric_arithmetic_and_missing_groups():
    value = metrics([0, 2], [1, 0], dummy=[1, 1], missing=[True, False])
    assert value["rmse"] == pytest.approx(np.sqrt(2.5))
    assert value["mae"] == 1.5
    assert value["dummy_relative_mae"] == 1.5
    assert value["row_groups"]["missing"]["mae"] == 1
    assert value["row_groups"]["observed"]["mae"] == 2


def test_selector_only_receives_inner_evidence():
    inner = {name: dict(metrics=dict(rmse=1, mae=1), seconds=.1) for name in ("linear", "extra_trees", "histogram15", "histogram31")}
    assert choose(inner, "rmse", 31)["selected"] == "linear"
    inner["histogram31"]["metrics"]["rmse"] = .5
    assert choose(inner, "rmse", 31)["selected"] == "histogram"
    assert choose(inner, "rmse", 15)["selected"] == "linear"
    inner["extra_trees"]["seconds"] = 20
    limited = choose(inner, "rmse", 31)
    assert limited["selected"] == "extra_trees"
    assert set(limited["excluded"]) == {"linear", "histogram"}
    assert digest({"b": 2, "a": 1}) == digest({"a": 1, "b": 2})


def test_study_keeps_evaluation_rows_outside_fits(monkeypatch):
    import scripts.third_regression as study
    calls = []
    def fake_fit(name, task, frame, target, evaluation, types, seed, final, route_plan):
        calls.append((name, set(frame.index), set(evaluation.index), final,
                      route_plan.experimental["histogram_max_leaf_nodes"]))
        assert not set(frame.index) & set(evaluation.index)
        assert len(target) == len(frame)
        return np.full(len(evaluation), float(np.mean(target))), {"fit_seconds": 0}
    monkeypatch.setattr(study, "fit_predict_fixed_candidate", fake_fit)
    case = study.run_case("numeric_interaction", SEEDS[0])
    assert len(calls) == 8
    assert [(name, capacity) for name, _, _, final, capacity in calls if final] == [
        ("linear", 15), ("extra_trees", 15), ("histogram", 15), ("histogram", 31)]
    assert all(evaluation == set(case["splits"]["test"]) for _, _, evaluation, final, _ in calls if final)
