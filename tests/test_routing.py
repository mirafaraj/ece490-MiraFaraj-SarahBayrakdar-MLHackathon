import json

import numpy as np
import pandas as pd
import pytest

from unknown_dataset.routing import load_router_config, make_route_plan, summarize_training
from unknown_dataset.policies import validate_config


def training(task="classification"):
    frame = pd.DataFrame({"x": np.arange(100), "category": ["001", "1"] * 50,
                          "constant": 1, "duplicate": np.arange(100)})
    return frame, summarize_training(task, frame, ["001", "1"] * 50)


def test_summary_and_plan_deterministic_json_and_training_only():
    frame, summary = training()
    assert summary.constant_columns == ["constant"]
    assert summary.duplicate_columns == ["duplicate"]
    assert summary.class_counts == [50, 50]
    plan = make_route_plan(summary)
    assert plan.to_dict() == make_route_plan(summarize_training("classification", frame, ["001", "1"] * 50)).to_dict()
    assert plan.candidates == ["extra_trees", "linear", "histogram"]
    assert plan.class_weight == "balanced"
    json.dumps(plan.to_dict(), allow_nan=False)


@pytest.mark.parametrize("config", [
    {"wat": 1}, {"limits": {"typo": 5}}, {"metrics": {"classification": "f1"}},
    {"limits": {"dense_cells_limit": float("inf")}}, {"limits": {"max_candidates": True}},
    {"limits": {"selection_sample_rows": 1.5}}, {"limits": {"selection_fraction": 1}},
    {"experimental": {"categorical_alphas": [0]}}, {"selection": {"near_tie_tolerance": -1}},
    {"experimental": {"categorical_enabled": "yes"}}, {"policy_version": "unknown"},
])
def test_strict_config_rejects_invalid_overrides(config):
    with pytest.raises(ValueError):
        validate_config(config)


def test_default_config_file_matches_builtins():
    assert load_router_config() == validate_config()


def test_dense_candidates_pruned_before_allocation_and_accuracy_weights():
    _, summary = training()
    plan = make_route_plan(summary, config={"metrics": {"classification": "accuracy"},
                                           "limits": {"dense_cells_limit": 100}})
    assert "histogram" not in plan.candidates
    assert "allocation" in plan.excluded["histogram"]
    assert plan.class_weight is None


def test_low_budget_and_rare_classes_keep_baseline():
    frame, summary = training()
    assert make_route_plan(summary, time_budget=3).candidates == ["extra_trees"]
    rare = summarize_training("classification", frame, [0] * 99 + [1])
    assert make_route_plan(rare).validation_mode == "none"


def test_clustering_numeric_geometry_eligibility():
    _, mixed = training("clustering")
    assert make_route_plan(mixed, n_clusters=2).candidates == ["kmeans"]
    numeric = summarize_training("clustering", pd.DataFrame({"x": np.arange(100)}))
    assert make_route_plan(numeric, n_clusters=2).candidates == ["kmeans", "gmm"]
    with pytest.raises(ValueError):
        make_route_plan(numeric, n_clusters=101)


def test_experimental_categorical_is_explicit_and_bounded():
    _, summary = training()
    assert "categorical" not in make_route_plan(summary).candidates
    plan = make_route_plan(summary, config={"experimental": {"categorical_enabled": True}})
    assert plan.candidates == ["extra_trees", "categorical", "linear"]
    numeric = summarize_training("classification", pd.DataFrame({"x": np.arange(100)}), [0, 1] * 50)
    plan = make_route_plan(numeric, config={"experimental": {"categorical_enabled": True}})
    assert "no nonconstant categorical" in plan.excluded["categorical"]


def test_working_memory_prunes_optional_candidates():
    _, summary = training()
    plan = make_route_plan(summary, config={"limits": {"working_memory_bytes": 1},
                                           "experimental": {"categorical_enabled": True}})
    assert plan.candidates == ["extra_trees"]
    assert all("memory" in reason for reason in plan.excluded.values())
