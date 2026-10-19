"""Behavioral boundary tests with generous budgets, separate from deadline tests."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import pickle
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.evaluate import evaluate_case, make_cases
from unknown_dataset.models import fit_predict
from unknown_dataset.routing import load_router_config
import unknown_dataset.supervised as supervised


def data(task):
    rng = np.random.default_rng(42)
    x = pd.DataFrame({"x": rng.normal(size=160), "z": rng.normal(size=160),
                      "category": rng.choice(["01", "1", "blue"], size=160), "constant": 1})
    x.loc[::11, "x"] = np.nan
    signal = x["x"].fillna(0).to_numpy() + 2 * (x["category"] == "blue").to_numpy() + .2 * x["z"].to_numpy()
    y = np.where(signal > 1, "positive", "negative") if task == "classification" else signal
    return x, y, {"x": "numeric", "z": "numeric", "category": "categorical", "constant": "numeric"}


def selection(info):
    return {key: info.get(key) for key in ("selected_candidate", "validation_scores", "candidate_errors",
        "selection_skipped", "dropped_constant_columns", "dropped_duplicate_columns", "transformed_features")}


@pytest.mark.parametrize("task", ["classification", "regression"])
@pytest.mark.parametrize("strategy", ["auto", "routed"])
def test_test_feature_values_cannot_change_fitted_state_or_selection(monkeypatch, task, strategy):
    train, labels, types = data(task)
    test = train.iloc[:20].copy()
    changed = test.copy()
    changed["x"] = np.arange(20) * 1000
    changed["z"] = -999
    changed["category"] = "unseen-test-category"
    fitted_states = []
    original = supervised._fit
    def record(*args, **kwargs):
        fitted = original(*args, **kwargs)
        # Includes cleaner decisions, fitted imputer/encoder/scaler, and estimator.
        # Retain objects and fingerprint after prediction too, so a bad transform
        # that mutates fitted state on prediction inputs cannot escape this check.
        fitted_states.append(fitted)
        return fitted
    monkeypatch.setattr(supervised, "_fit", record)
    config = load_router_config()
    first, first_info = fit_predict(task, train, labels, test, types, strategy=strategy,
        time_budget=120, seed=42, route_config=config)
    before = [hashlib.sha256(pickle.dumps(state)).hexdigest() for state in fitted_states]
    fitted_states.clear()
    second, second_info = fit_predict(task, train, labels, changed, types, strategy=strategy,
        time_budget=120, seed=42, route_config=config)
    assert before, "Must observe actual fitted candidate state"
    after = [hashlib.sha256(pickle.dumps(state)).hexdigest() for state in fitted_states]
    assert before == after
    assert selection(first_info) == selection(second_info)
    assert len(first) == len(second)


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_test_row_permutation_only_permutes_predictions(task):
    train, labels, types = data(task)
    test = train.iloc[:20].copy()
    order = np.random.default_rng(314).permutation(len(test))
    first, info = fit_predict(task, train, labels, test, types, strategy="routed", time_budget=120, seed=42)
    second, permuted_info = fit_predict(task, train, labels, test.iloc[order], types, strategy="routed", time_budget=120, seed=42)
    np.testing.assert_array_equal(first[order], second)
    assert selection(info) == selection(permuted_info)


def test_evaluation_label_mutation_cannot_change_cli_predictions_or_selection(tmp_path):
    case = make_cases(tmp_path / "input", 42)[0]
    first = evaluate_case(case, tmp_path / "first", 42, 140, 120, "routed")
    altered = deepcopy(case)
    altered["reference"] = np.asarray(case["reference"])[::-1].copy()
    second = evaluate_case(altered, tmp_path / "second", 42, 140, 120, "routed")
    assert first["valid"], first["error"]
    assert second["valid"], second["error"]
    assert (tmp_path / "first/predictions.csv").read_bytes() == (tmp_path / "second/predictions.csv").read_bytes()
    assert first["data_sha256"] == second["data_sha256"]
    assert selection(first["run_info"]["model"]) == selection(second["run_info"]["model"])


def test_categorical_counts_fit_inside_inner_fold_and_ignore_test_values(monkeypatch):
    from unknown_dataset.categorical import SmoothedCategoricalClassifier
    train, labels, types = data("classification")
    config = load_router_config()
    config["experimental"]["categorical_enabled"] = True
    config["limits"]["max_candidates"] = 4
    test = train.iloc[:20].copy()
    altered = test.copy()
    altered["category"] = "never-in-training"
    counts_states, fold_sizes = [], []
    original = SmoothedCategoricalClassifier.fit
    def capture(self, frame, target):
        result = original(self, frame, target)
        counts_states.append(hashlib.sha256(pickle.dumps((self.prior_, self.counts_))).hexdigest())
        fold_sizes.append(len(frame))
        assert sum(v.sum() for v in self.counts_.values()) == len(frame)
        return result
    monkeypatch.setattr(SmoothedCategoricalClassifier, "fit", capture)
    _, first = fit_predict("classification", train, labels, test, types, strategy="routed", route_config=config, time_budget=120)
    states = list(counts_states)
    assert states, "Experimental count candidate must be fitted"
    assert any(size < len(train) for size in fold_sizes), "Inner count fitting must use a proper fitting fold"
    counts_states.clear()
    _, second = fit_predict("classification", train, labels, altered, types, strategy="routed", route_config=config, time_budget=120)
    assert states == counts_states
    assert selection(first) == selection(second)
