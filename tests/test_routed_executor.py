"""Facade parity and preservation of complete predictions during optional work."""

import numpy as np
import pandas as pd
import pytest

from unknown_dataset.models import fit_predict


@pytest.mark.parametrize("task", ["classification", "regression", "clustering"])
def test_baseline_only_policy_preserves_existing_baseline(task):
    rng = np.random.default_rng(19)
    frame = pd.DataFrame({"x": rng.normal(size=90), "kind": ["a", "b", "c"] * 30})
    target = None if task == "clustering" else pd.Series(
        np.where(frame.x > 0, "01", "1") if task == "classification" else frame.x ** 2)
    kwargs = dict(task=task, X_train=frame, y_train=target,
                  X_test=None if task == "clustering" else frame.iloc[[6, 1, 7]],
                  n_clusters=3 if task == "clustering" else None,
                  feature_types={"x": "numeric", "kind": "categorical"}, seed=19)
    reference, _ = fit_predict(**kwargs, strategy="baseline")
    checkpoints = []
    actual, info = fit_predict(**kwargs, strategy="routed",
                               route_config={"limits": {"max_candidates": 1}},
                               checkpoint_callback=lambda pred, meta: checkpoints.append(pred.copy()))
    np.testing.assert_array_equal(actual, reference)
    np.testing.assert_array_equal(checkpoints[0], reference)
    assert len(checkpoints) == 1
    assert info["route_stage"] == "baseline_only"
    assert info["strategy"] == "routed"
    assert not info["fallback"]


def test_optional_error_retains_checkpointed_predictions(monkeypatch):
    from unknown_dataset import supervised

    def fail(*args, **kwargs):
        raise RuntimeError("optional fit failed")

    monkeypatch.setattr(supervised, "fit_predict_supervised", fail)
    frame = pd.DataFrame({"x": np.arange(100, dtype=float)})
    checkpoints = []
    predictions, info = fit_predict("regression", frame, pd.Series(np.arange(100) ** 2),
                                   frame.iloc[[50, 2]], strategy="routed",
                                   checkpoint_callback=lambda pred, meta: checkpoints.append(pred.copy()))
    np.testing.assert_array_equal(predictions, checkpoints[0])
    assert info["fallback_kind"] == "completed_baseline"
    assert info["fallback_reason"] == "RuntimeError: optional fit failed"
    assert info["route_stage"] == "optional_error"


def test_optional_refit_failure_is_reported_as_completed_baseline_fallback(monkeypatch):
    from unknown_dataset import supervised

    def recovered(*args, completed_baseline=None, **kwargs):
        predictions, info = completed_baseline
        return predictions, dict(info, refit_fallback_reason="candidate refit failed")

    monkeypatch.setattr(supervised, "fit_predict_supervised", recovered)
    frame = pd.DataFrame({"x": np.arange(100, dtype=float)})
    _, info = fit_predict("regression", frame, np.arange(100), frame.iloc[:2], strategy="routed")
    assert info["fallback"]
    assert info["fallback_kind"] == "completed_baseline"
    assert info["route_stage"] == "optional_refit_error"
