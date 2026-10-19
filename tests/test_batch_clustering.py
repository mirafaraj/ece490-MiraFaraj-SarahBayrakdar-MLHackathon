"""Isolation and leakage checks for opt-in clustering ablations."""
import numpy as np
import pandas as pd
import pytest
from types import SimpleNamespace

from unknown_dataset.clustering import fit_predict_clustering


def frame():
    values = np.random.default_rng(73).normal(size=(90, 3))
    values[::5, 0] = np.nan
    return pd.DataFrame(values)


def test_explicit_defaults_match_implicit_defaults():
    a, ia = fit_predict_clustering("clustering", frame(), n_clusters=3)
    b, ib = fit_predict_clustering("clustering", frame(), n_clusters=3,
        experimental_controls={"clustering_missing_indicators": True,
            "clustering_covariance": "full", "clustering_kmeans_only": False})
    np.testing.assert_array_equal(a, b)
    for key in ("model", "bic", "selection", "transformed_features"):
        assert ia[key] == ib[key]
    assert "experimental_controls" not in ia
    assert "phase_timings" not in ia


@pytest.mark.parametrize("controls", [
    {"clustering_missing_indicators": False},
    {"clustering_covariance": "diag"},
    {"clustering_kmeans_only": True}])
def test_each_alternative_ignores_labels_test_rows_and_preserves_input(controls):
    x = frame()
    original = x.copy(deep=True)
    a, ia = fit_predict_clustering("clustering", x, n_clusters=3,
        experimental_controls=controls)
    b, ib = fit_predict_clustering("clustering", x, y_train=object(),
        X_test=pd.DataFrame({"bad": ["arbitrary"]}), n_clusters=3,
        experimental_controls=controls)
    np.testing.assert_array_equal(a, b)
    pd.testing.assert_frame_equal(x, original)
    assert ia["bic"] == ib["bic"]
    assert len(np.unique(a)) == 3
    if controls.get("clustering_kmeans_only"):
        assert ia["bic"] == {}
    if controls.get("clustering_missing_indicators") is False:
        assert ia["transformed_features"] == 3


def test_instrumentation_does_not_change_result():
    a, ia = fit_predict_clustering("clustering", frame(), n_clusters=3)
    b, ib = fit_predict_clustering("clustering", frame(), n_clusters=3,
        experimental_controls={"runtime_instrumentation": True})
    np.testing.assert_array_equal(a, b)
    assert ia["bic"] == ib["bic"]
    assert set(ib["phase_timings"]) == {
        "preprocessing_seconds", "kmeans_seconds", "gmm_seconds"}
    assert all(v >= 0 for v in ib["phase_timings"].values())


@pytest.mark.parametrize("controls", [
    {"clustering_covariance": "tied"},
    {"clustering_missing_indicators": "false"},
    {"clustering_kmeans_only": 1}])
def test_invalid_controls_fail_clearly(controls):
    with pytest.raises(ValueError):
        fit_predict_clustering("clustering", frame(), n_clusters=3,
            experimental_controls=controls)


def test_route_controls_are_applied_and_explicit_override_wins():
    route = SimpleNamespace(experimental={"clustering_kmeans_only": True},
        candidates=["kmeans", "gmm"], to_dict=lambda: {"experimental": True})
    _, info = fit_predict_clustering("clustering", frame(), n_clusters=3,
        route_plan=route)
    assert info["bic"] == {}
    _, override = fit_predict_clustering("clustering", frame(), n_clusters=3,
        route_plan=route, experimental_controls={"clustering_kmeans_only": False})
    assert set(override["bic"]) == {"spherical", "full"}
