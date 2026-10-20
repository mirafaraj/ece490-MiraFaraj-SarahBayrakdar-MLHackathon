"""Indicator scaling boundaries, default invariance and training-only inputs."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from unknown_dataset.clustering import _scale_missing_indicators, fit_predict_clustering
from unknown_dataset.preprocessing import TableCleaner, build_preprocessor


@pytest.mark.parametrize("mixed", [False, True])
def test_only_numeric_indicators_scaled_after_standardization(mixed):
    frame = pd.DataFrame({"x": [1., 2., np.nan, 4.], "y": [9., 8., 7., 6.]})
    if mixed:
        frame["category"] = ["a", "b", "a", "b"]
    cleaner = TableCleaner().fit(frame)
    preprocessor = build_preprocessor(cleaner, scale_numeric=True)
    matrix = preprocessor.fit_transform(cleaner.transform(frame))
    assert _scale_missing_indicators(preprocessor, matrix, 1.) is matrix
    scaled = _scale_missing_indicators(preprocessor, matrix, .25)
    dense = matrix.toarray() if sparse.issparse(matrix) else matrix
    actual = scaled.toarray() if sparse.issparse(scaled) else scaled
    expected = dense.copy()
    expected[:, 2] *= .25
    np.testing.assert_array_equal(actual, expected)
    assert sparse.issparse(matrix) == sparse.issparse(scaled)


@pytest.mark.parametrize("value", [True, False, 0, -1, 1.01, float("nan"), float("inf"), "0.25", None])
def test_amplitude_rejects_invalid_controls(value):
    with pytest.raises(ValueError, match="amplitude"):
        fit_predict_clustering("clustering", pd.DataFrame({"x": range(50)}), n_clusters=2,
            experimental_controls={"clustering_indicator_amplitude": value})


@pytest.mark.parametrize("frame", [pd.DataFrame({"x": [1., 2., 3.]}),
    pd.DataFrame({"category": ["a", "b", "a"]}), pd.DataFrame({"x": [1., 1., 1.]})])
def test_no_numeric_indicators_remain_unchanged(frame):
    cleaner = TableCleaner().fit(frame)
    preprocessor = build_preprocessor(cleaner, scale_numeric=True)
    matrix = preprocessor.fit_transform(cleaner.transform(frame))
    assert _scale_missing_indicators(preprocessor, matrix, .25) is matrix


def test_default_one_amplitude_and_reference_labels_invariant():
    frame = pd.DataFrame({"x": [1., np.nan, 3., 4., 8., 9.] * 10,
                          "y": np.arange(60) % 7})
    before = frame.copy(deep=True)
    default, base = fit_predict_clustering("clustering", frame, n_clusters=2, seed=17,
                                         experimental_controls={"runtime_instrumentation": True})
    explicit, info = fit_predict_clustering("clustering", frame, n_clusters=2, seed=17,
        y_train=np.arange(60), X_test=pd.DataFrame({"wrong": [-999]}),
        experimental_controls={"clustering_indicator_amplitude": 1., "runtime_instrumentation": True})
    np.testing.assert_array_equal(default, explicit)
    for key in ("model", "selection", "bic", "gmm_diagnostics"):
        assert base[key] == info[key]
    assert "clustering_indicator_amplitude" not in info
    pd.testing.assert_frame_equal(frame, before)
    reduced, reduced_info = fit_predict_clustering("clustering", frame, n_clusters=2, seed=17,
        experimental_controls={"clustering_indicator_amplitude": .25})
    other, _ = fit_predict_clustering("clustering", frame, n_clusters=2, seed=17,
        y_train=np.arange(60)[::-1], X_test=pd.DataFrame({"wrong": [999]}),
        experimental_controls={"clustering_indicator_amplitude": .25})
    np.testing.assert_array_equal(reduced, other)
    assert reduced_info["clustering_indicator_amplitude"] == .25
    pd.testing.assert_frame_equal(frame, before)


def test_frozen_generator_and_protocol():
    path = Path(__file__).resolve().parents[1]/"scripts/third_clustering.py"
    spec = importlib.util.spec_from_file_location("third_clustering", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.PROTOCOL["geometry_calls"] == 48
    assert module.PROTOCOL["actual_baseline_calls"] == 24
    for family in module.FAMILIES:
        a, ref, k = module.generate(family, 99013, "informative")
        b, ref2, k2 = module.generate(family, 99013, "informative")
        pd.testing.assert_frame_equal(a, b)
        np.testing.assert_array_equal(ref, ref2)
        assert k == k2
