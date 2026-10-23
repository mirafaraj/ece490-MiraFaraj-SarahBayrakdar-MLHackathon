import json

import numpy as np
import pandas as pd
import pytest

from unknown_dataset.clustering import fit_predict_clustering


def test_mixed_candidate_preserves_order_input_and_is_deterministic():
    frame = pd.DataFrame({"x": [0, 1, np.nan, 9, 10, 11],
                          "category": ["a", "a", None, "b", "b", "b"]})
    original = frame.copy(deep=True)
    kwargs = dict(task="clustering", X_train=frame, n_clusters=2, seed=42,
                  feature_types={"x": "numeric", "category": "categorical"})
    labels, info = fit_predict_clustering(**kwargs)
    labels2, _ = fit_predict_clustering(**kwargs)
    assert np.array_equal(labels, labels2)
    assert labels.shape == (len(frame),)
    assert set(labels) == {0, 1}
    assert info["bic"] == {}
    pd.testing.assert_frame_equal(frame, original)
    json.dumps(info, allow_nan=False)


def test_constant_table_occupies_requested_groups_even_with_no_geometry():
    labels, info = fit_predict_clustering("clustering", pd.DataFrame({"x": [1] * 6}),
                                          n_clusters=6, seed=42)
    assert set(labels) == set(range(6))
    assert info["observed_clusters"] == 6


def test_numeric_covariance_candidate_does_not_use_supplied_reference_labels():
    rng = np.random.default_rng(42)
    frame = pd.DataFrame(rng.normal(size=(80, 2)), columns=["a", "b"])
    labels, info = fit_predict_clustering("clustering", frame, n_clusters=2, seed=42)
    labels2, _ = fit_predict_clustering("clustering", frame,
                                      y_train=object(), n_clusters=2, seed=42)
    assert np.array_equal(labels, labels2)
    assert set(labels) == {0, 1}
    assert info["bic"]
    json.dumps(info, allow_nan=False)


@pytest.mark.parametrize("k", [True, 0, 7, 1.5])
def test_invalid_group_count_rejected(k):
    with pytest.raises(ValueError):
        fit_predict_clustering("clustering", pd.DataFrame({"x": [1] * 6}), n_clusters=k)


def test_low_budget_skips_covariance_experiment():
    frame = pd.DataFrame(np.random.default_rng(42).normal(size=(80, 2)))
    _, info = fit_predict_clustering("clustering", frame, n_clusters=2, time_budget=2)
    assert not info["bic"]


def test_baseline_centroid_restart_recovers_noisy_blob_local_optimum(tmp_path):
    # Reproduce the previously observed numeric fixture without benchmark labels
    # entering the candidate. Other benchmark generators consume this RNG first.
    from sklearn.datasets import make_blobs
    from sklearn.metrics import adjusted_rand_score
    from unknown_dataset.models import fit_predict

    rng = np.random.default_rng(42)
    x, _ = make_blobs(n_samples=660, centers=4, n_features=4,
                      cluster_std=1.8, random_state=42)
    frame = pd.DataFrame(np.column_stack([x, rng.normal(0, 2, (660, 4))]))
    baseline, _ = fit_predict("clustering", frame, n_clusters=4, seed=42, strategy="baseline")
    candidate, info = fit_predict_clustering("clustering", frame, n_clusters=4, seed=42)
    assert adjusted_rand_score(baseline, candidate) == 1
    assert info["model"] == "GaussianMixtureFull"


def test_full_candidate_can_improve_geometry_without_baseline_agreement():
    from sklearn.metrics import adjusted_rand_score
    from unknown_dataset.models import fit_predict

    rng = np.random.default_rng(42)
    centers = [[-2, 2], [0, 0], [2, -2]]
    frame = pd.DataFrame(np.vstack([
        rng.multivariate_normal(center, [[2, 1.95], [1.95, 2]], 200)
        for center in centers]))
    truth = np.repeat(np.arange(3), 200)
    baseline, _ = fit_predict("clustering", frame, n_clusters=3, seed=42, strategy="baseline")
    candidate, info = fit_predict_clustering("clustering", frame, n_clusters=3, seed=42)
    assert adjusted_rand_score(truth, baseline) < .7
    assert adjusted_rand_score(truth, candidate) > .99
    assert info["model"] == "GaussianMixtureFull"


@pytest.mark.parametrize("failure", ["error", "nonconvergence"])
def test_failed_mixture_candidates_retain_completed_baseline(monkeypatch, failure):
    from unknown_dataset import clustering
    from unknown_dataset.models import fit_predict

    class UnusableMixture:
        def __init__(self, **kwargs):
            self.converged_ = False

        def fit(self, sample):
            if failure == "error":
                raise ValueError("singular candidate")
            return self

        def bic(self, sample):
            return 0.0

    monkeypatch.setattr(clustering, "GaussianMixture", UnusableMixture)
    frame = pd.DataFrame(np.random.default_rng(42).normal(size=(80, 2)))
    baseline, _ = fit_predict("clustering", frame, n_clusters=2, seed=42, strategy="baseline")
    candidate, info = fit_predict_clustering("clustering", frame, n_clusters=2, seed=42)
    assert np.array_equal(candidate, baseline)
    assert info["bic"] == {}


def test_full_covariance_compares_generic_and_centroid_start(monkeypatch):
    from unknown_dataset import clustering
    from sklearn.mixture import GaussianMixture

    starts = []

    def tracked_mixture(**kwargs):
        starts.append(kwargs)
        return GaussianMixture(**kwargs)

    monkeypatch.setattr(clustering, "GaussianMixture", tracked_mixture)
    frame = pd.DataFrame(np.random.default_rng(42).normal(size=(80, 2)))
    fit_predict_clustering("clustering", frame, n_clusters=2)
    full = [start for start in starts if start["covariance_type"] == "full"]
    assert [start["n_init"] for start in full] == [1, 1]
    assert full[0]["means_init"] is None
    assert full[1]["means_init"].shape == (2, 2)
