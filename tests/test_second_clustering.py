"""Observational diagnostics and second-batch harness correctness."""
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

from unknown_dataset.clustering import fit_predict_clustering


def _harness():
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    sys.path.insert(0, str(scripts))
    try:
        spec = importlib.util.spec_from_file_location("second_clustering", scripts / "second_clustering.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(scripts))


def test_diagnostics_observational_and_labels_unused():
    rng = np.random.default_rng(311)
    frame = pd.DataFrame(rng.normal(size=(90, 2)), columns=["a", "b"])
    before = frame.copy(deep=True)
    plain, base = fit_predict_clustering("clustering", frame, n_clusters=3,
                                        seed=12, time_budget=20)
    observed, info = fit_predict_clustering("clustering", frame,
        y_train=np.arange(90) % 3, X_test=pd.DataFrame({"unrelated": [-999]}),
        n_clusters=3, seed=12, time_budget=20,
        experimental_controls={"runtime_instrumentation": True})
    np.testing.assert_array_equal(plain, observed)
    pd.testing.assert_frame_equal(frame, before)
    for key in ("model", "selection", "bic", "transformed_features"):
        assert base[key] == info[key]
    assert "gmm_diagnostics" not in base
    assert len(info["gmm_diagnostics"]) == 3
    assert [d["restarts"] for d in info["gmm_diagnostics"]] == [2, 1, 1]
    for diagnostic in info["gmm_diagnostics"]:
        assert abs(sum(diagnostic["weights"]) - 1) < 1e-10
        assert diagnostic["iterations"] > 0
        assert np.isfinite(diagnostic["bic"])


def test_kmeans_ablation_has_no_gmm_diagnostics():
    frame = pd.DataFrame({"a": range(50), "b": np.arange(50) ** 2})
    _, info = fit_predict_clustering("clustering", frame, n_clusters=2,
        time_budget=20, experimental_controls={"runtime_instrumentation": True,
            "clustering_missing_indicators": False, "clustering_kmeans_only": True})
    assert info["gmm_diagnostics"] == []
    assert info["model"] == "KMeans"


def test_fresh_generators_are_deterministic_and_groups_inform_masks():
    harness = _harness()
    for family in harness.FAMILIES:
        a, groups, k = harness.fresh_case(family, 7201, "group_informative")
        b, other, other_k = harness.fresh_case(family, 7201, "group_informative")
        pd.testing.assert_frame_equal(a, b)
        np.testing.assert_array_equal(groups, other)
        assert k == other_k
        for j, column in enumerate(a.select_dtypes(include="number").columns):
            designated = groups == j % k
            assert a.loc[designated, column].isna().mean() > a.loc[~designated, column].isna().mean()


def test_historical_loss_count_and_input_fingerprints():
    import json
    harness = _harness()
    report = json.loads((harness.ROOT / "experiments/first-batch/clustering.json").read_text())
    losses = harness.historical_losses(report)
    assert len(losses) == 8
    for row, baseline in losses:
        case = {key: row[key] for key in ("family", "seed", "mechanism", "fraction")}
        frame, reference, k = harness.load_historical_case(case)
        assert harness.digest_frame(frame) == row["input_sha256"] == baseline["input_sha256"]
        regenerated, generated_reference, generated_k = harness.generate(**case)
        assert list(frame.columns) == list(regenerated.columns)
        assert frame.index.equals(regenerated.index)
        assert frame.dtypes.equals(regenerated.dtypes)
        np.testing.assert_array_equal(frame.isna(), regenerated.isna())
        # multivariate_normal uses platform/CPU-dependent LAPACK and BLAS.
        # Keep historical bytes exact; separately constrain regeneration drift.
        np.testing.assert_allclose(frame.to_numpy(), regenerated.to_numpy(),
                                   rtol=1e-14, atol=1e-14, equal_nan=True)
        np.testing.assert_array_equal(reference, generated_reference)
        assert k == generated_k


def test_historical_fixture_corruption_is_rejected(tmp_path, monkeypatch):
    harness = _harness()
    damaged = tmp_path / "damaged.npz"
    payload = bytearray(harness.HISTORICAL_INPUTS.read_bytes())
    payload[len(payload) // 2] ^= 1
    damaged.write_bytes(payload)
    monkeypatch.setattr(harness, "HISTORICAL_INPUTS", damaged)
    with pytest.raises(ValueError, match="fixture checksum mismatch"):
        harness.load_historical_case(dict(family="correlated", seed=6101,
                                          mechanism="feature", fraction=.3))


def test_gmm_seed_override_holds_sample_and_kmeans_centers_fixed(monkeypatch):
    import unknown_dataset.clustering as clustering
    harness = _harness()
    real_gmm = clustering.GaussianMixture
    records = []

    def factory(**kwargs):
        candidate = real_gmm(**kwargs)
        original_fit = candidate.fit

        def fit(sample):
            records.append((sample.copy(), kwargs["means_init"],
                            kwargs["random_state"], kwargs["n_init"]))
            return original_fit(sample)

        candidate.fit = fit
        return candidate

    monkeypatch.setattr(clustering, "GaussianMixture", factory)
    monkeypatch.setattr(harness, "GaussianMixture", factory)
    frame, reference, k = harness.generate("missing_axis", 6101, "none", 0.)
    common = dict(family="missing_axis", mechanism="none", fraction=0.,
                  seed=6101, policy="no_indicators", segment="test")
    harness.evaluate(frame, reference, k, **common)
    harness.evaluate(frame, reference, k, **common, gmm_seed_override=6102)
    assert len(records) == 6
    for first, second in zip(records[:3], records[3:]):
        np.testing.assert_array_equal(first[0], second[0])
        if first[1] is None:
            assert second[1] is None
        else:
            np.testing.assert_array_equal(first[1], second[1])
        assert first[2] == 6101 and second[2] == 6102
        assert first[3] == second[3]
