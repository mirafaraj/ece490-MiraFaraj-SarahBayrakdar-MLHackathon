"""Small, declared development ablations; references stay outside model calls.

Eight historical losses: fixed input hashes, 2x2, two identical repeats, then
two changed fitting seeds and two GMM-initialization-only seeds for
indicator-free full geometry only (96 calls).
Fresh parameter perturbations: six existing mechanisms x two seeds x random or
latent-group-dependent missingness x four policies (96 calls). Not confirmation.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score
from sklearn.mixture import GaussianMixture

from batch_clustering import FAMILIES, generate
from unknown_dataset.clustering import fit_predict_clustering

ROOT = Path(__file__).resolve().parents[1]
POLICIES = {
    "current": {},
    "no_indicators": {"clustering_missing_indicators": False},
    "kmeans_only": {"clustering_kmeans_only": True},
    "no_indicators_kmeans": {"clustering_missing_indicators": False,
                              "clustering_kmeans_only": True},
}
FRESH_SEEDS = (7201, 7211)
HISTORICAL_INPUTS = ROOT / "experiments/fixtures/historical-clustering-inputs-v1.npz"
HISTORICAL_INPUTS_SHA256 = "54cdc3d72271e2b452656ff0b0702883b1024f28ac8c8a4193b3c6ffa8fd3c8f"


def digest_frame(frame):
    return hashlib.sha256(pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes()).hexdigest()


def load_historical_case(case):
    """Replay exact evidence inputs, independent of BLAS/libm generation drift.

    The lossless, little-endian arrays retain the original pandas fingerprints.
    Fresh generation remains separate; never round inputs or rewrite old hashes.
    """
    if hashlib.sha256(HISTORICAL_INPUTS.read_bytes()).hexdigest() != HISTORICAL_INPUTS_SHA256:
        raise ValueError("Historical input fixture checksum mismatch")
    identity = json.dumps([case[key] for key in ("family", "seed", "mechanism", "fraction")],
                          separators=(",", ":"))
    with np.load(HISTORICAL_INPUTS, allow_pickle=False) as saved:
        metadata = json.loads(str(saved["metadata"]))
        record = metadata["cases"][identity]
        prefix = record["prefix"]
        frame = pd.DataFrame(saved[prefix + "_data"].copy(),
                             columns=saved[prefix + "_columns"].tolist())
        reference = saved[prefix + "_reference"].copy()
    if digest_frame(frame) != record["input_sha256"]:
        raise ValueError("Historical frame fingerprint mismatch")
    return frame, reference, record["clusters"]


def historical_losses(report):
    current = {(r["family"], r["mechanism"], r["fraction"], r["seed"]): r
               for r in report["rows"] if r["policy"] == "current"}
    return [(r, current[(r["family"], r["mechanism"], r["fraction"], r["seed"])])
            for r in report["rows"] if r["policy"] == "no_indicators" and
            r["ari"] < current[(r["family"], r["mechanism"], r["fraction"], r["seed"])]["ari"] - 1e-12]


def fresh_case(family, seed, mechanism):
    frame, reference, k = generate(family, seed, "none", 0.)
    rng = np.random.default_rng(seed + 919)
    numeric = list(frame.select_dtypes(include="number").columns)
    # New parameter setting: independent noise with SD=12% of each observed
    # feature's SD, before masking. Existing six mechanisms, not new families.
    for name in numeric:
        frame[name] += rng.normal(0., .12 * frame[name].std(), len(frame))
    if mechanism == "random":
        probabilities = np.full((len(frame), len(numeric)), .30)
    else:
        # Synthetic data-generation intervention: missingness itself carries
        # latent-group information. Reference groups are NEVER sent to fitting.
        probabilities = np.column_stack([
            np.where(reference == j % k, .65, .03) for j in range(len(numeric))])
    frame.loc[:, numeric] = frame[numeric].mask(rng.random(probabilities.shape) < probabilities)
    return frame, reference, k


def evaluate(frame, reference, k, *, family, mechanism, fraction, seed,
             policy, segment, repeat=0, fit_seed=None, gmm_seed_override=None):
    fit_seed = seed if fit_seed is None else fit_seed
    def gmm_factory(*args, **kwargs):
        kwargs["random_state"] = gmm_seed_override
        return GaussianMixture(*args, **kwargs)
    # Harness-only constructor interception: holds the public fitting seed,
    # KMeans initialization, GMM training sample and all restart counts fixed.
    context = (patch("unknown_dataset.clustering.GaussianMixture", gmm_factory)
               if gmm_seed_override is not None else nullcontext())
    started = time.monotonic()
    with context:
        prediction, info = fit_predict_clustering("clustering", frame,
            n_clusters=k, seed=fit_seed, time_budget=20,
            experimental_controls={**POLICIES[policy], "runtime_instrumentation": True})
    elapsed = time.monotonic() - started
    # Offline diagnostic: are returned groups following missingness patterns?
    masks = frame.isna().to_numpy(dtype=np.int64)
    mask_codes = masks @ (2 ** np.arange(masks.shape[1]))
    return dict(segment=segment, family=family, mechanism=mechanism,
        fraction=fraction, seed=seed, fit_seed=fit_seed,
        gmm_seed_override=gmm_seed_override, repeat=repeat, policy=policy,
        input_sha256=digest_frame(frame),
        reference=reference.tolist(), predictions=prediction.tolist(),
        prediction_sha256=hashlib.sha256(np.asarray(prediction, dtype="int64").tobytes()).hexdigest(),
        ari=float(adjusted_rand_score(reference, prediction)),
        ami=float(adjusted_mutual_info_score(reference, prediction)),
        mask_alignment_ari=float(adjusted_rand_score(mask_codes, prediction)),
        mask_alignment_ami=float(adjusted_mutual_info_score(mask_codes, prediction)),
        valid=bool(len(prediction) == len(frame) and len(np.unique(prediction)) == k),
        wall_seconds=elapsed, missingness=frame.isna().mean().to_dict(),
        missingness_by_reference_group={str(group): frame.loc[reference == group].isna().mean().to_dict()
                                        for group in np.unique(reference)},
        info=info)


def run(output):
    if output.exists():
        raise FileExistsError(f"Preserving existing evidence: {output}")
    historical = json.loads((ROOT / "experiments/first-batch/clustering.json").read_text())
    losses = historical_losses(historical)
    if len(losses) != 8:
        raise AssertionError(f"Expected eight historical losses, got {len(losses)}")
    sources = [Path(__file__), ROOT / "scripts/batch_clustering.py", HISTORICAL_INPUTS,
               ROOT / "src/unknown_dataset/clustering.py", ROOT / "src/unknown_dataset/preprocessing.py"]
    source_sha256 = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    rows, reproduction = [], []
    started = time.monotonic()
    for previous, baseline in losses:
        case = {key: previous[key] for key in ("family", "mechanism", "fraction", "seed")}
        frame, reference, k = load_historical_case(case)
        if digest_frame(frame) != previous["input_sha256"]:
            raise AssertionError(f"Historical input hash changed: {case}")
        for repeat in range(2):
            for policy in POLICIES:
                row = evaluate(frame, reference, k, **case, policy=policy,
                               segment="historical_2x2", repeat=repeat)
                rows.append(row)
                if policy in ("current", "no_indicators"):
                    old = baseline if policy == "current" else previous
                    reproduction.append(dict(**case, policy=policy, repeat=repeat,
                        input_match=row["input_sha256"] == old["input_sha256"],
                        prediction_match=row["prediction_sha256"] == old["prediction_sha256"],
                        ari_delta=row["ari"] - old["ari"]))
        for delta in (1, 2):
            rows.append(evaluate(frame, reference, k, **case, policy="no_indicators",
                segment="fitting_seed_sensitivity", fit_seed=case["seed"] + delta))
            rows.append(evaluate(frame, reference, k, **case, policy="no_indicators",
                segment="gmm_initialization_sensitivity", gmm_seed_override=case["seed"] + delta))
    print("historical reproduction/sensitivity completed", len(rows), flush=True)
    for family in FAMILIES:
        for seed in FRESH_SEEDS:
            for mechanism in ("random", "group_informative"):
                frame, reference, k = fresh_case(family, seed, mechanism)
                for policy in POLICIES:
                    rows.append(evaluate(frame, reference, k, family=family,
                        mechanism=mechanism, fraction=None, seed=seed,
                        policy=policy, segment="fresh_parameters"))
        print(family, "completed", len(rows), flush=True)
    report = dict(notice="Development evidence, direct calls only; no promotion or confirmation.",
        parameters=dict(fresh_seeds=FRESH_SEEDS, noise_sd_fraction=.12,
            random_mask_probability=.30, informative_mask_probabilities=[.65, .03],
            rows=240, model_budget_seconds=20, policies=POLICIES),
        source_sha256=source_sha256, historical_reproduction=reproduction,
        elapsed_seconds=time.monotonic()-started, rows=rows)
    if not all(r["input_match"] and r["prediction_match"] and abs(r["ari_delta"]) < 1e-12
               for r in reproduction):
        raise AssertionError("Historical reproduction failed; preserve diagnosis before interpreting")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path,
        default=Path("experiments/second-batch/clustering.json"))
    run(parser.parse_args().output)
