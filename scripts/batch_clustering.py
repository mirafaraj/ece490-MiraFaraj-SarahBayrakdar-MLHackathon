"""Bounded development ablations; external labels never enter model code.

Predeclared: six mechanisms, five missingness conditions, three development
seeds, four isolated policies, 240 rows. This is diagnostic direct-call evidence,
not unopened confirmation or a claim of official 60-second CLI compliance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs, make_moons
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score

from unknown_dataset.clustering import fit_predict_clustering

SEEDS = (6101, 6103, 6107)
FAMILIES = ("spherical", "correlated", "unequal_density", "mixed", "curved", "missing_axis")
CONDITIONS = (("none", 0.), ("random", .1), ("random", .3),
              ("feature", .1), ("feature", .3))
POLICIES = {"current": {}, "kmeans_only": {"clustering_kmeans_only": True},
            "no_indicators": {"clustering_missing_indicators": False},
            "diagonal": {"clustering_covariance": "diag"}}


def generate(family, seed, mechanism, fraction):
    rng = np.random.default_rng(seed)
    centers = [[-5., -4.], [0., 5.], [5., -3.]]
    if family == "curved":
        x, y = make_moons(n_samples=240, noise=.07, random_state=seed)
    elif family == "correlated":
        y = np.repeat(np.arange(3), 80)
        x = np.vstack([rng.multivariate_normal(c, [[2., 1.94], [1.94, 2.]], 80)
                       for c in [[-2., 2.], [0., 0.], [2., -2.]]])
    else:
        x, y = make_blobs(n_samples=[40, 80, 120] if family == "unequal_density" else 240,
            centers=centers, cluster_std=[.3, 1.8, .8] if family == "unequal_density" else .8,
            random_state=seed)
    frame = pd.DataFrame(x, columns=["x", "y"])
    if family == "missing_axis":
        # Additional irrelevant, heavily missing axis mirrors the mechanism of
        # inspected failures, while using new generated development rows.
        frame["missing_noise"] = rng.normal(size=len(frame))
        frame.loc[rng.random(len(frame)) < .5, "missing_noise"] = np.nan
    if family == "mixed":
        # Feature category derives from observed x, never from reference labels.
        frame["category"] = np.where(x[:, 0] > 0, "right", "left")
    numeric = frame.select_dtypes(include="number").columns
    if fraction:
        if mechanism == "random":
            mask = rng.random((len(frame), len(numeric))) < fraction
        else:
            # Exact per-column rate; missingness depends on a different observed
            # feature with small independent jitter, not on reference labels.
            scores = np.roll(x, 1, axis=1)
            mask = np.zeros((len(frame), len(numeric)), dtype=bool)
            for j in range(len(numeric)):
                order = np.argsort(scores[:, j % 2] + rng.normal(0, .2, len(frame)))
                mask[order[-round(fraction * len(frame)):], j] = True
        frame.loc[:, numeric] = frame[numeric].mask(mask)
    return frame, y, len(np.unique(y))


def run(output):
    if output.exists():
        raise FileExistsError(f"Preserving existing experiment report: {output}")
    root = Path(__file__).resolve().parents[1]
    sources = [Path(__file__), root / "src/unknown_dataset/clustering.py",
               root / "src/unknown_dataset/preprocessing.py"]
    fingerprints = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    rows = []
    started = time.monotonic()
    for family in FAMILIES:
        for mechanism, fraction in CONDITIONS:
            for seed in SEEDS:
                x, reference, k = generate(family, seed, mechanism, fraction)
                input_hash = hashlib.sha256(pd.util.hash_pandas_object(x, index=True).to_numpy().tobytes()).hexdigest()
                for policy, controls in POLICIES.items():
                    controls = {**controls, "runtime_instrumentation": True}
                    begin = time.monotonic()
                    pred, info = fit_predict_clustering("clustering", x,
                        n_clusters=k, seed=seed, time_budget=20,
                        experimental_controls=controls)
                    rows.append(dict(family=family, mechanism=mechanism, fraction=fraction,
                        seed=seed, policy=policy, ari=adjusted_rand_score(reference, pred),
                        input_sha256=input_hash,
                        ami=adjusted_mutual_info_score(reference, pred),
                        valid=bool(len(pred) == len(x) and len(np.unique(pred)) == k),
                        wall_seconds=time.monotonic()-begin,
                        prediction_sha256=hashlib.sha256(np.asarray(pred, dtype="int64").tobytes()).hexdigest(),
                        info=info))
        print(family, "completed", len(rows), "runs", flush=True)
    report = dict(notice="Development evidence only; small direct calls, no promotion.",
        seeds=SEEDS, families=FAMILIES, conditions=CONDITIONS, policies=POLICIES,
        rows_per_case=240, source_sha256=fingerprints,
        elapsed_seconds=time.monotonic()-started, rows=rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path,
        default=Path("experiments/first-batch/clustering.json"))
    args = parser.parse_args()
    run(args.output)
