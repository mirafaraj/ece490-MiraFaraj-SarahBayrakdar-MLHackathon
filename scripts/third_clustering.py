"""Frozen small amplitude study; reference groups are reporting-only."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.datasets import make_moons
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score

from unknown_dataset.clustering import fit_predict_clustering
from unknown_dataset.models import fit_predict

ROOT = Path(__file__).resolve().parents[1]
FAMILIES = ("correlated", "curved", "anisotropic_noise", "mixed")
SEEDS = (99013, 99041)
MASKS = ("none", "random", "informative")
PROTOCOL = dict(rows=240, families=FAMILIES, seeds=SEEDS, masks=MASKS,
    amplitudes=[1.0, .25], model_budget_seconds=20,
    geometry_calls=48, actual_baseline_calls=24,
    random_missing_probability=.30, informative_probabilities=[.65, .03],
    notice="Fresh development only, not confirmation; no parameter tuning.")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def serialize_frame(frame):
    return {"columns": list(frame.columns), "data": [
        [None if pd.isna(value) else value for value in row]
        for row in frame.itertuples(index=False, name=None)]}


def generate(family, seed, masking):
    if family not in FAMILIES or masking not in MASKS:
        raise ValueError("Unknown frozen generator condition")
    rng = np.random.default_rng(seed)
    if family == "curved":
        values, reference = make_moons(n_samples=240, noise=.08, random_state=seed)
        frame = pd.DataFrame(values, columns=["x", "y"])
        k = 2
    else:
        reference = np.repeat(np.arange(3), 80)
        z = rng.normal(size=(240, 2))
        if family == "correlated":
            # Explicit scalar transforms avoid platform-dependent LAPACK roots.
            centers = np.array([[-2., 2.], [0., 0.], [2., -2.]])
            values = centers[reference] + np.column_stack([
                np.sqrt(2.) * z[:, 0], np.sqrt(2.) * (.97*z[:, 0] + np.sqrt(1-.97**2)*z[:, 1])])
        elif family == "anisotropic_noise":
            centers = np.array([[-4., 0.], [0., 3.], [4., 0.]])
            values = centers[reference] + np.column_stack([
                1.5*z[:, 0] + .25*z[:, 1], .8*z[:, 0] + .45*z[:, 1]])
        else:
            centers = np.array([[-1., 0.], [0., 1.], [1., 0.]])
            values = centers[reference] + 1.1*z
        frame = pd.DataFrame(values, columns=["x", "y"])
        k = 3
        if family == "anisotropic_noise":
            for j in range(4):
                frame[f"noise{j}"] = rng.normal(size=240)
                frame.loc[rng.random(240) < .70, f"noise{j}"] = np.nan
        if family == "mixed":
            # Synthetic group/category association is part of generation only.
            category = reference.copy()
            replace = rng.random(240) < .20
            category[replace] = rng.integers(0, k, replace.sum())
            frame["category"] = np.array(["a", "b", "c"])[category]
    numeric = list(frame.select_dtypes(include="number").columns)
    if masking != "none":
        probabilities = np.full((len(frame), len(numeric)), .30)
        if masking == "informative":
            probabilities = np.column_stack([
                np.where(reference == j % k, .65, .03) for j in range(len(numeric))])
        frame.loc[:, numeric] = frame[numeric].mask(rng.random(probabilities.shape) < probabilities)
    return frame, reference, k


def sources():
    names = ["scripts/third_clustering.py", "src/unknown_dataset/clustering.py",
             "src/unknown_dataset/models.py", "src/unknown_dataset/preprocessing.py",
             "src/unknown_dataset/policies.py"]
    return {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in names}


def summarize(report):
    groups = {}
    for segment, rows in [("all", report["rows"])] + [
            (key, [r for r in report["rows"] if r["masking"] == key]) for key in MASKS] + [
            (key, [r for r in report["rows"] if r["family"] == key]) for key in FAMILIES]:
        groups[segment] = {}
        for policy in ("baseline", "current", "quarter"):
            chosen = [r for r in rows if r["policy"] == policy]
            counts = {}
            for row in chosen:
                counts[row["info"]["model"]] = counts.get(row["info"]["model"], 0)+1
            groups[segment][policy] = dict(n=len(chosen),
                ari=float(np.mean([r["ari"] for r in chosen])),
                ami=float(np.mean([r["ami"] for r in chosen])),
                median_seconds=float(np.median([r["wall_seconds"] for r in chosen])),
                max_seconds=max(r["wall_seconds"] for r in chosen),
                invalid=sum(not r["valid"] for r in chosen), selected_models=counts,
                repairs=sum(r["info"]["empty_clusters_repaired"] for r in chosen))
    indexed = {(r["family"], r["seed"], r["masking"], r["policy"]): r for r in report["rows"]}
    pairs = []
    for case in report["cases"]:
        key = (case["family"], case["seed"], case["masking"])
        a, b = indexed[key+("current",)], indexed[key+("quarter",)]
        pairs.append(dict(family=key[0], seed=key[1], masking=key[2],
            ari_delta=b["ari"]-a["ari"], ami_delta=b["ami"]-a["ami"],
            current_model=a["info"]["model"], quarter_model=b["info"]["model"]))
    return dict(groups=groups, pairs=pairs,
        severe_losses=[p for p in pairs if min(p["ari_delta"], p["ami_delta"]) <= -.10],
        elapsed_seconds=report["elapsed_seconds"])


def evaluate(frame, reference, k, family, seed, masking, policy):
    before = serialize_frame(frame)
    started = time.monotonic()
    if policy == "baseline":
        predictions, info = fit_predict("clustering", frame, n_clusters=k,
                                        time_budget=20, seed=seed)
    else:
        amplitude = 1.0 if policy == "current" else .25
        predictions, info = fit_predict_clustering("clustering", frame,
            n_clusters=k, time_budget=20, seed=seed,
            experimental_controls={"clustering_indicator_amplitude": amplitude,
                                   "runtime_instrumentation": True})
    elapsed = time.monotonic()-started
    if before != serialize_frame(frame):
        raise AssertionError("Model mutated input features")
    predictions = np.asarray(predictions).tolist()
    return dict(family=family, seed=seed, masking=masking, policy=policy,
        input_sha256=digest(before), predictions=predictions,
        prediction_sha256=digest(predictions),
        ari=float(adjusted_rand_score(reference, predictions)),
        ami=float(adjusted_mutual_info_score(reference, predictions)),
        valid=bool(len(predictions)==len(frame) and len(set(predictions))==k),
        wall_seconds=elapsed, info=info)


def run(output):
    summary_path = output.with_name(output.stem+"-summary.json")
    if output.exists() or summary_path.exists():
        raise FileExistsError("Preserving existing evidence")
    frozen = sources()
    report = dict(protocol=PROTOCOL, source_sha256=frozen, cases=[], rows=[])
    started = time.monotonic()
    for family in FAMILIES:
        for seed in SEEDS:
            for masking in MASKS:
                frame, reference, k = generate(family, seed, masking)
                payload = serialize_frame(frame)
                report["cases"].append(dict(family=family, seed=seed, masking=masking,
                    input=payload, input_sha256=digest(payload), reference=reference.tolist(),
                    numeric_missing_mask=frame.select_dtypes(include="number").isna().values.tolist(),
                    clusters=k))
                for policy in ("baseline", "current", "quarter"):
                    if sources() != frozen:
                        raise AssertionError("Source changed during frozen study")
                    report["rows"].append(evaluate(frame, reference, k, family, seed, masking, policy))
        print(family, len(report["rows"]), "completed", flush=True)
    report["elapsed_seconds"] = time.monotonic()-started
    if not all(row["valid"] for row in report["rows"]):
        raise AssertionError("Invalid clustering output")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    summary_path.write_text(json.dumps(summarize(report), indent=2, allow_nan=False), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("experiments/third-batch/clustering.json"))
    run(parser.parse_args().output)
