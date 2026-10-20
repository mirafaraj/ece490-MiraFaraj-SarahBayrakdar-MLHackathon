"""Frozen, bounded offline regression capacity experiment; no outer-label selection."""
from __future__ import annotations
import argparse
from collections import Counter
from dataclasses import replace
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from unknown_dataset.routing import make_route_plan, summarize_training
from unknown_dataset.supervised import fit_predict_fixed_candidate, select_candidate

SEEDS = (98003, 98017, 98029)
MECHANISMS = ("sparse_linear", "numeric_interaction", "mixed_category", "missing_product", "heavy_tail", "heteroscedastic")
VARIANTS = ("linear", "extra_trees", "histogram15", "histogram31")
TARGETS = MECHANISMS[1:]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def case_data(mechanism, seed, rows=1200):
    """Related synthetic mechanisms, parameters declared before first study fit."""
    if mechanism not in MECHANISMS:
        raise ValueError(mechanism)
    rng = np.random.default_rng(seed + 1000 * MECHANISMS.index(mechanism))
    x = rng.normal(size=(rows, 12))
    category = rng.integers(0, 5, rows)
    noise = rng.normal(size=rows)
    if mechanism == "sparse_linear":
        y = 5*x[:, 0] - 3*x[:, 1] + 2*x[:, 2] + .5*noise
    elif mechanism == "numeric_interaction":
        y = 5*x[:, 0]*x[:, 1] + 4*np.sin(2*x[:, 2]) + 3*(x[:, 3] > .25)*(x[:, 4] > -.25) + .7*noise
    elif mechanism == "mixed_category":
        y = (category-2)*x[:, 0]*3 + 4*np.sin(x[:, 1]*2) + 3*(category == 4)*(x[:, 2] > 0) + .7*noise
    elif mechanism == "missing_product":
        y = 5*x[:, 0]*x[:, 1] + 3*np.sin(2*x[:, 2]) + .7*noise
    elif mechanism == "heavy_tail":
        y = 4*np.sin(x[:, 0]*2) + 3*x[:, 1]*x[:, 2] + rng.standard_t(3, rows)*1.5
    else:
        y = 4*np.sin(2*x[:, 0]) + 3*x[:, 1]*x[:, 2] + (.3 + 2*np.abs(x[:, 3]))*noise
    frame = pd.DataFrame(x, columns=[f"x{i}" for i in range(12)])
    frame["category"] = np.asarray([f"level_{v}" for v in category], dtype=object)
    missing = rng.random((rows, 12)) < (.20 if mechanism == "missing_product" else .05)
    frame.iloc[:, :12] = np.where(missing, np.nan, x)
    types = {name: "numeric" for name in frame.columns[:-1]}
    types["category"] = "categorical"
    return frame, y, types


def split_indices(rows, seed):
    train, test = train_test_split(np.arange(rows), test_size=.25, random_state=seed)
    inner_train, inner_validation = train_test_split(train, test_size=.25, random_state=seed)
    return dict(train=train.tolist(), test=test.tolist(), inner_train=inner_train.tolist(), inner_validation=inner_validation.tolist())


def metrics(truth, prediction, dummy=None, missing=None):
    residual = np.asarray(prediction, dtype=float) - np.asarray(truth, dtype=float)
    absolute = np.abs(residual)
    result = dict(rmse=float(np.sqrt(np.mean(residual**2))), mae=float(np.mean(absolute)),
                  absolute_error_p90=float(np.quantile(absolute, .9)), absolute_error_p99=float(np.quantile(absolute, .99)))
    if dummy is not None:
        reference = metrics(truth, dummy)
        result["dummy_relative_rmse"] = result["rmse"] / reference["rmse"]
        result["dummy_relative_mae"] = result["mae"] / reference["mae"]
    if missing is not None:
        mask = np.asarray(missing, dtype=bool)
        result["row_groups"] = {name: dict(rows=int(group.sum()), **metrics(np.asarray(truth)[group], np.asarray(prediction)[group]))
                                for name, group in (("missing", mask), ("observed", ~mask)) if group.any()}
    return result


def choose(inner, metric, capacity, cap=20.0):
    """Descriptive elapsed-work replay; actual bounded CLI remains separate."""
    variants = dict(extra_trees="extra_trees", linear="linear", histogram=f"histogram{capacity}")
    scores, excluded, elapsed = {}, {}, 0.0
    for name, key in variants.items():
        if elapsed >= cap:
            excluded[name] = "shared selection cap exhausted"
            continue
        elapsed += inner[key]["seconds"]
        if inner[key].get("error"):
            excluded[name] = inner[key]["error"]
        else:
            scores[name] = inner[key]["metrics"][metric]
    name = select_candidate(scores, metric)
    return dict(selected=name, variant=variants[name], validation_scores=scores,
                excluded=excluded, selection_seconds=elapsed, selection_cap_seconds=cap)


def run_case(mechanism, seed):
    frame, target, types = case_data(mechanism, seed)
    splits = split_indices(len(frame), seed)
    inputs = dict(frame=dict(columns=frame.columns.tolist(), index=frame.index.tolist(),
                             data=frame.astype(object).where(frame.notna(), None).values.tolist()),
                  target=target.tolist(), feature_types=types)
    records = dict(mechanism=mechanism, seed=seed, inputs=inputs, input_sha256=digest(inputs),
                   splits=splits, split_sha256=digest(splits), inner={}, outer={}, policies={})
    started = time.monotonic()
    for partition, fit_key, test_key, final in (("inner", "inner_train", "inner_validation", False), ("outer", "train", "test", True)):
        fitting, evaluation = splits[fit_key], splits[test_key]
        training = frame.iloc[fitting]
        plan = make_route_plan(summarize_training("regression", training, target[fitting], types), time_budget=50)
        dummy = np.full(len(evaluation), float(target[fitting].mean()))
        for variant in VARIANTS:
            name = "histogram" if variant.startswith("histogram") else variant
            experimental = dict(plan.experimental)
            experimental["histogram_max_leaf_nodes"] = 31 if variant == "histogram31" else 15
            call_plan = replace(plan, experimental=experimental)
            call_started = time.monotonic()
            try:
                prediction, info = fit_predict_fixed_candidate(name, "regression", training, target[fitting], frame.iloc[evaluation],
                                                               types, seed, final=final, route_plan=call_plan)
                if not np.isfinite(prediction).all():
                    raise ValueError("nonfinite predictions")
                entry = dict(prediction=prediction.tolist(), prediction_sha256=digest(prediction.tolist()), truth=target[evaluation].tolist(),
                             metrics=metrics(target[evaluation], prediction, dummy, frame.iloc[evaluation].isna().any(axis=1).to_numpy()), info=info)
            except (ValueError, MemoryError) as exc:
                entry = dict(error=str(exc))
            entry["seconds"] = time.monotonic() - call_started
            records[partition][variant] = entry
    for metric in ("rmse", "mae"):
        for capacity in (15, 31):
            selected = choose(records["inner"], metric, capacity)
            outer = records["outer"][selected["variant"]]
            eligible = [v for v in ("linear", "extra_trees", f"histogram{capacity}") if "metrics" in records["outer"][v]]
            oracle = min(records["outer"][v]["metrics"][metric] for v in eligible)
            selected.update(metrics=outer["metrics"], selected_prediction_sha256=outer["prediction_sha256"],
                            regret=outer["metrics"][metric] - oracle, refit_seconds=outer["seconds"])
            records["policies"][f"{metric}_{capacity}"] = selected
    records["seconds"] = time.monotonic() - started
    return records


def summarize(cases):
    result = {}
    for metric in ("rmse", "mae"):
        rows = []
        for case in cases:
            before = case["policies"][f"{metric}_15"]["metrics"][metric]
            after = case["policies"][f"{metric}_31"]["metrics"][metric]
            baseline = case["outer"]["extra_trees"]["metrics"][metric]
            rows.append(dict(mechanism=case["mechanism"], seed=case["seed"], before=before, after=after,
                             relative_improvement=(before-after)/before, baseline=baseline,
                             current_vs_baseline=(baseline-before)/baseline, challenger_vs_baseline=(baseline-after)/baseline))
        family = {name: float(np.mean([v["relative_improvement"] for v in rows if v["mechanism"] == name])) for name in MECHANISMS}
        seed_effect = {str(seed): float(np.mean([v["relative_improvement"] for v in rows if v["seed"] == seed and v["mechanism"] in TARGETS])) for seed in SEEDS}
        target_gain = float(np.mean([family[name] for name in TARGETS]))
        result[metric] = dict(cases=rows, family_relative_improvements=family, target_relative_improvement=target_gain,
                              seed_relative_improvements=seed_effect,
                              gate_pass=target_gain >= .03 and sum(v > 0 for v in seed_effect.values()) >= 2 and family["sparse_linear"] >= -.02,
                              selections={str(cap): dict(Counter(c["policies"][f"{metric}_{cap}"]["selected"] for c in cases)) for cap in (15,31)})
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__), *sorted((ROOT/"src/unknown_dataset").glob("*.py"))]}
    cases = []
    started = time.monotonic()
    for mechanism in MECHANISMS:
        for seed in SEEDS:
            cases.append(run_case(mechanism, seed))
            print(f"{len(cases)}/18 {mechanism} {seed}", flush=True)
    for relative, expected in hashes.items():
        if hashlib.sha256((ROOT/relative).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"Source changed during study: {relative}")
    output = dict(protocol="reports/third-batch/regression.md", source_sha256=hashes,
                  environment=dict(python=platform.python_version(), platform=platform.platform(),
                                   packages={name: importlib.metadata.version(name) for name in ("numpy", "pandas", "scikit-learn")}),
                  seconds=time.monotonic()-started, fit_calls=144, cases=cases, summary=summarize(cases))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(output["summary"], indent=2))


if __name__ == "__main__":
    main()
