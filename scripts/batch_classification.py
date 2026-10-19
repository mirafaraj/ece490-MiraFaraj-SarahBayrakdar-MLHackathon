"""Small development-only factorial study; outer truth never enters selection.

Twelve balanced-design cases are mechanism probes, not independent confirmation.
Candidate statistics are fitted separately inside every inner fold.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
from unknown_dataset.routing import make_route_plan, summarize_training
from unknown_dataset.supervised import classification_metrics, fit_predict_fixed_candidate

METRICS = ("accuracy", "balanced_accuracy", "macro_f1")
FAMILIES = ("extra_trees", "linear", "histogram")


def case_data(index, seed, rows=600):
    """Cross class count/prevalence; rotate other factors without a full grid."""
    rng = np.random.default_rng(seed + index * 101)
    classes = 2 if index < 6 else 3
    prevalence = (.50, .80, .95)[(index // 2) % 3]
    missing = .20 if index % 2 else 0.0
    interaction = bool((index // 2 + index % 2) % 2)
    relevant = bool((index // 3 + index % 2) % 2)
    numeric = rng.normal(size=(rows, 10))
    category = rng.integers(0, 5, rows)
    signal = (numeric[:, 0] * numeric[:, 1] * 2 if interaction else
              numeric[:, 0] * 2 + numeric[:, 1])
    signal += (category - 2) * 1.2 if relevant else 0
    signal += rng.normal(0, .75, rows)
    # Classes are contiguous regions of a noisy score; 50% majority for 3-way
    # means remaining classes have 25% each.
    cuts = [prevalence] if classes == 2 else [prevalence, (1 + prevalence) / 2]
    labels = np.digitize(signal, np.quantile(signal, cuts)).astype(str)
    frame = pd.DataFrame(numeric, columns=[f"x{i}" for i in range(10)])
    frame[rng.random(frame.shape) < missing] = np.nan
    frame["category"] = [f"c{v}" for v in category]
    types = {name: "numeric" for name in frame.columns}
    types["category"] = "categorical"
    fitting, evaluation = train_test_split(np.arange(rows), test_size=.25,
                                           random_state=seed, stratify=labels)
    return (frame.iloc[fitting].reset_index(drop=True), labels[fitting],
            frame.iloc[evaluation].reset_index(drop=True), labels[evaluation], types,
            dict(classes=classes, majority=prevalence, missing=missing,
                 signal="interaction" if interaction else "linear", category_relevant=relevant))


def evaluate_case(index, seed, budget, supplied=None):
    train, labels, test, truth, types, definition = supplied or case_data(index, seed)
    plan = make_route_plan(summarize_training("classification", train, labels, types), budget)
    indices = np.arange(len(labels))
    folds = [train_test_split(indices, test_size=plan.selection_fraction,
              random_state=seed + fold, stratify=labels) for fold in range(2)]
    candidates = {}
    for family in FAMILIES:
        for weight in ("balanced", None):
            key = family + ":" + (weight or "unweighted")
            row = {"family": family, "weight": weight, "inner": []}
            for fit, validation in folds:
                started = time.perf_counter()
                prediction, _ = fit_predict_fixed_candidate(family, "classification", train.iloc[fit],
                    labels[fit], train.iloc[validation], types, seed, final=False,
                    route_plan=plan, class_weight=weight)
                row["inner"].append(dict(metrics=classification_metrics(labels[validation], prediction,
                    np.unique(labels)), seconds=time.perf_counter() - started))
            started = time.perf_counter()
            prediction, info = fit_predict_fixed_candidate(family, "classification", train, labels,
                test, types, seed, route_plan=plan, class_weight=weight)
            row.update(outer=classification_metrics(truth, prediction, np.unique(labels)),
                seconds=time.perf_counter() - started,
                prediction_sha256=hashlib.sha256(np.asarray(prediction).astype("U").tobytes()).hexdigest(),
                valid=bool(len(prediction) == len(test) and np.isin(prediction, labels).all()), fallback=info["fallback"])
            candidates[key] = row
    selections = []
    preference = {"linear": 0, "extra_trees": 1, "histogram": 2}
    baseline = candidates["extra_trees:balanced"]["outer"]
    for holdouts in (1, 2):
        for weighting in ("balanced", "unweighted", "both"):
            pool = [key for key in candidates if weighting == "both" or key.endswith(":" + weighting)]
            for objective in METRICS:
                # Only inner training-fold predictions influence this decision.
                selected, scores = select_offline(candidates, pool, objective, holdouts)
                quality = candidates[selected]["outer"]
                selections.append(dict(holdouts=holdouts, weighting=weighting, objective=objective,
                    selected=selected, validation_scores=scores, outer=quality,
                    gaps_to_best_evaluated={metric: max(candidates[k]["outer"][metric] for k in pool) - quality[metric] for metric in METRICS},
                    losses_vs_baseline={metric: baseline[metric] - quality[metric] for metric in METRICS}))
    inner_seconds = sum(f["seconds"] for row in candidates.values() for f in row["inner"])
    return dict(case=index, seed=seed, definition=definition, candidates=candidates, selections=selections,
                exhaustive_inner_seconds=inner_seconds,
                production_selection_cap_seconds=min(budget * plan.selection_budget_fraction, budget - plan.reserve_seconds))


def select_offline(candidates, pool, objective, holdouts):
    """Only training-fold metrics; outer scores are intentionally never read."""
    preference = {"linear": 0, "extra_trees": 1, "histogram": 2}
    scores = {key: float(np.mean([f["metrics"][objective] for f in candidates[key]["inner"][:holdouts]])) for key in pool}
    selected = min(pool, key=lambda key: (-scores[key], preference[candidates[key]["family"]], pool.index(key)))
    return selected, scores


def known_case(budget, model_seed=8081):
    from scripts.datasets import make_router_cases
    # Generate the inspected source tables in an isolated temporary directory;
    # no historical dataset or evaluator file is changed.
    workspace_temp = ROOT / "experiments/work"
    workspace_temp.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="classification-known-", dir=workspace_temp) as folder:
        cases = make_router_cases(Path(folder), 8081)
        case = next(case for case in cases if case["name"] == "missing_interactions")
        training = pd.read_csv(case["input"] / "train.csv", dtype={"target": str})
        test = pd.read_csv(case["input"] / "test.csv")
        labels = training.pop("target").to_numpy()
        data = (training, labels, test, np.asarray(case["reference"], dtype=str),
                {name: "numeric" for name in training.columns},
                {"name": "missing_interactions8081", "status": "previously inspected development evidence"})
        return evaluate_case(-1, model_seed, budget, supplied=data)


def summarize(results):
    summary = []
    for holdouts in (1, 2):
        for weighting in ("balanced", "unweighted", "both"):
            for objective in METRICS:
                rows = [row for result in results for row in result["selections"] if
                        (row["holdouts"], row["weighting"], row["objective"]) == (holdouts, weighting, objective)]
                summary.append(dict(holdouts=holdouts, weighting=weighting, objective=objective,
                    selection_frequency=dict(Counter(row["selected"] for row in rows)),
                    mean={metric: float(np.mean([r["outer"][metric] for r in rows])) for metric in METRICS},
                    mean_gap={metric: float(np.mean([r["gaps_to_best_evaluated"][metric] for r in rows])) for metric in METRICS},
                    worst_loss_vs_baseline={metric: float(max(r["losses_vs_baseline"][metric] for r in rows)) for metric in METRICS}))
    return summary


def stable_result(result):
    return {"candidates": {k: {"inner": [f["metrics"] for f in v["inner"]], "outer": v["outer"],
        "prediction_sha256": v["prediction_sha256"]} for k, v in result["candidates"].items()},
        "selections": result["selections"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=317)
    parser.add_argument("--cases", type=int, default=12)
    parser.add_argument("--repeat-cases", type=int, default=2)
    parser.add_argument("--budget", type=float, default=50)
    parser.add_argument("--output", type=Path, default=ROOT / "experiments/first-batch/classification.json")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output exists; choose a new path to preserve historical evidence")
    if not 1 <= args.cases <= 12 or not 0 <= args.repeat_cases <= args.cases:
        parser.error("cases must be 1..12; repeat cases must be 0..cases")
    started = time.perf_counter()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    recovery = args.output.with_name(args.output.stem + "-partial.jsonl")
    if recovery.exists():
        parser.error("Recovery artifact exists; choose a new output path")
    results = []
    for index in range(args.cases):
        results.append(evaluate_case(index, args.seed, args.budget))
        with recovery.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(results[-1], allow_nan=False) + "\n")
        print(f"case {index + 1}/{args.cases} completed", flush=True)
    repeats = []
    for index in range(args.repeat_cases):
        repeat = evaluate_case(index, args.seed, args.budget)
        repeats.append(dict(case=index, exact=stable_result(repeat) == stable_result(results[index])))
    known = known_case(args.budget)
    payload = dict(scope="development; offline exhaustive factorial, not 60-second CLI or confirmation evidence",
        seed=args.seed, source_sha256=hashlib.sha256((ROOT / "src/unknown_dataset/supervised.py").read_bytes()).hexdigest(),
        source_hashes={str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                       for path in sorted((ROOT / "src/unknown_dataset").glob("*.py"))},
        experiment_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        cases=results, known_inspected_case=known, summary=summarize(results), repeats=repeats, seconds=time.perf_counter() - started)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(dict(output=str(args.output), seconds=payload["seconds"], repeats=repeats)), flush=True)

if __name__ == "__main__":
    main()
