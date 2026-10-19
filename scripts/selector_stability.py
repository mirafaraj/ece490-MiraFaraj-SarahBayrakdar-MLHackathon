"""Offline repeated outer-split diagnostics. Never imported by production code.

Outer references enter only scoring. Retrospective gaps are diagnostic and never
influence production selection. Repeated splits overlap and are not new datasets.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import time

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.metrics import accuracy_score, balanced_accuracy_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
from scripts.datasets import make_broad_cases, make_router_cases
from scripts.evaluate import source_fingerprint, fallback_kind


def quality(task, metric, truth, predictions):
    if task == "classification":
        fn = accuracy_score if metric == "accuracy" else balanced_accuracy_score
        return float(fn(truth, predictions))
    if metric == "mae":
        return -float(mean_absolute_error(truth, predictions))
    return -float(np.sqrt(mean_squared_error(truth, predictions)))


def outer_data(case, split_seed):
    """Recombine harness-held references, then split before calling any selector."""
    meta = json.loads((case["input"] / "metadata.json").read_text())
    dtype = {c: str for c, t in meta["feature_types"].items() if t == "categorical"}
    if case["task"] == "classification":
        dtype["target"] = str
    train = pd.read_csv(case["input"] / "train.csv", dtype=dtype)
    features = pd.concat([train.drop(columns="target"), pd.read_csv(case["input"] / "test.csv", dtype=dtype)], ignore_index=True)
    target = np.concatenate([train["target"].to_numpy(), np.asarray(case["reference"], dtype=str if case["task"] == "classification" else float)])
    fitting, evaluation = train_test_split(np.arange(len(features)), test_size=.25, random_state=split_seed,
        stratify=target if case["task"] == "classification" else None)
    return features.iloc[fitting].reset_index(drop=True), target[fitting], features.iloc[evaluation].reset_index(drop=True), target[evaluation], meta["feature_types"]


def evaluate_split(case, split_seed, config, budget, metrics, explore_tie=False):
    from unknown_dataset.models import fit_predict
    from unknown_dataset.routing import summarize_training, make_route_plan
    from unknown_dataset.supervised import fit_predict_fixed_candidate, fit_predict_supervised

    task = case["task"]
    metric = metrics[task]
    train, labels, test, truth, types = outer_data(case, split_seed)
    input_hash = hashlib.sha256(pd.util.hash_pandas_object(pd.concat([train, test]), index=True).to_numpy().tobytes() + np.asarray(labels).astype(str).tobytes()).hexdigest()
    active = deepcopy(config)
    active["metrics"][task] = metric
    categorical = deepcopy(active)
    categorical["experimental"]["categorical_enabled"] = True
    categorical["limits"]["max_candidates"] = 4
    tied = deepcopy(active)
    tie_tolerance = .01 if task == "classification" else .01 * float(np.std(labels.astype(float)))
    tied["selection"]["near_tie_tolerance"] = tie_tolerance
    plan = make_route_plan(summarize_training(task, train, labels, types), budget, config=active)
    unrestricted = deepcopy(plan)
    unrestricted.candidates = ["extra_trees", "linear", "histogram"]
    unrestricted.excluded = {}
    rows = []
    strategies = ["dummy", "extra_trees", "linear", "histogram", "baseline", "auto", "routed", "unrestricted"]
    if task == "classification" and any(t == "categorical" for t in types.values()):
        strategies += ["categorical", "routed_categorical"]
    if explore_tie:
        strategies.append("routed_tie")
    for strategy in strategies:
        started = time.perf_counter()
        row = {"name": case["name"], "task": task, "split_seed": split_seed, "strategy": strategy,
               "metric": metric, "data_sha256": input_hash, "valid": False, "error": None, "score": None}
        try:
            if strategy == "dummy":
                model = DummyClassifier(strategy="prior") if task == "classification" else DummyRegressor(strategy="mean")
                model.fit(np.zeros((len(train), 1)), labels)
                predictions, info = model.predict(np.zeros((len(test), 1))), {"selected_candidate": "dummy"}
            elif strategy in {"extra_trees", "linear", "histogram", "categorical"}:
                predictions, info = fit_predict_fixed_candidate(strategy, task, train, labels, test, types,
                    seed=split_seed, metric=metric, route_plan=plan)
            elif strategy == "unrestricted":
                predictions, info = fit_predict_supervised(task, train, labels, test, types,
                    seed=split_seed, time_budget=budget, route_plan=unrestricted)
            else:
                predictions, info = fit_predict(task, train, labels, test, types, seed=split_seed,
                    time_budget=budget, strategy="routed" if strategy in {"routed_categorical", "routed_tie"} else strategy,
                    route_config=categorical if strategy == "routed_categorical" else tied if strategy == "routed_tie" else active)
            if len(predictions) != len(truth) or pd.isna(predictions).any():
                raise ValueError("Invalid prediction count or missing predictions")
            if task == "classification" and not np.isin(predictions, np.unique(labels)).all():
                raise ValueError("Unknown class")
            if task == "regression" and not np.isfinite(np.asarray(predictions, dtype=float)).all():
                raise ValueError("Nonfinite regression prediction")
            row.update(valid=True, score=quality(task, metric, truth, predictions), run_info={"model": info},
                selected_candidate=info.get("selected_candidate"), validation_scores=info.get("validation_scores", {}),
                candidate_errors=info.get("candidate_errors", {}), fallback=bool(info.get("fallback", False)))
            row["near_tie_tolerance"] = tie_tolerance if strategy == "routed_tie" else active["selection"]["near_tie_tolerance"]
            row["fallback_kind"] = fallback_kind(row)
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
        row["wall_seconds"] = time.perf_counter() - started
        # Direct-call timings exclude CLI imports/process startup and lack hard interruption.
        row["timing_scope"] = "offline direct fit/select/refit/predict, not full CLI or enforced deadline"
        row["budget_exceeded"] = row["wall_seconds"] > budget
        rows.append(row)
    return rows


def diagnostics(rows):
    groups = {}
    for row in rows:
        groups.setdefault((row["name"], row["split_seed"]), {})[row["strategy"]] = row
    gaps = []
    for (name, seed), group in groups.items():
        expected = {"dummy", "extra_trees", "linear", "histogram"}
        if "categorical" in group:
            expected.add("categorical")
        evaluated = [group[n]["score"] for n in expected if n in group and group[n]["valid"]]
        complete = len(evaluated) == len(expected)
        for strategy in ("auto", "routed", "unrestricted", "routed_categorical", "routed_tie"):
            if strategy not in group:
                continue
            row, base = group[strategy], group["baseline"]
            delta = row["score"] - base["score"] if row["valid"] and base["valid"] else None
            best = max([*evaluated, row["score"]]) if row["valid"] and evaluated else None
            raw_validation = row.get("validation_scores", {}).get(row.get("selected_candidate"))
            if raw_validation is not None:
                validation_quality = float(raw_validation)
                # Historical auto stores negative regression loss; routed stores natural loss.
                if row["task"] == "regression" and strategy != "auto":
                    validation_quality = -validation_quality
            else:
                validation_quality = None
            gaps.append({"name": name, "split_seed": seed, "task": row["task"], "strategy": strategy,
                "baseline_delta": delta, "outcome": "invalid" if delta is None else "tie" if abs(delta) <= 1e-9 else "win" if delta > 0 else "loss",
                "best_evaluated_gap": best - row["score"] if best is not None else None,
                "comparison_complete": complete, "missing_or_failed_candidates": [n for n in expected if n not in group or not group[n]["valid"]],
                "selected_candidate": row.get("selected_candidate"), "fallback_kind": row.get("fallback_kind"),
                "validation_quality": validation_quality,
                "validation_minus_outer_quality": validation_quality - row["score"] if validation_quality is not None and row["valid"] else None,
                "valid": row["valid"], "wall_seconds": row["wall_seconds"], "budget_exceeded": row["budget_exceeded"]})
    frequency = {}
    for strategy in ("auto", "routed", "unrestricted", "routed_categorical", "routed_tie"):
        frequency[strategy] = dict(Counter(r.get("selected_candidate") or "unknown" for r in rows if r["strategy"] == strategy and r["valid"]))
    return {"selection_frequencies": frequency, "per_split": gaps,
        "score_convention": "higher is better; regression scores are negative RMSE/MAE; retrospective best includes dummy and evaluated candidate families",
        "losses": [r for r in gaps if r["outcome"] == "loss"],
        "candidate_failures": [{k: r[k] for k in ("name", "split_seed", "strategy", "error")} for r in rows if not r["valid"]]}


def main():
    from unknown_dataset.routing import load_router_config
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "experiments/selector_stability.json")
    parser.add_argument("--router-config", type=Path)
    parser.add_argument("--split-seeds", type=int, nargs="+", default=[42, 314])
    parser.add_argument("--cases", nargs="+")
    parser.add_argument("--classification-metric", choices=["accuracy", "balanced_accuracy"], default="balanced_accuracy")
    parser.add_argument("--regression-metric", choices=["rmse", "mae"], default="rmse")
    parser.add_argument("--time-budget", type=float, default=120)
    parser.add_argument("--explore-tie", action="store_true", help="Development only: tolerance .01 accuracy points or .01 training-target std for regression")
    args = parser.parse_args()
    if 8081 in args.split_seeds:
        parser.error("8081 reserved for parent-controlled frozen CLI confirmation")
    if args.time_budget <= 0:
        parser.error("Positive budget required")
    config = load_router_config(args.router_config)
    report = {"notice": "Offline development only. Overlapping outer splits are not independent datasets. Retrospective best-evaluated gap is not a production oracle. Direct-call time is not CLI/deadline compliance.",
        "config": {"source_sha256": source_fingerprint(args.router_config), "router_config": config,
            "split_seeds": args.split_seeds, "data_seed": 42, "near_tie_experiment": {"enabled": args.explore_tie,
                "classification": .01, "regression": "0.01 times outer-training target std", "promotion": "zero default retained"}, "time_budget": args.time_budget}, "runs": []}
    with tempfile.TemporaryDirectory(dir=ROOT / "experiments") as directory:
        path = Path(directory)
        cases = [c for c in make_router_cases(path / "router", 42) if c["task"] != "clustering"]
        cases += [c for c in make_broad_cases(path / "broad", 42) if c["name"] in {"high_cardinality", "mixed_missing"}]
        if args.cases:
            cases = [c for c in cases if c["name"] in args.cases]
        if not cases:
            parser.error("No matching cases")
        for case in cases:
            for seed in args.split_seeds:
                rows = evaluate_split(case, seed, config, args.time_budget,
                    {"classification": args.classification_metric, "regression": args.regression_metric}, args.explore_tie)
                report["runs"].extend(rows)
                print(f"{case['name']} split={seed}: {len(rows)} strategies, {sum(r['valid'] for r in rows)} valid", flush=True)
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    report["diagnostics"] = diagnostics(report["runs"])
    report["config"]["source_sha256_after"] = source_fingerprint(args.router_config)
    report["config"]["source_unchanged"] = report["config"]["source_sha256"] == report["config"]["source_sha256_after"]
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    return 0 if report["config"]["source_unchanged"] and all(r["valid"] for r in report["runs"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
