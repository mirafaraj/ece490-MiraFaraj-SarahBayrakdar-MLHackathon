"""Bounded development replication; reference labels are reporting-only.

The exhaustive study fits each balanced candidate once per fold and final fit,
then reuses predictions for every metric. It is not 60-second CLI evidence.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
from scripts.batch_classification import case_data, FAMILIES, METRICS
from unknown_dataset.routing import make_route_plan, summarize_training
from unknown_dataset.supervised import (ClassificationControls, classification_metrics,
    fit_predict_fixed_candidate, fit_predict_supervised)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                    separators=(",", ":")).encode()).hexdigest()


def support(labels, classes):
    return [int(np.sum(np.asarray(labels) == label)) for label in classes]


def raw_metrics(truth, prediction, classes):
    """Independent confusion-matrix arithmetic, with sklearn BA support semantics."""
    lookup = {label: index for index, label in enumerate(classes)}
    matrix = np.zeros((len(classes), len(classes)), dtype=int)
    for target, predicted in zip(truth, prediction, strict=True):
        matrix[lookup[target], lookup[predicted]] += 1
    counts = matrix.sum(axis=1)
    diagonal = matrix.diagonal()
    recall = np.divide(diagonal, counts, out=np.zeros(len(classes)), where=counts > 0)
    denominator = counts + matrix.sum(axis=0)
    f1 = np.divide(2 * diagonal, denominator, out=np.zeros(len(classes)), where=denominator > 0)
    return dict(accuracy=float(diagonal.sum() / matrix.sum()),
                balanced_accuracy=float(recall[counts > 0].mean()),
                macro_f1=float(f1.mean()), class_recall=recall.tolist(), classes=list(classes))


def record(truth, prediction, classes):
    metrics = classification_metrics(truth, prediction, classes)
    verified = raw_metrics(truth, prediction, classes)
    for key in METRICS + ("class_recall",):
        if not np.allclose(metrics[key], verified[key], atol=1e-12, rtol=0):
            raise AssertionError(f"Independent metric mismatch: {key}")
    return dict(truth=np.asarray(truth).tolist(), prediction=np.asarray(prediction).tolist(),
                metrics=metrics, independent_metrics=verified, prediction_sha256=digest(np.asarray(prediction).tolist()))


def choose(training_records, objective, holdouts, cap=None):
    """Reads inner records only; each policy shares one total selection cap.

    Timing replay is descriptive, not a replacement for actual bounded calls.
    Missing/incomplete candidates cannot enter the comparison.
    """
    scores, excluded = {}, {}
    elapsed = 0.0
    for name in FAMILIES:
        folds = training_records[name]
        if len(folds) < holdouts:
            excluded[name] = "incomplete folds"
            continue
        complete = True
        for fold in folds[:holdouts]:
            if cap is not None and elapsed >= cap:
                complete = False
                break
            elapsed += fold["seconds"]
        if complete:
            scores[name] = float(np.mean([fold["metrics"][objective] for fold in folds[:holdouts]]))
        else:
            excluded[name] = "shared selection cap exhausted"
    preference = {"linear": 0, "extra_trees": 1, "histogram": 2}
    selected = min(scores, key=lambda name: (-scores[name], preference[name])) if scores else "extra_trees"
    return dict(selected=selected, validation_scores=scores, excluded_candidates=excluded,
                replay_seconds=elapsed, cap_seconds=cap)


def rare_data(seed, mechanism):
    """Separate balanced-tail interaction mechanisms, all 95% majority."""
    rng = np.random.default_rng(seed + 9001)
    numeric = rng.normal(size=(900, 10))
    if mechanism == "xor_tail":
        signal = numeric[:, 0] * numeric[:, 1] * 2 + rng.normal(0, .3, len(numeric))
    elif mechanism == "localized_tail":
        signal = -(numeric[:, 0] - 1.5) ** 2 - (numeric[:, 1] + 1.5) ** 2
        signal += rng.normal(0, .3, len(numeric))
    else:
        raise ValueError(mechanism)
    labels = np.digitize(signal, np.quantile(signal, [.95, .975])).astype(str)
    frame = pd.DataFrame(numeric, columns=[f"x{i}" for i in range(10)])
    types = {name: "numeric" for name in frame}
    fitting, evaluation = train_test_split(np.arange(len(labels)), test_size=.25,
                                           random_state=seed, stratify=labels)
    return (frame.iloc[fitting].reset_index(drop=True), labels[fitting],
            frame.iloc[evaluation].reset_index(drop=True), labels[evaluation], types,
            dict(classes=3, majority=.95, mechanism=mechanism))


def evaluate(data, seed, identity, budget=50):
    train, labels, test, truth, types, definition = data
    classes = np.unique(labels).tolist()
    plan = make_route_plan(summarize_training("classification", train, labels, types), budget)
    folds = [train_test_split(np.arange(len(labels)), test_size=plan.selection_fraction,
                             random_state=seed + fold, stratify=labels) for fold in range(2)]
    split_records = [dict(fitting=fit.tolist(), validation=validation.tolist(),
                         fitting_support=support(labels[fit], classes),
                         validation_support=support(labels[validation], classes)) for fit, validation in folds]
    # Nullable serialization gives platform-independent, complete input hashes.
    inputs = dict(train=json.loads(train.to_json(orient="split", double_precision=15)),
                  labels=labels.tolist(), test=json.loads(test.to_json(orient="split", double_precision=15)),
                  feature_types=types)
    candidates = {}
    for family in FAMILIES:
        row = dict(inner=[])
        for fit, validation in folds:
            started = time.perf_counter()
            prediction, info = fit_predict_fixed_candidate(family, "classification", train.iloc[fit],
                labels[fit], train.iloc[validation], types, seed, final=False,
                route_plan=plan, class_weight="balanced")
            seconds = time.perf_counter() - started
            row["inner"].append(dict(**record(labels[validation], prediction, classes), seconds=seconds))
        started = time.perf_counter()
        prediction, info = fit_predict_fixed_candidate(family, "classification", train, labels,
            test, types, seed, route_plan=plan, class_weight="balanced")
        row.update(outer=record(truth, prediction, classes), seconds=time.perf_counter() - started,
                   valid=bool(len(prediction) == len(test) and np.isin(prediction, labels).all()),
                   fallback=info["fallback"])
        candidates[family] = row
    training_records = {name: row["inner"] for name, row in candidates.items()}
    cap = min(budget * plan.selection_budget_fraction, budget - plan.reserve_seconds)
    selections = []
    minority = int(np.argmin(support(labels, classes)))
    for holdouts in (1, 2):
        for objective in METRICS:
            decision = choose(training_records, objective, holdouts, cap)
            quality = candidates[decision["selected"]]["outer"]["metrics"]
            selections.append(dict(**decision, holdouts=holdouts, objective=objective,
                outer=quality, minority_recall=quality["class_recall"][minority],
                gap={metric: max(row["outer"]["metrics"][metric] for row in candidates.values()) - quality[metric] for metric in METRICS},
                loss_vs_baseline={metric: candidates["extra_trees"]["outer"]["metrics"][metric] - quality[metric] for metric in METRICS}))
    return dict(identity=identity, seed=seed, definition=definition, inputs=inputs,
        input_sha256=digest(inputs), splits=split_records, split_sha256=digest(split_records),
        classes=classes, train_support=support(labels, classes), outer_support=support(truth, classes),
        candidates=candidates, selections=selections, selection_cap_seconds=cap)


def summarize(rows):
    result = []
    for holdouts in (1, 2):
        for objective in METRICS:
            selected = [s for row in rows for s in row["selections"]
                        if s["holdouts"] == holdouts and s["objective"] == objective]
            result.append(dict(holdouts=holdouts, objective=objective,
                selection_frequency=dict(Counter(s["selected"] for s in selected)),
                exclusions=sum(len(s["excluded_candidates"]) for s in selected),
                mean={m: float(np.mean([s["outer"][m] for s in selected])) for m in METRICS},
                mean_minority_recall=float(np.mean([s["minority_recall"] for s in selected])),
                mean_gap={m: float(np.mean([s["gap"][m] for s in selected])) for m in METRICS},
                worst_loss={m: float(max(s["loss_vs_baseline"][m] for s in selected)) for m in METRICS}))
    return result


def bounded_probe(seed, budget, holdouts, stress_cap):
    train, labels, test, truth, types, _ = case_data(10, seed)
    plan = make_route_plan(summarize_training("classification", train, labels, types), budget)
    if stress_cap:
        # Deliberate exhaustion probe; not a proposed deployment policy.
        plan = replace(plan, selection_budget_fraction=.02)
    started = time.perf_counter()
    prediction, info = fit_predict_supervised("classification", train, labels, test, types,
        time_budget=budget, seed=seed, route_plan=plan,
        classification_controls=ClassificationControls("balanced_accuracy", "balanced", holdouts))
    return dict(budget=budget, stress_cap=stress_cap, holdouts=holdouts, seconds=time.perf_counter() - started,
        valid=bool(len(prediction) == len(test) and np.isin(prediction, labels).all()),
        fallback=bool(info.get("fallback") or info.get("refit_fallback_reason")),
        result=record(truth, prediction, np.unique(labels).tolist()), info=info)


def stable(row):
    return dict(input_sha256=row["input_sha256"], split_sha256=row["split_sha256"],
        candidates={name: dict(inner=[f["prediction_sha256"] for f in value["inner"]],
                              outer=value["outer"]["prediction_sha256"]) for name, value in row["candidates"].items()},
        selections=[(s["holdouts"], s["objective"], s["selected"], s["validation_scores"]) for s in row["selections"]])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists; preserve historical evidence")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    seeds = (419, 733, 1201)
    source_hashes = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in [Path(__file__), ROOT / "scripts/batch_classification.py",
                     ROOT / "src/unknown_dataset/supervised.py"]}
    protocol_snapshot = (ROOT / "reports/second-batch/classification.md").read_text(encoding="utf-8")
    # Reuse completed identities after an interrupted run. A protocol/source
    # mismatch creates a distinct cache; historical evidence is never overwritten.
    scratch = ROOT / "experiments/work"
    scratch.mkdir(parents=True, exist_ok=True)
    cache_key = digest(dict(output=str(args.output.resolve()), hashes=source_hashes,
                            protocol=protocol_snapshot))[:16]
    recovery = scratch / f"{args.output.stem}-{cache_key}.jsonl"
    cached = {}
    if recovery.exists():
        for line in recovery.read_text(encoding="utf-8").splitlines():
            item = json.loads(line)
            cached[item["identity"]] = item["record"]
    def saved(identity, operation):
        if identity not in cached:
            result = operation()
            with recovery.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(dict(identity=identity, record=result), allow_nan=False,
                                        separators=(",", ":")) + "\n")
                handle.flush()
            cached[identity] = result
        print(identity, flush=True)
        return cached[identity]
    rows = []
    for seed in seeds:
        for index in range(12):
            identity = f"replication-{seed}-{index}"
            rows.append(saved(identity, lambda: evaluate(case_data(index, seed), seed, identity)))
    rare = [saved(f"{mechanism}-{seed}", lambda: evaluate(rare_data(seed, mechanism), seed, f"{mechanism}-{seed}"))
            for seed in seeds for mechanism in ("xor_tail", "localized_tail")]
    known = saved("inspected-case10-reproduction", lambda: evaluate(case_data(10, 317), 317, "inspected-case10-reproduction"))
    repeated = saved("repeat-419-10", lambda: evaluate(case_data(10, 419), 419, "repeat-419-10"))
    probes = [saved(f"bounded-{budget}-{stress}-{holdouts}", lambda: bounded_probe(419, budget, holdouts, stress))
              for budget, stress in ((20, False), (50, False), (50, True)) for holdouts in (1, 2)]
    payload = dict(scope="development exhaustive fixed balanced weights; no CLI/confirmation claim",
        seeds=seeds, cases=rows, rare=rare, inspected_reproduction=known,
        repeat=dict(exact=stable(repeated) == stable(rows[10]), record=repeated),
        summary=summarize(rows), rare_summary=summarize(rare), bounded_probes=probes,
        source_hashes=source_hashes,
        protocol_snapshot=protocol_snapshot, protocol_sha256=digest(protocol_snapshot),
        seconds=time.perf_counter() - started)
    args.output.write_text(json.dumps(payload, separators=(",", ":"), allow_nan=False), encoding="utf-8")
    print(json.dumps(dict(output=str(args.output), seconds=payload["seconds"], repeat=payload["repeat"]["exact"])))


if __name__ == "__main__":
    main()
