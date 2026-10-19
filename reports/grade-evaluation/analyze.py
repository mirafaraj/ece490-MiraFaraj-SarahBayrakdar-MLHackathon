"""Read saved paired CLI reports only; never import production/ML code or run fits.

Usage: python reports/grade-evaluation/analyze.py experiments/broad-comparison.json
       --summary experiments/comparison_summary.json --output reports/grade-evaluation/scorecard.json
No official point estimates are calculated. Repeated seeds are not independent families.
"""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, median
from copy import deepcopy
from promotion import evaluate
from prediction_evidence import enrich
from reference_evidence import pair_reference


METRICS = {"classification": "balanced_accuracy", "regression": "rmse", "clustering": "adjusted_rand_index"}


def outcome(value):
    return "tie" if abs(value) <= 1e-9 else "win" if value > 0 else "loss"


def analyze(report):
    groups = defaultdict(dict)
    for row in report["datasets"]:
        key = (row["seed"], row["task"], row["name"])
        if row["strategy"] in groups[key]:
            raise ValueError(f"Duplicate strategy for {key}")
        groups[key][row["strategy"]] = row
    pairs = []
    for (seed, task, name), strategies in groups.items():
        if set(strategies) != {"baseline", "auto"}:
            raise ValueError(f"Require exactly baseline and auto for {seed}/{task}/{name}")
        before, after = strategies["baseline"], strategies["auto"]
        if before["data_sha256"] != after["data_sha256"]:
            raise ValueError(f"Mismatched input hash: {seed}/{name}")
        if before["partition"] != after["partition"]:
            raise ValueError(f"Mismatched partition: {seed}/{name}")
        metric = METRICS[task]
        valid = bool(before["valid"] and after["valid"])
        record = dict(seed=seed, task=task, name=name, partition=before["partition"], valid_pair=valid,
                      metric=metric, baseline=before["metrics"].get(metric), auto=after["metrics"].get(metric),
                      baseline_seconds=before["wall_seconds"], auto_seconds=after["wall_seconds"],
                      auto_fallback=bool(after.get("fallback")),
                      auto_model=after.get("run_info", {}).get("model", {}).get("model"))
        if valid:
            delta = record["auto"] - record["baseline"]
            if task == "regression":
                delta = -delta
            record.update(quality_change=delta, outcome=outcome(delta))
            record["relative_rmse_reduction"] = 1 - record["auto"] / record["baseline"] if task == "regression" and record["baseline"] else None
            if task == "classification":
                record.update(baseline_accuracy=before["metrics"]["accuracy"], auto_accuracy=after["metrics"]["accuracy"])
                record["accuracy_change"] = record["auto_accuracy"] - record["baseline_accuracy"]
                record["accuracy_outcome"] = outcome(record["accuracy_change"])
                record["metric_reversal"] = delta * record["accuracy_change"] < -1e-12
        else:
            record.update(quality_change=None, relative_rmse_reduction=None, outcome="invalid")
        pairs.append(record)
    aggregates = []
    for partition, task in sorted({(p["partition"], p["task"]) for p in pairs}):
        rows = [p for p in pairs if (p["partition"], p["task"]) == (partition, task)]
        valid = [p for p in rows if p["valid_pair"]]
        aggregates.append(dict(partition=partition, task=task, cases=len(rows),
            outcomes={label: sum(p["outcome"] == label for p in rows) for label in ("win", "tie", "loss", "invalid")},
            mean_quality_change=mean(p["quality_change"] for p in valid) if valid and task != "regression" else None,
            mean_relative_rmse_reduction=mean(p["relative_rmse_reduction"] for p in valid) if valid and task == "regression" else None,
            mean_baseline_seconds=mean(p["baseline_seconds"] for p in rows), mean_auto_seconds=mean(p["auto_seconds"] for p in rows)))
    families = []
    for task, name in sorted({(p["task"], p["name"]) for p in pairs}):
        rows = [p for p in pairs if p["task"] == task and p["name"] == name and p["valid_pair"]]
        if not rows:
            continue
        families.append(dict(task=task, name=name, cases=len(rows), mean_baseline=mean(p["baseline"] for p in rows),
            mean_auto=mean(p["auto"] for p in rows),
            mean_relative_rmse_reduction=mean(p["relative_rmse_reduction"] for p in rows) if task == "regression" else None))
    runs = report["datasets"]
    classification = [p for p in pairs if p["task"] == "classification" and p["valid_pair"]]
    regression = [p for p in pairs if p["task"] == "regression" and p["valid_pair"]]
    return dict(notice="Development evidence only; no official grade estimate. Previously exposed confirmation seeds are development for future tuning.",
        source_sha256=report["config"]["source_sha256"], source_unchanged=report["config"].get("source_unchanged"),
        runs=len(runs), valid_runs=sum(bool(r["valid"]) for r in runs), fallback_runs=sum(bool(r.get("fallback")) for r in runs),
        wall_seconds=dict(min=min(r["wall_seconds"] for r in runs), max=max(r["wall_seconds"] for r in runs), median=median(r["wall_seconds"] for r in runs)),
        observed_shapes=dict(max_training_rows=max(r.get("run_info", {}).get("model", {}).get("training_rows", 0) for r in runs),
                             max_input_features=max(r.get("run_info", {}).get("model", {}).get("input_features", 0) for r in runs)),
        classification=dict(mean_accuracy_change=mean(p["accuracy_change"] for p in classification) if classification else None,
            mean_balanced_accuracy_change=mean(p["quality_change"] for p in classification) if classification else None,
            metric_reversals=[dict(seed=p["seed"], name=p["name"], accuracy_change=p["accuracy_change"], balanced_accuracy_change=p["quality_change"]) for p in classification if p["metric_reversal"]]),
        regression=dict(mean_relative_rmse_reduction=mean(p["relative_rmse_reduction"] for p in regression) if regression else None,
            excluding_linear_mean_relative_rmse_reduction=mean(p["relative_rmse_reduction"] for p in regression if p["name"] != "linear") if any(p["name"] != "linear" for p in regression) else None),
        aggregates=aggregates, families=families, pairs=pairs)


def validate_summary(result, summary):
    if result["source_sha256"] != summary["source_sha256"]:
        raise ValueError("Summary source hash mismatch")
    expected = {(r["partition"], r["task"]): r for r in summary["aggregates"]}
    actual = {(r["partition"], r["task"]): r for r in result["aggregates"]}
    if set(expected) != set(actual):
        raise ValueError("Summary partition/task mismatch")
    for key, row in actual.items():
        for field, value in row.items():
            prior = expected[key][field]
            if isinstance(value, float):
                equal = isinstance(prior, (int, float)) and math.isclose(value, prior, rel_tol=1e-10, abs_tol=1e-12)
            else:
                equal = value == prior
            if not equal:
                raise ValueError(f"Summary mismatch at {key}/{field}: {value!r} != {prior!r}")
    return "All aggregate fields independently reproduced from raw paired rows."


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate-strategy", default="auto")
    parser.add_argument("--baseline-strategy", default="baseline")
    parser.add_argument("--reference-report", type=Path, help="Saved frozen incumbent report on exactly the same inputs")
    parser.add_argument("--reference-run-strategy", help="Original strategy in reference report; baseline-strategy can be an alias such as incumbent")
    parser.add_argument("--classification-objective", choices=["accuracy", "balanced_accuracy"], help="Deprecated diagnostic label; does not select the promotion objective")
    parser.add_argument("--accuracy-loss-budget", type=float, help="Deprecated diagnostic metadata; never a promotion constraint")
    parser.add_argument("--balanced-accuracy-loss-budget", type=float, help="Deprecated diagnostic metadata; never a promotion constraint")
    parser.add_argument("--objective-policy", type=Path, help="Written official objective or declared provisional metric scenarios")
    parser.add_argument("--registry", type=Path, help="Read-only task incumbent/frontier registry")
    parser.add_argument("--manifest", type=Path, help="Frozen comparison pool and expected case inventory")
    parser.add_argument("--prediction-evidence", type=Path, help="Hash-bound evaluator-only truth/prediction evidence")
    parser.add_argument("--bootstrap-iterations", type=int, default=10000)
    parser.add_argument("--overwrite", action="store_true", help="Explicitly allow replacing an evaluator output")
    parser.add_argument("--enforce-evidence", action="store_true", help="Nonzero exit for invalid/incomplete evidence, independent of quality metrics")
    parser.add_argument("--enforce-accuracy-gate", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.overwrite:
        parser.error("Output already exists; choose a new filename to preserve historical evidence")
    inputs = [args.report, args.reference_report, args.summary, args.objective_policy, args.registry, args.manifest, args.prediction_evidence]
    if any(path and path.resolve() == args.output.resolve() for path in inputs):
        parser.error("Output must not replace an input artifact")
    source = args.report.read_bytes()
    if args.candidate_strategy == args.baseline_strategy:
        parser.error("Candidate strategy must differ from reference")
    if any(v is not None and (not math.isfinite(v) or v < 0) for v in (args.accuracy_loss_budget,args.balanced_accuracy_loss_budget)):
        parser.error("Loss budgets must be finite and nonnegative")
    if args.summary and (args.candidate_strategy != "auto" or args.baseline_strategy != "baseline"):
        parser.error("--summary validates only the legacy baseline/auto aggregate schema")
    parse_error = None
    try:
        report = json.loads(source)
    except (ValueError, UnicodeError) as exc:
        report = {}
        parse_error = str(exc)
    def optional_json(path):
        if path is None:
            return None
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError(f"Expected an object in {path}")
        return value
    try:
        policy, registry, manifest = [optional_json(p) for p in (args.objective_policy,args.registry,args.manifest)]
        if manifest and manifest.get("report_sha256") and manifest["report_sha256"] != hashlib.sha256(source).hexdigest():
            raise ValueError("Manifest report hash mismatch")
        if args.prediction_evidence:
            report = enrich(report,optional_json(args.prediction_evidence),hashlib.sha256(source).hexdigest())
        if args.reference_report:
            reference_source=args.reference_report.read_bytes()
            report=pair_reference(report,optional_json(args.reference_report),hashlib.sha256(reference_source).hexdigest(),
                                  args.baseline_strategy,args.reference_run_strategy or args.baseline_strategy,args.candidate_strategy)
        gate = evaluate(report, candidate=args.candidate_strategy, baseline=args.baseline_strategy,
                        classification_objective=args.classification_objective,
                        secondary_loss_budgets={k:v for k,v in {"accuracy":args.accuracy_loss_budget,"balanced_accuracy":args.balanced_accuracy_loss_budget}.items() if v is not None} or None,
                        policy=policy,registry=registry,manifest=manifest,bootstrap_iterations=args.bootstrap_iterations)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    if parse_error:
        gate["evidence_gaps"].append("Report JSON could not be parsed: " + parse_error)
    normalized = deepcopy(report) if isinstance(report, dict) else {}
    source_rows = normalized.get("datasets", [])
    normalized["datasets"] = [r for r in source_rows if isinstance(r, dict) and r.get("strategy") in {args.baseline_strategy, args.candidate_strategy}] if isinstance(source_rows, list) else []
    for row in normalized["datasets"]:
        if row["strategy"] == args.candidate_strategy:
            row["strategy"] = "auto"
        else:
            row["strategy"] = "baseline"
    result = {}
    try:
        result = analyze(normalized)
        # Legacy `auto` aggregate fields denote the requested candidate strategy.
        json.dumps(result, allow_nan=False)
    except (AttributeError, KeyError, ValueError, TypeError, ZeroDivisionError, OverflowError) as exc:
        result = {"legacy_analysis_status": "unavailable", "legacy_analysis_reason": str(exc)}
    result["candidate_strategy"] = args.candidate_strategy
    result["baseline_strategy"] = args.baseline_strategy
    result["legacy_analysis_notice"] = "Historical aggregate fields use balanced accuracy/RMSE/ARI; they are not official scores or promotion objectives. Use gate_assessment for current policy."
    result["gate_assessment"] = gate
    result["report_sha256"] = hashlib.sha256(source).hexdigest()
    result["evaluation_inputs"] = {name:{"path":str(path.resolve()),"sha256":hashlib.sha256(path.read_bytes()).hexdigest()} for name,path in
        (("policy",args.objective_policy),("registry",args.registry),("manifest",args.manifest),("prediction_evidence",args.prediction_evidence),("reference_report",args.reference_report)) if path}
    if args.summary:
        try:
            result["summary_validation"] = validate_summary(result, json.loads(args.summary.read_text(encoding="utf-8")))
        except (KeyError, ValueError, TypeError) as exc:
            result["summary_validation"] = "FAILED: " + str(exc)
            gate["evidence_gaps"].append("Paired summary verification failed.")
            gate["reliability"]["decision"] = "insufficient_evidence"
            gate["reliability"]["gaps"].append("Paired summary verification failed.")
            gate["promotion"]["decision"] = "insufficient_evidence"
            for task in gate["task_assessments"]:
                task["outcome"] = "inconclusive"
                task["development_decision"] = "insufficient_evidence"
                gate["promotion"]["task_decisions"][task["task"]] = "insufficient_evidence"
            if gate["measured_accuracy_gate"]["decision"] != "rejected":
                gate["measured_accuracy_gate"]["decision"] = "insufficient_evidence"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    console = {"output": str(args.output), "assessment_mode": gate["assessment_mode"],
                      "tasks": [{k:r[k] for k in ("task","outcome","development_decision")} for r in gate["task_assessments"]],
                      "reliability": gate["reliability"]["decision"], "promotion": gate["promotion"]}
    if args.enforce_accuracy_gate:
        console["explicit_strict_accuracy_diagnostic"] = gate["measured_accuracy_gate"]["decision"]
    print(json.dumps(console, indent=2))
    invalid_evidence=gate["reliability"]["decision"] in {"blocked","insufficient_evidence"}
    if args.enforce_evidence and invalid_evidence:
        return 2
    return 0 if not args.enforce_accuracy_gate or (not invalid_evidence and gate["measured_accuracy_gate"]["decision"] == "passed") else 2


if __name__ == "__main__":
    raise SystemExit(main())
