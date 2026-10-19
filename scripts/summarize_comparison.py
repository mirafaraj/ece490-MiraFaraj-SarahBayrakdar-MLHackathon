"""Pair identical baseline/candidate cases without averaging incompatible task metrics."""

import argparse
import json
from pathlib import Path
from statistics import mean


def summarize(report, reference="baseline", candidate_strategy="auto", classification_metric="balanced_accuracy"):
    if reference == candidate_strategy:
        raise ValueError("Comparison strategies must differ")
    pairs = {}
    for row in report["datasets"]:
        pairs.setdefault((row["seed"], row["task"], row["name"]), {})[row["strategy"]] = row
    results = []
    for (seed, task, name), strategies in pairs.items():
        if not {reference, candidate_strategy}.issubset(strategies):
            raise ValueError(f"Missing paired strategy: {seed}/{name}")
        baseline, candidate = strategies[reference], strategies[candidate_strategy]
        if baseline["data_sha256"] != candidate["data_sha256"]:
            raise ValueError(f"Different input rows: {seed}/{name}")
        valid = baseline["valid"] and candidate["valid"]
        metric = {"classification": classification_metric, "regression": "rmse",
                  "clustering": "adjusted_rand_index"}[task]
        before = baseline["metrics"].get(metric)
        after = candidate["metrics"].get(metric)
        change = ((before - after) if task == "regression" else (after - before)) if valid else None
        results.append({
            "seed": seed, "partition": baseline["partition"], "task": task, "name": name,
            "valid_pair": bool(valid), "metric": metric, reference: before, candidate_strategy: after,
            "quality_change": change,
            "relative_rmse_reduction": (1 - after / before) if valid and task == "regression" and before else None,
            "outcome": "invalid" if not valid else "tie" if abs(change) <= 1e-9 else "win" if change > 0 else "loss",
            reference + "_seconds": baseline["wall_seconds"], candidate_strategy + "_seconds": candidate["wall_seconds"],
            candidate_strategy + "_model": candidate.get("run_info", {}).get("model", {}).get("model"),
            candidate_strategy + "_fallback": candidate.get("fallback", False),
            "candidate_fallback_kind": candidate.get("fallback_kind", "emergency" if candidate.get("fallback") else "none"),
            "candidate_routing": candidate.get("routing", {}),
        })
    aggregates = []
    for partition in sorted({r["partition"] for r in results}):
        for task in ("classification", "regression", "clustering"):
            rows = [r for r in results if r["partition"] == partition and r["task"] == task]
            if not rows:
                continue
            valid = [r for r in rows if r["valid_pair"]]
            aggregates.append({
                "partition": partition, "task": task, "cases": len(rows),
                "outcomes": {label: sum(r["outcome"] == label for r in rows) for label in ("win", "tie", "loss", "invalid")},
                "mean_quality_change": mean(r["quality_change"] for r in valid) if valid and task != "regression" else None,
                "mean_relative_rmse_reduction": mean(r["relative_rmse_reduction"] for r in valid) if valid and task == "regression" else None,
                "mean_" + reference + "_seconds": mean(r[reference + "_seconds"] for r in rows),
                "mean_" + candidate_strategy + "_seconds": mean(r[candidate_strategy + "_seconds"] for r in rows),
                "checkpoint_recoveries": sum(r["candidate_fallback_kind"] == "completed_baseline" for r in rows),
                "emergency_fallbacks": sum(r["candidate_fallback_kind"] not in {"none", "completed_baseline"} for r in rows),
            })
    return {"notice": f"Paired offline metrics; no official contest points. Positive quality changes favor {candidate_strategy}.",
            "reference_strategy": reference, "candidate_strategy": candidate_strategy,
            "source_sha256": report["config"]["source_sha256"], "aggregates": aggregates, "pairs": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--output", type=Path, default=Path("experiments/comparison_summary.json"))
    parser.add_argument("--reference", choices=["baseline", "auto", "routed"], default="baseline")
    parser.add_argument("--candidate", choices=["baseline", "auto", "routed"], default="auto")
    parser.add_argument("--classification-metric", choices=["accuracy", "balanced_accuracy"], default="balanced_accuracy")
    args = parser.parse_args()
    result = summarize(json.loads(args.report.read_text(encoding="utf-8")), args.reference, args.candidate, args.classification_metric)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for row in result["aggregates"]:
        print(json.dumps(row))


if __name__ == "__main__":
    main()
