"""Separate 16-run optional refit-admission comparison; do not pool with imports."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import sys
import tempfile
import time
import zipfile

import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, recall_score

from batch_runtime import make_fixture, measured_process
from second_runtime import source_hashes as runtime_hashes

ROOT = Path(__file__).resolve().parents[1]


def plan_runs():
    rng = random.Random(20490)
    plans = []
    for budget, repeats in ((25, 5), (50, 3)):
        for repeat in range(repeats):
            policies = ["lazy", "lazy_admission"]
            rng.shuffle(policies)
            plans.extend(dict(shape="wider", budget=budget, repeat=repeat, policy=p) for p in policies)
    return plans


def source_hashes():
    result = runtime_hashes()
    result[Path(__file__).relative_to(ROOT).as_posix()] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--max-seconds", type=float, default=450)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("Use a new report path; never overwrite historical evidence")
    previous = json.loads(args.resume_from.read_text()) if args.resume_from else None
    hashes = source_hashes()
    if previous and previous["source_sha256"] != hashes:
        parser.error("Source/configuration changed since previous segment")
    planned = previous["unrun"] if previous else plan_runs()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    snapshot = args.output.parent / "runtime-admission-source.zip"
    if not previous:
        with zipfile.ZipFile(snapshot, "x", compression=zipfile.ZIP_DEFLATED) as archive:
            for name in hashes:
                archive.write(ROOT / name, name)
    archive_hash = hashlib.sha256(snapshot.read_bytes()).hexdigest()
    if previous and previous["snapshot_sha256"] != archive_hash:
        parser.error("Archived source bytes changed")
    report = dict(scope="separate controlled development: optional classification-only refit admission",
                  order_seed=20490, source_sha256=hashes, snapshot_sha256=archive_hash,
                  planned=planned, unrun=list(planned), rows=[],
                  environment=dict(python=sys.version, platform=platform.platform(), logical_cpus=os.cpu_count(),
                                   versions={name: importlib.metadata.version(name) for name in ("numpy", "pandas", "scipy", "scikit-learn")}),
                  notes=["Predeclared factor 1.5; no outer-reference-based selection or threshold tuning.",
                         "Both variants use lazy imports, same families/weights/objective/budgets.",
                         "Different frozen implementation from original60; compare these16 contemporaneously.",
                         "RSS may be unavailable; external process guard60seconds; native threads1."])
    if previous:
        report["resumed_from"] = str(args.resume_from)

    def save():
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")

    started = time.monotonic()
    temporary_root = ROOT / "experiments/work"
    temporary_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="second-admission-", dir=temporary_root) as temporary:
        work = Path(temporary)
        reference = make_fixture(work / "wider", "wider")
        report["references"] = reference.tolist()
        report["fixture_sha256"] = {name: hashlib.sha256((work / "wider" / name).read_bytes()).hexdigest()
                                    for name in ("metadata.json", "train.csv", "test.csv")}
        configs = {name: {"experimental": {"refit_admission": name == "lazy_admission"}}
                   for name in ("lazy", "lazy_admission")}
        report["configurations"] = configs
        report["configuration_sha256"] = {}
        for name, config in configs.items():
            path = work / f"{name}.json"
            path.write_text(json.dumps(config, sort_keys=True), encoding="utf-8")
            report["configuration_sha256"][name] = hashlib.sha256(path.read_bytes()).hexdigest()
        if previous and (previous["fixture_sha256"] != report["fixture_sha256"] or previous["configuration_sha256"] != report["configuration_sha256"]):
            raise ValueError("Fixture/experimental configuration bytes differ")
        save()
        for index, plan in enumerate(planned):
            if time.monotonic() - started >= args.max_seconds:
                break
            if source_hashes() != hashes:
                raise RuntimeError("Source/configuration changed during timed comparison")
            output = work / f"output-{index}"
            command = [sys.executable, str(ROOT / "main.py"), "--input", str(work / "wider"),
                       "--output", str(output), "--strategy", "routed", "--time-budget", str(plan["budget"]),
                       "--timings", "--lazy-router-imports", "--router-config", str(work / f"{plan['policy']}.json")]
            row = dict(plan, **measured_process(command), valid=False)
            row["implementation_failure"] = row["returncode"] != 0
            if row["returncode"] == 0:
                try:
                    predictions = pd.read_csv(output / "predictions.csv", dtype=str)["prediction"]
                    row["valid"] = bool(len(predictions) == len(reference) and predictions.isin(["majority", "minority"]).all())
                    row["run"] = json.loads((output / "run.json").read_text())
                    row["profile"] = json.loads((output / "profile.json").read_text())
                    model = row["run"]["model"]
                    issues = []
                    if model.get("route_stage") == "optional_error":
                        issues.append("optional_error: " + model.get("fallback_reason", "unspecified"))
                    if plan["policy"] == "lazy_admission" and model.get("route_stage") == "completed_selection":
                        admission = model.get("refit_admission", {})
                        if admission.get("deadline_source") != "parent_absolute":
                            issues.append("Completed admission did not use the parent absolute deadline")
                        if not {"eligible_candidates", "rejected_candidates", "selected_before", "selected_after"} <= admission.keys():
                            issues.append("Completed admission is missing eligibility/selection diagnostics")
                    row["implementation_errors"] = issues
                    row["implementation_failure"] = bool(issues)
                    row["predictions"] = predictions.tolist()
                    row["prediction_sha256"] = hashlib.sha256((output / "predictions.csv").read_bytes()).hexdigest()
                    row.update(accuracy=float(accuracy_score(reference, predictions)),
                               balanced_accuracy=float(balanced_accuracy_score(reference, predictions)),
                               macro_f1=float(f1_score(reference, predictions, average="macro")),
                               minority_recall=float(recall_score(reference, predictions, pos_label="minority")))
                except Exception as exc:
                    row["valid"] = False
                    row["implementation_failure"] = True
                    row["validation_error"] = str(exc)
            report["rows"].append(row)
            report["unrun"] = planned[index + 1:]
            report["elapsed_seconds"] = time.monotonic() - started
            save()
    report["elapsed_seconds"] = time.monotonic() - started
    save()
    print(json.dumps(dict(completed=len(report["rows"]), unrun=len(report["unrun"]), seconds=report["elapsed_seconds"])))


if __name__ == "__main__":
    main()
