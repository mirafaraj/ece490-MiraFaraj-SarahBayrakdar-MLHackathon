"""Frozen, randomized serial runtime comparison with resumable evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import sys
import tempfile
import time

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, recall_score

from batch_runtime import make_fixture, measured_process

ROOT = Path(__file__).resolve().parents[1]


def plan_runs():
    rng = random.Random(10490)
    plans = []
    for shape in ("normal", "wider"):
        for budget in (10, 25, 50):
            for repeat in range(5 if (shape, budget) == ("wider", 25) else 3):
                policies = ["routed", "lazy_routed", "auto"]
                rng.shuffle(policies)
                plans.extend(dict(shape=shape, budget=budget, repeat=repeat, policy=p) for p in policies)
    return plans


def source_hashes():
    files = [ROOT / "main.py", Path(__file__), ROOT / "scripts/batch_runtime.py",
             *sorted((ROOT / "src").rglob("*.py")), *sorted((ROOT / "configs").glob("*.json"))]
    return {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--max-seconds", type=float, default=300)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("Use a new output path; historical evidence is never overwritten")
    previous = json.loads(args.resume_from.read_text()) if args.resume_from else None
    hashes = source_hashes()
    if previous and previous["source_sha256"] != hashes:
        parser.error("Source/configuration changed since previous segment")
    planned = previous["unrun"] if previous else plan_runs()
    report = dict(scope="controlled development; metrics provisional; no admission changes",
                  order_seed=10490, source_sha256=hashes, planned=planned, unrun=list(planned), rows=[],
                  environment=dict(python=sys.version, platform=platform.platform(), logical_cpus=os.cpu_count()),
                  notes=["Identical two fixtures from first batch; evaluation labels remain harness-only.",
                         "Three repeats per cell, wider25 extended to five; serial randomized policies.",
                         "RSS may be unavailable; 60-second guard enforced by reused offline process monitor."])
    if previous:
        report["resumed_from"] = str(args.resume_from)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")

    started = time.monotonic()
    temporary_root = ROOT / "experiments/work"
    temporary_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="second-runtime-", dir=temporary_root) as temporary:
        work = Path(temporary)
        labels = {shape: make_fixture(work / shape, shape) for shape in ("normal", "wider")}
        report["fixture_sha256"] = {shape: {name: hashlib.sha256((work / shape / name).read_bytes()).hexdigest()
                                           for name in ("metadata.json", "train.csv", "test.csv")} for shape in labels}
        report["references"] = {shape: values.tolist() for shape, values in labels.items()}
        if previous and previous["fixture_sha256"] != report["fixture_sha256"]:
            raise ValueError("Fixture bytes differ")
        save()
        for index, plan in enumerate(planned):
            if time.monotonic() - started >= args.max_seconds:
                break
            if source_hashes() != hashes:
                raise RuntimeError("Source changed during serial timing; stop rather than mix implementations")
            output = work / f"output-{index}"
            command = [sys.executable, str(ROOT / "main.py"), "--input", str(work / plan["shape"]),
                       "--output", str(output), "--strategy", "auto" if plan["policy"] == "auto" else "routed",
                       "--time-budget", str(plan["budget"]), "--timings"]
            if plan["policy"] == "lazy_routed":
                command.append("--lazy-router-imports")
            row = dict(plan, **measured_process(command), valid=False)
            if row["returncode"] == 0:
                try:
                    predictions = pd.read_csv(output / "predictions.csv", dtype=str)["prediction"]
                    reference = labels[plan["shape"]]
                    row["valid"] = bool(len(predictions) == len(reference) and predictions.isin(["majority", "minority"]).all())
                    row["run"] = json.loads((output / "run.json").read_text())
                    row["profile"] = json.loads((output / "profile.json").read_text())
                    row["predictions"] = predictions.tolist()
                    row["prediction_sha256"] = hashlib.sha256((output / "predictions.csv").read_bytes()).hexdigest()
                    for metric, value in dict(accuracy=accuracy_score(reference, predictions),
                                             balanced_accuracy=balanced_accuracy_score(reference, predictions),
                                             macro_f1=f1_score(reference, predictions, average="macro"),
                                             minority_recall=recall_score(reference, predictions, pos_label="minority")).items():
                        row[metric] = float(value)
                except Exception as exc:
                    row["valid"] = False
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
