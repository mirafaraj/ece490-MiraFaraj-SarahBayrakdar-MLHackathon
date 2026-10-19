"""Bounded serial CLI timing study; labels live only in this offline harness.

Shapes are development fixtures, not claims about unpublished official limits.
All planned runs are recorded; the global cap can leave explicit unrun rows.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import platform
import subprocess
import sys
import tempfile
import time

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, recall_score

ROOT = Path(__file__).resolve().parents[1]


def plan_runs(repeats=3):
    rng = random.Random(490)
    plans = []
    for shape in ("normal", "wider"):
        for budget in (10, 25, 50):
            for repeat in range(repeats):
                strategies = ["baseline", "auto", "routed"]
                rng.shuffle(strategies)
                plans.extend(dict(shape=shape, budget=budget, repeat=repeat, strategy=s)
                             for s in strategies)
    return plans


def make_fixture(directory, shape):
    rng = np.random.default_rng(490)
    rows, columns = (96, 8) if shape == "normal" else (660, 110)
    features = rng.normal(size=(rows + 120, columns))
    signal = features[:, 0] + features[:, 1] * .65 + rng.normal(scale=.25, size=len(features))
    labels = np.where(signal > .7, "minority", "majority")
    features[rng.random(features.shape) < .10] = np.nan
    frame = pd.DataFrame(features, columns=[f"x{i}" for i in range(columns)])
    train = frame.iloc[:rows].copy()
    train["target"] = labels[:rows]
    directory.mkdir()
    train.to_csv(directory / "train.csv", index=False)
    frame.iloc[rows:].to_csv(directory / "test.csv", index=False)
    (directory / "metadata.json").write_text(json.dumps({
        "task": "classification", "target": "target",
        "feature_types": {c: "numeric" for c in frame.columns}}), encoding="utf-8")
    return labels[rows:]


def measured_process(command, timeout=60):
    try:
        import psutil
    except ImportError:
        psutil = None
    started = time.monotonic()
    environment = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          text=True, env=environment) as process:
        peak = None
        while process.poll() is None:
            if psutil is not None:
                try:
                    parent = psutil.Process(process.pid)
                    rss = sum(p.memory_info().rss for p in [parent, *parent.children(recursive=True)]
                              if p.is_running())
                    peak = max(peak or 0, rss)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            if time.monotonic() - started > timeout:
                # This helper is offline only; terminate descendants as well.
                if psutil is not None:
                    try:
                        for child in psutil.Process(process.pid).children(recursive=True):
                            child.kill()
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
                process.kill()
                stdout, stderr = process.communicate()
                return dict(returncode=process.returncode, external_timeout=True,
                            external_seconds=time.monotonic() - started,
                            peak_process_tree_rss_bytes=peak, stderr=stderr[-4000:])
            time.sleep(.025)
        stdout, stderr = process.communicate()
    return dict(returncode=process.returncode, external_timeout=False,
                external_seconds=time.monotonic() - started,
                peak_process_tree_rss_bytes=peak, stderr=stderr[-4000:])


def summarize(rows):
    summaries = []
    for shape in ("normal", "wider"):
        for budget in (10, 25, 50):
            for strategy in ("baseline", "auto", "routed"):
                group = [r for r in rows if (r["shape"], r["budget"], r["strategy"]) == (shape, budget, strategy)]
                valid = [r for r in group if r["valid"]]
                if not group:
                    continue
                entry = dict(shape=shape, budget=budget, strategy=strategy, runs=len(group),
                             valid=len(valid), failures=len(group) - len(valid),
                             mean_external_seconds=float(np.mean([r["external_seconds"] for r in group])),
                             external_over_60=sum(r["external_seconds"] >= 60 for r in group),
                             completed_baseline_recoveries=sum(r.get("run", {}).get("model", {}).get("checkpoint_recovered", False) for r in group),
                             fallback_runs=sum(r.get("run", {}).get("model", {}).get("fallback", False) for r in group),
                             emergency_fallbacks=sum(str(r.get("run", {}).get("model", {}).get("model", "")).startswith("Emergency") for r in group),
                             repeat_prediction_hashes=sorted({r["prediction_sha256"] for r in valid}))
                for metric in ("accuracy", "balanced_accuracy", "macro_f1", "minority_recall"):
                    entry["mean_" + metric] = float(np.mean([r[metric] for r in valid])) if valid else None
                summaries.append(entry)
    return summaries


def controlled_recovery(source, reference, checkpoint):
    """Fit a real baseline, force optional failure, then simulate deadline recovery.

    This is a capability/quality example, not a natural deadline probability.
    Process termination itself is covered by the existing spawned-worker tests.
    """
    from unittest.mock import patch

    sys.path.insert(0, str(ROOT / "src"))
    from unknown_dataset.checkpoints import write_checkpoint
    from unknown_dataset.contract import load_dataset
    from unknown_dataset.fallback import fallback_predictions
    from unknown_dataset.routed import fit_predict_routed
    from unknown_dataset.runner import _recover_checkpoint

    data = load_dataset(source)
    with patch("unknown_dataset.supervised.fit_predict_supervised",
               side_effect=RuntimeError("controlled optional-stage failure")):
        predictions, info = fit_predict_routed(
            data.task, data.X_train, data.y_train, data.X_test, data.feature_types,
            time_budget=25, seed=42,
            checkpoint_callback=lambda p, i: write_checkpoint(checkpoint, data, p, i))
    recovered = _recover_checkpoint(checkpoint, data, "simulated worker deadline")
    emergency, emergency_info = fallback_predictions(
        task=data.task, X_train=data.X_train, y_train=data.y_train, X_test=data.X_test,
        n_clusters=None, seed=42, reason="controlled comparison")
    results = {"scope": "controlled failure and simulated deadline, not natural frequency",
               "optional_failure_retained_baseline": info.get("fallback_kind") == "completed_baseline",
               "recovered_equals_completed_predictions": bool(np.array_equal(predictions, recovered["predictions"])),
               "emergency_model": emergency_info}
    for name, values in (("completed_and_recovered", recovered["predictions"]), ("emergency", emergency)):
        results[name] = {"accuracy": accuracy_score(reference, values),
                         "balanced_accuracy": balanced_accuracy_score(reference, values),
                         "macro_f1": f1_score(reference, values, average="macro"),
                         "minority_recall": recall_score(reference, values, pos_label="minority")}
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-seconds", type=float, default=300)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--resume-from", type=Path,
                        help="Run only the explicitly unrun suffix in a preserved report")
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("--output already exists; use a new report path to preserve evidence")
    previous = json.loads(args.resume_from.read_text()) if args.resume_from else None
    planned = previous["unrun"] if previous else plan_runs(args.repeats)
    report = {"scope": "development evidence; official size/memory/metric limits unavailable",
              "order_seed": 490, "sample_period_seconds": .025, "planned": planned,
              "rows": [], "notes": ["RSS is sampled sum of resident process-tree memory, not unique memory.",
                "Fine phases available in routed; baseline/auto have worker total only.",
                "Natural fallback frequency does not measure expected production recovery benefit."]}
    files = [ROOT / "main.py", Path(__file__).resolve(), *sorted((ROOT / "src").rglob("*.py")),
             *sorted((ROOT / "configs").glob("*.json"))]
    report["source_sha256"] = {str(path.relative_to(ROOT)).replace("\\", "/"):
                               hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
    if previous:
        # The offline harness gained resume support; production/config bytes must
        # remain identical. Both harness hashes remain visible in their reports.
        changed = [name for name, digest in previous["source_sha256"].items()
                   if name != "scripts/batch_runtime.py" and report["source_sha256"].get(name) != digest]
        if changed:
            parser.error(f"Cannot resume after production/config changes: {changed}")
        report["resumed_from"] = str(args.resume_from)
    report["environment"] = {"python": sys.version, "platform": platform.platform(),
                             "processor": platform.processor(), "logical_cpus": os.cpu_count(),
                             "package_versions": {name: importlib.metadata.version(name)
                                                   for name in ("numpy", "pandas", "scikit-learn")}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    started = time.monotonic()
    temporary_root = ROOT / "experiments" / "work"
    temporary_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="first-batch-runtime-", dir=temporary_root) as temporary:
        work = Path(temporary)
        labels = {shape: make_fixture(work / shape, shape) for shape in ("normal", "wider")}
        report["fixture_sha256"] = {shape: {
            name: hashlib.sha256((work / shape / name).read_bytes()).hexdigest()
            for name in ("metadata.json", "train.csv", "test.csv")} for shape in labels}
        if previous and report["fixture_sha256"] != previous["fixture_sha256"]:
            raise ValueError("Resumed fixture bytes differ from the original study")
        for index, plan in enumerate(planned):
            if time.monotonic() - started >= args.max_seconds:
                report["unrun"] = planned[index:]
                break
            output = work / f"run-{index}"
            command = [sys.executable, str(ROOT / "main.py"), "--input", str(work / plan["shape"]),
                       "--output", str(output), "--strategy", plan["strategy"],
                       "--time-budget", str(plan["budget"]), "--timings"]
            row = dict(plan, **measured_process(command))
            row["valid"] = False
            if row["returncode"] == 0 and (output / "predictions.csv").exists():
                try:
                    predictions = pd.read_csv(output / "predictions.csv", dtype=str)["prediction"]
                    reference = labels[plan["shape"]]
                    row["valid"] = len(predictions) == len(reference) and predictions.isin(["majority", "minority"]).all().item()
                    row["run"] = json.loads((output / "run.json").read_text())
                    row["accuracy"] = accuracy_score(reference, predictions)
                    row["balanced_accuracy"] = balanced_accuracy_score(reference, predictions)
                    row["macro_f1"] = f1_score(reference, predictions, average="macro")
                    row["minority_recall"] = recall_score(reference, predictions, pos_label="minority")
                    row["prediction_sha256"] = hashlib.sha256((output / "predictions.csv").read_bytes()).hexdigest()
                except Exception as exc:
                    row["validation_error"] = str(exc)
            report["rows"].append(row)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
        if time.monotonic() - started < args.max_seconds:
            try:
                report["controlled_recovery"] = controlled_recovery(
                    work / "normal", labels["normal"], work / "controlled-checkpoint.json")
            except Exception as exc:
                report["controlled_recovery"] = {"error": f"{type(exc).__name__}: {exc}"}
    report["total_seconds"] = time.monotonic() - started
    report.setdefault("unrun", [])
    report["summary"] = summarize(report["rows"])
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"completed": len(report["rows"]), "planned": len(planned),
                      "seconds": report["total_seconds"]}))


if __name__ == "__main__":
    main()
