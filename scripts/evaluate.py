"""Reproducible development benchmark; these are NOT official contest scores."""
from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, adjusted_rand_score, balanced_accuracy_score, mean_squared_error, r2_score

try:
    from .datasets import make_broad_cases, make_router_cases
except ImportError:
    from datasets import make_broad_cases, make_router_cases

ROOT = Path(__file__).resolve().parents[1]


def make_cases(root: Path, seed: int) -> list[dict]:
    """Create mixed nonlinear supervised tasks and separated synthetic groups."""
    cases = []
    for offset, task in enumerate(("classification", "regression", "clustering")):
        rng = np.random.default_rng(seed + offset)
        n = 600 if task != "clustering" else 360
        if task == "clustering":
            truth = np.repeat(np.arange(3), n // 3)
            centers = np.array([[-5, -4], [0, 5], [5, -3]])
            xy = centers[truth] + rng.normal(0, 0.65, (n, 2))
            category = np.array(["west", "north", "east"])[truth].astype(object)
        else:
            xy = rng.normal(size=(n, 2))
            category = rng.choice(["standard", "premium", "basic"], size=n).astype(object)
            signal = 2.8 * xy[:, 0] ** 2 + 1.8 * xy[:, 1] + 2 * (category == "premium")
            truth = (signal + rng.normal(0, 0.4, n) > 4.2).astype(int) if task == "classification" else signal + rng.normal(0, 0.3, n)
        # Targets are generated before corruption, so missingness is realistic.
        frame = pd.DataFrame({"row_id": np.arange(n), "x": xy[:, 0], "large_scale": xy[:, 1] * 1000,
                              "segment": category, "noise": rng.normal(size=n), "constant": 7.0,
                              "all_missing": np.nan})
        frame.loc[rng.random(n) < 0.08, "x"] = np.nan
        frame.loc[rng.random(n) < 0.06, "segment"] = None
        frame["x_duplicate"] = frame["x"]
        types = {c: "categorical" if c == "segment" else "numeric" for c in frame.columns if c != "row_id"}
        metadata = {"task": task, "feature_types": types, "id_column": "row_id"}
        folder = root / task
        folder.mkdir(parents=True, exist_ok=True)
        if task == "clustering":
            metadata["n_clusters"] = 3
            frame.to_csv(folder / "train.csv", index=False)
            reference = truth
        else:
            train, test = frame.iloc[:450].copy(), frame.iloc[450:].copy()
            train["target"] = truth[:450]
            test.loc[test.index[:8], "segment"] = "unseen_at_training"
            train.to_csv(folder / "train.csv", index=False)
            test.to_csv(folder / "test.csv", index=False)
            metadata["target"] = "target"
            reference = truth[450:]
        (folder / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        cases.append({"task": task, "input": folder, "reference": reference,
                      "training_target": None if task == "clustering" else truth[:450]})
    return cases


def git_commit() -> str | None:
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, timeout=5)
        return result.stdout.strip() if result.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def source_fingerprint(router_config: Path | None = None) -> str:
    digest = hashlib.sha256()
    for path in sorted([ROOT / "main.py", *ROOT.glob("src/**/*.py"), *ROOT.glob("scripts/*.py"), *ROOT.glob("configs/*.json")]):
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(path.read_bytes())
    if router_config is not None:
        digest.update(b"explicit-router-config")
        digest.update(router_config.read_bytes())
    return digest.hexdigest()


def validate_profile(case: dict, profile: dict) -> None:
    """Independently check provisional semantics from raw CSV, without production imports."""
    folder = case["input"]
    metadata = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
    types = metadata.get("feature_types")
    lexical = [c for c, kind in (types or {}).items() if kind == "categorical"]
    if metadata["task"] == "classification":
        lexical.append(metadata["target"])
    train = pd.read_csv(folder / "train.csv", dtype={c: str for c in lexical})
    target = train[metadata["target"]] if "target" in metadata else None
    excluded = [metadata[key] for key in ("target", "id_column") if key in metadata]
    features = train.drop(columns=excluded)
    types = types or {c: "numeric" if pd.api.types.is_numeric_dtype(features[c]) else "categorical" for c in features}
    def check(actual, expected, field):
        if isinstance(expected, float):
            valid = actual is not None and np.isclose(actual, expected, rtol=1e-7, atol=1e-9)
        else:
            valid = actual == expected
        if not valid:
            raise ValueError(f"Profile mismatch at {field}: {actual!r} != {expected!r}")
    check(profile.get("row_count"), len(features), "row_count")
    check(profile.get("feature_count"), len(features.columns), "feature_count")
    check(profile.get("missing_count"), int(features.isna().sum().sum()), "missing_count")
    check(profile.get("feature_type_counts"), {k: list(types.values()).count(k) for k in ("numeric", "categorical")}, "feature_type_counts")
    check(profile.get("constant_columns"), [c for c in features if features[c].nunique() <= 1], "constant_columns")
    duplicates, representatives = [], []
    for column in features:
        duplicate = next((c for c in representatives if features[c].equals(features[column])), None)
        if duplicate is None:
            representatives.append(column)
        else:
            duplicates.append({"column": column, "duplicate_of": duplicate})
    check(profile.get("duplicate_columns"), duplicates, "duplicate_columns")
    def summary(series, actual, prefix):
        values = series.dropna().to_numpy(dtype=float)
        expected = ({"min": float(np.min(values)), "max": float(np.max(values)), "mean": float(np.mean(values)),
                     "median": float(np.median(values)), "std": float(np.std(values))} if len(values)
                    else dict.fromkeys(["min", "max", "mean", "median", "std"]))
        for key, value in expected.items():
            check(actual.get(key), value, prefix + "." + key)
    for column in features:
        series, entry = features[column], profile.get("columns", {}).get(column, {})
        for key, value in {"type": types[column], "missing_count": int(series.isna().sum()),
                           "missing_fraction": float(series.isna().mean()), "unique_count": int(series.nunique()),
                           "constant": series.nunique() <= 1, "all_missing": bool(series.isna().all())}.items():
            check(entry.get(key), value, column + "." + key)
        if types[column] == "numeric":
            summary(series, entry.get("numeric_summary", {}), column)
    if target is not None:
        actual = profile.get("target", {})
        for key, value in {"name": target.name, "missing_count": int(target.isna().sum()), "unique_count": int(target.nunique())}.items():
            check(actual.get(key), value, "target." + key)
        numeric = pd.api.types.is_numeric_dtype(target)
        if numeric:
            summary(target, actual.get("numeric_summary", {}), "target")
        if not numeric or target.nunique() <= 50:
            distribution = {row["value"]: row["count"] for row in actual.get("distribution", [])}
            check(distribution, target.value_counts().to_dict(), "target.distribution")


def classification_metrics(truth, pred, training_target):
    # sklearn treats fractional labels as continuous; paired encoding preserves identity.
    levels = list(pd.unique(np.concatenate([np.asarray(training_target), np.asarray(truth), np.asarray(pred)])))
    mapping = {value: index for index, value in enumerate(levels)}
    truth_codes = [mapping[value] for value in truth]
    pred_codes = [mapping[value] for value in pred]
    majority = pd.Series(training_target).mode().iloc[0]
    return {"accuracy": accuracy_score(truth_codes, pred_codes),
            "balanced_accuracy": balanced_accuracy_score(truth_codes, pred_codes),
            "majority_accuracy": accuracy_score(truth_codes, [mapping[majority]] * len(truth))}


def evaluate_case(case: dict, output: Path, seed: int, timeout: float, budget: float, strategy: str = "baseline", router_config: Path | None = None) -> dict:
    command = [sys.executable, str(ROOT / "main.py"), "--input", str(case["input"]),
               "--output", str(output), "--seed", str(seed), "--time-budget", str(budget), "--strategy", strategy]
    if router_config is not None and strategy == "routed":
        command.extend(["--router-config", str(router_config.resolve())])
    digest = hashlib.sha256()
    for file in sorted(case["input"].glob("*")):
        digest.update(file.name.encode())
        digest.update(file.read_bytes())
    result = {"name": case.get("name", case["task"]), "family": case.get("family", "smoke"),
              "task": case["task"], "seed": seed, "strategy": strategy, "source": case.get("source", "synthetic"),
              "partition": case.get("partition", "confirmation" if seed == 8081 else "development"),
              "data_sha256": digest.hexdigest(), "valid": False, "metrics": {}, "error": None}
    start = time.perf_counter()
    try:
        environment = os.environ.copy()
        environment.update({"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
        completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True,
                                   timeout=timeout, env=environment)
        result["wall_seconds"] = round(time.perf_counter() - start, 4)
        result["exit_code"] = completed.returncode
        if completed.returncode:
            raise ValueError(f"CLI failed: {completed.stderr[-3000:]}")
        predictions = pd.read_csv(output / "predictions.csv", dtype={"prediction": str} if case["task"] == "classification" else None)
        if list(predictions.columns) != ["prediction"] or len(predictions) != len(case["reference"]):
            raise ValueError("Invalid prediction schema or row count")
        pred, truth = predictions["prediction"].to_numpy(), case["reference"]
        if pd.isna(pred).any():
            raise ValueError("Predictions contain missing values")
        profile = json.loads((output / "profile.json").read_text(encoding="utf-8"))
        info = json.loads((output / "run.json").read_text(encoding="utf-8"))
        if not isinstance(profile, dict) or not profile or not isinstance(info, dict):
            raise ValueError("Profile and run metadata must be JSON objects")
        validate_profile(case, profile)
        task = case["task"]
        if task == "classification":
            training_target = np.asarray([str(v) for v in case["training_target"]])
            truth = np.asarray([str(v) for v in truth])
            if not set(pred).issubset(set(training_target)):
                raise ValueError("Predicted class absent from training labels")
            result["metrics"] = classification_metrics(truth, pred, training_target)
        elif task == "regression":
            if not np.isfinite(pred.astype(float)).all():
                raise ValueError("Non-finite numeric predictions")
            result["metrics"] = {"rmse": float(np.sqrt(mean_squared_error(truth, pred))), "r2": r2_score(truth, pred),
                                 "mean_baseline_rmse": float(np.sqrt(mean_squared_error(truth, np.full(len(truth), np.mean(case["training_target"])))))}
        else:
            k = case.get("n_clusters", 3)
            if not np.isfinite(pred.astype(float)).all() or not np.equal(pred, pred.astype(int)).all() or set(pred) != set(range(k)):
                raise ValueError(f"Expected integer labels occupying exactly {k} clusters in range [0, {k})")
            result["metrics"] = {"adjusted_rand_index": adjusted_rand_score(truth, pred)}
        result.update(valid=True, profile_status="Provisional raw-data semantics checked; official semantics unverified", run_info=info,
                      fallback=bool(info.get("model", {}).get("fallback", False)))
        model = info.get("model", {})
        result["fallback_kind"] = fallback_kind(result)
        result["routing"] = model.get("routing", {})
        result["resource_exclusions"] = result["routing"].get("excluded", {})
    except subprocess.TimeoutExpired:
        result["error"] = f"CLI exceeded {timeout} second subprocess timeout"
    except (OSError, ValueError, TypeError, KeyError) as exc:
        result["error"] = str(exc)
    result.setdefault("wall_seconds", round(time.perf_counter() - start, 4))
    return result


def fallback_kind(row: dict) -> str:
    """Separate retained completed predictions from weak emergency output."""
    model = row.get("run_info", {}).get("model", {})
    if not row.get("fallback", model.get("fallback", False)):
        return "none"
    if model.get("checkpoint_recovered") or model.get("fallback_kind") == "completed_baseline":
        return "completed_baseline"
    return model.get("fallback_kind", "emergency")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "experiments" / "baseline.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--suite", choices=["smoke", "broad", "router"], default="smoke")
    parser.add_argument("--strategies", choices=["baseline", "auto", "routed"], nargs="+", default=["baseline"])
    parser.add_argument("--router-config", type=Path)
    parser.add_argument("--cases", nargs="+", help="Only these named cases")
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--time-budget", type=float, default=50)
    args = parser.parse_args()
    if args.timeout <= 0 or args.time_budget <= 0:
        parser.error("Timeout and time budget must be positive")
    report = {"notice": "Offline metrics only; not official contest scores or the official checker. Seed 8081 is predeclared confirmation; all other seeds are development. Independence requires frozen code before confirmation.",
              "config": {"seed": args.seed, "timeout_seconds": args.timeout, "time_budget_seconds": args.time_budget,
                         "git_commit": git_commit(), "python": sys.version, "generator_version": 3,
                         "source_sha256": source_fingerprint(args.router_config), "platform": sys.platform,
                         "router_config": str(args.router_config.resolve()) if args.router_config else None,
                         "router_config_sha256": hashlib.sha256(args.router_config.read_bytes()).hexdigest() if args.router_config else None,
                         "suite": args.suite, "seeds": args.seeds or [args.seed], "strategies": args.strategies}}
    report["config"]["dependencies"] = {name: version(name) for name in ("numpy", "pandas", "scipy", "scikit-learn")}
    work_root = ROOT / "experiments" / "work"
    work_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="unknown-dataset-eval-", dir=work_root) as temporary:
        work = Path(temporary)
        report["datasets"] = []
        for seed in args.seeds or [args.seed]:
            case_root = work / str(seed)
            generator = {"broad": make_broad_cases, "smoke": make_cases, "router": make_router_cases}[args.suite]
            cases = generator(case_root, seed)
            if args.cases:
                cases = [case for case in cases if case.get("name", case["task"]) in args.cases]
                if not cases:
                    parser.error("No requested cases found in selected suite")
            for case in cases:
                for strategy in args.strategies:
                    output = work / f"{seed}-{case.get('name', case['task'])}-{strategy}-output"
                    row = evaluate_case(case, output, seed, args.timeout, args.time_budget, strategy, args.router_config)
                    report["datasets"].append(row)
                    print(f"{row['name']} seed={seed} {strategy}: valid={row['valid']} time={row['wall_seconds']:.2f}s metrics={row['metrics']} error={row['error']}", flush=True)
                    args.output.parent.mkdir(parents=True, exist_ok=True)
                    args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    report["summary"] = {}
    report["config"]["source_sha256_after"] = source_fingerprint(args.router_config)
    report["config"]["source_unchanged"] = report["config"]["source_sha256"] == report["config"]["source_sha256_after"]
    for strategy in args.strategies:
        report["summary"][strategy] = {}
        for task in ("classification", "regression", "clustering"):
            rows = [r for r in report["datasets"] if r["strategy"] == strategy and r["task"] == task]
            valid = [r for r in rows if r["valid"]]
            report["summary"][strategy][task] = {"cases": len(rows), "valid": len(valid),
                "fallbacks": sum(bool(r.get("fallback")) for r in valid),
                "checkpoint_recoveries": sum(fallback_kind(r) == "completed_baseline" for r in valid),
                "emergency_fallbacks": sum(fallback_kind(r) not in {"none", "completed_baseline"} for r in valid),
                "mean_wall_seconds": float(np.mean([r["wall_seconds"] for r in rows])) if rows else None}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Report: {args.output.resolve()}")
    if not report["config"]["source_unchanged"]:
        print("Source files changed during evaluation; rerun against a frozen source tree.")
    return 0 if report["config"]["source_unchanged"] and all(row["valid"] for row in report["datasets"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
