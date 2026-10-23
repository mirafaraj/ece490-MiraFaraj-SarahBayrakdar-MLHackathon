"""Small paired checkpoint/default verification; arrays stay in local run folders."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(phase):
    from unknown_dataset.contract import load_dataset
    from unknown_dataset.models import fit_predict
    target = HERE / "local" / ("defaults-" + phase)
    target.mkdir(parents=True, exist_ok=False)
    previous = json.loads((HERE / "defaults-before.json").read_text()) if phase == "after" else None
    rows = []
    for task, name in [("classification", "wine"), ("regression", "diabetes"), ("clustering", "iris")]:
        source = ROOT / f"experiments/real-data-check/inputs/real-{task}-{name}-42"
        data = load_dataset(source)
        policies = ["baseline", "auto", "routed"] + (["submission", None] if phase == "after" else [])
        for policy in policies:
            label = policy or "omitted"
            output = target / f"{task}-{label}"
            command = [sys.executable, str(ROOT / "main.py"), "--input", str(source),
                       "--output", str(output), "--time-budget", "50", "--seed", "42"]
            if policy is not None:
                command += ["--strategy", policy]
            env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
            start = time.monotonic()
            result = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True, timeout=60)
            (target / f"{task}-{label}.log").write_text(result.stdout + result.stderr)
            assert result.returncode == 0, (command, result.stderr)
            wall = time.monotonic() - start
            assert wall < 60
            run_info = json.loads((output / "run.json").read_text())
            info = run_info["model"]
            assert not info.get("fallback"), info
            api_kwargs = dict(task=data.task, X_train=data.X_train, y_train=data.y_train,
                X_test=data.X_test, feature_types=data.feature_types, n_clusters=data.n_clusters,
                seed=42, time_budget=50)
            if policy is not None:
                api_kwargs["strategy"] = policy
            pred, api_info = fit_predict(**api_kwargs)
            api_hash = hashlib.sha256(json.dumps(pred.tolist(), ensure_ascii=False, allow_nan=False).encode()).hexdigest()
            row = dict(task=task, strategy=label, prediction_sha256=sha(output / "predictions.csv"),
                profile_sha256=sha(output / "profile.json"), selected=info["model"], api_sha256=api_hash,
                api_selected=api_info["model"], seconds=wall, fallback=info.get("fallback", False))
            if previous and policy in ("baseline", "auto", "routed"):
                old = next(r for r in previous if r["task"] == task and r["strategy"] == policy)
                for key in ("prediction_sha256", "profile_sha256", "selected", "api_sha256", "api_selected"):
                    assert row[key] == old[key], (task, policy, key)
            rows.append(row)
            print(task, label, "verified", round(wall, 3), flush=True)
    if phase == "after":
        for task in ("classification", "regression", "clustering"):
            subset = {r["strategy"]: r for r in rows if r["task"] == task}
            delegated = "baseline" if task == "classification" else "auto"
            for key in ("prediction_sha256", "profile_sha256", "selected", "api_sha256", "api_selected"):
                assert subset["omitted"][key] == subset["submission"][key] == subset[delegated][key], (task, key)
    (HERE / f"defaults-{phase}.json").write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["before", "after"])
    run(parser.parse_args().phase)
