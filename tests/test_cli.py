"""Exercise the actual entry point and spawned worker, including on Windows."""
import csv
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def invoke(input_dir, output_dir):
    environment = os.environ.copy()
    environment.update({"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
    return subprocess.run([sys.executable, str(ROOT / "main.py"), "--input", str(input_dir),
                           "--output", str(output_dir), "--time-budget", "20", "--seed", "7"],
                          cwd=ROOT, env=environment, capture_output=True, text=True, timeout=30)


def test_cli_writes_all_artifacts_in_requested_directory(tmp_path):
    source, output = tmp_path / "input", tmp_path / "output"
    source.mkdir()
    (source / "metadata.json").write_text(json.dumps({"task": "classification", "target": "label",
        "feature_types": {"x": "numeric", "category": "categorical"}}))
    (source / "train.csv").write_text("x,category,label\n1,a,no\n2,a,no\n3,b,no\n8,b,yes\n9,c,yes\n10,c,yes\n")
    (source / "test.csv").write_text("category,x\nnew,1\na,9\n")
    completed = invoke(source, output)
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == ""
    with (output / "predictions.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert all(set(row) == {"prediction"} and row["prediction"] in {"yes", "no"} for row in rows)
    profile = json.loads((output / "profile.json").read_text())
    assert profile["row_count"] == 6
    assert profile["feature_count"] == 2
    run = json.loads((output / "run.json").read_text())
    assert run["task"] == "classification"
    assert run["seed"] == 7


def test_cli_rejects_invalid_input_without_prediction_artifact(tmp_path):
    source, output = tmp_path / "input", tmp_path / "output"
    source.mkdir()
    (source / "metadata.json").write_text(json.dumps({"task": "unsupported"}))
    completed = invoke(source, output)
    assert completed.returncode != 0
    assert "task" in completed.stderr.lower()
    assert not (output / "predictions.csv").exists()
