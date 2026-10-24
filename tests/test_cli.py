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
    assert profile["columns"]["category"]["categorical_summary"] == {
        "top_values": [{"value": "a", "count": 2}, {"value": "b", "count": 2}, {"value": "c", "count": 2}],
        "omitted_category_count": 0, "omitted_value_count": 0,
    }
    assert "requested_cluster_count" not in profile
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


def test_cli_profiles_requested_clusters_and_full_categorical_counts(tmp_path):
    source, output = tmp_path / "input", tmp_path / "output"
    source.mkdir()
    (source / "metadata.json").write_text(json.dumps({"task": "clustering", "n_clusters": 3,
        "feature_types": {"x": "numeric", "kind": "categorical"}}))
    (source / "train.csv").write_text("x,kind\n0,a\n0,a\n0,a\n0,b\n0,b\n0,\n")
    completed = invoke(source, output)
    assert completed.returncode == 0, completed.stderr
    profile = json.loads((output / "profile.json").read_text())
    assert profile["requested_cluster_count"] == 3 and profile["row_count"] == 6
    assert "observed_cluster_count" not in profile
    kind = profile["columns"]["kind"]
    assert kind["missing_count"] == 1
    assert kind["categorical_summary"] == {"top_values": [{"value": "a", "count": 3}, {"value": "b", "count": 2}],
        "omitted_category_count": 0, "omitted_value_count": 0}
