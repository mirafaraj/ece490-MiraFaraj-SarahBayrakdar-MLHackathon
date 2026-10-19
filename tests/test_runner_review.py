"""Regression checks from independent execution review."""

import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]


def test_failed_rerun_clears_only_generated_artifacts(tmp_path):
    source, output = tmp_path / "input", tmp_path / "output"
    source.mkdir()
    output.mkdir()
    (source / "metadata.json").write_text('{"task":"invalid"}')
    for name in ("predictions.csv", "profile.json", "run.json"):
        (output / name).write_text("old run")
    (output / "notes.txt").write_text("user file")
    completed = subprocess.run(
        [sys.executable, str(ROOT / "main.py"), "--input", str(source), "--output", str(output)],
        capture_output=True, text=True, timeout=15,
    )
    assert completed.returncode == 1
    assert {item.name for item in output.iterdir()} == {"notes.txt"}


def test_real_worker_deadline_publishes_marked_fallback(tmp_path):
    source, output = tmp_path / "input", tmp_path / "output"
    source.mkdir()
    (source / "metadata.json").write_text(json.dumps({
        "task": "classification", "target": "label", "feature_types": {"x": "numeric"},
    }))
    (source / "train.csv").write_text("x,label\n1,a\n2,a\n3,b\n")
    (source / "test.csv").write_text("x\n4\n5\n")
    helper = tmp_path / "sleeping_worker.py"
    helper.write_text(
        "import sys, time\n"
        f"sys.path.insert(0, {str(ROOT / 'src')!r})\n"
        "from unknown_dataset import runner\n"
        "def sleeper(*args):\n    time.sleep(60)\n"
        "if __name__ == '__main__':\n"
        "    runner._model_worker = sleeper\n"
        "    raise SystemExit(runner.main())\n"
    )
    environment = os.environ.copy()
    environment.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    started = time.monotonic()
    completed = subprocess.run(
        [sys.executable, str(helper), "--input", str(source), "--output", str(output),
         "--time-budget", "4"],
        capture_output=True, text=True, env=environment, timeout=12,
    )
    assert completed.returncode == 0, completed.stderr
    assert time.monotonic() - started < 10
    info = json.loads((output / "run.json").read_text())
    assert info["model"]["fallback"] is True
    assert info["model"]["fallback_reason"] == "model worker deadline"
    assert (output / "predictions.csv").read_text().splitlines() == ["prediction", "a", "a"]
