"""Recovery is useful only for complete, task-valid worker predictions."""

import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from unknown_dataset import checkpoints


ROOT = Path(__file__).resolve().parents[1]


def dataset(task="classification"):
    return SimpleNamespace(task=task, X_train=pd.DataFrame({"x": range(4)}),
                           X_test=pd.DataFrame({"x": [4, 5]}),
                           y_train=pd.Series(["01", "1", "01", "1"]), n_clusters=2)


@pytest.mark.parametrize("task,predictions", [
    ("classification", [["01"], ["1"]]),
    ("classification", ["01"]),
    ("classification", ["missing", "1"]),
    ("classification", [None, "1"]),
    ("classification", [1, "01"]),
    ("regression", [float("nan"), 1]),
    ("regression", [float("inf"), 1]),
    ("regression", ["1", "2"]),
    ("clustering", [0, 0, 0, 0]),
    ("clustering", [0.0, 1.0, 0.0, 1.0]),
    ("clustering", [0, 1, 2, 0]),
])
def test_checkpoint_rejects_task_invalid_predictions(tmp_path, task, predictions):
    frame = dataset(task)
    path = tmp_path / "checkpoint.json"
    with pytest.raises(ValueError):
        checkpoints.write_checkpoint(path, frame, predictions, {"model": "baseline"})
    assert not path.exists()


@pytest.mark.parametrize("task,predictions", [
    ("classification", ["01", "1"]),
    ("regression", [1.5, 2.5]),
    ("clustering", [0, 1, 0, 1]),
])
def test_checkpoint_roundtrip_preserves_predictions(tmp_path, task, predictions):
    frame = dataset(task)
    path = tmp_path / "checkpoint.json"
    checkpoints.write_checkpoint(path, frame, np.asarray(predictions), {"model": "baseline"})
    result = checkpoints.load_checkpoint(path, frame)
    assert result["predictions"].tolist() == predictions
    assert result["model_info"] == {"model": "baseline"}


def test_atomic_failure_keeps_previous_checkpoint(tmp_path, monkeypatch):
    path = tmp_path / "checkpoint.json"
    checkpoints.write_checkpoint(path, dataset(), ["01", "1"], {"model": "first"})
    previous = path.read_bytes()

    def fail_replace(*args):
        raise OSError("forced publication failure")

    monkeypatch.setattr(checkpoints.os, "replace", fail_replace)
    with pytest.raises(OSError, match="publication"):
        checkpoints.write_checkpoint(path, dataset(), ["1", "01"], {"model": "second"})
    assert path.read_bytes() == previous
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize("contents", [
    '{"version":1,',
    '{"version":2,"predictions":["01","1"],"model_info":{}}',
    '{"version":1,"predictions":["bad","1"],"model_info":{}}',
    '{"version":1,"predictions":["01","1"],"model_info":null}',
])
def test_corrupt_checkpoint_is_not_recovered(tmp_path, contents):
    from unknown_dataset.runner import _recover_checkpoint

    path = tmp_path / "checkpoint.json"
    path.write_text(contents)
    assert _recover_checkpoint(path, dataset(), "test") is None


def invoke_worker(tmp_path, mode, strategy="routed"):
    source, output = tmp_path / "input", tmp_path / "output"
    source.mkdir()
    (source / "metadata.json").write_text(json.dumps({
        "task": "classification", "target": "label", "feature_types": {"x": "numeric"}}))
    (source / "train.csv").write_text("x,label\n1,a\n2,a\n3,b\n4,b\n")
    (source / "test.csv").write_text("x\n5\n6\n")
    helper = tmp_path / "controlled_worker.py"
    # Run a genuine spawned process; only the model work is controlled. Distinct
    # baseline predictions prove recovery was used instead of majority fallback.
    helper.write_text(
        "import sys, time, os, pickle\n"
        f"sys.path.insert(0, {str(ROOT / 'src')!r})\n"
        "from unknown_dataset import runner\n"
        "from unknown_dataset.checkpoints import write_checkpoint\n"
        f"MODE = {mode!r}\n"
        "def controlled(dataset, budget, seed, result_path, strategy, *extra):\n"
        "    if extra and MODE not in ('error_empty', 'corrupt_error'):\n"
        "        write_checkpoint(extra[1], dataset, ['b', 'a'], {'model': 'CompletedTestBaseline'})\n"
        "    if MODE == 'timeout':\n"
        "        time.sleep(60)\n"
        "    if MODE == 'crash':\n"
        "        os._exit(7)\n"
        "    if MODE == 'corrupt_error':\n"
        "        open(extra[1], 'w').write('{broken')\n"
        "    if MODE == 'invalid_final':\n"
        "        result = {'predictions': ['unknown', 'a'], 'model_info': {}}\n"
        "    elif MODE == 'success':\n"
        "        result = {'predictions': ['a', 'b'], 'model_info': {'model': 'Final'}}\n"
        "    else:\n"
        "        result = {'error': 'optional model failed'}\n"
        "    with open(result_path, 'wb') as handle:\n"
        "        pickle.dump(result, handle)\n"
        "if __name__ == '__main__':\n"
        "    runner._model_worker = controlled\n"
        "    raise SystemExit(runner.main())\n"
    )
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    command = [sys.executable, str(helper), "--input", str(source), "--output", str(output),
               "--time-budget", "8", "--strategy", strategy]
    result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=20)
    return result, output


@pytest.mark.parametrize("mode", ["timeout", "error", "crash", "invalid_final"])
def test_real_spawn_recovers_completed_baseline(tmp_path, mode):
    result, output = invoke_worker(tmp_path, mode)
    assert result.returncode == 0, result.stderr
    run = json.loads((output / "run.json").read_text())
    info = run["model"]
    assert info["model"] == "CompletedTestBaseline"
    assert info["fallback"] is True
    assert info["fallback_kind"] == "completed_baseline"
    assert info["checkpoint_recovered"] is True
    assert info["fallback_reason"]
    assert (output / "predictions.csv").read_text().splitlines() == ["prediction", "b", "a"]


@pytest.mark.parametrize("mode", ["error_empty", "corrupt_error"])
def test_failure_without_valid_checkpoint_is_nonzero(tmp_path, mode):
    result, output = invoke_worker(tmp_path, mode)
    assert result.returncode == 1
    assert "optional model failed" in result.stderr
    assert not (output / "predictions.csv").exists()


@pytest.mark.parametrize("strategy", ["baseline", "auto", "routed"])
def test_successful_final_output_takes_priority(tmp_path, strategy):
    result, output = invoke_worker(tmp_path, "success", strategy)
    assert result.returncode == 0, result.stderr
    run = json.loads((output / "run.json").read_text())
    assert run["model"] == {"model": "Final"}
    assert run["strategy"] == strategy
    assert (output / "predictions.csv").read_text().splitlines() == ["prediction", "a", "b"]


def test_real_model_worker_passes_new_arguments_only_when_routed(tmp_path, monkeypatch):
    from unknown_dataset import models, runner

    observed = []

    def predict(**kwargs):
        observed.append(kwargs)
        if "checkpoint_callback" in kwargs:
            kwargs["checkpoint_callback"](["01", "1"], {"model": "baseline"})
        return ["01", "1"], {"model": "final"}

    monkeypatch.setattr(models, "fit_predict", predict)
    frame = dataset()
    frame.feature_types = {"x": "numeric"}
    for strategy in ("baseline", "auto", "routed"):
        runner._model_worker(frame, 5, 42, str(tmp_path / f"{strategy}.pkl"), strategy,
                             {"test": True}, str(tmp_path / "checkpoint.json"))
    assert "route_config" not in observed[0]
    assert "checkpoint_callback" not in observed[1]
    assert observed[2]["route_config"] == {"test": True}
    assert checkpoints.load_checkpoint(tmp_path / "checkpoint.json", frame)["model_info"] == {"model": "baseline"}
