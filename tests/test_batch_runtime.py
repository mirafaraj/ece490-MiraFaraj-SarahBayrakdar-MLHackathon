"""Optional instrumentation must preserve predictions and completed recovery."""
import json
import pickle
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from unknown_dataset import models, routed, runner, supervised
from unknown_dataset.checkpoints import write_checkpoint
from unknown_dataset.contract import Dataset
from unknown_dataset.routing import load_router_config


def fixture():
    frame = pd.DataFrame({"x": np.arange(60, dtype=float)})
    return Dataset("classification", {}, frame, pd.Series(["a", "b"] * 30),
                   frame.iloc[:4], {"x": "numeric"}, None)


@pytest.mark.parametrize("optional_fails", [False, True])
def test_routed_timings_are_optional_and_preserve_predictions(monkeypatch, tmp_path, optional_fails):
    data = fixture()
    expected = np.array(["a", "b", "a", "b"])
    monkeypatch.setattr(supervised, "fit_baseline_supervised",
                        lambda **kwargs: (expected.copy(), {"model": "test_baseline"}))

    def optional(**kwargs):
        if optional_fails:
            raise RuntimeError("controlled optional failure")
        return expected[::-1].copy(), {"model": "test_optional"}

    monkeypatch.setattr(supervised, "fit_predict_supervised", optional)
    results = []
    for enabled in (False, True):
        config = load_router_config()
        config["experimental"]["runtime_instrumentation"] = enabled
        checkpoint = tmp_path / f"checkpoint-{enabled}.json"
        predictions, info = routed.fit_predict_routed(
            data.task, data.X_train, data.y_train, data.X_test, data.feature_types,
            time_budget=50, route_config=config,
            checkpoint_callback=lambda p, i: write_checkpoint(checkpoint, data, p, i))
        results.append(predictions)
        assert ("route_phase_seconds" in info) is enabled
        recovered = runner._recover_checkpoint(checkpoint, data, "controlled deadline")
        np.testing.assert_array_equal(recovered["predictions"], expected)
        assert recovered["model_info"]["checkpoint_recovered"] is True
        if enabled:
            assert {"plan", "baseline", "checkpoint", "optional_total"} <= info["route_phase_seconds"].keys()
            assert all(value >= 0 for value in info["route_phase_seconds"].values())
        assert info.get("fallback", False) is optional_fails
    np.testing.assert_array_equal(*results)


def test_worker_timing_metadata_is_opt_in(monkeypatch, tmp_path):
    data = fixture()
    monkeypatch.setattr(models, "fit_predict", lambda **kwargs: (np.array(["a"] * 4), {"model": "test"}))
    for enabled in (False, True):
        output = tmp_path / f"result-{enabled}.pkl"
        runner._model_worker(data, 50, 42, str(output), "baseline", instrument_runtime=enabled)
        with output.open("rb") as handle:
            result = pickle.load(handle)
        assert ("worker_phase_seconds" in result) is enabled
        assert result["predictions"].tolist() == ["a"] * 4
        if enabled:
            json.dumps(result["worker_phase_seconds"], allow_nan=False)


def test_cli_timing_flag_preserves_predictions(tmp_path):
    source = tmp_path / "input"
    source.mkdir()
    (source / "metadata.json").write_text(json.dumps({"task": "classification", "target": "label",
                                                      "feature_types": {"x": "numeric"}}))
    (source / "train.csv").write_text("x,label\n" + "".join(f"{i},{'a' if i < 48 else 'b'}\n" for i in range(96)))
    (source / "test.csv").write_text("x\n2\n90\n")
    root = Path(__file__).resolve().parents[1]
    environment = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    outputs = []
    for enabled in (False, True):
        output = tmp_path / str(enabled)
        command = [sys.executable, str(root / "main.py"), "--input", str(source), "--output", str(output),
                   # This checks optional-selection instrumentation, not low-budget
                   # admission. Leave room for cold imports and the 17s route gate.
                   "--strategy", "routed", "--time-budget", "50"] + (["--timings"] if enabled else [])
        result = subprocess.run(command, capture_output=True, text=True, env=environment, timeout=60)
        assert result.returncode == 0, result.stderr
        outputs.append((output / "predictions.csv").read_bytes())
        run = json.loads((output / "run.json").read_text())
        assert run["model"]["validation_scores"]
        assert ("phase_seconds" in run) is enabled
        status = json.loads(result.stderr.strip().splitlines()[-1])
        assert ("output_serialization_seconds" in status) is enabled
    assert outputs[0] == outputs[1]
