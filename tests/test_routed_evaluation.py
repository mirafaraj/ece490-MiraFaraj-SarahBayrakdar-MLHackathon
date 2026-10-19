"""Independent checks of offline router evaluation plumbing."""
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.datasets import make_router_cases
from scripts.evaluate import fallback_kind, source_fingerprint
from scripts.summarize_comparison import summarize


def test_router_suite_balanced_offline_repeatable_without_label_files(tmp_path):
    left = make_router_cases(tmp_path / "left", 42)
    right = make_router_cases(tmp_path / "right", 42)
    assert len(left) == 9
    assert {t: sum(c["task"] == t for c in left) for t in ("classification", "regression", "clustering")} == dict.fromkeys(["classification", "regression", "clustering"], 3)
    assert sum("real" in c["source"] for c in left) == 1
    for a, b in zip(left, right):
        assert a["partition"] == "development"
        assert (a["input"] / "train.csv").read_bytes() == (b["input"] / "train.csv").read_bytes()
        np.testing.assert_array_equal(a["reference"], b["reference"])
        files = {p.name for p in a["input"].iterdir()}
        assert files <= {"metadata.json", "train.csv", "test.csv"}
        if a["task"] != "clustering":
            assert "target" not in pd.read_csv(a["input"] / "test.csv")
    sparse = next(c for c in left if c["name"] == "sparse_linear_target")
    assert (pd.read_csv(sparse["input"] / "test.csv").to_numpy() == 0).mean() > .8
    missing = next(c for c in left if c["name"] == "missing_cluster_axes")
    assert pd.read_csv(missing["input"] / "train.csv").isna().any().any()


def test_router_config_contents_affect_fingerprint(tmp_path):
    config = tmp_path / "policy.json"
    config.write_text('{"a":1}')
    before = source_fingerprint(config)
    config.write_text('{"a":2}')
    assert before != source_fingerprint(config)


def _row(strategy, value, kind="none"):
    return {"seed": 42, "task": "classification", "name": "case", "strategy": strategy,
            "partition": "development", "data_sha256": "same", "valid": True,
            "metrics": {"balanced_accuracy": value, "accuracy": 1-value}, "wall_seconds": 1,
            "fallback": kind != "none", "fallback_kind": kind}


def test_comparison_can_select_pair_from_three_and_count_recovery():
    report = {"config": {"source_sha256": "source"}, "datasets": [
        _row("baseline", .5), _row("auto", .6), _row("routed", .7, "completed_baseline")]}
    result = summarize(report, "auto", "routed")
    assert result["pairs"][0]["quality_change"] == pytest.approx(.1)
    assert result["aggregates"][0]["checkpoint_recoveries"] == 1
    assert result["aggregates"][0]["emergency_fallbacks"] == 0
    assert summarize(report)["pairs"][0]["auto"] == .6
    assert summarize(report, "baseline", "routed", "accuracy")["pairs"][0]["outcome"] == "loss"
    report["datasets"][2]["data_sha256"] = "different"
    with pytest.raises(ValueError, match="Different input"):
        summarize(report, "baseline", "routed")


@pytest.mark.parametrize("info,expected", [({}, "none"), ({"fallback": True}, "emergency"),
    ({"fallback": True, "checkpoint_recovered": True}, "completed_baseline")])
def test_fallback_provenance(info, expected):
    assert fallback_kind({"run_info": {"model": info}}) == expected


def test_evaluator_cli_advertises_router_flags():
    script = Path(__file__).resolve().parents[1] / "scripts" / "evaluate.py"
    result = subprocess.run([sys.executable, str(script), "--help"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "router" in result.stdout and "routed" in result.stdout and "--router-config" in result.stdout
