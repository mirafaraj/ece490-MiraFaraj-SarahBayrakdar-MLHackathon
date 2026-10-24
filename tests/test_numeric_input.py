"""Declared numeric features are tolerant; targets and infinities stay strict."""
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from unknown_dataset.contract import load_dataset
from unknown_dataset.preprocessing import TableCleaner, build_preprocessor
from unknown_dataset.profiling import build_profile


def make_input(folder, task="classification"):
    metadata = {"task": task, "target": "target", "id_column": "id",
                "feature_types": {"x": "numeric", "empty": "numeric", "kind": "categorical"}}
    (folder / "metadata.json").write_text(json.dumps(metadata))
    (folder / "train.csv").write_text(
        "id,x,empty,kind,target\nrow-a,1, ,01,01\nrow-b,  ,bad,1,1\n"
        "row-c,oops,,  ,01\nrow-d,,\t,nan-like,1\nrow-e,2.5,?,a,01\nrow-f,-3, ,b,1\n")
    (folder / "test.csv").write_text("id,x,empty,kind\ntest-a,\t,?,01\ntest-b,bad, ,1\ntest-c,4,,  \n")
    return folder


def test_counts_profiles_and_untouched_lexical_columns(tmp_path):
    make_input(tmp_path)
    originals = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    data = load_dataset(tmp_path)
    assert data.numeric_feature_coercions == {
        "train": {"x": {"already_missing_count": 1, "coerced_whitespace_count": 1, "coerced_invalid_count": 1},
                  "empty": {"already_missing_count": 1, "coerced_whitespace_count": 3, "coerced_invalid_count": 2}},
        "test": {"x": {"already_missing_count": 0, "coerced_whitespace_count": 1, "coerced_invalid_count": 1},
                 "empty": {"already_missing_count": 1, "coerced_whitespace_count": 1, "coerced_invalid_count": 1}}}
    assert data.X_train.x.dropna().tolist() == [1., 2.5, -3.]
    assert data.X_test.x.dropna().tolist() == [4.]
    assert data.X_train.kind.tolist() == ["01", "1", "  ", "nan-like", "a", "b"]
    assert data.y_train.tolist() == ["01", "1", "01", "1", "01", "1"]
    assert "id" not in data.X_train and "id" not in data.X_test
    assert originals == {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    profile = build_profile(data.X_train, data.feature_types, data.y_train)
    assert profile["columns"]["x"]["missing_count"] == 3
    assert profile["columns"]["empty"]["missing_count"] == 6
    assert profile["columns"]["empty"]["all_missing"]
    cleaner = TableCleaner(data.feature_types).fit(data.X_train)
    processor = build_preprocessor(cleaner).fit(cleaner.transform(data.X_train))
    assert processor.transform(cleaner.transform(data.X_test)).shape[0] == 3


@pytest.mark.parametrize("split", ["train", "test"])
@pytest.mark.parametrize("token", ["inf", "-inf", "Infinity"])
def test_infinite_numeric_features_still_fail(tmp_path, split, token):
    make_input(tmp_path)
    p = tmp_path / (split + ".csv")
    text = p.read_text().replace("row-a,1,", f"row-a,{token},") if split == "train" else p.read_text().replace("test-c,4,", f"test-c,{token},")
    p.write_text(text)
    with pytest.raises(ValueError, match="infinity"):
        load_dataset(tmp_path)


@pytest.mark.parametrize("target", ["", "bad", "  ", "inf", "-inf"])
def test_invalid_regression_targets_are_not_coerced(tmp_path, target):
    make_input(tmp_path, "regression")
    frame = pd.read_csv(tmp_path / "train.csv", dtype=str)
    frame["target"] = [target, "1", "2", "3", "4", "5"]
    frame.to_csv(tmp_path / "train.csv", index=False)
    with pytest.raises(ValueError):
        load_dataset(tmp_path)


def test_missing_classification_target_still_fails(tmp_path):
    make_input(tmp_path)
    p = tmp_path / "train.csv"
    p.write_text(p.read_text().replace("row-a,1, ,01,01", "row-a,1, ,01,"))
    with pytest.raises(ValueError, match="labels must not be missing"):
        load_dataset(tmp_path)


def test_inferred_columns_do_not_enable_numeric_coercion(tmp_path):
    make_input(tmp_path)
    path = tmp_path / "metadata.json"
    meta = json.loads(path.read_text()); del meta["feature_types"]
    path.write_text(json.dumps(meta))
    data = load_dataset(tmp_path)
    assert data.feature_types["x"] == "categorical"
    assert "oops" in data.X_train.x.tolist()
    assert data.numeric_feature_coercions == {}


def test_cli_publishes_counts_and_full_profile(tmp_path):
    source = tmp_path / "input"; source.mkdir(); make_input(source)
    output = tmp_path / "output"
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    done = subprocess.run([sys.executable, str(Path(__file__).resolve().parents[1] / "main.py"),
        "--input", str(source), "--output", str(output), "--time-budget", "20"],
        env=env, capture_output=True, text=True, timeout=30)
    assert done.returncode == 0, done.stderr
    assert json.loads((output / "run.json").read_text())["numeric_feature_coercions"] == load_dataset(source).numeric_feature_coercions
    profile = json.loads((output / "profile.json").read_text())
    assert profile["columns"]["empty"]["all_missing"] and profile["columns"]["x"]["missing_count"] == 3
    assert set(pd.read_csv(output / "predictions.csv", dtype=str).prediction) <= {"01", "1"}
