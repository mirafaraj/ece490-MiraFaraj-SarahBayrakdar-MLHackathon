"""Regression cases discovered by independent data/preprocessing review."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from unknown_dataset.contract import load_dataset, write_outputs
from unknown_dataset.preprocessing import TableCleaner, build_preprocessor
import unknown_dataset.contract as contract


@pytest.mark.parametrize("labels", [["01", "1"], ["0.25", "1.75"]])
def test_csv_class_labels_preserve_lexical_values(tmp_path, labels):
    (tmp_path / "metadata.json").write_text(json.dumps({"task": "classification", "target": "label"}))
    pd.DataFrame({"x": [0, 1], "label": labels}).to_csv(tmp_path / "train.csv", index=False)
    pd.DataFrame({"x": [0]}).to_csv(tmp_path / "test.csv", index=False)
    dataset = load_dataset(tmp_path)
    assert dataset.y_train.tolist() == labels
    write_outputs(tmp_path / "output", dataset.y_train.to_numpy(), {}, {})
    output = pd.read_csv(tmp_path / "output" / "predictions.csv", dtype=str)
    assert output["prediction"].tolist() == labels


def test_inferred_categorical_test_values_use_training_lexical_type(tmp_path):
    (tmp_path / "metadata.json").write_text(json.dumps({"task": "classification", "target": "label"}))
    (tmp_path / "train.csv").write_text("category,x,label\na,0,no\n1,1,yes\n")
    (tmp_path / "test.csv").write_text("category,x\n1,0\n,1\n")
    dataset = load_dataset(tmp_path)
    assert dataset.feature_types["category"] == "categorical"
    assert dataset.X_test["category"].iloc[0] == dataset.X_train["category"].iloc[1] == "1"
    assert pd.isna(dataset.X_test["category"].iloc[1])


def test_numeric_missingness_retains_predictive_signal():
    frame = pd.DataFrame({"x": [7.0, np.nan, 7.0, np.nan]})
    cleaner = TableCleaner({"x": "numeric"}).fit(frame)
    transformed = build_preprocessor(cleaner).fit_transform(cleaner.transform(frame))
    np.testing.assert_array_equal(transformed[:, 1], [0, 1, 0, 1])
    assert not np.array_equal(transformed[0], transformed[1])


def test_real_missing_token_category_stays_distinct_from_missing_values():
    frame = pd.DataFrame({"category": ["__MISSING__", np.nan, "value:__MISSING__"]})
    cleaner = TableCleaner({"category": "categorical"}).fit(frame)
    transformed = build_preprocessor(cleaner).fit_transform(cleaner.transform(frame)).toarray()
    assert len(np.unique(transformed, axis=0)) == 3


def test_staged_write_failure_preserves_previous_complete_result(tmp_path, monkeypatch):
    write_outputs(tmp_path, ["old"], {"old": True}, {"old": True})
    previous = {name: (tmp_path / name).read_bytes() for name in ("predictions.csv", "profile.json", "run.json")}
    original_write = Path.write_text

    def fail_staging(self, *args, **kwargs):
        if self.name == "run.json" and self.parent.name.startswith(".result-"):
            raise OSError("simulated staging failure")
        return original_write(self, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", fail_staging)
    with pytest.raises(OSError, match="staging failure"):
        write_outputs(tmp_path, ["new"], {"new": True}, {"new": True})
    assert {name: (tmp_path / name).read_bytes() for name in previous} == previous


def test_partial_publish_failure_removes_mixed_results(tmp_path, monkeypatch):
    write_outputs(tmp_path, ["old"], {"old": True}, {"old": True})
    original_replace = contract.os.replace

    def fail_publish(source, destination):
        if Path(destination).name == "run.json":
            raise OSError("simulated publish failure")
        return original_replace(source, destination)

    monkeypatch.setattr(contract.os, "replace", fail_publish)
    with pytest.raises(OSError, match="publish failure"):
        write_outputs(tmp_path, ["new"], {"new": True}, {"new": True})
    assert all(not (tmp_path / name).exists() for name in ("predictions.csv", "profile.json", "run.json"))
