import json

import numpy as np
import pandas as pd
import pytest

from unknown_dataset.contract import load_dataset, write_outputs


@pytest.fixture
def input_dir(tmp_path):
    pd.DataFrame({"id": [10, 11, 12], "value": [2, None, 5], "category": ["a", "b", "a"],
                  "label": ["yes", "no", "yes"]}).to_csv(tmp_path / "train.csv", index=False)
    pd.DataFrame({"category": ["new", "a"], "id": [21, 20], "value": [8, 1]}).to_csv(tmp_path / "test.csv", index=False)
    (tmp_path / "metadata.json").write_text(json.dumps({"task": "classification", "target": "label",
        "id_column": "id", "feature_types": {"value": "numeric", "category": "categorical"}}))
    return tmp_path


def test_loader_excludes_labels_and_ids_and_preserves_row_order(input_dir):
    data = load_dataset(input_dir)
    assert list(data.X_train) == ["value", "category"]
    assert list(data.X_test) == ["value", "category"]
    assert data.X_test["value"].tolist() == [8, 1]
    assert data.X_test["category"].tolist() == ["new", "a"]
    assert data.y_train.tolist() == ["yes", "no", "yes"]
    assert data.X_train["value"].isna().sum() == 1


@pytest.mark.parametrize("fault", ["missing_target", "test_label", "feature_mismatch", "partial_types", "duplicate_id", "infinite_numeric", "invalid_task"])
def test_invalid_contract_is_rejected(input_dir, fault):
    metadata_path = input_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    if fault == "missing_target":
        metadata["target"] = "absent"
    elif fault == "partial_types":
        metadata["feature_types"] = {"value": "numeric"}
    elif fault == "invalid_task":
        metadata["task"] = "forecasting"
    else:
        name = "train.csv" if fault == "duplicate_id" else "test.csv"
        frame = pd.read_csv(input_dir / name)
        if fault == "test_label":
            frame["label"] = "yes"
        elif fault == "feature_mismatch":
            frame = frame.drop(columns="value")
        elif fault == "duplicate_id":
            frame["id"] = 1
        elif fault == "infinite_numeric":
            frame["value"] = np.inf
        frame.to_csv(input_dir / name, index=False)
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError):
        load_dataset(input_dir)


def test_invalid_json_is_rejected(input_dir):
    (input_dir / "metadata.json").write_text("{ broken")
    with pytest.raises(json.JSONDecodeError):
        load_dataset(input_dir)


@pytest.mark.parametrize("clusters", [0, 4, True, 1.5])
def test_invalid_cluster_count_is_rejected(input_dir, clusters):
    (input_dir / "metadata.json").write_text(json.dumps({"task": "clustering", "n_clusters": clusters}))
    with pytest.raises(ValueError):
        load_dataset(input_dir)


@pytest.mark.parametrize("predictions", [[np.nan], [np.inf], [[1, 2]], []])
def test_output_rejects_invalid_predictions(tmp_path, predictions):
    with pytest.raises(ValueError):
        write_outputs(tmp_path / "out", predictions, {}, {})


def test_output_preserves_label_order_and_serializes_numpy(tmp_path):
    write_outputs(tmp_path, np.array(["z", "a", "z"]), {"count": np.int64(3)}, {"ok": True})
    assert pd.read_csv(tmp_path / "predictions.csv")["prediction"].tolist() == ["z", "a", "z"]
    assert json.loads((tmp_path / "profile.json").read_text()) == {"count": 3}
