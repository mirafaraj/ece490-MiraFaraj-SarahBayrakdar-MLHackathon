"""Provisional file contract; adapt this module to the official kickoff schema."""
from __future__ import annotations

import csv
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class Dataset:
    task: str
    metadata: dict[str, Any]
    X_train: pd.DataFrame
    y_train: pd.Series | None
    X_test: pd.DataFrame | None
    feature_types: dict[str, str]
    n_clusters: int | None


def _read_table(path: Path, categorical_columns: list[str] | None = None) -> pd.DataFrame:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        header = next(csv.reader(handle), [])
    if not header or any(not name for name in header) or len(header) != len(set(header)):
        raise ValueError(f"{path.name} requires nonempty, unique column names")
    frame = pd.read_csv(path, dtype={name: str for name in categorical_columns or []})
    if frame.empty:
        raise ValueError(f"{path.name} must contain at least one row")
    return frame


def _numeric(series: pd.Series, name: str) -> pd.Series:
    try:
        result = pd.to_numeric(series, errors="raise")
    except (ValueError, TypeError) as exc:
        raise ValueError(f"Declared numeric column {name!r} contains nonnumeric values") from exc
    if np.isinf(result.to_numpy(dtype=float, na_value=np.nan)).any():
        raise ValueError(f"Numeric column {name!r} contains infinity")
    return result


def load_dataset(input_dir: str | Path) -> Dataset:
    directory = Path(input_dir)
    with (directory / "metadata.json").open(encoding="utf-8-sig") as handle:
        metadata = json.load(handle)
    if not isinstance(metadata, dict):
        raise ValueError("metadata.json must contain an object")
    task = metadata.get("task")
    if not isinstance(task, str) or task not in {"classification", "regression", "clustering"}:
        raise ValueError("task must be classification, regression, or clustering")
    declared = metadata.get("feature_types")
    categorical_columns = [name for name, kind in declared.items() if kind == "categorical"] if isinstance(declared, dict) else []
    target = metadata.get("target")
    # Classification labels are opaque CSV tokens: numeric inference can collapse
    # distinct classes such as "01" and "1" before the estimator sees them.
    lexical_columns = categorical_columns + ([target] if task == "classification" and isinstance(target, str) else [])
    train = _read_table(directory / "train.csv", lexical_columns)
    id_column = metadata.get("id_column")
    if id_column is not None:
        if not isinstance(id_column, str) or id_column not in train:
            raise ValueError("id_column must name a train.csv column")
        if train[id_column].isna().any() or train[id_column].duplicated().any():
            raise ValueError("Training IDs must be nonmissing and unique")
    y = None
    n_clusters = None
    excluded = [id_column] if id_column else []
    if task != "clustering":
        if not isinstance(target, str) or target not in train or target == id_column:
            raise ValueError("Supervised target must name a non-ID train.csv column")
        y = train[target].copy()
        if y.isna().any():
            raise ValueError("Training labels must not be missing")
        if task == "regression":
            y = _numeric(y, target)
        elif pd.api.types.is_numeric_dtype(y):
            _numeric(y, target)
        excluded.append(target)
    else:
        if target is not None:
            raise ValueError("Clustering metadata must not declare a target")
        n_clusters = metadata.get("n_clusters")
        if isinstance(n_clusters, bool) or not isinstance(n_clusters, int) or not 1 <= n_clusters <= len(train):
            raise ValueError("n_clusters must be an integer between 1 and the number of rows")
    X_train = train.drop(columns=excluded).copy()
    if X_train.shape[1] == 0:
        raise ValueError("At least one feature column is required")
    if declared is None:
        feature_types = {
            column: "numeric" if pd.api.types.is_numeric_dtype(X_train[column]) else "categorical"
            for column in X_train
        }
    else:
        if not isinstance(declared, dict) or set(declared) != set(X_train.columns):
            raise ValueError("feature_types must map every feature exactly once, excluding target and ID")
        if any(not isinstance(kind, str) or kind not in {"numeric", "categorical"} for kind in declared.values()):
            raise ValueError("feature_types values must be numeric or categorical")
        feature_types = dict(declared)
    X_test = None
    if task != "clustering":
        # Inferred training categoricals must also be read lexically in test.csv;
        # independent numeric inference can turn category "1" into "1.0".
        test_categories = [name for name, kind in feature_types.items() if kind == "categorical"]
        test = _read_table(directory / "test.csv", test_categories)
        if target in test:
            raise ValueError("test.csv must not contain the target")
        if id_column and id_column in test:
            if test[id_column].isna().any() or test[id_column].duplicated().any():
                raise ValueError("Test IDs must be nonmissing and unique")
            test = test.drop(columns=[id_column])
        if set(test.columns) != set(X_train.columns):
            raise ValueError("Train and test feature columns must match exactly")
        X_test = test.loc[:, X_train.columns].copy()
    for column, kind in feature_types.items():
        if kind == "numeric":
            X_train[column] = _numeric(X_train[column], column)
            if X_test is not None:
                X_test[column] = _numeric(X_test[column], column)
        else:
            # Normalize categorical values identically across separate CSV parses.
            X_train[column] = X_train[column].map(lambda value: str(value) if pd.notna(value) else np.nan)
            if X_test is not None:
                X_test[column] = X_test[column].map(lambda value: str(value) if pd.notna(value) else np.nan)
    return Dataset(task, metadata, X_train, y, X_test, feature_types, n_clusters)


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Not JSON serializable: {type(value).__name__}")


def write_outputs(output_dir: str | Path, predictions: Any, profile: dict, run_info: dict) -> None:
    """Validate values before writing; the runner validates task-specific row counts."""
    values = np.asarray(predictions)
    if values.ndim != 1 or not len(values):
        raise ValueError("Predictions must be a nonempty one-dimensional array")
    if pd.isna(values).any():
        raise ValueError("Predictions must not contain missing values")
    if any(isinstance(value, (float, np.floating)) and not np.isfinite(value) for value in values):
        raise ValueError("Predictions must not contain infinity")
    serialized = {
        "profile.json": json.dumps(profile, indent=2, allow_nan=False, default=_json_default),
        "run.json": json.dumps(run_info, indent=2, allow_nan=False, default=_json_default),
    }
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    names = ("profile.json", "run.json", "predictions.csv")
    # Finish every write in a staging directory before publishing any artifact.
    # Commit predictions last so their presence signals a complete result.
    with tempfile.TemporaryDirectory(prefix=".result-", dir=directory) as staging_dir:
        staging = Path(staging_dir)
        pd.DataFrame({"prediction": values}).to_csv(staging / "predictions.csv", index=False)
        for name, content in serialized.items():
            (staging / name).write_text(content + "\n", encoding="utf-8")
        try:
            for name in names:
                os.replace(staging / name, directory / name)
        except OSError:
            # A multi-file publish cannot be atomic. Remove known artifacts if
            # replacement fails partway, so callers never consume mixed runs.
            for name in names:
                try:
                    (directory / name).unlink(missing_ok=True)
                except OSError:
                    pass
            raise
