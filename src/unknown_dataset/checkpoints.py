"""Validated, atomic completed-result checkpoints in the private worker directory."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


def validate_predictions(dataset, predictions):
    """Return a validated 1-D vector for both final outputs and recovery."""
    # Preserve opaque class token types instead of coercing a mixed vector such
    # as [1, "01"] into strings and silently accepting integer 1 as class "1".
    values = np.asarray(predictions, dtype=object if dataset.task == "classification" else None)
    expected = len(dataset.X_train if dataset.task == "clustering" else dataset.X_test)
    if values.ndim != 1 or len(values) != expected:
        raise ValueError(f"Expected {expected} one-dimensional predictions")
    if pd.isna(values).any():
        raise ValueError("Predictions must not contain missing values")
    if dataset.task == "classification":
        # Membership depends on distinct labels, not their repeated row counts.
        # Preserve scalar types: object predictions must not become strings.
        if not np.isin(values, pd.unique(np.asarray(dataset.y_train))).all():
            raise ValueError("Predicted class is absent from the training labels")
        if any(isinstance(value, (float, np.floating)) and not np.isfinite(value) for value in values):
            raise ValueError("Predictions must be finite")
    elif dataset.task == "regression":
        if values.dtype.kind not in "iuf" or not np.isfinite(values).all():
            raise ValueError("Regression predictions must be finite numbers")
    elif dataset.task == "clustering":
        if values.dtype.kind not in "iu" or set(values.tolist()) != set(range(dataset.n_clusters)):
            raise ValueError("Cluster labels must occupy every integer group from 0 to k-1")
    else:
        raise ValueError(f"Unsupported task: {dataset.task}")
    return values


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Not JSON serializable: {type(value).__name__}")


def write_checkpoint(path, dataset, predictions, model_info):
    """Publish only complete valid results; a failed update preserves the old one."""
    values = validate_predictions(dataset, predictions)
    if not isinstance(model_info, dict):
        raise ValueError("Checkpoint model_info must be an object")
    payload = json.dumps({"version": 1, "predictions": values.tolist(), "model_info": model_info},
                         allow_nan=False, default=_json_default)
    path = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=".checkpoint-", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def load_checkpoint(path, dataset):
    """Read only private local JSON checkpoints; reject partial or invalid contents."""
    with Path(path).open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict) or payload.get("version") != 1:
        raise ValueError("Unsupported checkpoint payload")
    if not isinstance(payload.get("model_info"), dict):
        raise ValueError("Invalid checkpoint model_info")
    # Enforce JSON-safe metadata as well as predictions before it reaches outputs.
    json.dumps(payload["model_info"], allow_nan=False)
    payload["predictions"] = validate_predictions(dataset, payload.get("predictions"))
    return {"predictions": payload["predictions"], "model_info": payload["model_info"]}
