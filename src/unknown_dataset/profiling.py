"""JSON-safe descriptive statistics on training data before model preparation."""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


def _finite(value: Any) -> float | None:
    number = float(value)
    return number if math.isfinite(number) else None


def _numeric_summary(series: pd.Series) -> dict:
    clean = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return {key: None for key in ("min", "max", "mean", "std", "median")}
    return {
        "min": _finite(clean.min()), "max": _finite(clean.max()),
        "mean": _finite(clean.mean()), "std": _finite(clean.std(ddof=0)),
        "median": _finite(clean.median()),
    }


def _distribution(series: pd.Series) -> list[dict]:
    # Records avoid collisions between JSON object keys such as numeric and string labels.
    return [
        {"value": value.item() if isinstance(value, np.generic) else value, "count": int(count)}
        for value, count in series.dropna().value_counts(sort=False).items()
    ]


def build_profile(X: pd.DataFrame, feature_types: dict[str, str], y: pd.Series | None = None) -> dict:
    if set(feature_types) != set(X.columns):
        raise ValueError("feature_types must cover exactly the profile feature columns")
    if any(not isinstance(kind, str) or kind not in {"numeric", "categorical"} for kind in feature_types.values()):
        raise ValueError("Unsupported feature type")
    rows = len(X)
    columns = {}
    constants = []
    duplicate_columns = []
    # Hash buckets avoid quadratic full-column comparisons for ordinary wide tables.
    buckets: dict[tuple, list[str]] = {}
    for column in X:
        series = X[column]
        missing = int(series.isna().sum())
        unique = int(series.nunique(dropna=True))
        constant = unique <= 1
        if constant:
            constants.append(column)
        entry = {
            "type": feature_types[column], "missing_count": missing,
            "missing_fraction": missing / rows if rows else 0.0,
            "unique_count": unique, "constant": constant,
            "all_missing": missing == rows,
        }
        if feature_types[column] == "numeric":
            entry["numeric_summary"] = _numeric_summary(series)
        columns[column] = entry
        # Series.equals deliberately requires matching dtype and matching missing positions.
        signature = (str(series.dtype), int(pd.util.hash_pandas_object(series, index=False).sum()))
        candidates = buckets.setdefault(signature, [])
        match = next((other for other in candidates if series.equals(X[other])), None)
        if match is not None:
            duplicate_columns.append({"column": column, "duplicate_of": match})
        else:
            candidates.append(column)
    profile = {
        "schema_version": "provisional-1", "row_count": rows, "feature_count": X.shape[1],
        "feature_type_counts": {kind: list(feature_types.values()).count(kind) for kind in ("numeric", "categorical")},
        "missing_count": int(X.isna().to_numpy().sum()),
        "constant_definition": "At most one distinct nonmissing value; includes all-missing columns.",
        "constant_columns": constants, "duplicate_columns": duplicate_columns, "columns": columns,
    }
    if y is not None:
        if len(y) != rows:
            raise ValueError("Target and feature row counts must match")
        numeric = pd.api.types.is_numeric_dtype(y)
        target = {"name": y.name, "missing_count": int(y.isna().sum()), "unique_count": int(y.nunique(dropna=True))}
        if numeric:
            target["numeric_summary"] = _numeric_summary(y)
        if not numeric or target["unique_count"] <= 50:
            target["distribution"] = _distribution(y)
        profile["target"] = target
    return profile
