"""Deterministic, training-only summaries and explainable task route plans."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np

from .policies import validate_config
from .preprocessing import TableCleaner


@dataclass
class RoutingSummary:
    task: str
    rows: int
    input_features: int
    retained_features: int
    numeric_features: int
    categorical_features: int
    estimated_encoded_width: int
    missing_fraction: float
    constant_columns: list
    duplicate_columns: list
    class_counts: list[int]
    categorical_cardinalities: dict

    def to_dict(self):
        return asdict(self)


@dataclass
class RoutePlan:
    task: str
    policy_version: str
    metric: str
    candidates: list[str]
    excluded: dict[str, str]
    validation_mode: str
    selection_sample_rows: int
    selection_fraction: float
    final_tree_count: int
    dense_cells_limit: int
    class_weight: str | None
    selection_budget_fraction: float
    reserve_seconds: float
    decisions: list[str]
    summary: dict
    limits: dict
    near_tie_tolerance: float
    experimental: dict

    def to_dict(self):
        return asdict(self)


def load_router_config(path=None):
    """Read repository defaults; installed packages fall back to identical built-ins.

    An explicit path must exist. Partial configurations merge into versioned defaults.
    """
    from .lightweight_config import load_router_config as load_lightweight

    return load_lightweight(path)


def summarize_training(task, X_train, y_train=None, feature_types=None):
    if task not in {"classification", "regression", "clustering"}:
        raise ValueError(f"Unsupported task: {task}")
    if not len(X_train):
        raise ValueError("Training data must contain rows")
    cleaner = TableCleaner(feature_types).fit(X_train)
    clean = cleaner.transform(X_train)
    # Numeric imputation may add one missingness indicator per column. Categories
    # include an imputation category and are bounded by the encoder's cap of 32.
    width = 2 * len(cleaner.numeric_columns_)
    width += sum(min(32, int(clean[c].nunique(dropna=True)) + 1) for c in cleaner.categorical_columns_)
    counts = []
    if task in {"classification", "regression"}:
        if y_train is None or len(y_train) != len(X_train):
            raise ValueError("Aligned training labels are required")
        if task == "classification":
            _, values = np.unique(np.asarray(y_train), return_counts=True)
            counts = [int(c) for c in values]
    missing = float(clean.isna().to_numpy().mean()) if clean.shape[1] else 0.0
    return RoutingSummary(task, len(X_train), X_train.shape[1], len(cleaner.retained_columns_),
                          len(cleaner.numeric_columns_), len(cleaner.categorical_columns_),
                          max(1, width), missing, cleaner.constant_columns_, cleaner.duplicate_columns_, counts,
                          {str(c): int(clean[c].nunique(dropna=False)) for c in cleaner.categorical_columns_})


def make_route_plan(summary, time_budget=50.0, n_clusters=None, config=None):
    if isinstance(time_budget, bool) or not math.isfinite(time_budget) or time_budget <= 0:
        raise ValueError("Time budget must be finite and positive")
    configuration = load_router_config() if config is None else validate_config(config)
    limits = configuration["limits"]
    task = summary.task
    if task not in configuration["metrics"]:
        raise ValueError("Unsupported task")
    if task == "clustering" and (isinstance(n_clusters, bool) or not isinstance(n_clusters, (int, np.integer)) or not 1 <= n_clusters <= summary.rows):
        raise ValueError("Clustering requires 1 <= n_clusters <= training rows")
    metric = configuration["metrics"][task]
    baseline = "kmeans" if task == "clustering" else "extra_trees"
    candidates, excluded = [baseline], {}
    decisions = ["Eligibility uses training summary and configured resources only", "Baseline is always eligible"]
    reason = None
    if time_budget < limits["min_selection_seconds"] + limits["reserve_seconds"]:
        reason = "insufficient time after output/refit reserve"
    elif summary.rows > limits["max_training_rows"] or summary.input_features > limits["max_input_features"]:
        reason = "input exceeds optional candidate shape limits"
    elif summary.rows < limits["min_validation_rows"] or summary.retained_features == 0:
        reason = "too few rows or no informative features"
    elif task == "classification" and (len(summary.class_counts) < 2 or min(summary.class_counts) < 3 or len(summary.class_counts) > min(limits["max_classes"], summary.rows // 5)):
        reason = "single, rare, or excessive classes cannot support reliable stratification"
    optional = ["gmm"] if task == "clustering" else ["linear", "histogram"]
    experimental = configuration["experimental"]
    if task == "classification" and experimental["categorical_enabled"]:
        # Make the cheap experiment eligible before expensive optional estimators;
        # keep the overall configured portfolio cap, including the baseline.
        optional.insert(0, "categorical")
    dense_cells = summary.rows * summary.estimated_encoded_width
    for name in optional:
        skip = reason
        if skip is None and name == "categorical":
            eligible = [v for v in summary.categorical_cardinalities.values() if 1 < v <= experimental["categorical_max_cardinality"]]
            if not eligible:
                skip = "no nonconstant categorical feature within cardinality limit"
            elif min(len(eligible), experimental["categorical_max_features"]) * max(eligible) * max(1, len(summary.class_counts)) * 16 > limits["working_memory_bytes"]:
                skip = "categorical count allocation exceeds memory limit"
        if skip is None and name in {"histogram", "gmm"} and (dense_cells > limits["dense_cells_limit"] or dense_cells * 8 * 6 > limits["working_memory_bytes"]):
            skip = "dense allocation estimate exceeds cell or working memory limit"
        if skip is None and name == "linear" and summary.rows * min(summary.estimated_encoded_width, summary.retained_features * 2) * 16 > limits["working_memory_bytes"]:
            skip = "sparse working allocation estimate exceeds memory limit"
        if skip is None and name == "gmm" and (summary.categorical_features or summary.rows > limits["gmm_max_rows"] or summary.numeric_features > limits["gmm_max_features"] or n_clusters > limits["gmm_max_clusters"] or n_clusters < 2):
            skip = "mixture requires bounded numeric-only rows, features, and clusters"
        if skip is None and len(candidates) >= limits["max_candidates"]:
            skip = "candidate count cap"
        if skip:
            excluded[name] = skip
        else:
            candidates.append(name)
    validation = ("bic" if task == "clustering" else "stratified_holdout" if task == "classification" else "holdout") if len(candidates) > 1 else "none"
    if task == "classification":
        decisions.append("Class weights follow configured objective; no threshold tuning")
    return RoutePlan(task, configuration["policy_version"], metric, candidates, excluded, validation,
                     min(summary.rows, limits["selection_sample_rows"]), limits["selection_fraction"],
                     limits["low_budget_tree_count"] if time_budget < 15 else limits["final_tree_count"],
                     limits["dense_cells_limit"], "balanced" if task == "classification" and metric == "balanced_accuracy" else None,
                     limits["selection_budget_fraction"], limits["reserve_seconds"], decisions,
                     summary.to_dict(), limits, float(configuration["selection"]["near_tie_tolerance"]), experimental)
