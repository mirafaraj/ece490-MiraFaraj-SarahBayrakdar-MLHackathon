"""Versioned, provisional objectives and resource rules; no dataset-specific rules."""
from __future__ import annotations

from copy import deepcopy
import math

DEFAULT_CONFIG = {
    "policy_version": "1.0",
    "metrics": {"classification": "balanced_accuracy", "regression": "rmse", "clustering": "bic"},
    "selection": {"near_tie_tolerance": 0.0},
    "experimental": {"categorical_enabled": False, "categorical_max_features": 4,
                     "categorical_max_cardinality": 512, "categorical_alphas": [1.0, 10.0]},
    "limits": {
        "min_selection_seconds": 12.0, "reserve_seconds": 5.0,
        "selection_fraction": 0.25, "selection_budget_fraction": 0.4,
        "selection_sample_rows": 4000, "max_training_rows": 100000,
        "max_input_features": 1000, "max_candidates": 3,
        "dense_cells_limit": 2000000, "working_memory_bytes": 268435456,
        "min_validation_rows": 60, "max_classes": 30,
        "final_tree_count": 96, "low_budget_tree_count": 32,
        "gmm_max_rows": 5000, "gmm_max_features": 12, "gmm_max_clusters": 8,
    },
}


def validate_config(config=None):
    """Merge strict partial overrides; reject typos and invalid objectives early."""
    result = deepcopy(DEFAULT_CONFIG)
    if config is None:
        return result
    if not isinstance(config, dict):
        raise ValueError("Router configuration must be an object")
    if set(config) - set(result):
        raise ValueError("Unknown router configuration keys")
    for key, value in config.items():
        if key == "policy_version":
            if value != DEFAULT_CONFIG[key]:
                raise ValueError("Unsupported router policy version")
            continue
        if not isinstance(value, dict) or set(value) - set(result[key]):
            raise ValueError(f"Unknown or malformed router {key}")
        result[key].update(value)
    allowed = {"classification": {"accuracy", "balanced_accuracy"},
               "regression": {"rmse", "mae"}, "clustering": {"bic"}}
    for task, metric in result["metrics"].items():
        if not isinstance(metric, str) or metric not in allowed[task]:
            raise ValueError(f"Unsupported {task} objective: {metric}")
    for key, value in result["limits"].items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"Router limit {key} must be finite and positive")
        if isinstance(DEFAULT_CONFIG["limits"][key], int) and not isinstance(value, int):
            raise ValueError(f"Router limit {key} must be an integer")
    for key in ("selection_fraction", "selection_budget_fraction"):
        if not 0 < result["limits"][key] < 1:
            raise ValueError(f"Router limit {key} must be between zero and one")
    if result["limits"]["max_candidates"] > 4:
        raise ValueError("Router supports at most four candidates")
    tolerance = result["selection"]["near_tie_tolerance"]
    if isinstance(tolerance, bool) or not isinstance(tolerance, (float, int)) or not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("Near-tie tolerance must be finite and nonnegative")
    experimental = result["experimental"]
    if not isinstance(experimental["categorical_enabled"], bool):
        raise ValueError("categorical_enabled must be boolean")
    for key in ("categorical_max_features", "categorical_max_cardinality"):
        if isinstance(experimental[key], bool) or not isinstance(experimental[key], int) or experimental[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    alphas = experimental["categorical_alphas"]
    if not isinstance(alphas, list) or not 1 <= len(alphas) <= 4 or any(isinstance(a, bool) or not isinstance(a, (int, float)) or not math.isfinite(a) or a <= 0 for a in alphas):
        raise ValueError("categorical_alphas must contain one to four finite positive values")
    return result
