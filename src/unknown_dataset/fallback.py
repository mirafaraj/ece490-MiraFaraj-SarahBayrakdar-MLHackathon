"""Lightweight emergency outputs without importing sklearn or scipy."""

from __future__ import annotations

import numpy as np


def fallback_predictions(task, X_train, y_train=None, X_test=None,
                         feature_types=None, n_clusters=None,
                         time_budget=0.0, seed=42, reason="worker deadline"):
    """Emergency valid outputs after a failed or terminated modelling worker.

    These intentionally weak predictions preserve the submission contract, not
    model quality. The runner should report their use prominently.
    """
    if task == "classification":
        if y_train is None or X_test is None or not len(y_train):
            raise ValueError("Classification fallback requires training labels and test rows")
        labels, counts = np.unique(np.asarray(y_train), return_counts=True)
        predictions = np.repeat(labels[np.argmax(counts)], len(X_test))
        name = "EmergencyMajorityClass"
    elif task == "regression":
        if y_train is None or X_test is None or not len(y_train):
            raise ValueError("Regression fallback requires training labels and test rows")
        target = np.asarray(y_train, dtype=float)
        if not np.isfinite(target).all():
            raise ValueError("Regression labels must be finite numbers")
        # Divide before summing to avoid overflow on otherwise finite labels.
        predictions = np.full(len(X_test), np.sum(target / len(target)), dtype=float)
        name = "EmergencyMeanRegressor"
    elif task == "clustering":
        if not isinstance(n_clusters, (int, np.integer)) or not 1 <= n_clusters <= len(X_train):
            raise ValueError("n_clusters must be between one and the number of training rows")
        predictions = np.arange(len(X_train), dtype=np.int64) % n_clusters
        name = "EmergencyRoundRobinClusters"
    else:
        raise ValueError(f"Unsupported task: {task}")
    return predictions, {
        "model": name, "fallback": True, "fallback_reason": str(reason),
        "seed": int(seed), "training_rows": len(X_train),
        "time_budget_seconds": float(time_budget),
    }


