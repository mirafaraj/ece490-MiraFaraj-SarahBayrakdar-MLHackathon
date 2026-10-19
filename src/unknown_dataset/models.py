"""Bounded, deterministic MVP models for the three supported task families."""

from __future__ import annotations

import time

import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.preprocessing import LabelEncoder
from threadpoolctl import threadpool_limits

from .fallback import fallback_predictions  # Re-export the public fallback API.
from .preprocessing import TableCleaner, build_preprocessor


def fit_predict(task, X_train, y_train=None, X_test=None, feature_types=None,
                n_clusters=None, time_budget=50.0, seed=42, strategy="baseline",
                route_config=None, checkpoint_callback=None):
    """Fit on training features and return ordered predictions plus run metadata.

    The caller owns the hard process deadline. The budget here bounds estimator
    complexity; it is not a promise that an arbitrary input finishes in time.
    """
    if strategy not in {"baseline", "auto", "routed"}:
        raise ValueError(f"Unknown strategy: {strategy}")
    if strategy == "routed":
        from .routed import fit_predict_routed

        return fit_predict_routed(task=task, X_train=X_train, y_train=y_train, X_test=X_test,
                                  feature_types=feature_types, n_clusters=n_clusters,
                                  time_budget=time_budget, seed=seed, route_config=route_config,
                                  checkpoint_callback=checkpoint_callback)
    if strategy == "auto":
        if task == "clustering":
            from .clustering import fit_predict_clustering as candidate
        else:
            from .supervised import fit_predict_supervised as candidate
        return candidate(task=task, X_train=X_train, y_train=y_train, X_test=X_test,
                         feature_types=feature_types, n_clusters=n_clusters,
                         time_budget=time_budget, seed=seed)
    started = time.monotonic()
    if task not in {"classification", "regression", "clustering"}:
        raise ValueError(f"Unsupported task: {task}")
    if len(X_train) == 0:
        raise ValueError("Training data must contain at least one row")
    if time_budget <= 0:
        raise ValueError("Model time budget must be positive")
    if task != "clustering" and (y_train is None or X_test is None):
        raise ValueError("Supervised tasks require labels and test features")
    if task != "clustering" and len(y_train) != len(X_train):
        raise ValueError("Training labels must align with training rows")
    if task == "clustering":
        if not isinstance(n_clusters, (int, np.integer)) or not 1 <= n_clusters <= len(X_train):
            raise ValueError("n_clusters must be between one and the number of training rows")

    # Cap native BLAS/OpenMP as well as sklearn's explicit worker count.
    with threadpool_limits(limits=1):
        cleaner = TableCleaner(feature_types).fit(X_train)
        clean_train = cleaner.transform(X_train)
        preprocessor = build_preprocessor(cleaner, scale_numeric=(task == "clustering"))
        train = preprocessor.fit_transform(clean_train)
        if task != "clustering" and len(X_test):
            test = preprocessor.transform(cleaner.transform(X_test))

        tree_count = 32 if time_budget < 15 else 96
        tree_parameters = dict(
            n_estimators=tree_count, max_depth=24, min_samples_leaf=1,
            n_jobs=1, random_state=seed,
        )
        if task == "classification":
            target = np.asarray(y_train)
            label_encoder = LabelEncoder()
            encoded_target = label_encoder.fit_transform(target)
            if len(label_encoder.classes_) == 1 or not cleaner.retained_columns_:
                model = DummyClassifier(strategy="prior")
            else:
                model = ExtraTreesClassifier(**tree_parameters, class_weight="balanced")
            model.fit(train, encoded_target)
            predictions = (label_encoder.inverse_transform(model.predict(test))
                           if len(X_test) else np.array([], dtype=target.dtype))
        elif task == "regression":
            target = np.asarray(y_train, dtype=float)
            if not np.isfinite(target).all():
                raise ValueError("Regression labels must be finite numbers")
            if not cleaner.retained_columns_:
                model = DummyRegressor(strategy="mean")
            else:
                model = ExtraTreesRegressor(**tree_parameters)
            model.fit(train, target)
            predictions = model.predict(test) if len(X_test) else np.array([], dtype=float)
        else:
            if len(X_train) <= 5000 and train.shape[1] <= 256:
                model = KMeans(n_clusters=n_clusters, n_init=5, max_iter=100, random_state=seed)
            else:
                model = MiniBatchKMeans(
                    n_clusters=n_clusters, n_init=3, max_iter=100,
                    batch_size=max(1024, 3 * n_clusters), random_state=seed,
                    reassignment_ratio=0.0,
                )
            predictions = model.fit_predict(train)
            # Coincident points can leave k-means clusters empty. The contract
            # requests k nonempty groups, so split ties deterministically.
            repaired_clusters = 0
            rng = np.random.default_rng(seed)
            counts = np.bincount(predictions, minlength=n_clusters)
            for missing_label in np.flatnonzero(counts == 0):
                donor = int(np.argmax(counts))
                row = int(rng.choice(np.flatnonzero(predictions == donor)))
                predictions[row] = missing_label
                counts[donor] -= 1
                counts[missing_label] += 1
                repaired_clusters += 1

        info = {
            "model": type(model).__name__, "seed": int(seed), "cpu_threads": 1,
            "training_rows": len(X_train), "input_features": X_train.shape[1],
            "retained_features": len(cleaner.retained_columns_),
            "transformed_features": int(train.shape[1]),
            "dropped_constant_columns": cleaner.constant_columns_,
            "dropped_duplicate_columns": cleaner.duplicate_columns_,
            "categorical_max_categories": 32,
            "time_budget_seconds": float(time_budget),
            "model_seconds": time.monotonic() - started,
        }
        if isinstance(model, (ExtraTreesClassifier, ExtraTreesRegressor)):
            info["n_estimators"] = tree_count
            info["max_depth"] = 24
        if task == "clustering":
            info["requested_clusters"] = int(n_clusters)
            info["observed_clusters"] = int(len(np.unique(predictions)))
            info["empty_clusters_repaired"] = repaired_clusters
        return np.asarray(predictions), info
