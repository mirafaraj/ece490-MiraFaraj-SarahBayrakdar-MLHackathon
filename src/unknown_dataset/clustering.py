"""Opt-in, bounded geometric clustering experiment; no evaluation labels used."""

from __future__ import annotations

import time

import numpy as np
from scipy import sparse
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.mixture import GaussianMixture
from threadpoolctl import threadpool_limits

from .preprocessing import TableCleaner, build_preprocessor


def _occupy_groups(labels, k, seed):
    labels = np.asarray(labels, dtype=np.int64).copy()
    counts = np.bincount(labels, minlength=k)
    rng = np.random.default_rng(seed)
    repaired = 0
    for missing in np.flatnonzero(counts == 0):
        donor = int(np.argmax(counts))
        row = int(rng.choice(np.flatnonzero(labels == donor)))
        labels[row] = missing
        counts[donor] -= 1
        counts[missing] += 1
        repaired += 1
    return labels, repaired


def fit_predict_clustering(task, X_train, y_train=None, X_test=None,
                           feature_types=None, n_clusters=None,
                           time_budget=50.0, seed=42, route_plan=None):
    """Compare covariance geometry only for small, numeric tables.

    The caller retains responsibility for a hard worker deadline. GMM model
    selection uses BIC on at most 2,000 training rows, never reference labels.
    Full covariance compares one generic and one baseline-centroid EM start.
    BIC measures density fit rather than agreement with semantic ground truth;
    consequently this strategy remains opt-in until independently validated.
    """
    started = time.monotonic()
    if task != "clustering":
        raise ValueError("This candidate supports clustering only")
    if not len(X_train):
        raise ValueError("Training data must contain at least one row")
    if (isinstance(n_clusters, (bool, np.bool_)) or
            not isinstance(n_clusters, (int, np.integer)) or
            not 1 <= n_clusters <= len(X_train)):
        raise ValueError("n_clusters must be between one and the number of rows")
    if not np.isfinite(time_budget) or time_budget <= 0:
        raise ValueError("Model time budget must be finite and positive")
    with threadpool_limits(limits=1):
        cleaner = TableCleaner(feature_types).fit(X_train)
        matrix = build_preprocessor(cleaner, scale_numeric=True).fit_transform(
            cleaner.transform(X_train))
        if len(X_train) <= 5000 and matrix.shape[1] <= 256:
            model = KMeans(n_clusters=n_clusters, n_init=5, max_iter=100, random_state=seed)
        else:
            model = MiniBatchKMeans(n_clusters=n_clusters, n_init=3, max_iter=100,
                batch_size=max(1024, 3 * n_clusters), random_state=seed,
                reassignment_ratio=0.0)
        predictions = model.fit_predict(matrix)
        selected = type(model).__name__
        selection = "baseline geometry; covariance comparison not eligible"
        bics = {}
        # Bound dense conversion, covariance dimension, EM iterations, and fit rows.
        eligible = (not cleaner.categorical_columns_ and
                    1 <= len(cleaner.numeric_columns_) <= 16 and
                    n_clusters <= 12 and max(40, 10 * n_clusters) <= len(X_train) <= 20000 and
                    time_budget - (time.monotonic() - started) >= 12)
        if route_plan is not None:
            eligible = eligible and "gmm" in route_plan.candidates
        if eligible:
            dense = matrix.toarray() if sparse.issparse(matrix) else np.asarray(matrix)
            rng = np.random.default_rng(seed)
            sample = dense[rng.choice(len(dense), min(2000, len(dense)), replace=False)]
            models = {}
            # Spend the same two full-covariance restarts on complementary
            # basins: a generic start and the already fitted baseline centers.
            # Sample order can otherwise send both generic starts to the same
            # inferior density optimum, discarding a useful completed result.
            starts = (("spherical", None, 2), ("full", None, 1),
                      ("full", model.cluster_centers_, 1))
            for covariance, means, restarts in starts:
                if time_budget - (time.monotonic() - started) < 4:
                    break
                candidate = GaussianMixture(n_components=n_clusters,
                    covariance_type=covariance, max_iter=80, n_init=restarts,
                    means_init=means,
                    reg_covar=1e-5, random_state=seed)
                try:
                    candidate.fit(sample)
                    score = float(candidate.bic(sample))
                    if (candidate.converged_ and np.isfinite(score) and
                            score < bics.get(covariance, float("inf"))):
                        bics[covariance] = score
                        models[covariance] = candidate
                except (ValueError, np.linalg.LinAlgError):
                    continue
            if set(bics) == {"spherical", "full"}:
                margin = max(10.0, 0.05 * abs(bics["spherical"]))
                selection = "full covariance BIC gain below conservative margin"
                if bics["full"] < bics["spherical"] - margin:
                    predictions = models["full"].predict(dense)
                    selected = "GaussianMixtureFull"
                    selection = "full covariance improves BIC by more than margin"
        predictions, repaired = _occupy_groups(predictions, n_clusters, seed)
    return predictions, {
        "model": selected, "strategy": "geometry", "experimental": True,
        "selection": selection, "bic": bics,
        "covariance_initialization": "generic plus baseline centroids",
        "cpu_threads": 1, "seed": int(seed),
        "training_rows": len(X_train), "input_features": X_train.shape[1],
        "transformed_features": int(matrix.shape[1]),
        "requested_clusters": int(n_clusters),
        "observed_clusters": int(len(np.unique(predictions))),
        "empty_clusters_repaired": repaired,
        "model_seconds": time.monotonic() - started,
        "time_budget_seconds": float(time_budget),
        **({"routing": route_plan.to_dict()} if route_plan is not None else {}),
    }
