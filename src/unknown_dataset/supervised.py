"""Bounded holdout selection with preprocessing fitted inside each split."""
from __future__ import annotations

import time
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, recall_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from threadpoolctl import threadpool_limits

from .preprocessing import TableCleaner, build_preprocessor


_DENSE_CELLS = 2_000_000
_OBJECTIVE_WEIGHT = object()


@dataclass(frozen=True)
class ClassificationControls:
    """Explicit development controls; omitted controls retain the incumbent path.

    These do not change repository configuration or the assessment entry point.
    A shared elapsed-time cap covers all candidates and all inner holdouts.
    """
    objective: str = "balanced_accuracy"
    weighting: str = "objective"
    holdouts: int = 1

    def __post_init__(self):
        if self.objective not in {"accuracy", "balanced_accuracy", "macro_f1"}:
            raise ValueError("Unsupported experimental classification objective")
        if self.weighting not in {"objective", "balanced", "unweighted", "both"}:
            raise ValueError("Unsupported experimental classification weighting")
        if type(self.holdouts) is not int or self.holdouts not in {1, 2}:
            raise ValueError("Experimental holdouts must be 1 or 2")


def classification_metrics(target, prediction, labels):
    """Offline/inner-fold diagnostics with a fixed training-derived class order."""
    encoder = LabelEncoder().fit(np.asarray(labels))
    target, prediction = encoder.transform(target), encoder.transform(prediction)
    return {
        "accuracy": float(accuracy_score(target, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(target, prediction)),
        "macro_f1": float(f1_score(target, prediction, labels=np.arange(len(encoder.classes_)), average="macro", zero_division=0)),
        "class_recall": recall_score(target, prediction, labels=np.arange(len(encoder.classes_)), average=None, zero_division=0).tolist(),
        "classes": encoder.classes_.tolist(),
    }


def _matrix(cleaner, preprocessor, frame, dense=False, dense_cells_limit=None):
    dense_cells_limit = _DENSE_CELLS if dense_cells_limit is None else dense_cells_limit
    result = preprocessor.transform(cleaner.transform(frame))
    if dense:
        if result.shape[0] * result.shape[1] > dense_cells_limit:
            raise ValueError("Dense candidate exceeds the matrix cell cap")
        if hasattr(result, "toarray"):
            result = result.toarray()
    return result


def _fit(name, task, frame, labels, feature_types, seed, final=False, final_tree_count=96,
         class_weight="balanced", dense_cells_limit=None):
    dense_cells_limit = _DENSE_CELLS if dense_cells_limit is None else dense_cells_limit
    cleaner = TableCleaner(feature_types).fit(frame)
    preprocessor = build_preprocessor(cleaner, scale_numeric=(name == "linear"))
    matrix = preprocessor.fit_transform(cleaner.transform(frame))
    dense = name == "histogram"
    if dense:
        if matrix.shape[0] * matrix.shape[1] > dense_cells_limit:
            raise ValueError("Dense candidate exceeds the matrix cell cap")
        if hasattr(matrix, "toarray"):
            matrix = matrix.toarray()
    classification = task == "classification"
    if not cleaner.retained_columns_ or (classification and len(np.unique(labels)) == 1):
        estimator = DummyClassifier(strategy="prior") if classification else DummyRegressor(strategy="mean")
    elif name == "extra_trees":
        kwargs = dict(n_estimators=final_tree_count if final else 40, max_depth=24, n_jobs=1, random_state=seed)
        estimator = ExtraTreesClassifier(class_weight=class_weight, **kwargs) if classification else ExtraTreesRegressor(**kwargs)
    elif name == "linear":
        estimator = (LogisticRegression(class_weight=class_weight, max_iter=200, random_state=seed)
                     if classification else Ridge(alpha=1.0, solver="lsqr"))
    elif name == "histogram":
        kwargs = dict(max_iter=100 if final else 60, max_leaf_nodes=15, max_depth=8,
                      min_samples_leaf=max(5, min(20, len(frame) // 20)), early_stopping=False,
                      random_state=seed)
        estimator = (HistGradientBoostingClassifier(class_weight=class_weight, **kwargs)
                     if classification else HistGradientBoostingRegressor(**kwargs))
    else:
        raise ValueError(f"Unknown supervised candidate {name}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        estimator.fit(matrix, labels)
    return cleaner, preprocessor, estimator, dense, matrix.shape[1]


def fit_predict_supervised(task, X_train, y_train=None, X_test=None, feature_types=None,
                           n_clusters=None, time_budget=50, seed=42, route_plan=None,
                           completed_baseline=None, classification_controls=None,
                           execution_deadline=None):
    """Choose using train-only holdout, then refit once on all supplied labels.

    The caller must enforce a process deadline. Checks between candidate fits
    cannot interrupt a running sklearn estimator.
    """
    if classification_controls is None and route_plan is not None and task == "classification":
        experimental = route_plan.experimental
        weighting = experimental.get("classification_weighting", "objective")
        holdouts = experimental.get("classification_holdouts", 1)
        objective = experimental.get("classification_objective", "route")
        if weighting != "objective" or holdouts != 1 or objective != "route":
            classification_controls = ClassificationControls(
                route_plan.metric if objective == "route" else objective, weighting, holdouts)
    if classification_controls is not None:
        if task != "classification" or route_plan is None:
            raise ValueError("Classification controls require a classification route plan")
        if route_plan.experimental.get("refit_admission", False):
            raise ValueError("refit_admission supports the original selector only")
        return _fit_predict_classification_experiment(X_train, y_train, X_test, feature_types,
            time_budget, seed, route_plan, classification_controls, completed_baseline)
    if route_plan is not None:
        return _fit_predict_routed_supervised(task, X_train, y_train, X_test, feature_types,
                                             time_budget, seed, route_plan, completed_baseline,
                                             execution_deadline)
    started = time.monotonic()
    if task not in {"classification", "regression"}:
        raise ValueError("Supervised selection requires classification or regression")
    if y_train is None or X_test is None or len(y_train) != len(X_train) or not len(X_train):
        raise ValueError("Aligned nonempty training data, labels, and test features are required")
    if time_budget <= 0:
        raise ValueError("Model time budget must be positive")
    encoder = None
    if task == "classification":
        encoder = LabelEncoder().fit(np.asarray(y_train))
        target = encoder.transform(np.asarray(y_train))
        _, counts = np.unique(target, return_counts=True)
        rare = len(counts) < 2 or counts.min() < 3 or len(counts) > min(30, len(target) // 5)
    else:
        target = np.asarray(y_train, dtype=float)
        if not np.isfinite(target).all():
            raise ValueError("Regression labels must be finite numbers")
        rare = False
    selected = "extra_trees"
    scores = {}
    errors = {}
    skip = None
    if time_budget < 12:
        skip = "low_time_budget"
    elif len(target) < 60 or rare:
        skip = "tiny_data_or_rare_classes"
    elif len(target) > 100_000 or X_train.shape[1] > 1000:
        skip = "large_input"
    with threadpool_limits(limits=1):
        final_tree_count = 32 if time_budget < 15 else 96
        if skip is None:
            indices = np.arange(len(target))
            if len(indices) > 4000:
                indices, _ = train_test_split(indices, train_size=4000, random_state=seed,
                                              stratify=target if task == "classification" else None)
            try:
                fitting, validation = train_test_split(indices, test_size=0.25, random_state=seed,
                    stratify=target[indices] if task == "classification" else None)
            except ValueError:
                # Downsampling can leave only one example of a rare class.
                skip = "sample_too_small_for_stratification"
        if skip is None:
            best = -np.inf
            for name in ("extra_trees", "linear", "histogram"):
                if time.monotonic() - started > time_budget * 0.4:
                    errors[name] = "selection budget exhausted; reserved remaining time for refit"
                    break
                try:
                    fitted = _fit(name, task, X_train.iloc[fitting], target[fitting], feature_types, seed)
                    cleaner, processor, estimator, dense, _ = fitted
                    prediction = estimator.predict(_matrix(cleaner, processor, X_train.iloc[validation], dense))
                    score = (balanced_accuracy_score(target[validation], prediction) if task == "classification"
                             else -float(np.sqrt(mean_squared_error(target[validation], prediction))))
                    scores[name] = float(score)
                    if score > best + 1e-8:
                        best, selected = score, name
                except (ValueError, MemoryError) as exc:
                    errors[name] = str(exc)
        refit_fallback = None
        try:
            fitted = _fit(selected, task, X_train, target, feature_types, seed, final=True,
                          final_tree_count=final_tree_count)
        except (ValueError, MemoryError) as exc:
            if selected == "extra_trees":
                raise
            refit_fallback = str(exc)
            selected = "extra_trees"
            fitted = _fit(selected, task, X_train, target, feature_types, seed, final=True,
                          final_tree_count=final_tree_count)
        cleaner, processor, estimator, dense, width = fitted
        if len(X_test):
            # Dense prediction is batched independently of evaluation row count.
            batch = max(1, _DENSE_CELLS // max(1, width)) if dense else len(X_test)
            predictions = np.concatenate([
                estimator.predict(_matrix(cleaner, processor, X_test.iloc[offset:offset + batch], dense))
                for offset in range(0, len(X_test), batch)
            ])
            if encoder is not None:
                predictions = encoder.inverse_transform(predictions.astype(int))
        else:
            predictions = np.array([], dtype=np.asarray(y_train).dtype)
    return predictions, {
        "model": type(estimator).__name__, "strategy": "auto", "selected_candidate": selected,
        "selection_metric": "balanced_accuracy" if task == "classification" else "negative_rmse",
        "validation_scores": scores, "candidate_errors": errors, "selection_skipped": skip,
        "refit_fallback_reason": refit_fallback, "fallback": False,
        "training_rows": len(X_train), "input_features": X_train.shape[1],
        "retained_features": len(cleaner.retained_columns_), "transformed_features": int(width),
        "dropped_constant_columns": cleaner.constant_columns_, "dropped_duplicate_columns": cleaner.duplicate_columns_,
        "seed": int(seed), "cpu_threads": 1, "time_budget_seconds": float(time_budget),
        "model_seconds": time.monotonic() - started,
        "n_estimators": getattr(estimator, "n_estimators", None),
    }


def _validate_supervised(task, X_train, y_train, X_test, time_budget):
    if task not in {"classification", "regression"}:
        raise ValueError("Supervised selection requires classification or regression")
    if y_train is None or X_test is None or len(y_train) != len(X_train) or not len(X_train):
        raise ValueError("Aligned nonempty training features, labels and test features required")
    if not np.isfinite(time_budget) or time_budget <= 0:
        raise ValueError("Model budget must be finite and positive")


def metric_score(metric, target, prediction):
    """Return named metric in its natural direction (losses are positive)."""
    if metric in {"accuracy", "balanced_accuracy"}:
        # Class identities may be nonintegral numbers. Encode jointly solely for
        # metric calculation so sklearn does not interpret them as regression.
        encoder = LabelEncoder().fit(np.concatenate([np.asarray(target), np.asarray(prediction)]))
        target, prediction = encoder.transform(target), encoder.transform(prediction)
    if metric == "accuracy":
        return float(accuracy_score(target, prediction))
    if metric == "balanced_accuracy":
        return float(balanced_accuracy_score(target, prediction))
    if metric == "mae":
        return float(mean_absolute_error(target, prediction))
    if metric == "rmse":
        return float(np.sqrt(mean_squared_error(target, prediction)))
    raise ValueError(f"Unknown supervised objective: {metric}")


def select_candidate(scores, metric, tolerance=0.0):
    """Prefer simpler candidates only within an explicitly configured metric gap.

    Compare to the best absolute score, rather than sequential pairwise ties,
    to prevent chained near-ties accepting a materially worse candidate.
    """
    if not scores:
        return "extra_trees"
    higher = metric in {"accuracy", "balanced_accuracy"}
    best = max(scores.values()) if higher else min(scores.values())
    eligible = [name for name, value in scores.items()
                if (best - value if higher else value - best) <= tolerance]
    order = {"categorical": 0, "linear": 1, "extra_trees": 2, "histogram": 3}
    return min(eligible, key=lambda name: (order[name], name))


def _categorical_features(frame, feature_types, plan):
    experimental = plan.experimental
    features = []
    for name in frame.columns:
        declared = (feature_types or {}).get(name)
        is_category = declared == "categorical" or (declared is None and not pd.api.types.is_numeric_dtype(frame[name]))
        if is_category and 1 < frame[name].nunique(dropna=False) <= experimental["categorical_max_cardinality"]:
            features.append(name)
    # Feature order is input schema order, never association with validation labels.
    return features[:experimental["categorical_max_features"]]


def fit_predict_fixed_candidate(name, task, X_train, y_train, X_test, feature_types=None,
                                seed=42, metric=None, final=True, route_plan=None,
                                categorical_feature=None, categorical_alpha=10.0,
                                class_weight=_OBJECTIVE_WEIGHT):
    """Fit a fixed candidate for offline paired evaluation or routed final refit.

    Evaluation references are deliberately absent from this API. A categorical
    feature defaults to first fitting-data-eligible column; no feature search is
    performed by this fixed-candidate entry point.
    """
    from .routing import make_route_plan, summarize_training
    from .categorical import SmoothedCategoricalClassifier
    started = time.monotonic()
    _validate_supervised(task, X_train, y_train, X_test, 50)
    if route_plan is None:
        route_plan = make_route_plan(summarize_training(task, X_train, y_train, feature_types),
                                    config={"metrics": {task: metric}} if metric else None)
    metric = metric or route_plan.metric
    if metric not in ({"accuracy", "balanced_accuracy"} if task == "classification" else {"rmse", "mae"}):
        raise ValueError("Objective does not match task")
    encoder = LabelEncoder().fit(np.asarray(y_train)) if task == "classification" else None
    target = encoder.transform(np.asarray(y_train)) if encoder is not None else np.asarray(y_train, dtype=float)
    if encoder is None and not np.isfinite(target).all():
        raise ValueError("Regression labels must be finite")
    weight = "balanced" if task == "classification" and metric == "balanced_accuracy" else None
    if class_weight is not _OBJECTIVE_WEIGHT:
        if task != "classification" or class_weight not in {None, "balanced"}:
            raise ValueError("Class-weight override requires classification and balanced or None")
        weight = class_weight
    fitting_started = time.monotonic()
    with threadpool_limits(limits=1):
        if name == "categorical":
            if task != "classification":
                raise ValueError("Categorical count candidate is classification-only")
            eligible = _categorical_features(X_train, feature_types, route_plan)
            if categorical_feature is None:
                if not eligible:
                    raise ValueError("No eligible categorical feature")
                categorical_feature = eligible[0]
            if categorical_feature not in eligible:
                raise ValueError("Categorical feature is not eligible in fitting data")
            estimator = SmoothedCategoricalClassifier(categorical_feature, categorical_alpha, weight,
                route_plan.experimental["categorical_max_cardinality"]).fit(X_train, target)
            cleaner = None
            width = len(estimator.counts_)
        else:
            cleaner, processor, estimator, dense, width = _fit(name, task, X_train, target,
                feature_types, seed, final=final, final_tree_count=route_plan.final_tree_count,
                class_weight=weight, dense_cells_limit=route_plan.dense_cells_limit)
        fitting_seconds = time.monotonic() - fitting_started
        prediction_started = time.monotonic()
        # Bound transformation and probability allocation for every model family.
        prediction_width = max(1, len(estimator.classes_)) if name == "categorical" else max(1, width)
        batch = max(1, route_plan.dense_cells_limit // prediction_width)
        pieces = []
        for offset in range(0, len(X_test), batch):
            frame = X_test.iloc[offset:offset + batch]
            if name == "categorical":
                values = estimator.predict(frame)
            else:
                values = estimator.predict(_matrix(cleaner, processor, frame, dense,
                                                    route_plan.dense_cells_limit))
            pieces.append(values)
        predictions = np.concatenate(pieces) if pieces else np.array([], dtype=target.dtype)
        if encoder is not None:
            predictions = encoder.inverse_transform(predictions.astype(int))
    return predictions, {
        "model": type(estimator).__name__, "selected_candidate": name, "strategy": "routed",
        "selection_metric": metric, "class_weight": weight, "fallback": False,
        "training_rows": len(X_train), "input_features": X_train.shape[1],
        "retained_features": len(cleaner.retained_columns_) if cleaner is not None else 1,
        "transformed_features": int(width), "seed": int(seed), "cpu_threads": 1,
        "dropped_constant_columns": cleaner.constant_columns_ if cleaner is not None else [],
        "dropped_duplicate_columns": cleaner.duplicate_columns_ if cleaner is not None else [],
        "n_estimators": getattr(estimator, "n_estimators", None),
        "candidate_parameters": {"feature": str(categorical_feature), "alpha": float(categorical_alpha)} if name == "categorical" else {},
        "fit_seconds": fitting_seconds, "prediction_seconds": time.monotonic() - prediction_started,
        "model_seconds": time.monotonic() - started, "routing": route_plan.to_dict(),
    }


def fit_baseline_supervised(task, X_train, y_train=None, X_test=None, feature_types=None,
                            n_clusters=None, time_budget=50.0, seed=42, route_plan=None):
    """Completed checkpoint baseline with the plan's objective and tree count."""
    from .routing import make_route_plan, summarize_training
    _validate_supervised(task, X_train, y_train, X_test, time_budget)
    if route_plan is None:
        route_plan = make_route_plan(summarize_training(task, X_train, y_train, feature_types), time_budget)
    if route_plan.task != task:
        raise ValueError("Route plan task does not match supplied task")
    prediction, info = fit_predict_fixed_candidate("extra_trees", task, X_train, y_train, X_test,
        feature_types, seed, route_plan=route_plan)
    info.update(validation_scores={}, selection_skipped="completed_baseline", time_budget_seconds=float(time_budget))
    return prediction, info


def _fit_predict_routed_supervised(task, X_train, y_train, X_test, feature_types,
                                   time_budget, seed, route_plan, completed_baseline=None,
                                   execution_deadline=None):
    _validate_supervised(task, X_train, y_train, X_test, time_budget)
    if route_plan.task != task:
        raise ValueError("Route plan task does not match supplied task")
    if completed_baseline is not None:
        if not isinstance(completed_baseline, tuple) or len(completed_baseline) != 2:
            raise ValueError("Completed baseline must be a predictions/info tuple")
        values, metadata = completed_baseline
        values = np.asarray(values)
        if values.ndim != 1 or len(values) != len(X_test) or not isinstance(metadata, dict):
            raise ValueError("Completed baseline has invalid shape or metadata")
        if task == "classification" and not np.isin(values, np.asarray(y_train)).all():
            raise ValueError("Completed baseline includes unknown classes")
        if task == "regression" and not np.isfinite(values.astype(float)).all():
            raise ValueError("Completed baseline includes nonfinite predictions")
    def reuse_baseline():
        values, metadata = completed_baseline
        info = dict(metadata)
        info.update(baseline_reused=True, baseline_fit_seconds=info.get("fit_seconds", 0.0),
                    fit_seconds=0.0, prediction_seconds=0.0)
        return np.asarray(values).copy(), info
    if route_plan.validation_mode == "none" or len(route_plan.candidates) <= 1:
        if completed_baseline is not None:
            return reuse_baseline()
        return fit_baseline_supervised(task, X_train, y_train, X_test, feature_types,
                                      time_budget=time_budget, seed=seed, route_plan=route_plan)
    started = time.monotonic()
    scores, timings, errors, parameters = {}, {}, {}, {}
    indices = np.arange(len(X_train))
    target = np.asarray(y_train)
    skip = None
    try:
        if len(indices) > route_plan.selection_sample_rows:
            indices, _ = train_test_split(indices, train_size=route_plan.selection_sample_rows,
                random_state=seed, stratify=target if task == "classification" else None)
        fitting, validation = train_test_split(indices, test_size=route_plan.selection_fraction,
            random_state=seed, stratify=target[indices] if task == "classification" else None)
    except ValueError:
        skip = "sample_too_small_for_stratification"
    selection_limit = max(0.0, min(time_budget * route_plan.selection_budget_fraction,
                                 time_budget - route_plan.reserve_seconds))
    if skip is None:
        for name in route_plan.candidates:
            if time.monotonic() - started >= selection_limit:
                errors[name] = "selection budget exhausted; reserve retained for final fit"
                continue
            candidate_started = time.monotonic()
            try:
                variants = [(None, 10.0)]
                if name == "categorical":
                    variants = [(feature, alpha) for feature in _categorical_features(
                        X_train.iloc[fitting], feature_types, route_plan)
                        for alpha in route_plan.experimental["categorical_alphas"]]
                    if not variants:
                        raise ValueError("No eligible categorical feature in fitting fold")
                for feature, alpha in variants:
                    if time.monotonic() - started >= selection_limit:
                        break
                    prediction, _ = fit_predict_fixed_candidate(name, task,
                        X_train.iloc[fitting], target[fitting], X_train.iloc[validation],
                        feature_types, seed, final=False, route_plan=route_plan,
                        categorical_feature=feature, categorical_alpha=alpha)
                    value = metric_score(route_plan.metric, target[validation], prediction)
                    if not np.isfinite(value):
                        raise ValueError("Candidate metric is nonfinite")
                    better = name not in scores or (value > scores[name] if task == "classification" else value < scores[name])
                    if better:
                        scores[name] = value
                        parameters[name] = {"categorical_feature": feature, "categorical_alpha": alpha}
            except (ValueError, MemoryError) as exc:
                errors[name] = str(exc)
            timings[name] = time.monotonic() - candidate_started
    selection_seconds = time.monotonic() - started
    selected = select_candidate(scores, route_plan.metric, route_plan.near_tie_tolerance)
    admission = None
    if task == "classification" and route_plan.experimental.get("refit_admission", False):
        if completed_baseline is None:
            admission = {"skipped": "no completed baseline available"}
        else:
            deadline = started + time_budget if execution_deadline is None else execution_deadline
            if isinstance(deadline, bool) or not np.isfinite(deadline):
                raise ValueError("Execution deadline must be finite")
            selected, admission = admit_refit_candidate(scores, route_plan.metric, timings,
                deadline - time.monotonic() - route_plan.reserve_seconds,
                route_plan.near_tie_tolerance)
            admission["deadline_source"] = "local_budget" if execution_deadline is None else "parent_absolute"
    refit_fallback = None
    try:
        if selected == "extra_trees" and completed_baseline is not None:
            prediction, info = reuse_baseline()
        else:
            prediction, info = fit_predict_fixed_candidate(selected, task, X_train, y_train, X_test,
                feature_types, seed, route_plan=route_plan, **parameters.get(selected, {}))
    except (ValueError, MemoryError) as exc:
        if selected == "extra_trees":
            raise
        refit_fallback = str(exc)
        if completed_baseline is not None:
            prediction, info = reuse_baseline()
        else:
            prediction, info = fit_baseline_supervised(task, X_train, y_train, X_test, feature_types,
                time_budget=time_budget, seed=seed, route_plan=route_plan)
    info.update(validation_scores=scores, candidate_seconds=timings, candidate_errors=errors,
                selection_seconds=selection_seconds, selection_skipped=skip,
                refit_fallback_reason=refit_fallback, time_budget_seconds=float(time_budget),
                model_seconds=time.monotonic() - started,
                selection_direction="maximize" if task == "classification" else "minimize",
                near_tie_tolerance=route_plan.near_tie_tolerance)
    if admission is not None:
        info["refit_admission"] = admission
    if route_plan.experimental.get("runtime_instrumentation", False):
        info["phase_seconds"] = {"selection": selection_seconds,
            "refit": info.get("fit_seconds", 0.0),
            "prediction": info.get("prediction_seconds", 0.0)}
    return prediction, info


def admit_refit_candidate(scores, metric, timings, headroom, tolerance=0.0):
    """Training-only refit eligibility, with an already completed tree baseline.

    The fixed 1.5 multiplier was declared from first-batch fit-cost ratios, not
    optimized against reference labels. It is a heuristic, not an execution
    guarantee; the parent still enforces the hard deadline and owns recovery.
    """
    available = max(0.0, float(headroom))
    estimates, rejected, eligible = {}, {}, {}
    for name, score in scores.items():
        elapsed = timings.get(name)
        estimate = (0.0 if name == "extra_trees" else
                    1.5 * float(elapsed) if elapsed is not None and np.isfinite(elapsed) and elapsed >= 0
                    else None)
        estimates[name] = estimate
        if estimate is not None and estimate <= available:
            eligible[name] = score
        else:
            rejected[name] = "estimated refit exceeds headroom" if estimate is not None else "missing finite fit cost"
    selected = select_candidate(eligible, metric, tolerance)
    return selected, {"multiplier": 1.5, "headroom_seconds": available,
        "estimated_refit_seconds": estimates, "eligible_candidates": list(eligible),
        "rejected_candidates": rejected,
        "selected_before": select_candidate(scores, metric, tolerance), "selected_after": selected}


def _fit_predict_classification_experiment(X_train, y_train, X_test, feature_types,
                                           time_budget, seed, route_plan, controls,
                                           completed_baseline=None):
    """Optional factorial selector; no evaluation labels or features enter selection."""
    if not isinstance(controls, ClassificationControls):
        raise ValueError("classification_controls must be ClassificationControls")
    _validate_supervised("classification", X_train, y_train, X_test, time_budget)
    if route_plan.task != "classification":
        raise ValueError("Route plan task does not match classification")
    started = time.monotonic()
    target = np.asarray(y_train)
    labels = np.unique(target)
    if completed_baseline is not None:
        if not isinstance(completed_baseline, tuple) or len(completed_baseline) != 2:
            raise ValueError("Completed baseline must be a predictions/info tuple")
        values, metadata = completed_baseline
        values = np.asarray(values)
        if values.ndim != 1 or len(values) != len(X_test) or not isinstance(metadata, dict) or not np.isin(values, target).all():
            raise ValueError("Completed baseline has invalid shape, classes, or metadata")
    weights = (["balanced", None] if controls.weighting == "both" else
               ["balanced"] if controls.weighting == "balanced" else
               [None] if controls.weighting == "unweighted" else
               ["balanced" if controls.objective == "balanced_accuracy" else None])
    candidates = [name for name in route_plan.candidates if name in {"extra_trees", "linear", "histogram"}]
    variants = [(name, weight) for name in candidates for weight in weights]
    indices = np.arange(len(target))
    folds, scores, details, errors, timings = [], {}, {}, {}, {}
    skip = "route_validation_disabled" if route_plan.validation_mode == "none" else None
    if skip is None:
        try:
            if len(indices) > route_plan.selection_sample_rows:
                indices, _ = train_test_split(indices, train_size=route_plan.selection_sample_rows,
                    random_state=seed, stratify=target)
            for fold in range(controls.holdouts):
                folds.append(train_test_split(indices, test_size=route_plan.selection_fraction,
                    random_state=seed + fold, stratify=target[indices]))
        except ValueError:
            skip = "sample_too_small_for_stratification"
    selection_limit = max(0.0, min(time_budget * route_plan.selection_budget_fraction,
                                   time_budget - route_plan.reserve_seconds))
    with threadpool_limits(limits=1):
        if skip is None:
            for name, weight in variants:
                key = name + ":" + (weight or "unweighted")
                candidate_started = time.monotonic()
                records = []
                try:
                    for fold, (fitting, validation) in enumerate(folds):
                        if time.monotonic() - started >= selection_limit:
                            raise ValueError("selection budget exhausted; incomplete candidate excluded")
                        prediction, _ = fit_predict_fixed_candidate(name, "classification",
                            X_train.iloc[fitting], target[fitting], X_train.iloc[validation], feature_types,
                            seed, final=False, route_plan=route_plan, class_weight=weight)
                        record = classification_metrics(target[validation], prediction, labels)
                        record.update(fold=fold, fitting_rows=len(fitting), validation_rows=len(validation))
                        records.append(record)
                    scores[key] = float(np.mean([record[controls.objective] for record in records]))
                except (ValueError, MemoryError) as exc:
                    errors[key] = str(exc)
                details[key] = records
                timings[key] = time.monotonic() - candidate_started
        selection_seconds = time.monotonic() - started
        # Preserve incumbent simple-candidate preference on exact ties, with
        # weighting ties resolved in declared order. No near-tie expansion.
        order = {"linear": 0, "extra_trees": 1, "histogram": 2}
        selected = min(scores, key=lambda key: (-scores[key], order[key.split(":")[0]],
            list(scores).index(key))) if scores else "extra_trees:" + (weights[0] or "unweighted")
        name, weight_name = selected.split(":")
        weight = None if weight_name == "unweighted" else weight_name
        def final_fit(candidate):
            if (candidate == "extra_trees" and completed_baseline is not None
                    and metadata.get("class_weight", _OBJECTIVE_WEIGHT) == weight):
                reused = dict(metadata)
                reused.update(baseline_reused=True, baseline_fit_seconds=reused.get("fit_seconds", 0.0),
                              fit_seconds=0.0, prediction_seconds=0.0)
                return values.copy(), reused
            return fit_predict_fixed_candidate(candidate, "classification", X_train, y_train,
                X_test, feature_types, seed, route_plan=route_plan, class_weight=weight)
        refit_fallback = None
        try:
            prediction, info = final_fit(name)
        except (ValueError, MemoryError) as exc:
            if name == "extra_trees" and completed_baseline is None:
                raise
            refit_fallback = str(exc)
            if completed_baseline is not None:
                # Recovery must not launch another fit after a failed refit.
                # A safe completed checkpoint takes precedence over the planned
                # experimental weight, which is recorded as actually recovered.
                prediction, info = values.copy(), dict(metadata)
                info.update(baseline_reused=True, baseline_fit_seconds=info.get("fit_seconds", 0.0),
                            fit_seconds=0.0, prediction_seconds=0.0)
                weight_name = info.get("class_weight") or "unweighted"
            else:
                prediction, info = final_fit("extra_trees")
            selected = "extra_trees:" + weight_name
    info.update(experimental_classification=True, selection_metric=controls.objective,
        classification_weighting=controls.weighting, selection_holdouts=controls.holdouts,
        selected_variant=selected, validation_scores=scores, validation_fold_metrics=details,
        candidate_errors=errors, candidate_seconds=timings, selection_seconds=selection_seconds,
        selection_skipped=skip, refit_fallback_reason=refit_fallback,
        time_budget_seconds=float(time_budget), model_seconds=time.monotonic() - started,
        selection_direction="maximize", near_tie_tolerance=0.0,
        selection_time_limit_seconds=selection_limit)
    if route_plan.experimental.get("runtime_instrumentation", False):
        info["phase_seconds"] = {"selection": selection_seconds,
            "refit": info["fit_seconds"], "prediction": info["prediction_seconds"]}
    return prediction, info
