"""Fixed engineering safeguards for costly supervised inputs, not size guarantees."""
from __future__ import annotations


def effective_strategy(task, strategy):
    return ('baseline' if task == 'classification' else 'auto') if strategy == 'submission' else strategy


def needs_bounded_forest(task, rows, features, strategy):
    strategy = effective_strategy(task, strategy)
    costly = rows * features >= 2_500_000
    return costly and (task == 'regression' or
                       task == 'classification' and rows >= 50_000 and strategy in {'baseline', 'routed'})


def guard_histogram(rows, features):
    return features >= 256 and rows * features >= 2_500_000


def fitting_indices(labels, task, seed, limit=20_000):
    """Deterministic bounded sample; classification retains every observed class."""
    import numpy as np
    target = np.asarray(labels)
    if len(target) <= limit:
        return np.arange(len(target))
    rng = np.random.default_rng(seed)
    if task != 'classification':
        return np.sort(rng.choice(len(target), limit, replace=False))
    classes, inverse, counts = np.unique(target, return_inverse=True, return_counts=True)
    if len(classes) > limit:
        raise ValueError('Cannot preserve all classes within the fitting-row cap')
    ideal = counts * (limit / len(target))
    allocated = np.maximum(1, np.floor(ideal).astype(int))
    while allocated.sum() > limit:
        eligible = np.flatnonzero(allocated > 1)
        allocated[eligible[np.argmax((allocated-ideal)[eligible])]] -= 1
    while allocated.sum() < limit:
        eligible = np.flatnonzero(allocated < counts)
        allocated[eligible[np.argmax((ideal-allocated)[eligible])]] += 1
    return np.sort(np.concatenate([rng.choice(np.flatnonzero(inverse == i), int(n), replace=False)
                                   for i,n in enumerate(allocated)]))


def fit_predict_bounded(task, X_train, y_train, X_test, feature_types, seed,
                        time_budget, mode='forest', checkpoint_callback=None):
    import time
    import numpy as np
    from sklearn.preprocessing import LabelEncoder
    from threadpoolctl import threadpool_limits
    started = time.monotonic()
    indices = fitting_indices(y_train, task, seed)
    frame = X_train.iloc[indices]
    target = np.asarray(y_train)[indices]
    if mode not in {'forest', 'linear'}:
        raise ValueError('Unknown bounded recovery mode')
    with threadpool_limits(limits=1):
        if mode == 'forest':
            from .models import fit_predict
            prediction, info = fit_predict(task, frame, target, X_test, feature_types,
                seed=seed, time_budget=time_budget, strategy='baseline', _bounded_forest=True)
        else:
            from .supervised import _fit, _matrix
            encoder = LabelEncoder().fit(target) if task == 'classification' else None
            encoded = encoder.transform(target) if encoder is not None else target.astype(float)
            cleaner, processor, estimator, dense, width = _fit('linear', task, frame, encoded,
                                                               feature_types, seed, final=True)
            batch = max(1, 2_000_000 // max(1, width))
            pieces = [estimator.predict(_matrix(cleaner, processor, X_test.iloc[i:i+batch]))
                      for i in range(0, len(X_test), batch)]
            prediction = np.concatenate(pieces) if pieces else np.array([], dtype=target.dtype)
            if encoder is not None:
                prediction = encoder.inverse_transform(prediction.astype(int))
            info = dict(model=type(estimator).__name__, selected_candidate='linear',
                        training_rows=len(frame), input_features=X_train.shape[1])
    info.update(reliability_guard=True, fitting_rows=len(frame), full_training_rows=len(X_train),
                bounded_sample_seed=int(seed), reliability_mode=mode,
                fallback=mode == 'linear', model_seconds=time.monotonic()-started)
    if mode == 'linear':
        info['fallback_kind'] = 'bounded_linear'
    if checkpoint_callback is not None:
        checkpoint_callback(prediction, info)
    return prediction, info
