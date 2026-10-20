"""Frozen small-leaf development study; outer references never select models."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from unknown_dataset.supervised import _fit, _matrix

SEEDS = (98003, 98017, 98029)
MECHANISMS = ('linear_control', 'localized_interaction', 'weak_noisy')
OBJECTIVES = ('accuracy', 'balanced_accuracy', 'macro_f1')
VARIANTS = ('extra_trees', 'linear', 'histogram', 'histogram_leaf5')

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()

def case_data(seed, mechanism, classes):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(600, 6))
    category = rng.integers(0, 4, 600)
    other = rng.integers(0, 3, 600)
    if mechanism == 'linear_control':
        score = 1.5*x[:, 0] - x[:, 1] + .7*(category == 0) + rng.normal(0, .1, 600)
    elif mechanism == 'localized_interaction':
        score = x[:, 0]*x[:, 1] + .7*((category == 0) & (x[:, 2] > 0)) + rng.normal(0, .1, 600)
    elif mechanism == 'weak_noisy':
        score = .15*x[:, 0] + rng.normal(0, 1, 600)
    else:
        raise ValueError(mechanism)
    minority = np.argsort(score, kind='stable')[-30:]
    y = np.full(600, 'majority', dtype=object)
    if classes == 2:
        y[minority] = 'rare_a'
    elif classes == 3:
        order = minority[np.argsort(x[minority, 3], kind='stable')]
        y[order[:15]], y[order[15:]] = 'rare_a', 'rare_b'
    else:
        raise ValueError(classes)
    x[rng.random(x.shape) < .03] = np.nan
    frame = pd.DataFrame(x, columns=[f'x{i}' for i in range(6)])
    frame['category'] = [f'c{i}' for i in category]
    frame['other'] = [f'o{i}' for i in other]
    features = {column: ('numeric' if column.startswith('x') else 'categorical') for column in frame}
    return frame, y, features

def metrics(truth, prediction, probabilities, classes):
    lookup = {label: i for i, label in enumerate(classes)}
    target = np.array([lookup[x] for x in truth])
    pred = np.array([lookup[x] for x in prediction])
    matrix = np.zeros((len(classes), len(classes)), dtype=int)
    np.add.at(matrix, (target, pred), 1)
    supports = matrix.sum(axis=1)
    recalls = np.divide(matrix.diagonal(), supports, out=np.zeros(len(classes)), where=supports > 0)
    denom = supports + matrix.sum(axis=0)
    f1 = np.divide(2*matrix.diagonal(), denom, out=np.zeros(len(classes)), where=denom > 0)
    probabilities = np.asarray(probabilities)
    onehot = np.eye(len(classes))[target]
    return {'accuracy': float(np.mean(target == pred)), 'balanced_accuracy': float(recalls[supports > 0].mean()),
        'macro_f1': float(f1.mean()), 'class_recall': recalls.tolist(), 'class_support': supports.tolist(),
        'minority_recall': float(recalls[1:].mean()),
        'log_loss': float(-np.log(np.clip(probabilities[np.arange(len(target)), target], 1e-15, 1)).mean()),
        'brier_multiclass': float(np.square(probabilities-onehot).sum(axis=1).mean())}

def record(truth, prediction, probability, classes, seconds):
    prediction = np.asarray(prediction).tolist()
    return {'truth': np.asarray(truth).tolist(), 'prediction': prediction, 'probability': np.asarray(probability).tolist(),
        'prediction_sha256': digest(prediction), 'metrics': metrics(truth, prediction, probability, classes), 'seconds': seconds}

def select(inner, weighting, objective, replacement=False, cap=20.0):
    names = ('extra_trees', 'linear', 'histogram_leaf5' if replacement else 'histogram')
    scores, excluded, elapsed = {}, {}, 0.
    for name in names:
        key = name + ':' + weighting
        if elapsed >= cap:
            excluded[key] = 'shared selection cap exhausted'
            continue
        row = inner[key]
        elapsed += row['seconds']
        scores[key] = row['metrics'][objective]
    preference = {'linear': 0, 'extra_trees': 1, 'histogram': 2, 'histogram_leaf5': 2}
    selected = min(scores, key=lambda key: (-scores[key], preference[key.split(':')[0]])) if scores else 'extra_trees:'+weighting
    return {'selected': selected, 'validation_scores': scores, 'excluded': excluded, 'selection_seconds': elapsed}

def bootstrap(inner, classes, seed):
    rng = np.random.default_rng(seed + 42000)
    truth = np.asarray(next(iter(inner.values()))['truth'])
    groups = [np.flatnonzero(truth == label) for label in classes]
    output = {key: {objective: [] for objective in OBJECTIVES} for key in inner}
    frequencies = {f'{weight}:{objective}:{pool}': {} for weight in ('balanced', 'unweighted')
        for objective in OBJECTIVES for pool in ('current', 'leaf5')}
    for _ in range(200):
        index = np.concatenate([rng.choice(group, len(group), replace=True) for group in groups])
        resampled = {}
        for key, row in inner.items():
            value = metrics(truth[index], np.asarray(row['prediction'])[index], np.asarray(row['probability'])[index], classes)
            resampled[key] = {'seconds': row['seconds'], 'metrics': value}
            for objective in OBJECTIVES:
                output[key][objective].append(value[objective])
        for policy, counts in frequencies.items():
            weight, objective, pool = policy.split(':')
            selected = select(resampled, weight, objective, pool == 'leaf5')['selected']
            counts[selected] = counts.get(selected, 0) + 1
    intervals = {key: {objective: {'p025': float(np.quantile(values, .025)), 'p975': float(np.quantile(values, .975)), 'std': float(np.std(values))}
        for objective, values in row.items()} for key, row in output.items()}
    return {'metric_intervals': intervals, 'selection_counts': frequencies}

def run_case(seed, mechanism, class_count):
    frame, labels, features = case_data(seed, mechanism, class_count)
    outer_train, outer_test = train_test_split(np.arange(len(frame)), test_size=.25, random_state=seed, stratify=labels)
    inner_train, inner_test = train_test_split(outer_train, test_size=.25, random_state=seed, stratify=labels[outer_train])
    classes = sorted(set(labels[outer_train]))
    inner, outer = {}, {}
    for name in VARIANTS:
        for weighting in ('balanced', 'unweighted'):
            key = name + ':' + weighting
            for final, fit_indices, test_indices, destination in ((False, inner_train, inner_test, inner), (True, outer_train, outer_test, outer)):
                started = time.monotonic()
                kwargs = {'histogram_min_samples_leaf': 5} if name == 'histogram_leaf5' else {}
                with threadpool_limits(limits=1):
                    cleaner, processor, estimator, dense, _ = _fit('histogram' if name == 'histogram_leaf5' else name,
                        'classification', frame.iloc[fit_indices], labels[fit_indices], features, seed,
                        final=final, class_weight='balanced' if weighting == 'balanced' else None, **kwargs)
                    matrix = _matrix(cleaner, processor, frame.iloc[test_indices], dense)
                    prediction = estimator.predict(matrix)
                    probability = estimator.predict_proba(matrix)
                if list(estimator.classes_) != classes:
                    raise AssertionError('Class/probability alignment changed')
                destination[key] = record(labels[test_indices], prediction, probability, classes, time.monotonic()-started)
    policies = {}
    for weighting in ('balanced', 'unweighted'):
        for objective in OBJECTIVES:
            for replacement in (False, True):
                policy = select(inner, weighting, objective, replacement)
                chosen = policy['selected']
                pool = [name+':'+weighting for name in ('extra_trees', 'linear', 'histogram_leaf5' if replacement else 'histogram')]
                policy.update(outer_metrics=outer[chosen]['metrics'], selected_prediction_sha256=outer[chosen]['prediction_sha256'],
                    retrospective_gap=max(outer[key]['metrics'][objective] for key in pool)-outer[chosen]['metrics'][objective],
                    baseline_delta={metric: outer[chosen]['metrics'][metric]-outer['extra_trees:balanced']['metrics'][metric] for metric in OBJECTIVES})
                policies[f'{weighting}:{objective}:{"leaf5" if replacement else "current"}'] = policy
    serial = frame.astype(object).where(pd.notna(frame), None).to_dict(orient='split')
    input_data = {'frame': serial, 'labels': labels.tolist(), 'feature_types': features}
    return {'seed': seed, 'mechanism': mechanism, 'class_count': class_count, 'inputs': input_data,
        'input_sha256': digest(input_data), 'splits': {name: indices.tolist() for name, indices in
        [('outer_train', outer_train), ('outer_test', outer_test), ('inner_train', inner_train), ('inner_test', inner_test)]},
        'classes': classes, 'inner': inner, 'outer': outer, 'policies': policies,
        'bootstrap_inner_diagnostics': bootstrap(inner, classes, seed), 'valid': True, 'fallback': False}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    sources = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in
        [Path(__file__).resolve(), ROOT/'src/unknown_dataset/supervised.py', ROOT/'src/unknown_dataset/preprocessing.py']}
    protocol = {'seeds': SEEDS, 'mechanisms': MECHANISMS, 'class_counts': [2, 3], 'rows': 600,
        'outer_fraction': .25, 'inner_fraction': .25, 'bootstrap_repetitions': 200, 'selection_cap': 20., 'source_sha256': sources}
    cache = ROOT/'experiments/work'/('third-classification-'+digest(protocol)[:16])
    cache.mkdir(parents=True, exist_ok=True)
    started, cases = time.monotonic(), []
    for mechanism in MECHANISMS:
        for count in (2, 3):
            for seed in SEEDS:
                for relative, expected in sources.items():
                    if hashlib.sha256((ROOT/relative).read_bytes()).hexdigest() != expected:
                        raise RuntimeError('Source changed during frozen study: '+relative)
                path = cache/f'{mechanism}-{count}-{seed}.json'
                if path.exists():
                    case = json.loads(path.read_text())
                else:
                    case = run_case(seed, mechanism, count)
                    pending = path.with_suffix('.tmp')
                    pending.write_text(json.dumps(case, allow_nan=False, separators=(',', ':')))
                    pending.replace(path)
                cases.append(case)
                print(f'{len(cases)}/18 {mechanism}/{count}/{seed}', flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump({'protocol': protocol, 'cases': cases, 'seconds': time.monotonic()-started, 'fits': len(cases)*16}, stream, allow_nan=False, separators=(',', ':'))

if __name__ == '__main__':
    main()
