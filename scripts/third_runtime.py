"""Conditional paired CLI checks for frozen primary changes; no production tuning."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
import tempfile
import time

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score
from sklearn.model_selection import train_test_split

from batch_runtime import measured_process

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from unknown_dataset.supervised import classification_metrics


def plans(tasks):
    rng = random.Random(30490)
    result = []
    for task in tasks:
        for budget in (25, 50):
            for repeat in range(3):
                policies = ['baseline', 'current', 'challenger']
                rng.shuffle(policies)
                result.extend(dict(task=task, budget=budget, repeat=repeat, policy=p) for p in policies)
    return result


def fixture(task, directory):
    rng = np.random.default_rng(30491)
    if task == 'classification':
        from third_classification import case_data
        frame, labels, types = case_data(98003, 'localized_interaction', 3)
        train, test = train_test_split(np.arange(len(frame)), test_size=.25, random_state=98003, stratify=labels)
        extra, k = 102, None
    elif task == 'regression':
        from third_regression import case_data, split_indices
        frame, labels, types = case_data('mixed_category', 98003)
        splits = split_indices(len(frame), 98003)
        train, test = splits['train'], splits['test']
        extra, k = 97, None
    else:
        from third_clustering import generate
        frame, labels, k = generate('anisotropic_noise', 99013, 'random')
        frame = pd.concat([frame] * 3, ignore_index=True)
        labels = np.tile(labels, 3)
        types = {name: 'numeric' for name in frame}
        train, test, extra = np.arange(len(frame)), None, 4
    noise = rng.normal(size=(len(frame), extra))
    noise[rng.random(noise.shape) < .1] = np.nan
    names = [f'nuisance{i}' for i in range(extra)]
    frame = pd.concat([frame, pd.DataFrame(noise, columns=names)], axis=1)
    types.update({name: 'numeric' for name in names})
    directory.mkdir()
    training = frame.iloc[train].copy()
    metadata = dict(task=task, feature_types=types)
    if task != 'clustering':
        training['target'] = labels[train]
        metadata['target'] = 'target'
        frame.iloc[test].to_csv(directory / 'test.csv', index=False)
        reference = np.asarray(labels)[test]
    else:
        metadata['n_clusters'] = k
        reference = labels
    training.to_csv(directory / 'train.csv', index=False)
    (directory / 'metadata.json').write_text(json.dumps(metadata), encoding='utf-8')
    return np.asarray(reference).tolist(), k


def sources():
    files = [ROOT / 'main.py', Path(__file__), ROOT / 'scripts/batch_runtime.py',
             *(ROOT / 'scripts').glob('third_*.py'), *(ROOT / 'src').rglob('*.py'),
             *(ROOT / 'configs').glob('*.json')]
    return {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(files))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tasks', nargs='+', choices=('classification', 'regression', 'clustering'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume-from', type=Path)
    parser.add_argument('--max-seconds', type=float, default=300)
    parser.add_argument('--classification-weighting', choices=('balanced', 'unweighted'), default='unweighted')
    parser.add_argument('--regression-objective', choices=('rmse', 'mae'), default='rmse')
    args = parser.parse_args()
    if args.output.exists() or len(set(args.tasks)) != len(args.tasks):
        parser.error('Choose a new report and unique task list')
    configurations = {}
    for task in args.tasks:
        current = {'experimental': {'runtime_instrumentation': True}}
        if task == 'classification':
            current['experimental']['classification_weighting'] = args.classification_weighting
        elif task == 'regression':
            current['metrics'] = {'regression': args.regression_objective}
        challenger = json.loads(json.dumps(current))
        key, value = {'classification': ('histogram_min_samples_leaf', 5),
                      'regression': ('histogram_max_leaf_nodes', 31),
                      'clustering': ('clustering_indicator_amplitude', .25)}[task]
        challenger['experimental'][key] = value
        configurations[task] = {'current': current, 'challenger': challenger}
    hashes = sources()
    previous = json.loads(args.resume_from.read_bytes()) if args.resume_from else None
    inventory = plans(args.tasks)
    if previous and (previous['source_sha256'] != hashes or previous['configurations'] != configurations
                     or previous['inventory'] != inventory):
        parser.error('Source, configuration or planned inventory changed; cannot resume')
    remaining = previous['unrun'] if previous else inventory
    report = dict(scope='Conditional development CLI checks; no confirmation or promotion',
                  source_sha256=hashes, configurations=configurations, inventory=inventory,
                  unrun=list(remaining), rows=[], references={}, fixture_sha256={})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    work = ROOT / 'experiments/work'
    work.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    def save():
        report['seconds'] = time.monotonic() - started
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    with tempfile.TemporaryDirectory(prefix='third-cli-', dir=work) as name:
        directory = Path(name)
        cluster_counts = {}
        for task in args.tasks:
            report['references'][task], cluster_counts[task] = fixture(task, directory / task)
            report['fixture_sha256'][task] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((directory / task).iterdir())}
            for policy, config in configurations[task].items():
                (directory / f'{task}-{policy}.json').write_text(json.dumps(config), encoding='utf-8')
        if previous and previous['fixture_sha256'] != report['fixture_sha256']:
            raise ValueError('Fixture bytes changed on resume')
        save()
        for index, plan in enumerate(remaining):
            if time.monotonic() - started >= args.max_seconds:
                break
            if sources() != hashes:
                raise RuntimeError('Frozen source changed during CLI study')
            task, policy = plan['task'], plan['policy']
            output = directory / f'output-{index}'
            command = [sys.executable, str(ROOT / 'main.py'), '--input', str(directory / task),
                       '--output', str(output), '--time-budget', str(plan['budget']), '--timings',
                       '--strategy', 'baseline' if policy == 'baseline' else 'routed']
            if policy != 'baseline':
                command += ['--lazy-router-imports', '--router-config', str(directory / f'{task}-{policy}.json')]
            row = dict(plan, **measured_process(command), valid=False)
            if row['returncode'] == 0 and not row['external_timeout']:
                try:
                    raw = pd.read_csv(output / 'predictions.csv', dtype=str)['prediction']
                    truth = report['references'][task]
                    assert len(raw) == len(truth)
                    if task == 'classification':
                        predictions = raw.tolist()
                        assert set(predictions) <= set(truth)
                        metrics = classification_metrics(truth, predictions, sorted(set(truth)))
                    elif task == 'regression':
                        from third_regression import metrics as regression_metrics
                        predictions = raw.astype(float).tolist()
                        assert np.isfinite(predictions).all()
                        metrics = regression_metrics(truth, predictions)
                    else:
                        predictions = raw.astype(int).tolist()
                        assert set(predictions) == set(range(cluster_counts[task]))
                        metrics = dict(ari=float(adjusted_rand_score(truth, predictions)),
                                       ami=float(adjusted_mutual_info_score(truth, predictions)))
                    row.update(valid=True, predictions=predictions, metrics=metrics,
                        run=json.loads((output / 'run.json').read_bytes()), profile=json.loads((output / 'profile.json').read_bytes()))
                    model = row['run'].get('model', {})
                    row['implementation_failure'] = model.get('route_stage') == 'optional_error'
                except Exception as exc:
                    row.update(valid=False, validation_error=f'{type(exc).__name__}: {exc}')
            report['rows'].append(row)
            report['unrun'] = remaining[index + 1:]
            save()
    save()
    print(json.dumps({'completed': len(report['rows']), 'unrun': len(report['unrun']), 'seconds': report['seconds']}))


if __name__ == '__main__':
    main()
