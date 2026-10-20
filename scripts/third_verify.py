"""Reconstruct primary-study evidence from saved arrays; never fit models."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.second_verify_evidence import adjusted_rand, classification_metrics


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def close(a, b):
    assert math.isclose(a, b, abs_tol=1e-11, rel_tol=1e-11), (a, b)


def classification(record, classes):
    truth, prediction, probability = record['truth'], record['prediction'], record['probability']
    m = classification_metrics(truth, prediction, classes)
    for name in ('accuracy', 'balanced_accuracy', 'macro_f1'):
        close(m[name], record['metrics'][name])
    for a, b in zip(m['class_recall'], record['metrics']['class_recall'], strict=True):
        close(a, b)
    assert record['prediction_sha256'] == digest(prediction)
    lookup = {label: i for i, label in enumerate(classes)}
    assert len(probability) == len(truth) == len(prediction)
    logloss, brier = [], []
    for target, row in zip(truth, probability, strict=True):
        assert len(row) == len(classes) and all(math.isfinite(p) and 0 <= p <= 1 for p in row)
        close(sum(row), 1)
        logloss.append(-math.log(max(1e-15, min(1, row[lookup[target]]))))
        brier.append(sum((p - int(i == lookup[target])) ** 2 for i, p in enumerate(row)))
    close(sum(logloss) / len(truth), record['metrics']['log_loss'])
    close(sum(brier) / len(truth), record['metrics']['brier_multiclass'])


def regression(record):
    truth, prediction = record['truth'], record['prediction']
    assert len(truth) == len(prediction) and record['prediction_sha256'] == digest(prediction)
    delta = [p - t for p, t in zip(prediction, truth, strict=True)]
    close(math.sqrt(math.fsum(x*x for x in delta) / len(delta)), record['metrics']['rmse'])
    close(math.fsum(abs(x) for x in delta) / len(delta), record['metrics']['mae'])


def verify(document):
    counts = Counter()
    if document.get('fits') == 288:
        for case in document['cases']:
            assert digest(case['inputs']) == case['input_sha256']
            splits = case['splits']
            outer = splits['outer_train']
            assert sorted(outer + splits['outer_test']) == list(range(600))
            assert sorted(splits['inner_train'] + splits['inner_test']) == sorted(outer)
            labels = case['inputs']['labels']
            for partition, index in (('inner', splits['inner_test']), ('outer', splits['outer_test'])):
                for record in case[partition].values():
                    assert record['truth'] == [labels[i] for i in index]
                    classification(record, case['classes'])
                    counts['classification_records'] += 1
            for key, policy in case['policies'].items():
                objective = key.split(':')[1]
                scores = policy['validation_scores']
                for name, score in scores.items():
                    close(score, case['inner'][name]['metrics'][objective])
                preference = {'linear': 0, 'extra_trees': 1, 'histogram': 2, 'histogram_leaf5': 2}
                selected = min(scores, key=lambda n: (-scores[n], preference[n.split(':')[0]]))
                assert selected == policy['selected']
                assert policy['outer_metrics'] == case['outer'][selected]['metrics']
                counts['training_only_classification_selections'] += 1
    elif 'fit_calls' in document:
        for case in document['cases']:
            assert digest(case['inputs']) == case['input_sha256']
            assert digest(case['splits']) == case['split_sha256']
            s = case['splits']
            assert sorted(s['train'] + s['test']) == list(range(1200))
            assert sorted(s['inner_train'] + s['inner_validation']) == sorted(s['train'])
            for part, indices in (('inner', s['inner_validation']), ('outer', s['test'])):
                for record in case[part].values():
                    assert record['truth'] == [case['inputs']['target'][i] for i in indices]
                    regression(record)
                    counts['regression_records'] += 1
            for key, policy in case['policies'].items():
                metric, capacity = key.split('_')
                mapping = dict(linear='linear', extra_trees='extra_trees', histogram='histogram'+capacity)
                for name, score in policy['validation_scores'].items():
                    close(score, case['inner'][mapping[name]]['metrics'][metric])
                preference = {'linear': 0, 'extra_trees': 1, 'histogram': 2}
                selected = min(policy['validation_scores'], key=lambda n: (policy['validation_scores'][n], preference[n]))
                assert policy['selected'] == selected
                assert policy['metrics'] == case['outer'][mapping[selected]]['metrics']
                counts['training_only_regression_selections'] += 1
    else:
        cases = {(c['family'], c['seed'], c['masking']): c for c in document['cases']}
        for case in cases.values():
            assert digest(case['input']) == case['input_sha256']
        for row in document['rows']:
            case = cases[(row['family'], row['seed'], row['masking'])]
            close(adjusted_rand(case['reference'], row['predictions']), row['ari'])
            assert digest(row['predictions']) == row['prediction_sha256']
            assert len(row['predictions']) == len(case['reference'])
            assert set(row['predictions']) == set(range(case['clusters']))
            counts['clustering_records'] += 1
    assert counts, 'Unrecognized evidence schema'
    return dict(counts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('inputs', type=Path, nargs='+')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Choose a new output')
    result = []
    for path in args.inputs:
        raw = path.read_bytes()
        result.append(dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(), checks=verify(json.loads(raw))))
    args.output.write_text(json.dumps(dict(status='passed', documents=result), indent=2), encoding='utf-8')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
