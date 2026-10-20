"""Read-only, independent arithmetic checks of second-batch saved evidence.

This tool performs no fits and writes only a new verification report. It never
passes reference arrays into model or selection code.
"""
from collections import Counter
import argparse
import hashlib
import json
import math
from pathlib import Path


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                    separators=(',', ':')).encode()).hexdigest()


def label_key(value):
    return json.dumps(value, sort_keys=True)


def classification_metrics(truth, prediction, classes):
    if not truth or len(truth) != len(prediction):
        raise AssertionError('Nonempty aligned predictions required')
    keys = [label_key(c) for c in classes]
    index = {key: i for i, key in enumerate(keys)}
    matrix = [[0] * len(keys) for _ in keys]
    for actual, predicted in zip(truth, prediction, strict=True):
        matrix[index[label_key(actual)]][index[label_key(predicted)]] += 1
    supports = [sum(row) for row in matrix]
    columns = [sum(row[i] for row in matrix) for i in range(len(keys))]
    recall = [matrix[i][i] / n if n else 0.0 for i, n in enumerate(supports)]
    f1 = [2 * matrix[i][i] / (n + columns[i]) if n + columns[i] else 0.0
          for i, n in enumerate(supports)]
    return dict(accuracy=sum(matrix[i][i] for i in range(len(keys))) / len(truth),
                balanced_accuracy=sum(r for r, n in zip(recall, supports) if n) / sum(n > 0 for n in supports),
                macro_f1=sum(f1) / len(f1), class_recall=recall)


def adjusted_rand(truth, prediction):
    if not truth or len(truth) != len(prediction):
        raise AssertionError('Nonempty aligned clustering rows required')
    joint = Counter((label_key(a), label_key(b)) for a, b in zip(truth, prediction, strict=True))
    left, right = Counter(map(label_key, truth)), Counter(map(label_key, prediction))
    choose = lambda n: n * (n - 1) / 2
    pairs = choose(len(truth))
    if not pairs:
        return 1.0
    same = sum(map(choose, joint.values()))
    a, b = sum(map(choose, left.values())), sum(map(choose, right.values()))
    expected = a * b / pairs
    denominator = (a + b) / 2 - expected
    return (same - expected) / denominator if denominator else 1.0


def verify_document(document):
    counts = Counter()

    def equal(actual, expected, name):
        if isinstance(expected, list):
            assert len(actual) == len(expected), name
            for a, b in zip(actual, expected, strict=True):
                equal(a, b, name)
        else:
            assert math.isfinite(actual) and math.isclose(actual, expected, abs_tol=1e-10, rel_tol=0), name
        counts['arithmetic_checks'] += 1

    if 'references' in document and 'rows' in document:
        for row in document['rows']:
            if not row.get('valid'):
                counts['runtime_invalid_rows'] += 1
                continue
            references = document['references']
            truth = references[row['shape']] if isinstance(references, dict) else references
            classes = sorted(set(truth))
            computed = classification_metrics(truth, row['predictions'], classes)
            for key in ('accuracy', 'balanced_accuracy', 'macro_f1'):
                equal(row[key], computed[key], key)
            if 'minority_recall' in row:
                equal(row['minority_recall'], computed['class_recall'][classes.index('minority')], 'minority_recall')
            counts['runtime_prediction_records'] += 1

    def visit(value):
        if isinstance(value, list):
            for item in value:
                visit(item)
            return
        if not isinstance(value, dict):
            return
        if {'truth', 'prediction', 'metrics'} <= value.keys():
            metrics = value['metrics']
            independent = classification_metrics(value['truth'], value['prediction'], metrics['classes'])
            for key, expected in independent.items():
                equal(metrics[key], expected, key)
                if 'independent_metrics' in value:
                    equal(value['independent_metrics'][key], expected, key)
            assert digest(value['prediction']) == value['prediction_sha256']
            counts['classification_prediction_records'] += 1
        if {'inputs', 'input_sha256', 'splits', 'split_sha256', 'candidates', 'classes'} <= value.keys():
            assert digest(value['inputs']) == value['input_sha256']
            assert digest(value['splits']) == value['split_sha256']
            labels = value['inputs']['labels']
            for fold_index, split in enumerate(value['splits']):
                fit, validation = split['fitting'], split['validation']
                assert not set(fit) & set(validation)
                assert sorted(fit + validation) == list(range(len(labels)))
                for indices, key in ((fit, 'fitting_support'), (validation, 'validation_support')):
                    assert split[key] == [sum(labels[i] == c for i in indices) for c in value['classes']]
                for candidate in value['candidates'].values():
                    assert candidate['inner'][fold_index]['truth'] == [labels[i] for i in validation]
            outer_truths = [candidate['outer']['truth'] for candidate in value['candidates'].values()]
            assert all(t == outer_truths[0] for t in outer_truths)
            for selection in value['selections']:
                scores = selection['validation_scores']
                for candidate, score in scores.items():
                    records = value['candidates'][candidate]['inner'][:selection['holdouts']]
                    equal(score, sum(r['metrics'][selection['objective']] for r in records) / len(records), 'inner_score')
                order = {'linear': 0, 'extra_trees': 1, 'histogram': 2}
                selected = min(scores, key=lambda c: (-scores[c], order[c])) if scores else 'extra_trees'
                assert selection['selected'] == selected
                assert selection['outer'] == value['candidates'][selected]['outer']['metrics']
                counts['training_only_selections_reconstructed'] += 1
            counts['case_input_and_split_records'] += 1
        if {'reference', 'predictions', 'ari'} <= value.keys():
            equal(value['ari'], adjusted_rand(value['reference'], value['predictions']), 'ari')
            counts['clustering_prediction_records'] += 1
        for nested in value.values():
            if isinstance(nested, (list, dict)):
                visit(nested)

    visit(document)
    return dict(counts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('inputs', nargs='+', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Choose a new report path; evidence is never overwritten')
    rows = []
    for path in args.inputs:
        data = path.read_bytes()
        rows.append(dict(path=str(path), sha256=hashlib.sha256(data).hexdigest(),
                         checks=verify_document(json.loads(data))))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(status='passed', documents=rows), indent=2), encoding='utf-8')
    print(json.dumps(dict(status='passed', documents=rows)))


if __name__ == '__main__':
    main()
