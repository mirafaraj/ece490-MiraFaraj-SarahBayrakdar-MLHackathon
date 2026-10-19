"""Compare deterministic predictions with the committed pre-batch source.

This is a development compatibility check, not a model quality comparison.
Run serially, outside timing experiments. Historical source is read from Git.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile


CHECKPOINT = '0ad79a77844ef1bf5b15750a3e39a0ab4a7a06ed'
ROOT = Path(__file__).resolve().parents[1]
PROBE = r'''
import hashlib, json
import numpy as np
import pandas as pd
from unknown_dataset.models import fit_predict
rng = np.random.default_rng(70421)
x = pd.DataFrame(rng.normal(size=(180, 5)), columns=list('abcde'))
x.loc[::13, 'a'] = np.nan
signal = x.a.fillna(0).to_numpy() * 3 + x.b.to_numpy() - x.c.to_numpy() * .2
rows=[]
for task in ('classification', 'regression', 'clustering'):
    for strategy in ('baseline', 'auto', 'routed'):
        labels = np.where(signal > 1, '001', '1') if task == 'classification' else signal
        kwargs = dict(task=task, X_train=x, y_train=labels if task != 'clustering' else None,
                      X_test=x.iloc[:24].copy() if task != 'clustering' else None,
                      n_clusters=3 if task == 'clustering' else None, seed=5041,
                      time_budget=50, strategy=strategy)
        prediction, info = fit_predict(**kwargs)
        rows.append(dict(task=task, strategy=strategy, prediction=np.asarray(prediction).tolist(),
                         selected=info.get('selected_candidate', info.get('model')),
                         scores=info.get('validation_scores'), bic=info.get('bic')))
for mechanism in ('missing_product', 'heavy_tail', 'sparse_linear'):
    local_rng = np.random.default_rng(94017)
    frame = pd.DataFrame(local_rng.normal(size=(180, 24)))
    if mechanism == 'sparse_linear':
        frame *= local_rng.random(frame.shape) < .15
        target = frame.iloc[:, :8].to_numpy() @ np.arange(1, 9)
    elif mechanism == 'heavy_tail':
        target = 4 * frame[0] - 3 * frame[1] + local_rng.standard_t(2, len(frame))
    else:
        target = 3 * frame[0] * frame[1] + np.sin(3 * frame[2])
        frame[local_rng.random(frame.shape) < .12] = np.nan
    for strategy in ('baseline', 'auto', 'routed'):
        prediction, info = fit_predict('regression', frame, np.asarray(target), frame.iloc[:24].copy(),
                                       strategy=strategy, time_budget=50, seed=5041)
        rows.append(dict(task='regression', mechanism=mechanism, strategy=strategy,
                         prediction=np.asarray(prediction).tolist(),
                         selected=info.get('selected_candidate', info.get('model')),
                         scores=info.get('validation_scores'), bic=info.get('bic')))
print(json.dumps(rows, allow_nan=False))
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output already exists; choose a new evidence path')
    paths = subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', CHECKPOINT], cwd=ROOT).decode().splitlines()
    work = ROOT / 'experiments' / 'work'
    work.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='default-equivalence-', dir=work) as temporary:
        frozen = Path(temporary)
        for path in paths:
            if path.startswith(('src/', 'configs/')):
                target = frozen / path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(subprocess.check_output(['git', 'show', f'{CHECKPOINT}:{path}'], cwd=ROOT))
        results = []
        for source in (frozen, ROOT):
            env = dict(os.environ, PYTHONPATH=str(source / 'src'), OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
            results.append(json.loads(subprocess.check_output([sys.executable, '-c', PROBE], cwd=source, env=env, timeout=120)))
    source_hashes = {str(p.relative_to(ROOT)).replace('\\', '/'): hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in sorted((ROOT / 'src').rglob('*.py'))}
    report = dict(checkpoint=CHECKPOINT, equivalent=results[0] == results[1], frozen=results[0], current=results[1], source_hashes=source_hashes)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(dict(equivalent=report['equivalent'], comparisons=len(results[0]))))
    return 0 if report['equivalent'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
