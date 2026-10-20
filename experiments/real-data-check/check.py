"""Read-only model comparison: every prediction comes from the actual CLI.

All files this script creates, including downloads and logs, stay beside it.
Resume by running the same command; completed run identities are never repeated.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import gc
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import random
import signal
import socket
import statistics
import subprocess
import sys
import time

sys.dont_write_bytecode = True
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'

import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.metrics import (accuracy_score, adjusted_mutual_info_score, adjusted_rand_score,
                             balanced_accuracy_score, f1_score, mean_absolute_error,
                             mean_squared_error, r2_score, recall_score)
from sklearn.model_selection import train_test_split

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
SEEDS = (42, 314, 2027)
STRATEGIES = ('baseline', 'auto', 'routed')
METRICS = {
    'classification': ('accuracy', 'balanced_accuracy', 'macro_f1', 'minority_recall'),
    'regression': ('rmse', 'mae', 'r2'), 'clustering': ('ari', 'ami')}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, data):
    pending = path.with_suffix(path.suffix + '.pending')
    pending.write_text(json.dumps(data, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(pending, path)


def protected_files():
    names = subprocess.check_output(['git', 'ls-files', '--cached', '--others', '--exclude-standard', '-z'], cwd=ROOT).decode().split('\0')
    return {name: sha(ROOT / name) for name in names if name and not name.replace('\\', '/').startswith('experiments/real-data-check/') and (ROOT/name).is_file()}


def sources():
    files = [ROOT/'main.py', *(ROOT/'src').rglob('*.py'), *(ROOT/'configs').glob('*.json')]
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in files}


def line(path, text):
    return next(i for i, value in enumerate((ROOT/path).read_text(encoding='utf-8').splitlines(), 1) if text in value)


def facts():
    runner, contract = 'src/unknown_dataset/runner.py', 'src/unknown_dataset/contract.py'
    noflags = subprocess.run([sys.executable, '-B', str(ROOT/'main.py')], cwd=ROOT,
                             capture_output=True, text=True, timeout=20)
    text = f'''# What the repository actually does

## Command and default

From the repository folder, the no-flags command in this workspace is:

```powershell
.\\.venv312\\Scripts\\python.exe main.py
```

It does **not run a model**: it exits with code {noflags.returncode}, because input
and output folders are required ({runner}:{line(runner, '"--input"')}-{line(runner, '"--output"')}).
The minimum usable form is:

```powershell
.\\.venv312\\Scripts\\python.exe main.py --input PATH_TO_INPUT --output PATH_TO_OUTPUT
```

Omitting only the strategy chooses **baseline**. Exact source at
`{runner}:{line(runner, 'choices=("baseline", "auto", "routed")')}`:

```python
parser.add_argument("--strategy", choices=("baseline", "auto", "routed"), default="baseline")
```

The default budget is 50 seconds and seed is 42 ({runner}:{line(runner, '"--time-budget"')},
{line(runner, '"--seed"')}). The Python model API also defaults to baseline
(`src/unknown_dataset/models.py:{line('src/unknown_dataset/models.py', 'strategy="baseline"')}`).

## Published instructions versus code

The README explicitly says the official kickoff contract is unavailable
(`README.md:{line('README.md', '**Status:**')}`). Therefore **official organizer
compatibility cannot be certified**. The following checks concern the repo's
published, provisional instructions, not unprovided organizer rules.

| Item | Repository instructions | Actual implementation | Finding |
| --- | --- | --- | --- |
| Input files | README.md:{line('README.md', 'The input folder contains')} | {contract}:{line(contract, 'directory / "metadata.json"')}, {line(contract, 'train = _read_table')}, {line(contract, 'test = _read_table')} | Matches: metadata.json, train.csv; supervised tasks also require test.csv. |
| Metadata | README.md:{line('README.md', '`task` is')} and {line('README.md', '`feature_types`, if provided')} | {contract}:{line(contract, 'task = metadata.get')}, {line(contract, 'feature_types must map every feature')}, {line(contract, 'n_clusters = metadata.get')} | Matches: task; supervised target; optional ID; exact feature type map; clustering k. |
| Hidden test labels | README.md:{line('README.md', 'the target; test tables must omit it')} | {contract}:{line(contract, 'if target in test:')} | Matches: test.csv containing target is rejected. |
| Labels/order | README.md:{line('README.md', 'The file adapter treats classification labels')} | {contract}:{line(contract, 'lexical_columns =')}, {line(contract, 'X_test = test.loc')} | Matches: lexical class labels preserved; feature order aligned; row order retained. |
| Outputs | README.md:{line('README.md', '- `predictions.csv`')}-{line('README.md', '- `run.json`')} | {contract}:{line(contract, 'names = ("profile.json"')}, {line(contract, 'pd.DataFrame({"prediction": values})')} | Matches: one prediction column plus profile.json and run.json. |
| Invalid input | README.md:{line('README.md', 'Malformed input and model errors')} | {runner}:{line(runner, 'print(f"ERROR:')} | Nonzero errors are implemented. |

### Mismatches and qualifications

1. **No-flags execution is not supported.** The README's example supplies the
   required flags; there is no default input folder. Its `data/example` path is an
   example, not a bundled runnable dataset (existence here: {(ROOT/'data/example').exists()}).
2. **Error wording is too broad.** README.md:{line('README.md', 'Malformed input and model errors')}
   says model errors return nonzero. Routed can instead return successful recovered
   baseline predictions after optional errors (`src/unknown_dataset/routed.py:{line('src/unknown_dataset/routed.py', 'except Exception as exc:')}`;
   `{runner}:{line(runner, 'except Exception as exc:')}`). Earlier README paragraphs
   describe recovery correctly. Consumers must inspect run.json, not only exit code.
3. **Some documentation status is stale.** README.md:{line('README.md', 'clean environment and the CI workflow')}
   describes CI as a remaining check; the approved portability checkpoint subsequently
   passed GitHub Actions. Old benchmark/test counts are historical, not current totals.
4. README.md:{line('README.md', 'Missing labels and nonfinite numeric inputs')}
   should be read with the following sentence: numeric feature NaNs are allowed;
   infinities and missing targets are rejected ({contract}:{line(contract, 'if np.isinf')},
   {line(contract, 'if y.isna().any():')}). This is ambiguous wording, not an observed format defect.

No instruction, contract, configuration or model file was edited for this check.
'''
    (HERE/'FACTS.md').write_text(text, encoding='utf-8')
    save(HERE/'no-flags.json', dict(command=[sys.executable, 'main.py'], returncode=noflags.returncode,
                                  stdout=noflags.stdout, stderr=noflags.stderr))


def types_of(frame):
    return {name: 'numeric' if pd.api.types.is_numeric_dtype(frame[name]) else 'categorical' for name in frame}


def write_case(name, task, seed, frame, target, group='real', train=None, test=None, details=None):
    identity = f'{group}-{task}-{name}-{seed}'
    directory = HERE/'inputs'/identity
    directory.mkdir(parents=True, exist_ok=True)
    def preserve_csv(table, path):
        if path.exists():
            assert path.read_text(encoding='utf-8') == table.to_csv(index=False).replace('\r\n','\n'), path
        else:
            table.to_csv(path,index=False)
    frame = frame.copy()
    frame.columns = [str(c) for c in frame]
    assert frame.columns.is_unique and '__target__' not in frame
    target = np.asarray(target)
    metadata = dict(task=task, feature_types=types_of(frame))
    if task == 'clustering':
        fitting = frame
        truth = target.tolist()
        metadata['n_clusters'] = int(len(np.unique(target)))
        training_labels = None
    else:
        fitting = frame.iloc[train].copy()
        fitting['__target__'] = target[train]
        preserve_csv(frame.iloc[test],directory/'test.csv')
        metadata['target'] = '__target__'
        truth = target[test].tolist()
        training_labels = target[train].tolist() if task == 'classification' else None
    preserve_csv(fitting,directory/'train.csv')
    if (directory/'metadata.json').exists():
        assert json.loads((directory/'metadata.json').read_bytes()) == metadata
    else:
        (directory/'metadata.json').write_text(json.dumps(metadata), encoding='utf-8')
    reference = dict(truth=truth, training_labels=training_labels)
    refpath = HERE/'references'/f'{identity}.json'
    refpath.parent.mkdir(exist_ok=True)
    if refpath.exists():
        assert json.loads(refpath.read_bytes()) == reference
    else:
        save(refpath, reference)
    case = dict(id=identity, group=group, task=task, dataset=name, seed=seed,
                input=str(directory.relative_to(HERE)), reference=str(refpath.relative_to(HERE)),
                training_rows=len(fitting), evaluation_rows=len(truth), features=len(frame.columns),
                k=metadata.get('n_clusters'), details=details or {},
                input_sha256={p.name: sha(p) for p in sorted(directory.iterdir())}, reference_sha256=sha(refpath))
    assert refpath.parent != directory and '__target__' not in frame
    return case


def real_cases(download):
    cache = HERE/'download-cache'
    cache.mkdir(exist_ok=True)
    provenance, cases = [], []
    bundled = {}
    for name in ('breast_cancer', 'wine', 'digits', 'iris', 'diabetes'):
        bunch = getattr(datasets, 'load_'+name)(as_frame=True)
        bundled[name] = (bunch.data, bunch.target)
    classification = [(name, *bundled[name], {'source': 'sklearn bundled real dataset'})
                      for name in ('breast_cancer', 'wine', 'digits', 'iris')]
    for name, data_id, offset in (('mammography', 310, 9101), ('blood_transfusion', 1464, 9102)):
        try:
            if not download:
                raise RuntimeError('Network downloads not enabled; no prior downloaded-source manifest')
            print('Downloading OpenML', name, flush=True)
            bunch = datasets.fetch_openml(data_id=data_id, as_frame=True, data_home=str(cache), n_retries=1)
            x, y = bunch.data, bunch.target
            if isinstance(y, pd.DataFrame):
                raise ValueError('Multiple targets unsupported')
            keep = y.notna()
            x, y = x.loc[keep].reset_index(drop=True), y.loc[keep].reset_index(drop=True)
            details = dict(source='OpenML real dataset', data_id=data_id,
                           reported_name=bunch.details.get('name'), url=f'https://www.openml.org/d/{data_id}')
        except Exception as exc:
            x, original = bundled['breast_cancer']
            rng = np.random.default_rng(offset)
            negative = np.flatnonzero(np.asarray(original) == 1)
            positive = rng.choice(np.flatnonzero(np.asarray(original) == 0), round(len(negative)/19), replace=False)
            selected = rng.permutation(np.r_[negative, positive])
            x = x.iloc[selected].reset_index(drop=True)
            y = pd.Series(np.where(np.asarray(original)[selected] == 0, 'positive', 'negative'))
            details = dict(source='Fallback: subsampled real breast_cancer; NOT an independent dataset',
                           unavailable=name, error=f'{type(exc).__name__}: {exc}',
                           positive_fraction=float(np.mean(y == 'positive')), subsample_seed=offset)
            name = name+'_fallback_bc5pct'
        classification.append((name, x, y, details))
    regression = [('diabetes', *bundled['diabetes'], {'source': 'sklearn bundled real dataset'})]
    try:
        print('Loading California housing', flush=True)
        bunch = datasets.fetch_california_housing(data_home=str(cache), as_frame=True,
                                                  download_if_missing=download, n_retries=1)
        regression.append(('california_housing', bunch.data, bunch.target,
                           {'source': 'sklearn fetched real dataset'}))
    except Exception as exc:
        provenance.append(dict(dataset='california_housing', status='unavailable', error=f'{type(exc).__name__}: {exc}'))
    try:
        if not download:
            raise RuntimeError('Download disabled and categorical dataset not bundled')
        print('Downloading categorical house_prices regression', flush=True)
        bunch = datasets.fetch_openml(data_id=42165, as_frame=True, data_home=str(cache), n_retries=1)
        y = pd.to_numeric(bunch.target, errors='raise')
        keep = y.notna()
        regression.append(('house_prices', bunch.data.loc[keep].reset_index(drop=True), y.loc[keep].reset_index(drop=True),
                           dict(source='OpenML real categorical regression', data_id=42165,
                                url='https://www.openml.org/d/42165', reported_name=bunch.details.get('name'))))
    except Exception as exc:
        provenance.append(dict(dataset='categorical_regression', status='unavailable', error=f'{type(exc).__name__}: {exc}'))
    for task, pool in (('classification', classification), ('regression', regression)):
        for name, x, y, details in pool:
            target = np.asarray(y).astype(str) if task == 'classification' else np.asarray(y, dtype=float)
            provenance.append(dict(dataset=name, task=task, status='available', rows=len(x), features=len(x.columns),
                                   categories=sum(v == 'categorical' for v in types_of(x).values()),
                                   source_class_counts=dict(Counter(target.tolist())) if task == 'classification' else None, **details))
            for seed in SEEDS:
                train, test = train_test_split(np.arange(len(x)), test_size=.25, random_state=seed,
                                              stratify=target if task == 'classification' else None)
                cases.append(write_case(name, task, seed, x.reset_index(drop=True), target, train=train, test=test, details=details))
    for name in ('iris', 'wine', 'digits'):
        x, target = bundled[name]
        for seed in SEEDS:
            order = np.random.default_rng(seed).permutation(len(x))
            cases.append(write_case(name, 'clustering', seed, x.iloc[order].reset_index(drop=True),
                                    np.asarray(target)[order], details={'source': 'sklearn bundled real dataset',
                                    'rows': 'All unlabeled feature rows; seed changes row order and model initialization',
                                    'k': 'True class count supplied, labels remain evaluator-only'}))
    return cases, provenance


def stress_case(name, task, rows, columns, special=None):
    seed = 7001
    rng = np.random.default_rng(seed)
    test_rows = 2000 if rows >= 100000 else 1000 if rows >= 5000 else 200 if rows >= 500 else 20
    count = rows if task == 'clustering' else rows+test_rows
    x = rng.normal(size=(count, columns))
    signal_value = x[:, 0]*2 + x[:, 1] + .4*x[:, 2]*x[:, 3]
    if task == 'classification':
        if special == 'many_classes':
            labels = np.asarray([f'class_{i}' for i in np.arange(count) % 25])
            x[:, 0] += (np.arange(count) % 25)*.2
        elif special == 'lexical':
            tokens = np.array(['01', '1', '001', '1.0', '0001', '1e0'])
            labels = tokens[np.digitize(signal_value, np.quantile(signal_value, np.arange(1,6)/6))]
        else:
            labels = np.where(signal_value > 0, 'positive', 'negative')
    elif task == 'regression':
        labels = signal_value + rng.normal(scale=.3, size=count)
    else:
        labels = np.arange(count) % 3
        x[:, :2] += np.array([[-3., -3.], [0., 3.], [3., -3.]])[labels]
    frame = pd.DataFrame(x, columns=[f'x{i}' for i in range(columns)])
    if special == 'missing40':
        frame = frame.mask(rng.random(frame.shape) < .4)
    if special == 'categorical5000':
        frame['category5000'] = [f'level_{i}' for i in range(count)]
    if special == 'constants_duplicates':
        frame['constant'] = 7.
        frame['exact_duplicate_x0'] = frame['x0']
    if special == 'all_missing':
        frame['all_missing'] = np.nan
    return write_case(name, task, seed, frame, labels, group='stress',
                      train=np.arange(rows) if task != 'clustering' else None,
                      test=np.arange(rows, count) if task != 'clustering' else None,
                      details=dict(synthetic=True, special=special, specification=f'{rows} training rows x {columns} generated numeric features',
                                   categorical_note='5000 distinct training categories and 1000 unseen test categories' if special == 'categorical5000' else None))


def prepare(download):
    if (HERE/'manifest.json').exists():
        raise FileExistsError('Manifest already exists; use --run or --report to preserve it')
    current = protected_files()
    facts()
    socket.setdefaulttimeout(30)
    cases, provenance = real_cases(download)
    for rows, columns in ((100000, 50), (5000, 500)):
        for task in ('classification', 'regression', 'clustering'):
            print('Preparing stress', task, rows, columns, flush=True)
            cases.append(stress_case(f'shape_{rows}x{columns}', task, rows, columns))
            gc.collect()
    for spec in [('missing40', 'classification', 1000, 20, 'missing40'),
                 ('categorical5000', 'classification', 5000, 4, 'categorical5000'),
                 ('constant_duplicate', 'clustering', 1000, 8, 'constants_duplicates'),
                 ('tiny50', 'classification', 50, 8, None),
                 ('classes25', 'classification', 1250, 12, 'many_classes'),
                 ('numeric_looking_labels', 'classification', 1200, 8, 'lexical'),
                 ('all_missing_column', 'regression', 1000, 8, 'all_missing')]:
        cases.append(stress_case(*spec))
    order = []
    rng = random.Random(49073)
    for case in cases:
        policies = list(STRATEGIES)
        rng.shuffle(policies)
        order.extend(dict(case_id=case['id'], strategy=strategy) for strategy in policies)
    manifest = dict(seeds=SEEDS, strategies=STRATEGIES, internal_budget=50, external_limit=60,
                    source_sha256=sources(), harness_sha256=sha(Path(__file__)), protected_start=current,
                    checkpoint=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT).decode().strip(),
                    environment=dict(python=sys.version, platform=platform.platform(),
                        packages={name: importlib.metadata.version(name) for name in ('numpy','pandas','scipy','scikit-learn')}),
                    minority_recall_definition='Mean recall for the least frequent training class(es); ties averaged',
                    clustering_protocol='All feature rows, no reference labels in inputs; row permutation and model seed change',
                    provenance=provenance, cases=cases, inventory=order)
    assert current == protected_files(), 'Files outside real-data-check changed during preparation'
    save(HERE/'manifest.json', manifest)
    print(json.dumps({'prepared_cases':len(cases), 'planned_CLI_runs':len(order), 'downloads':provenance}), flush=True)


def kill_tree(process):
    if os.name == 'nt':
        subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'], capture_output=True)
        if process.poll() is None:
            process.kill()
    else:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    process.wait(timeout=10)


def retry_downloads():
    """Correct pre-benchmark acquisition failures; preserve old manifest/inputs."""
    assert not existing_rows(), 'Cannot replace dataset inventory after model runs'
    path = HERE/'manifest.json'
    manifest = json.loads(path.read_bytes())
    assert sources() == manifest['source_sha256']
    attempt=1
    while (HERE/f'preparation-attempt-{attempt}.json').exists():
        attempt+=1
    backup = HERE/f'preparation-attempt-{attempt}.json'
    backup.write_bytes(path.read_bytes())
    socket.setdefaulttimeout(30)
    cases, provenance = real_cases(True)
    cases += [c for c in manifest['cases'] if c['group']=='stress']
    rng=random.Random(49073)
    inventory=[]
    for case in cases:
        strategies=list(STRATEGIES)
        rng.shuffle(strategies)
        inventory.extend(dict(case_id=case['id'],strategy=s) for s in strategies)
    manifest.update(cases=cases,provenance=provenance,inventory=inventory,harness_sha256=sha(Path(__file__)),
                    preparation_correction='Retry count 0 was invalid; corrected to1 before any benchmark. First manifest and all inputs preserved.')
    assert protected_files()==manifest['protected_start']
    save(path,manifest)
    print(json.dumps({'cases':len(cases),'runs':len(inventory),'provenance':provenance}),flush=True)


def scores(task, truth, pred, training_labels=None):
    if task == 'classification':
        classes = sorted(set(training_labels))
        counts = Counter(training_labels)
        minority = [c for c in classes if counts[c] == min(counts.values())]
        recall = recall_score(truth, pred, labels=classes, average=None, zero_division=0)
        return dict(accuracy=float(accuracy_score(truth, pred)),
                    balanced_accuracy=float(balanced_accuracy_score(truth, pred)),
                    macro_f1=float(f1_score(truth, pred, labels=classes, average='macro', zero_division=0)),
                    minority_recall=float(np.mean([recall[classes.index(c)] for c in minority])),
                    class_recall={c:float(r) for c,r in zip(classes,recall)}, minority_classes=minority)
    if task == 'regression':
        return dict(rmse=float(np.sqrt(mean_squared_error(truth,pred))), mae=float(mean_absolute_error(truth,pred)), r2=float(r2_score(truth,pred)))
    return dict(ari=float(adjusted_rand_score(truth,pred)), ami=float(adjusted_mutual_info_score(truth,pred)))


def existing_rows():
    path = HERE/'results.jsonl'
    rows = []
    if path.exists():
        with path.open(encoding='utf-8') as stream:
            for number, text in enumerate(stream, 1):
                if text.strip():
                    try:
                        rows.append(json.loads(text))
                    except json.JSONDecodeError as exc:
                        raise ValueError(f'Incomplete result line {number}; preserve and inspect before resuming') from exc
    identities = [(r['case_id'],r['strategy']) for r in rows]
    assert len(identities) == len(set(identities))
    return rows


def run_case(case, strategy):
    reference = json.loads((HERE/case['reference']).read_bytes())
    for name, expected in case['input_sha256'].items():
        assert sha(HERE/case['input']/name) == expected
    assert sha(HERE/case['reference']) == case['reference_sha256']
    directory = HERE/'runs'/case['id']/strategy
    if directory.exists():
        raise FileExistsError(f'Unrecorded output exists; inspect rather than overwrite: {directory}')
    directory.mkdir(parents=True)
    command = [sys.executable, '-B', str(ROOT/'main.py'), '--input', str(HERE/case['input']),
               '--output', str(directory), '--strategy', strategy, '--seed', str(case['seed']), '--time-budget', '50']
    save(directory/'command.json', command)
    started = time.perf_counter()
    with (directory/'stdout.txt').open('w',encoding='utf-8') as out, (directory/'stderr.txt').open('w',encoding='utf-8') as err:
        process = subprocess.Popen(command,cwd=ROOT,stdout=out,stderr=err,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),
                                   start_new_session=os.name != 'nt')
        timeout = False
        try:
            returncode = process.wait(timeout=60)
        except subprocess.TimeoutExpired:
            timeout = True
            kill_tree(process)
            returncode = process.returncode
        elapsed = time.perf_counter()-started
    result = dict(case_id=case['id'], group=case['group'], task=case['task'], dataset=case['dataset'], seed=case['seed'],
                  strategy=strategy, wall_seconds=elapsed, over_45_seconds=elapsed>45,
                  external_timeout=timeout, returncode=returncode, valid=False,
                  deadline_pass=not timeout and elapsed<=60, output=str(directory.relative_to(HERE)))
    if returncode == 0 and not timeout:
        try:
            table = pd.read_csv(directory/'predictions.csv',dtype=str)
            assert list(table.columns)==['prediction'] and len(table)==len(reference['truth'])
            assert not table.prediction.isna().any()
            if case['task']=='classification':
                prediction=table.prediction.tolist()
                assert set(prediction)<=set(reference['training_labels'])
            elif case['task']=='regression':
                prediction=table.prediction.astype(float).tolist()
                assert np.isfinite(prediction).all()
            else:
                numeric=pd.to_numeric(table.prediction,errors='raise').to_numpy()
                assert np.isfinite(numeric).all() and np.equal(numeric,np.floor(numeric)).all()
                prediction=numeric.astype(int).tolist()
                assert set(prediction)==set(range(case['k']))
            info=json.loads((directory/'run.json').read_bytes())
            profile=json.loads((directory/'profile.json').read_bytes())
            assert info['task']==case['task'] and info['strategy']==strategy
            assert profile['row_count']==case['training_rows'] and profile['feature_count']==case['features']
            model=info['model']
            fallback=bool(model.get('fallback',False) or model.get('checkpoint_recovered',False) or model.get('refit_fallback_reason'))
            result.update(valid=True,metrics=scores(case['task'],reference['truth'],prediction,reference['training_labels']),
                          selected_candidate=model.get('selected_candidate',model.get('model')),
                          model=model.get('model'),fallback=fallback,recovered=bool(model.get('checkpoint_recovered',False)),
                          fallback_kind=model.get('fallback_kind') or ('emergency' if str(model.get('model','')).startswith('Emergency') else 'refit_fallback' if model.get('refit_fallback_reason') else None),
                          fallback_reason=model.get('fallback_reason') or model.get('refit_fallback_reason'),
                          model_info=model,prediction_sha256=sha(directory/'predictions.csv'))
        except Exception as exc:
            result['validation_error']=f'{type(exc).__name__}: {exc}'
    if not result['valid']:
        result['stderr']=(directory/'stderr.txt').read_text(encoding='utf-8')[-6000:]
    result['pass']=result['valid'] and result['deadline_pass']
    return result


def run(section, max_runs):
    manifest=json.loads((HERE/'manifest.json').read_bytes())
    assert sources()==manifest['source_sha256'], 'Production source/config changed; do not mix results'
    assert sha(Path(__file__))==manifest['harness_sha256'], 'Harness changed; do not mix results'
    done=existing_rows()
    identities={(r['case_id'],r['strategy']) for r in done}
    cases={c['id']:c for c in manifest['cases']}
    remaining=[p for p in manifest['inventory'] if (p['case_id'],p['strategy']) not in identities and
               (section=='all' or cases[p['case_id']]['group']==section)]
    for index, plan in enumerate(remaining[:max_runs] if max_runs else remaining):
        assert sources()==manifest['source_sha256']
        case=cases[plan['case_id']]
        print(f"RUN {len(done)+1}/{len(manifest['inventory'])}: {case['task']} {case['dataset']} seed={case['seed']} {plan['strategy']}",flush=True)
        row=run_case(case,plan['strategy'])
        with (HERE/'results.jsonl').open('a',encoding='utf-8') as stream:
            stream.write(json.dumps(row,allow_nan=False,separators=(',',':'))+'\n')
            stream.flush()
            os.fsync(stream.fileno())
        done.append(row)
        print(f"  valid={row['valid']} pass={row['pass']} wall={row['wall_seconds']:.2f}s selected={row.get('selected_candidate')} fallback={row.get('fallback_kind')}",flush=True)
        status=dict(completed=len(done),planned=len(manifest['inventory']),last=row['case_id'],last_strategy=row['strategy'])
        save(HERE/'progress.json',status)
    report()


def quality_gain(metric, base, candidate):
    return (base-candidate)/abs(base) if metric in ('rmse','mae') and base else candidate-base if metric not in ('rmse','mae') else 0.


def report():
    manifest=json.loads((HERE/'manifest.json').read_bytes())
    rows=existing_rows()
    index={(r['case_id'],r['strategy']):r for r in rows}
    cases={c['id']:c for c in manifest['cases']}
    summary, losses = {}, []
    for task, metrics in METRICS.items():
        taskcases=[c for c in manifest['cases'] if c['group']=='real' and c['task']==task]
        header=['dataset','seed']+[f'{s}.{m}' for s in STRATEGIES for m in (*metrics,'wall_seconds','valid','fallback','selected')]
        table=[]
        for case in taskcases:
            entry={'dataset':case['dataset'],'seed':case['seed']}
            for strategy in STRATEGIES:
                row=index.get((case['id'],strategy),{})
                for m in metrics:entry[f'{strategy}.{m}']=row.get('metrics',{}).get(m)
                entry.update({f'{strategy}.wall_seconds':row.get('wall_seconds'),f'{strategy}.valid':row.get('valid'),
                              f'{strategy}.fallback':row.get('fallback_kind'),f'{strategy}.selected':row.get('selected_candidate')})
            table.append(entry)
        with (HERE/f'{task}-table.csv').open('w',newline='',encoding='utf-8') as stream:
            writer=csv.DictWriter(stream,fieldnames=header);writer.writeheader();writer.writerows(table)
        md=[f'# {task.title()}: all real dataset/seed rows',
            'Columns group strategy × metric. Empty cells mean unrun/invalid, never a hidden zero.',
            '| Dataset / seed | '+' | '.join(f'{s}: '+', '.join(metrics)+'; seconds; valid; fallback; selected' for s in STRATEGIES)+' |',
            '|---|'+'---|'*3]
        for case in taskcases:
            values=[]
            for s in STRATEGIES:
                r=index.get((case['id'],s))
                if not r:values.append('UNRUN');continue
                q=', '.join(f'{r["metrics"][m]:.5f}' for m in metrics) if r['valid'] else 'INVALID'
                values.append(f'{q}; {r["wall_seconds"]:.2f}s; {r["valid"]}; {r.get("fallback_kind") or "none"}; {r.get("selected_candidate", "—")}')
            md.append(f'| {case["dataset"]} / {case["seed"]} | '+' | '.join(values)+' |')
        (HERE/f'{task}-table.md').write_text('\n\n'.join(md[:2])+'\n\n'+'\n'.join(md[2:])+'\n',encoding='utf-8')
        summary[task]={}
        for strategy in STRATEGIES:
            observed=[index[(c['id'],strategy)] for c in taskcases if (c['id'],strategy) in index]
            valid=[r for r in observed if r['valid']]
            entry=dict(runs=len(observed),valid=len(valid),mean_wall_seconds=statistics.mean(r['wall_seconds'] for r in observed) if observed else None,
                       maximum_wall_seconds=max((r['wall_seconds'] for r in observed),default=None),
                       fallbacks=sum(bool(r.get('fallback')) for r in observed),metrics={})
            for metric in metrics:
                paired=[]
                for c in taskcases:
                    base=index.get((c['id'],'baseline')); other=index.get((c['id'],strategy))
                    if base and other and base['valid'] and other['valid']:
                        gain=quality_gain(metric,base['metrics'][metric],other['metrics'][metric])
                        paired.append(dict(dataset=c['dataset'],seed=c['seed'],gain=gain,
                                           baseline=base['metrics'][metric],candidate=other['metrics'][metric]))
                        if gain < -1e-12:
                            losses.append(dict(task=task,strategy=strategy,metric=metric,**paired[-1]))
                bydataset={}
                for r in paired:bydataset.setdefault(r['dataset'],[]).append(r['gain'])
                datasetmeans={name:statistics.mean(v) for name,v in bydataset.items()}
                entry['metrics'][metric]=dict(mean=statistics.mean(r['metrics'][metric] for r in valid) if valid else None,
                    paired_cases=len(paired),mean_gain_vs_baseline=statistics.mean(datasetmeans.values()) if datasetmeans else None,
                    worst_case=min(paired,key=lambda r:r['gain']) if paired else None,
                    dataset_mean_gain=datasetmeans,
                    worst_dataset_mean=min(datasetmeans.items(),key=lambda item:item[1]) if datasetmeans else None)
            summary[task][strategy]=entry
    stress=[r for r in rows if r['group']=='stress']
    stress_fields=['dataset','task','strategy','wall_seconds','valid','deadline_pass','pass','fallback_kind','recovered','selected_candidate']
    with (HERE/'stress-table.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=stress_fields,extrasaction='ignore');writer.writeheader();writer.writerows(stress)
    stress_md=['# Stress checks: actual CLI, 60-second external guard',
               '| Case | Task | Strategy | Seconds | Valid | Pass | Fallback | Selected |', '|---|---|---|---:|---|---|---|---|']
    for r in stress:stress_md.append(f'| {r["dataset"]} | {r["task"]} | {r["strategy"]} | {r["wall_seconds"]:.2f} | {r["valid"]} | {r["pass"]} | {r.get("fallback_kind") or "none"} | {r.get("selected_candidate","—")} |')
    (HERE/'stress-table.md').write_text('\n'.join(stress_md)+'\n',encoding='utf-8')
    maximum={s:max((r['wall_seconds'] for r in rows if r['strategy']==s),default=None) for s in STRATEGIES}
    stress_maximum={s:max((r['wall_seconds'] for r in stress if r['strategy']==s),default=None) for s in STRATEGIES}
    alerts=[{k:r.get(k) for k in ('case_id','strategy','wall_seconds','valid','external_timeout','returncode','fallback_kind','validation_error','stderr')}
            for r in rows if not r['pass'] or r['over_45_seconds']]
    current=protected_files()
    original=manifest['protected_start']
    changed={p for p in original if current.get(p)!=original[p]}
    added=set(current)-set(original)
    preserved=not changed and not added
    result=dict(completed=len(rows),planned=len(manifest['inventory']),real=summary,losses=losses,alerts=alerts,
                maximum_wall_seconds=maximum,stress_maximum_seconds=stress_maximum,
                protected_files_preserved=preserved,unexpected_changed=sorted(changed),unexpected_added=sorted(added),
                source_matches=sources()==manifest['source_sha256'])
    save(HERE/'summary.json',result)
    text=['# Real-data comparison and stress results',f'Completed **{len(rows)}/{len(manifest["inventory"])}** actual CLI runs.',
          'All figures are measurements on this machine, not official hackathon points. Classification uses stratified75/25 splits; regression uses75/25 splits. Clustering receives all feature rows and k, never labels.',
          f'Seeds: {SEEDS}. Internal budget50s; external guard60s. No lazy imports or experimental controls enabled. Current working-tree code is frozen by hashes, including existing uncommitted third-batch controls in their disabled defaults.',
          '## Facts and complete tables',
          'See [FACTS.md](FACTS.md), [classification](classification-table.md), [regression](regression-table.md), [clustering](clustering-table.md), and [stress](stress-table.md). CSV counterparts are suitable for Excel/Colab.',
          'Minority recall means recall averaged over the least frequent training class(es), ties averaged. It is descriptive on balanced datasets.',
          '## Task means and worst losses',
          'Classification gains are percentage-point differences; clustering gains are absolute ARI/AMI differences. Regression RMSE/MAE gains are within-dataset relative reductions: positive is better. Raw mean regression errors mix different target units and should not determine the winner.',
          '| Task | Strategy | Metric | Mean metric | Mean gain vs baseline | Worst dataset mean gain | Worst single dataset/seed gain |',
          '|---|---|---|---:|---:|---|---|']
    for task,strategies in summary.items():
        for s,e in strategies.items():
            for metric,q in e['metrics'].items():
                if q['mean'] is None:continue
                if not q['paired_cases']:
                    text.append(f'| {task} | {s} | {metric} | {q["mean"]:.5f} | insufficient paired evidence | — | — |')
                    continue
                factor=100 if task=='classification' or metric in ('rmse','mae') else 1
                suffix=' pp' if task=='classification' else '%' if metric in ('rmse','mae') else ''
                worst=q['worst_case'];wd=q['worst_dataset_mean']
                text.append(f'| {task} | {s} | {metric} | {q["mean"]:.5f} | {factor*q["mean_gain_vs_baseline"]:+.3f}{suffix} | {wd[0]} {factor*wd[1]:+.3f}{suffix} | {worst["dataset"]}/{worst["seed"]}: {factor*worst["gain"]:+.3f}{suffix} |')
    text+=['## Every measured loss versus baseline',
           '| Task | Dataset/seed | Strategy | Metric | Baseline | Candidate | Gain (negative = loss) |',
           '|---|---|---|---|---:|---:|---:|']
    for r in sorted(losses,key=lambda r:(r['task'],r['dataset'],r['seed'],r['strategy'],r['metric'])):
        text.append(f'| {r["task"]} | {r["dataset"]}/{r["seed"]} | {r["strategy"]} | {r["metric"]} | {r["baseline"]:.5f} | {r["candidate"]:.5f} | {r["gain"]:+.5f} |')
    text+=['## Crashes, invalid results, and calls over45seconds', 'No such alerts.' if not alerts else 'All alerts are listed below; valid deadline fallback is not labeled a crash.']
    for r in alerts:text.append(f'- {r["case_id"]}, {r["strategy"]}: {r["wall_seconds"]:.2f}s; valid={r["valid"]}; timeout={r["external_timeout"]}; fallback={r["fallback_kind"]}; error={r.get("validation_error") or r.get("stderr") or "none"}.')
    text+=['## Maximum full-process wall time',f'All runs: {maximum}. Stress only: {stress_maximum}.',
           '## Dataset provenance and unavailable work']
    for p in manifest['provenance']:text.append('- '+json.dumps(p,ensure_ascii=False))
    text+=['## Preservation',f'Outside-folder files preserved: **{preserved}**. Production source/config hashes match: **{result["source_matches"]}**.',
           'No models, defaults, configurations, existing evaluation reports, commits or remote branches were changed. Inputs/references/logs/downloads and this harness all stay under experiments/real-data-check/.',
           '## Interpretation', 'See STUDENT_SUMMARY.md for the final plain-English evidence and recommendations. Until all planned runs complete, these tables are partial and cannot establish a winner.',
           'Three seeds are repeated splits of the same dataset, not independent new datasets. No model was tuned against these held-out results. Official scoring, clean assessment environment and memory compliance remain unverified.']
    (HERE/'REPORT.md').write_text('\n\n'.join(text[:8])+'\n\n'+'\n'.join(text[8:])+'\n',encoding='utf-8')
    print(json.dumps({'completed':len(rows),'planned':len(manifest['inventory']),'preserved':preserved,'alerts':len(alerts)}),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--allow-download',action='store_true')
    parser.add_argument('--run',action='store_true')
    parser.add_argument('--report',action='store_true')
    parser.add_argument('--retry-downloads',action='store_true')
    parser.add_argument('--section',choices=('all','real','stress'),default='all')
    parser.add_argument('--max-runs',type=int)
    args=parser.parse_args()
    if args.prepare:prepare(args.allow_download)
    if args.retry_downloads:retry_downloads()
    if args.run:run(args.section,args.max_runs)
    elif args.report:report()
    if not any((args.prepare,args.run,args.report,args.retry_downloads)):parser.print_help()


if __name__=='__main__':
    main()
