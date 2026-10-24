"""Selection-fit recovery must never replace a successful full-data result."""
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from unknown_dataset import checkpoints, runner, supervised


def data():
    return SimpleNamespace(task='regression', X_train=pd.DataFrame({'x': range(80)}),
        X_test=pd.DataFrame({'x': [81, 82]}), y_train=pd.Series(np.arange(80.)),
        n_clusters=None, feature_types={'x': 'numeric'})


def info():
    return dict(model='Ridge', checkpoint_kind='selection_model', fitting_scope='selection_partition',
                training_rows=60, available_training_rows=80)


@pytest.mark.parametrize('contents', [None, '{broken', '{"version":2}'])
def test_missing_or_corrupt_selection_checkpoint(tmp_path, contents):
    path = tmp_path / 'checkpoint.json'
    if contents is not None:
        path.write_text(contents)
    assert runner._recover_checkpoint(path, data(), 'deadline') is None


def test_selection_recovery_reports_actual_rows_and_kind(tmp_path):
    path = tmp_path / 'checkpoint.json'
    checkpoints.write_checkpoint(path, data(), [1., 2.], info())
    result = runner._recover_checkpoint(path, data(), 'deadline')
    assert result['predictions'].tolist() == [1., 2.]
    assert result['model_info']['training_rows'] == 60
    assert result['model_info']['fallback_kind'] == 'selection_model'
    assert result['model_info']['checkpoint_recovered'] is True
    assert result['model_info']['fitting_scope'] == 'selection_partition'


@pytest.mark.parametrize('patch', [dict(training_rows=True), dict(training_rows=0),
    dict(training_rows=81), dict(available_training_rows=79), dict(fitting_scope='full_data')])
def test_invalid_fitting_metadata_is_not_recovered(tmp_path, patch):
    path = tmp_path / 'checkpoint.json'
    checkpoints.write_checkpoint(path, data(), [1., 2.], dict(info(), **patch))
    assert runner._recover_checkpoint(path, data(), 'deadline') is None


@pytest.mark.parametrize('task', ['classification', 'regression'])
def test_checkpoint_adds_no_fit_and_keeps_full_predictions(task, monkeypatch):
    d = data()
    if task == 'classification':
        d.y_train = pd.Series(['01', '1'] * 40)
    calls = []
    original = supervised._fit
    def record(*args, **kwargs):
        calls.append((args[0], len(args[2]), kwargs.get('final', False)))
        return original(*args, **kwargs)
    monkeypatch.setattr(supervised, '_fit', record)
    kwargs = dict(task=task, X_train=d.X_train, y_train=d.y_train, X_test=d.X_test,
                  feature_types=d.feature_types)
    before, old = supervised.fit_predict_supervised(**kwargs)
    old_calls = list(calls); calls.clear(); saved = []
    after, new = supervised.fit_predict_supervised(**kwargs,
        checkpoint_callback=lambda pred, metadata: saved.append((pred.copy(), metadata)))
    np.testing.assert_array_equal(after, before)
    assert calls == old_calls
    assert new['selected_candidate'] == old['selected_candidate']
    assert new['validation_scores'] == old['validation_scores']
    assert len(saved) == 1
    assert saved[0][1]['training_rows'] == 60
    assert saved[0][1]['available_training_rows'] == 80
    assert saved[0][1]['checkpoint_kind'] == 'selection_model'
    assert saved[0][1]['fit_seconds'] >= 0
    assert new['training_rows'] == 80 and not new['fallback']
    if task == 'classification':
        assert set(saved[0][0]) <= {'01', '1'}


def test_checkpoint_is_published_before_final_fit_failure(monkeypatch):
    d = data(); saved = []; original = supervised._fit
    def fail_final(*args, **kwargs):
        if kwargs.get('final'):
            assert len(saved) == 1
            raise RuntimeError('full fit interrupted')
        return original(*args, **kwargs)
    monkeypatch.setattr(supervised, '_fit', fail_final)
    with pytest.raises(RuntimeError, match='interrupted'):
        supervised.fit_predict_supervised(task=d.task, X_train=d.X_train, y_train=d.y_train,
            X_test=d.X_test, feature_types=d.feature_types,
            checkpoint_callback=lambda pred, metadata: saved.append((pred, metadata)))
    assert saved[0][1]['training_rows'] == 60


def test_checkpoint_publication_failure_does_not_prevent_full_fit():
    d = data()
    def fail(*args):
        raise OSError('disk publication failed')
    predictions, metadata = supervised.fit_predict_supervised(task=d.task, X_train=d.X_train,
        y_train=d.y_train, X_test=d.X_test, feature_types=d.feature_types, checkpoint_callback=fail)
    assert len(predictions) == 2 and metadata['training_rows'] == 80
    assert metadata['selection_checkpoint_error'] == 'OSError: disk publication failed'
    assert not metadata['fallback']

@pytest.mark.parametrize('mode', ['success', 'checkpoint_timeout', 'absent_timeout', 'corrupt_timeout'])
def test_spawn_full_result_precedence_and_timeout_safety(tmp_path, mode):
    import os
    from pathlib import Path
    import subprocess
    import sys
    root = Path(__file__).resolve().parents[1]
    inp = tmp_path / 'input'; inp.mkdir(); out = tmp_path / 'output'
    (inp / 'metadata.json').write_text(json.dumps(dict(task='regression', target='y', feature_types={'x': 'numeric'})))
    (inp / 'train.csv').write_text('x,y\n1,1\n2,2\n3,3\n4,4\n')
    (inp / 'test.csv').write_text('x\n5\n6\n')
    script = tmp_path / 'worker.py'
    script.write_text(
        'import sys,time,pickle\n'
        f'sys.path.insert(0,{str(root / "src")!r})\n'
        'from unknown_dataset import runner\n'
        'from unknown_dataset.checkpoints import write_checkpoint\n'
        f'MODE={mode!r}\n'
        'def worker(dataset,budget,seed,result_path,strategy,*extra):\n'
        '    if MODE in ("success","checkpoint_timeout"):\n'
        '        write_checkpoint(extra[1],dataset,[5.,6.],dict(model="LearnedSelection",checkpoint_kind="selection_model",fitting_scope="selection_partition",training_rows=3,available_training_rows=4))\n'
        '    if MODE=="corrupt_timeout":\n'
        '        open(extra[1],"w").write("{broken")\n'
        '    if MODE!="success":\n'
        '        time.sleep(60)\n'
        '    with open(result_path,"wb") as f:\n'
        '        pickle.dump(dict(predictions=[7.,8.],model_info=dict(model="FullFit")),f)\n'
        'if __name__=="__main__":\n'
        '    runner._model_worker=worker\n'
        '    raise SystemExit(runner.main())\n')
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    process = subprocess.run([sys.executable, '-B', str(script), '--input', str(inp),
        '--output', str(out), '--time-budget', '8'], env=env, capture_output=True, text=True, timeout=20)
    assert process.returncode == 0, process.stderr
    metadata = json.loads((out / 'run.json').read_text())['model']
    pred = pd.read_csv(out / 'predictions.csv').iloc[:, 0].tolist()
    if mode == 'success':
        assert metadata == dict(model='FullFit') and pred == [7., 8.]
    elif mode == 'checkpoint_timeout':
        assert metadata['model'] == 'LearnedSelection' and pred == [5., 6.]
        assert metadata['fallback_kind'] == 'selection_model' and metadata['training_rows'] == 3
    else:
        assert metadata['model'] == 'EmergencyMeanRegressor' and pred == [2.5, 2.5]
