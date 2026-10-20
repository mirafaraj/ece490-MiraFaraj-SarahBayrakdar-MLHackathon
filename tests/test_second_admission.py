"""Admission uses measured costs and an absolute deadline, never references."""
import json
import pickle

import numpy as np
import pandas as pd
import pytest

from unknown_dataset import models, runner, supervised
from unknown_dataset.policies import validate_config
from unknown_dataset.routing import make_route_plan, summarize_training


SCORES = {'extra_trees': .5, 'linear': .82, 'histogram': .9}
COSTS = {'extra_trees': .2, 'linear': .3, 'histogram': 8.0}


def test_admission_selects_best_feasible_training_score():
    selected, info = supervised.admit_refit_candidate(SCORES, 'balanced_accuracy', COSTS, 5)
    assert selected == 'linear'
    assert info['selected_before'] == 'histogram'
    assert info['estimated_refit_seconds']['histogram'] == 12
    assert info['rejected_candidates'] == {'histogram': 'estimated refit exceeds headroom'}
    assert set(info['eligible_candidates']) == {'extra_trees', 'linear'}


def test_generous_headroom_keeps_histogram_and_no_headroom_reuses_baseline():
    assert supervised.admit_refit_candidate(SCORES, 'accuracy', COSTS, 30)[0] == 'histogram'
    assert supervised.admit_refit_candidate(SCORES, 'accuracy', COSTS, -1)[0] == 'extra_trees'
    selected, info = supervised.admit_refit_candidate(SCORES, 'accuracy', {}, 5)
    assert selected == 'extra_trees'
    assert info['estimated_refit_seconds']['linear'] is None
    json.dumps(info, allow_nan=False)


@pytest.mark.parametrize('value', [1, 'true', None])
def test_admission_control_strict_and_disabled_by_default(value):
    assert validate_config()['experimental']['refit_admission'] is False
    with pytest.raises(ValueError):
        validate_config({'experimental': {'refit_admission': value}})


@pytest.mark.parametrize('other', [{'classification_holdouts': 2}, {'categorical_enabled': True},
                                  {'classification_weighting': 'both'}, {'classification_objective': 'macro_f1'}])
def test_unsupported_portfolio_combinations_fail_explicitly(other):
    with pytest.raises(ValueError):
        validate_config({'experimental': {'refit_admission': True, **other}})


def test_real_selector_uses_caller_deadline_not_local_budget(monkeypatch):
    frame = pd.DataFrame({'x': np.arange(100, dtype=float)})
    labels = np.array(['01', '1'] * 50)
    plan = make_route_plan(summarize_training('classification', frame, labels),
                           config={'experimental': {'refit_admission': True}})
    clock = [0.0]
    final_names = []

    def fit(name, task, train, target, test, *args, **kwargs):
        if kwargs.get('final', True):
            final_names.append(name)
        else:
            clock[0] += COSTS[name]
        return np.resize(np.asarray(target), len(test)), {
            'selected_candidate': name, 'fit_seconds': .1, 'prediction_seconds': 0,
        }

    monkeypatch.setattr(supervised.time, 'monotonic', lambda: clock[0])
    monkeypatch.setattr(supervised, 'fit_predict_fixed_candidate', fit)
    # Scores are supplied in deterministic fit order, while the clock records
    # real total validation work (8.5 seconds in this controlled fixture).
    scores = iter(SCORES.values())
    monkeypatch.setattr(supervised, 'metric_score', lambda *args: next(scores))
    baseline = (labels[:8].copy(), {'selected_candidate': 'extra_trees'})
    _, info = supervised.fit_predict_supervised('classification', frame, labels, frame.iloc[:8],
        route_plan=plan, completed_baseline=baseline, time_budget=50, execution_deadline=16)
    assert final_names == ['linear']
    assert info['refit_admission']['deadline_source'] == 'parent_absolute'
    assert info['refit_admission']['headroom_seconds'] == pytest.approx(2.5)
    assert info['validation_scores'] == SCORES


def test_worker_forwards_absolute_deadline_only_when_supplied(monkeypatch, tmp_path):
    from types import SimpleNamespace
    observed = []
    monkeypatch.setattr(models, 'fit_predict', lambda **kwargs: (observed.append(kwargs) or ['01'], {}))
    data = SimpleNamespace(task='classification', X_train=pd.DataFrame({'x': [1]}),
        y_train=['01'], X_test=pd.DataFrame({'x': [2]}), feature_types={'x': 'numeric'}, n_clusters=None)
    for deadline in (None, 16.0):
        result = tmp_path / f'{deadline}.pkl'
        runner._model_worker(data, 50, 42, str(result), 'routed', {}, str(tmp_path / 'checkpoint'),
                             execution_deadline=deadline)
        with result.open('rb') as handle:
            assert 'error' not in pickle.load(handle)
    assert 'execution_deadline' not in observed[0]
    assert observed[1]['execution_deadline'] == 16.0


def test_deadline_reaches_routed_selector_through_dispatch(monkeypatch):
    from unknown_dataset import routed
    observed = []
    frame = pd.DataFrame({'x': np.arange(100, dtype=float)})
    labels = np.array(['01', '1'] * 50)
    monkeypatch.setattr(supervised, 'fit_baseline_supervised',
                        lambda **kwargs: (labels[:8], {'model': 'completed'}))

    def optional(**kwargs):
        observed.append(kwargs)
        return labels[:8], {'model': 'selected'}

    monkeypatch.setattr(supervised, 'fit_predict_supervised', optional)
    for deadline in (None, 16.0):
        models.fit_predict('classification', frame, labels, frame.iloc[:8],
                           strategy='routed', execution_deadline=deadline)
    assert 'execution_deadline' not in observed[0]
    assert observed[1]['execution_deadline'] == 16.0


def test_explicit_experimental_selector_cannot_silently_ignore_admission():
    frame = pd.DataFrame({'x': np.arange(100, dtype=float)})
    labels = np.array(['01', '1'] * 50)
    plan = make_route_plan(summarize_training('classification', frame, labels),
                           config={'experimental': {'refit_admission': True}})
    with pytest.raises(ValueError, match='original selector'):
        supervised.fit_predict_supervised('classification', frame, labels, frame.iloc[:8],
            route_plan=plan, classification_controls=supervised.ClassificationControls())
