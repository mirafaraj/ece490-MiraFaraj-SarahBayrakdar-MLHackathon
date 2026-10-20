"""Task-local histogram controls preserve defaults and candidate portfolios."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from unknown_dataset.policies import validate_config
from unknown_dataset.routing import make_route_plan, summarize_training
from unknown_dataset import supervised


@pytest.mark.parametrize('key,value', [
    ('histogram_min_samples_leaf', True), ('histogram_min_samples_leaf', 10),
    ('histogram_min_samples_leaf', 5.0), ('histogram_max_leaf_nodes', 30),
    ('histogram_max_leaf_nodes', True), ('histogram_max_leaf_nodes', 31.0),
    ('clustering_indicator_amplitude', True), ('clustering_indicator_amplitude', 0),
    ('clustering_indicator_amplitude', 1.1), ('clustering_indicator_amplitude', float('nan')),
])
def test_invalid_controls_are_rejected(key, value):
    with pytest.raises(ValueError):
        validate_config({'experimental': {key: value}})


def test_defaults_and_portfolios_are_preserved():
    config = validate_config()
    assert config['experimental']['histogram_min_samples_leaf'] is None
    assert config['experimental']['histogram_max_leaf_nodes'] == 15
    assert config['experimental']['clustering_indicator_amplitude'] == 1
    frame = pd.DataFrame({'x': np.arange(450, dtype=float)})
    labels = np.arange(450) % 2
    for task in ('classification', 'regression'):
        summary = summarize_training(task, frame, labels)
        original = make_route_plan(summary, 50)
        changed = make_route_plan(summary, 50, config={'experimental': {
            'histogram_min_samples_leaf': 5, 'histogram_max_leaf_nodes': 31,
            'clustering_indicator_amplitude': .25}})
        for key in ('candidates', 'selection_budget_fraction', 'reserve_seconds', 'metric'):
            assert getattr(original, key) == getattr(changed, key)


@pytest.mark.parametrize('task,controls,minimum,nodes', [
    ('classification', {}, 20, 15),
    ('classification', {'histogram_min_samples_leaf': 5}, 5, 15),
    ('classification', {'histogram_max_leaf_nodes': 31}, 20, 15),
    ('regression', {}, 20, 15),
    ('regression', {'histogram_max_leaf_nodes': 31}, 20, 31),
    ('regression', {'histogram_min_samples_leaf': 5}, 20, 15),
])
def test_controls_reach_only_relevant_estimator(monkeypatch, task, controls, minimum, nodes):
    recorded = []
    class Estimator:
        def __init__(self, **kwargs):
            recorded.append(kwargs)
        def fit(self, matrix, labels):
            self.classes_ = np.unique(labels)
            return self
        def predict(self, matrix):
            return np.zeros(matrix.shape[0], dtype=int)
    monkeypatch.setattr(supervised, 'HistGradientBoostingClassifier', Estimator)
    monkeypatch.setattr(supervised, 'HistGradientBoostingRegressor', Estimator)
    frame = pd.DataFrame({'x': np.arange(450, dtype=float)})
    labels = np.arange(450) % 2
    plan = make_route_plan(summarize_training(task, frame, labels), 50,
                           config={'experimental': controls})
    for final in (False, True):
        prediction, info = supervised.fit_predict_fixed_candidate(
            'histogram', task, frame, labels, frame.iloc[:3], seed=42,
            route_plan=plan, final=final)
        assert len(prediction) == 3
        assert recorded[-1]['min_samples_leaf'] == minimum
        assert recorded[-1]['max_leaf_nodes'] == nodes
        assert recorded[-1]['max_iter'] == (100 if final else 60)
        assert recorded[-1]['max_depth'] == 8
        assert recorded[-1]['early_stopping'] is False


def test_admission_does_not_silently_combine_with_new_classifier():
    with pytest.raises(ValueError, match='original three-family selector'):
        validate_config({'experimental': {'refit_admission': True,
                                         'histogram_min_samples_leaf': 5}})
