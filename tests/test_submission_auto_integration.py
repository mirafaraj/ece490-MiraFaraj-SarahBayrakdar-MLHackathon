"""Submission delegates without changing existing strategies or their controls."""
import inspect

import numpy as np
import pandas as pd
import pytest

from unknown_dataset.models import fit_predict
from unknown_dataset.reliability import effective_strategy


@pytest.mark.parametrize('task', ['classification', 'regression', 'clustering'])
def test_default_submission_and_auto_predictions_and_selection_match(task):
    rng = np.random.default_rng(47131)
    frame = pd.DataFrame(rng.normal(size=(90, 4)), columns=list('abcd'))
    target = None if task == 'clustering' else (
        np.where(frame.a + frame.b > 0, 'yes', 'no') if task == 'classification'
        else frame.a * 2 + frame.b)
    args = dict(task=task, X_train=frame, y_train=target,
                X_test=None if task == 'clustering' else frame.iloc[:12],
                n_clusters=3 if task == 'clustering' else None, seed=47131)
    results = [fit_predict(**args, **option) for option in
               ({}, {'strategy': 'submission'}, {'strategy': 'auto'})]
    for prediction, info in results[1:]:
        np.testing.assert_array_equal(prediction, results[0][0])
        for key in ('model', 'selected_candidate'):
            assert info.get(key) == results[0][1].get(key)


def test_existing_explicit_strategy_names_and_api_default_are_preserved():
    assert inspect.signature(fit_predict).parameters['strategy'].default == 'submission'
    assert 'classification_metric' not in inspect.signature(fit_predict).parameters
    for task in ('classification', 'regression', 'clustering'):
        for strategy in ('baseline', 'auto', 'routed'):
            assert effective_strategy(task, strategy) == strategy
        assert effective_strategy(task, 'submission') == 'auto'
