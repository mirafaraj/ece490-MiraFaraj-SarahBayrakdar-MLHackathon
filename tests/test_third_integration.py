"""Optional primary changes retain training-only selection and fixed CLI inventory."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from third_runtime import plans
from unknown_dataset.routing import make_route_plan, summarize_training
from unknown_dataset.supervised import fit_predict_supervised


def test_conditional_cli_inventory_is_paired_and_deterministic():
    rows = plans(['classification'])
    assert rows == plans(['classification'])
    assert len(rows) == 18
    assert len({(r['budget'], r['repeat'], r['policy']) for r in rows}) == 18
    assert {r['policy'] for r in rows} == {'baseline', 'current', 'challenger'}


@pytest.mark.parametrize('task', ['classification', 'regression'])
def test_optional_selection_does_not_depend_on_test_features(task):
    rng = np.random.default_rng(9137)
    train = pd.DataFrame(rng.normal(size=(260, 4)), columns=list('abcd'))
    train.loc[::19, 'a'] = np.nan
    signal = train.a.fillna(0) * train.b + train.c
    labels = np.where(signal > 1, '01', '1') if task == 'classification' else signal.to_numpy()
    config = {'experimental': {'histogram_min_samples_leaf': 5}} if task == 'classification' else {'experimental': {'histogram_max_leaf_nodes': 31}}
    plan = make_route_plan(summarize_training(task, train, labels), 50, config=config)
    first = train.iloc[:16].copy()
    changed = first * 100 + 300
    _, a = fit_predict_supervised(task, train, labels, first, seed=83, time_budget=50, route_plan=plan)
    _, b = fit_predict_supervised(task, train, labels, changed, seed=83, time_budget=50, route_plan=plan)
    assert a['selected_candidate'] == b['selected_candidate']
    assert a['validation_scores'] == b['validation_scores']
    assert a['candidate_errors'] == b['candidate_errors']
