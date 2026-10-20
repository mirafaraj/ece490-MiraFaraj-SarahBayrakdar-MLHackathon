from copy import deepcopy
import numpy as np
from scripts.third_classification import case_data, metrics, select

def test_frozen_cases_exact_class_support_and_mixed_columns():
    for count in (2, 3):
        frame, labels, features = case_data(98003, 'localized_interaction', count)
        assert frame.shape == (600, 8)
        assert np.sum(labels == 'majority') == 570
        assert sorted(np.unique(labels, return_counts=True)[1]) == ([30, 570] if count == 2 else [15, 15, 570])
        assert set(features.values()) == {'numeric', 'categorical'}
        other, target, _ = case_data(98003, 'localized_interaction', count)
        assert frame.equals(other) and np.array_equal(labels, target)

def test_probability_metrics_hand_computation():
    result = metrics(['majority', 'rare_a'], ['majority', 'majority'], [[.75, .25], [.75, .25]], ['majority', 'rare_a'])
    assert result['accuracy'] == .5 and result['balanced_accuracy'] == .5
    assert result['macro_f1'] == 1/3 and result['minority_recall'] == 0
    assert result['brier_multiclass'] == .625

def test_selection_uses_inner_metrics_only_and_replaces_histogram():
    inner = {name+':balanced': {'seconds': .1, 'metrics': {'accuracy': value}, 'outer': value} for name, value in
        [('extra_trees', .5), ('linear', .6), ('histogram', .7), ('histogram_leaf5', .8)]}
    assert select(inner, 'balanced', 'accuracy')['selected'] == 'histogram:balanced'
    before = select(inner, 'balanced', 'accuracy', True)
    changed = deepcopy(inner)
    for row in changed.values(): row['outer'] = {'truth': ['changed']}
    assert before == select(changed, 'balanced', 'accuracy', True)
    assert 'histogram:balanced' not in before['validation_scores']

def test_cap_excludes_late_candidate():
    inner = {name+':balanced': {'seconds': 5., 'metrics': {'accuracy': value}} for name, value in
        [('extra_trees', .5), ('linear', .6), ('histogram', .7)]}
    chosen = select(inner, 'balanced', 'accuracy', cap=8.)
    assert chosen['selected'] == 'linear:balanced'
    assert chosen['excluded'] == {'histogram:balanced': 'shared selection cap exhausted'}
