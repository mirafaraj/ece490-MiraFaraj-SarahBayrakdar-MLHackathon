import pytest

from scripts.second_verify_evidence import adjusted_rand, classification_metrics


def test_independent_rare_class_metric_arithmetic():
    metrics = classification_metrics(['a', 'a', 'a', 'b'], ['a', 'a', 'b', 'a'], ['a', 'b'])
    assert metrics['accuracy'] == .5
    assert metrics['balanced_accuracy'] == pytest.approx(1 / 3)
    assert metrics['macro_f1'] == pytest.approx(1 / 3)
    assert metrics['class_recall'] == [2 / 3, 0]


def test_lexical_labels_remain_distinct():
    metrics = classification_metrics(['01', '1'], ['01', '1'], ['01', '1'])
    assert metrics['accuracy'] == metrics['macro_f1'] == 1


def test_independent_ari_matches_hand_computed_cross_partition():
    assert adjusted_rand([0, 0, 1, 1], [0, 1, 0, 1]) == pytest.approx(-.5)
    assert adjusted_rand([0, 0, 1, 1], [9, 9, 8, 8]) == 1
