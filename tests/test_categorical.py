import numpy as np
import pandas as pd
import pytest

from unknown_dataset.categorical import SmoothedCategoricalClassifier


def test_smoothed_counts_and_unseen_prior_hand_calculation():
    X = pd.DataFrame({"category": ["a", "a", "b", "b"]})
    model = SmoothedCategoricalClassifier("category", alpha=2).fit(X, ["001", "001", "001", "1"])
    proba = model.predict_proba(pd.DataFrame({"category": ["a", "b", "unseen"]}))
    assert np.allclose(proba, [[0.875, 0.125], [0.625, 0.375], [0.75, 0.25]])
    assert np.allclose(proba.sum(axis=1), 1)
    assert model.predict(pd.DataFrame({"category": ["unseen"]})).tolist() == ["001"]


def test_missing_namespace_and_lexical_categories_are_distinct():
    X = pd.DataFrame({"c": [None, "missing", "001", "1", "__MISSING__"] * 4})
    model = SmoothedCategoricalClassifier("c", alpha=0.1).fit(X, [0, 1, 2, 3, 4] * 4)
    assert len(model.counts_) == 5
    assert model.predict(X.iloc[:5]).tolist() == list(range(5))


def test_fold_counts_unchanged_by_validation_feature_changes():
    fitting = pd.DataFrame({"c": ["a", "a", "b", None]})
    model = SmoothedCategoricalClassifier("c").fit(fitting, [0, 1, 1, 1])
    prior = model.prior_.copy()
    before = {k: v.copy() for k, v in model.counts_.items()}
    model.predict_proba(pd.DataFrame({"c": ["unseen", "unseen", "a"]}))
    assert np.array_equal(prior, model.prior_)
    assert before.keys() == model.counts_.keys()
    assert all(np.array_equal(v, model.counts_[k]) for k, v in before.items())


def test_balanced_decision_does_not_change_posterior():
    X = pd.DataFrame({"c": ["a"] * 8 + ["b"] * 2})
    y = [0] * 9 + [1]
    model = SmoothedCategoricalClassifier("c", alpha=1, class_weight="balanced").fit(X, y)
    assert model.predict(X.iloc[-1:]).tolist() == [1]
    assert np.isclose(model.predict_proba(X.iloc[-1:]).sum(), 1)


def test_single_class_fractional_labels_and_empty_predictions():
    X = pd.DataFrame({"c": [None, "a"]})
    model = SmoothedCategoricalClassifier("c").fit(X, [0.25, 0.25])
    assert model.predict(X).tolist() == [0.25, 0.25]
    assert model.predict(X.iloc[:0]).shape == (0,)


def test_cardinality_and_alpha_guards():
    X = pd.DataFrame({"c": ["a", "b"]})
    with pytest.raises(ValueError, match="cardinality"):
        SmoothedCategoricalClassifier("c", max_cardinality=1).fit(X, [0, 1])
    with pytest.raises(ValueError, match="alpha"):
        SmoothedCategoricalClassifier("c", alpha=0).fit(X, [0, 1])
