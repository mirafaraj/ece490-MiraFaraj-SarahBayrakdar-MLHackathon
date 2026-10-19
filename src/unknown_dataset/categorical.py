"""Experimental fitting-fold-only single-category count classifier."""
from __future__ import annotations

import math
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_is_fitted


def category_token(value):
    """Disjoint namespaces preserve lexical values, including leading zeroes."""
    return ("missing", "") if pd.isna(value) else ("value", str(value))


class SmoothedCategoricalClassifier(ClassifierMixin, BaseEstimator):
    """P(k|c)=(n(c,k)+alpha*prior[k])/(n(c)+alpha), unseen -> prior.

    Feature and alpha are hyperparameters; callers must select them using fitting
    folds and validation rows, never counts computed over a full outer dataset.
    """

    def __init__(self, feature, alpha=10.0, class_weight=None, max_cardinality=512):
        self.feature = feature
        self.alpha = alpha
        self.class_weight = class_weight
        self.max_cardinality = max_cardinality

    def fit(self, X, y):
        if isinstance(self.alpha, bool) or not isinstance(self.alpha, (float, int)) or not math.isfinite(self.alpha) or self.alpha <= 0:
            raise ValueError("alpha must be finite and positive")
        if self.class_weight not in {None, "balanced"}:
            raise ValueError("class_weight must be None or balanced")
        if isinstance(self.max_cardinality, bool) or not isinstance(self.max_cardinality, int) or self.max_cardinality < 1:
            raise ValueError("max_cardinality must be a positive integer")
        if len(X) != len(y) or not len(X):
            raise ValueError("Aligned nonempty fitting features and labels required")
        self.encoder_ = LabelEncoder().fit(np.asarray(y))
        self.classes_ = self.encoder_.classes_
        target = self.encoder_.transform(np.asarray(y))
        counts = np.bincount(target, minlength=len(self.classes_)).astype(float)
        self.prior_ = counts / counts.sum()
        self.counts_ = {}
        for value, label in zip(X[self.feature], target):
            token = category_token(value)
            if token not in self.counts_:
                if len(self.counts_) >= self.max_cardinality:
                    raise ValueError("Categorical cardinality exceeds configured limit")
                self.counts_[token] = np.zeros(len(self.classes_), dtype=float)
            self.counts_[token][label] += 1
        return self

    def predict_proba(self, X):
        check_is_fitted(self, "prior_")
        result = np.empty((len(X), len(self.classes_)), dtype=float)
        for row, value in enumerate(X[self.feature]):
            counts = self.counts_.get(category_token(value))
            result[row] = self.prior_ if counts is None else (counts + self.alpha * self.prior_) / (counts.sum() + self.alpha)
        return result

    def predict(self, X):
        probabilities = self.predict_proba(X)
        # Balanced objective changes decisions, not the reported count posterior.
        if self.class_weight == "balanced":
            probabilities = probabilities / self.prior_[None, :]
        return self.classes_[np.argmax(probabilities, axis=1)]
