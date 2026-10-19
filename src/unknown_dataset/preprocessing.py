"""Train-fitted mixed-table preprocessing shared by the baseline models."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class TableCleaner(BaseEstimator, TransformerMixin):
    """Normalize values and learn removable columns from training data only."""

    def __init__(self, feature_types=None):
        self.feature_types = feature_types

    def _clean(self, X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Model features must be a pandas DataFrame")
        missing = set(self.columns_) - set(X.columns)
        if missing:
            raise ValueError(f"Missing feature columns: {sorted(missing)}")
        result = X.loc[:, self.columns_].copy()
        for name in self.columns_:
            if self.types_[name] == "numeric":
                result[name] = pd.to_numeric(result[name], errors="coerce").astype(float)
                result[name] = result[name].replace([np.inf, -np.inf], np.nan)
            else:
                # Strings prevent sklearn sorting errors on mixed scalar types.
                result[name] = result[name].map(
                    # Reserve a disjoint namespace for genuine category values;
                    # no user string can collide with the imputation token.
                    lambda value: np.nan if pd.isna(value) else "value:" + str(value)
                ).astype(object)
        return result

    def fit(self, X, y=None):
        self.columns_ = list(X.columns)
        provided = self.feature_types or {}
        self.types_ = {
            name: provided.get(
                name, "numeric" if pd.api.types.is_numeric_dtype(X[name]) else "categorical"
            )
            for name in self.columns_
        }
        invalid = set(self.types_.values()) - {"numeric", "categorical"}
        if invalid:
            raise ValueError(f"Unsupported feature types: {sorted(invalid)}")
        clean = self._clean(X)
        self.constant_columns_ = [c for c in self.columns_ if clean[c].nunique(dropna=False) <= 1]
        self.duplicate_columns_ = []
        retained = []
        # Hashing limits expensive equality checks to likely exact matches.
        signatures = {}
        for name in self.columns_:
            if name in self.constant_columns_:
                continue
            signature = (self.types_[name], int(pd.util.hash_pandas_object(clean[name], index=False).sum()))
            candidates = signatures.setdefault(signature, [])
            if any(clean[name].equals(clean[other]) for other in candidates):
                self.duplicate_columns_.append(name)
            else:
                retained.append(name)
                candidates.append(name)
        self.retained_columns_ = retained
        self.numeric_columns_ = [c for c in retained if self.types_[c] == "numeric"]
        self.categorical_columns_ = [c for c in retained if self.types_[c] == "categorical"]
        return self

    def transform(self, X):
        return self._clean(X).loc[:, self.retained_columns_]


class ConstantFeature(BaseEstimator, TransformerMixin):
    """Keep estimators usable when training contains no informative columns."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.zeros((len(X), 1), dtype=np.float64)


def build_preprocessor(cleaner, scale_numeric=False):
    """Build transformations using column decisions learned by ``cleaner.fit``."""
    transformers = []
    if cleaner.numeric_columns_:
        numeric = [("impute", SimpleImputer(strategy="median", keep_empty_features=True, add_indicator=True))]
        if scale_numeric:
            numeric.append(("scale", StandardScaler()))
        transformers.append(("numeric", Pipeline(numeric), cleaner.numeric_columns_))
    if cleaner.categorical_columns_:
        categorical = Pipeline([
            ("impute", SimpleImputer(strategy="constant", fill_value="__MISSING__", keep_empty_features=True)),
            ("encode", OneHotEncoder(
                handle_unknown="infrequent_if_exist", max_categories=32,
                sparse_output=True, dtype=np.float64,
            )),
        ])
        transformers.append(("categorical", categorical, cleaner.categorical_columns_))
    if not transformers:
        return ConstantFeature()
    return ColumnTransformer(transformers, remainder="drop", sparse_threshold=1.0)
