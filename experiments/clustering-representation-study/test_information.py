"""Information tracing and reference isolation; no preprocessing changes."""
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from unknown_dataset.clustering import fit_predict_clustering, _scale_missing_indicators
from unknown_dataset.preprocessing import TableCleaner, build_preprocessor


def test_observed_constant_columns_keep_distinct_missing_masks():
    frame = pd.DataFrame({"a": [7., np.nan, 7., np.nan, 7., np.nan],
        "b": [np.nan, 7., 7., np.nan, np.nan, 7.], "constant": 7., "empty": np.nan})
    frame["same_mask_duplicate"] = frame.a
    cleaner = TableCleaner().fit(frame)
    assert cleaner.retained_columns_ == ["a", "b"]
    assert cleaner.duplicate_columns_ == ["same_mask_duplicate"]
    assert set(cleaner.constant_columns_) == {"constant", "empty"}
    processor = build_preprocessor(cleaner, scale_numeric=True)
    matrix = processor.fit_transform(cleaner.transform(frame))
    imputer = processor.named_transformers_["numeric"].named_steps["impute"]
    np.testing.assert_array_equal(imputer.indicator_.features_, [0, 1])
    # Values become constant after imputation, but the two independent masks survive.
    np.testing.assert_array_equal(matrix[:, :2], np.zeros((6, 2)))
    for j, column in enumerate(["a", "b"]):
        mask = frame[column].isna().to_numpy()
        np.testing.assert_array_equal(matrix[:, j + 2] > 0, mask)
    np.testing.assert_array_equal(_scale_missing_indicators(processor, matrix, .25)[:, 2:], matrix[:, 2:] * .25)


def test_categorical_missingness_and_observed_values_remain_distinct():
    frame = pd.DataFrame({"category": ["same", None, "same", None], "value": [1., 2., 3., 4.]})
    cleaner = TableCleaner().fit(frame)
    processor = build_preprocessor(cleaner, scale_numeric=True)
    matrix = processor.fit_transform(cleaner.transform(frame))
    matrix = matrix.toarray() if sparse.issparse(matrix) else matrix
    block = processor.output_indices_["categorical"]
    assert not np.array_equal(matrix[0, block], matrix[1, block])
    scaled = _scale_missing_indicators(processor, matrix, .25)
    np.testing.assert_array_equal(scaled[:, block], matrix[:, block])


@pytest.mark.parametrize("controls", [{}, {"clustering_indicator_amplitude": .25},
    {"clustering_missing_indicators": False}])
def test_references_and_test_features_do_not_change_clustering(controls):
    rng = np.random.default_rng(104729)
    frame = pd.DataFrame(rng.normal(size=(90, 2)), columns=["a", "b"])
    frame.loc[::3, "a"] = np.nan
    original = frame.copy(deep=True)
    first, info = fit_predict_clustering("clustering", frame, n_clusters=3, seed=104729,
        time_budget=20, experimental_controls=controls)
    changed, changed_info = fit_predict_clustering("clustering", frame, n_clusters=3, seed=104729,
        time_budget=20, experimental_controls=controls, y_train=rng.permutation(90),
        X_test=pd.DataFrame({"unrelated": [-999]}))
    np.testing.assert_array_equal(first, changed)
    for key in ("model", "selection", "bic", "transformed_features"):
        assert info[key] == changed_info[key]
    pd.testing.assert_frame_equal(frame, original)
