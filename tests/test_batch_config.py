"""Experimental controls remain opt-in and strictly validated."""
import pytest

from unknown_dataset.policies import validate_config


def test_batch_defaults_preserve_policy():
    config = validate_config()
    assert config['metrics'] == {'classification': 'balanced_accuracy', 'regression': 'rmse', 'clustering': 'bic'}
    experimental = config['experimental']
    assert experimental['classification_weighting'] == 'objective'
    assert experimental['classification_holdouts'] == 1
    assert experimental['classification_objective'] == 'route'
    assert experimental['clustering_missing_indicators'] is True
    assert experimental['clustering_covariance'] == 'full'
    assert experimental['clustering_kmeans_only'] is False
    assert experimental['runtime_instrumentation'] is False


@pytest.mark.parametrize('key,value', [
    ('classification_weighting', 'auto'), ('classification_objective', 'official'),
    ('classification_holdouts', True), ('classification_holdouts', 3),
    ('clustering_missing_indicators', 0), ('clustering_covariance', 'tied'),
    ('clustering_kmeans_only', 'false'), ('runtime_instrumentation', 1),
])
def test_bad_batch_controls_rejected(key, value):
    with pytest.raises(ValueError):
        validate_config({'experimental': {key: value}})
