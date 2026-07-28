"""Smoke tests for registry discovery and estimator compatibility."""

import numpy as np

from src.models import available_models, create_model


def test_core_sklearn_models_expose_probabilities():
    X = np.array(
        [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0], [0.1, 0.2], [0.9, 0.8]]
    )
    y = np.array([0, 0, 1, 1, 0, 1])
    for name in ("logistic_regression",):
        model, _ = create_model(name, 42)
        model.fit(X, y)
        proba = model.predict_proba(X)
        assert proba.shape == (len(X), 2)
        np.testing.assert_allclose(proba.sum(axis=1), 1)


def test_curated_neural_architectures_are_registered():
    names = set(available_models())
    assert names == {
        "logistic_regression",
        "extra_trees",
        "xgboost",
        "feature_mlp",
        "band_electrode_cnn",
        "fft_lstm",
        "ft_transformer",
        "tabicl",
    }


def test_standard_suite_contains_only_curated_models():
    assert available_models() == [
        "logistic_regression",
        "extra_trees",
        "xgboost",
        "feature_mlp",
        "band_electrode_cnn",
        "fft_lstm",
        "ft_transformer",
        "tabicl",
    ]
