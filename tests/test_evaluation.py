"""Contract tests for shared binary probability evaluation."""

import numpy as np
import pytest

from src.evaluation import evaluate_probabilities, normalize_probabilities


def test_probability_metrics_and_baseline():
    y = np.array([0, 0, 0, 1])
    proba = np.array([[0.9, 0.1], [0.8, 0.2], [0.6, 0.4], [0.1, 0.9]])
    result = evaluate_probabilities(y, proba)
    assert result["accuracy"] == 1.0
    assert result["majority_accuracy"] == 0.75
    assert result["accuracy_lift"] == 0.25
    assert result["confusion_matrix"] == [[3, 0], [0, 1]]


def test_invalid_probabilities_are_rejected():
    with pytest.raises(ValueError):
        evaluate_probabilities(np.array([0]), np.array([[0.2, 0.2]]))


def test_small_probability_drift_is_normalized_exactly():
    probabilities = np.array([[0.70002, 0.29999], [0.19999, 0.80002]])
    normalized = normalize_probabilities(probabilities)

    np.testing.assert_allclose(normalized.sum(axis=1), 1.0, rtol=0, atol=1e-15)
    result = evaluate_probabilities(np.array([0, 1]), probabilities)
    assert result["accuracy"] == 1.0
