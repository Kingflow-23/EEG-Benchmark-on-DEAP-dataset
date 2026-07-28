"""Contract tests for shared binary probability evaluation."""

import numpy as np
import pytest

from src.evaluation import evaluate_probabilities


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
