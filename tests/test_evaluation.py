"""Contract tests for shared binary probability evaluation."""

import numpy as np
import pytest

from src.evaluation import evaluate_probabilities


def test_probability_metrics_and_baseline():
    y = np.array([0, 0, 0, 1])
    proba = np.array([[.9, .1], [.8, .2], [.6, .4], [.1, .9]])
    result = evaluate_probabilities(y, proba)
    assert result["accuracy"] == 1.0
    assert result["majority_accuracy"] == .75
    assert result["accuracy_lift"] == .25
    assert result["confusion_matrix"] == [[3, 0], [0, 1]]


def test_invalid_probabilities_are_rejected():
    with pytest.raises(ValueError):
        evaluate_probabilities(np.array([0]), np.array([[.2, .2]]))
