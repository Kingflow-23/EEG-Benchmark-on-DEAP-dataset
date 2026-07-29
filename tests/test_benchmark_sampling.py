"""Tests for deterministic model-specific benchmark sampling."""

import numpy as np

from src.benchmark import _model_sample_indices


def _joint_labels(repeats: int) -> np.ndarray:
    """Return balanced rows covering every Valence/Arousal combination."""
    return np.tile(np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), (repeats, 1))


def test_tabicl_sampling_is_capped_stratified_and_deterministic():
    labels = _joint_labels(100)
    first = _model_sample_indices("tabicl", labels, labels, 42, 40, 20)
    second = _model_sample_indices("tabicl", labels, labels, 42, 40, 20)

    assert len(first[0]) == 40
    assert len(first[1]) == 20
    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])
    assert set(map(tuple, labels[first[0]])) == {(0, 0), (0, 1), (1, 0), (1, 1)}
    assert set(map(tuple, labels[first[1]])) == {(0, 0), (0, 1), (1, 0), (1, 1)}


def test_other_models_keep_every_selected_row():
    train = _joint_labels(11)
    test = _joint_labels(7)
    train_idx, test_idx = _model_sample_indices(
        "feature_mlp", train, test, 42, 4, 4
    )

    np.testing.assert_array_equal(train_idx, np.arange(len(train)))
    np.testing.assert_array_equal(test_idx, np.arange(len(test)))
