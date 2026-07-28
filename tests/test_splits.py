"""Regression tests for deterministic split membership and leakage guards."""

import numpy as np

from src.preprocessing.build_dataset import assert_no_leak, make_test_mask


def test_repo_mask_matches_every_fourth_window():
    idx = np.arange(8)
    mask = make_test_mask("repo", idx, idx, np.ones(8))
    assert mask.tolist() == [True, False, False, False, True, False, False, False]


def test_group_splits_use_declared_ids():
    idx = np.arange(4)
    assert make_test_mask("trial", idx, np.array([1, 2, 3, 4]), np.ones(4), held_out_trials=[2]).tolist() == [False, True, False, False]
    assert make_test_mask("subject", idx, idx, np.array([1, 2, 3, 4]), held_out_subjects=[3]).tolist() == [False, False, True, False]


def test_nonleaky_grouped_split_passes():
    assert_no_leak("subject", np.array([1, 2]), np.array([3]), np.array([1, 2]), np.array([3]))
