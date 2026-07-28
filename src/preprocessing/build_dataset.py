"""
Stage 2 -- pool the 32 per-subject feature caches into train/test arrays.

Three split modes, selected by ``split_mode``:

``repo``
    Every 4th window of each subject goes to test (``i % 4 == 0`` in
    ``PreProcessing/BuildDataset.py``). This is the split behind the published
    92-94%. It leaks heavily: at a 2 s window with a 0.125 s step, consecutive
    windows share 93.75% of their raw samples, so every test window overlaps
    ~75% with training windows drawn from the same trial and the same subject.
    Reproduced so the paper's numbers can be regenerated, not to be reported as
    a generalisation estimate.

``trial``
    Whole videos held out. No window from a test trial appears in training.
    Same subjects on both sides, so this measures generalisation to unseen
    stimuli for a known viewer.

``subject``
    Whole subjects held out. Nothing from a test participant is ever seen in
    training. This is the honest number for the deployment case -- showing an
    ad to a viewer the model was not trained on.

Outputs go to ``output/DATASET/<split_mode>/`` as ``.npy`` (not ``.npz``) so the
training loop can memory-map them; the pooled feature matrix is ~400 MB in
float32 and does not need to sit in RAM twice.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np

from ..config import (
    DATASET_PATH, DEFAULT_SPLIT, FEATURES_PATH, HELD_OUT_SUBJECTS,
    HELD_OUT_TRIALS, N_FEATURES, N_SUBJECTS, SPLIT_MODES,
)
from ..utils import human_bytes
from .fft_features import extraction_config, features_path_for, load_extraction_config


# --------------------------------------------------------------------------- #
# Split logic
# --------------------------------------------------------------------------- #
def make_test_mask(
    split_mode: str,
    index_in_subject: np.ndarray,
    trial_id: np.ndarray,
    subject_id: np.ndarray,
    held_out_trials: Iterable[int] = HELD_OUT_TRIALS,
    held_out_subjects: Iterable[int] = HELD_OUT_SUBJECTS,
) -> np.ndarray:
    """Return the test-membership mask for one participant cache.

    All arrays describe the same rows. ``index_in_subject`` is required only by
    the reference ``repo`` split; trial and subject modes use their named IDs.
    Optional held-out collections make the pure split logic easy to test.
    """
    if split_mode == "repo":
        # Faithful to BuildDataset.py: `for i in range(sub.shape[0]): if i % 4 == 0`
        return (index_in_subject % 4) == 0
    if split_mode == "trial":
        return np.isin(trial_id, np.asarray(list(held_out_trials), dtype=trial_id.dtype))
    if split_mode == "subject":
        return np.isin(subject_id, np.asarray(list(held_out_subjects), dtype=subject_id.dtype))
    raise ValueError(f"Unknown split_mode {split_mode!r}; expected one of {SPLIT_MODES}.")


def assert_no_leak(
    split_mode: str,
    trial_tr: np.ndarray, trial_te: np.ndarray,
    subj_tr: np.ndarray, subj_te: np.ndarray,
) -> None:
    """Assert that the grouping unit is disjoint across train and test.

    The check applies to ``trial`` and ``subject``. The ``repo`` split is exempt
    because overlapping groups are its documented reference behavior.
    """
    if split_mode == "trial":
        overlap = set(np.unique(trial_te)) & set(np.unique(trial_tr))
        if overlap:
            raise AssertionError(f"trial split leaked trials into both sides: {sorted(overlap)}")
    elif split_mode == "subject":
        overlap = set(np.unique(subj_te)) & set(np.unique(subj_tr))
        if overlap:
            raise AssertionError(f"subject split leaked subjects into both sides: {sorted(overlap)}")
    # "repo" leaks by construction -- nothing to assert.


# --------------------------------------------------------------------------- #
# Assembly
# --------------------------------------------------------------------------- #
def split_dir(split_mode: str, base: Path = DATASET_PATH) -> Path:
    """Return the directory containing one assembled split."""
    return Path(base) / split_mode


def build_dataset(
    subject_ids: Optional[Iterable[int]] = None,
    split_mode: str = DEFAULT_SPLIT,
    features_dir: Path = FEATURES_PATH,
    out_base: Path = DATASET_PATH,
    overwrite: bool = False,
    verbose: bool = True,
) -> Path:
    """Pool participant caches and persist one reproducible split.

    Parameters
    ----------
    subject_ids
        Participants to include; ``None`` means all 32 DEAP participants.
    split_mode
        ``repo``, ``trial``, or ``subject`` as described in the module header.
    features_dir, out_base
        Input cache root and output dataset root.
    overwrite
        Rebuild even when ``meta.json`` describes the same participant set.
    verbose
        Print participant counts and the final class distribution.

    Returns
    -------
    pathlib.Path
        Directory containing six arrays and ``meta.json``.

    Notes
    -----
    Feature arrays are ``float32 (n, 160)``; binary labels are ``int8 (n, 2)``
    ordered ``[Valence, Arousal]``. Metadata columns are documented in the
    emitted ``meta.json`` and retained for downstream aggregation.
    """
    if split_mode not in SPLIT_MODES:
        raise ValueError(f"Unknown split_mode {split_mode!r}; expected one of {SPLIT_MODES}.")

    subject_ids = (list(subject_ids) if subject_ids is not None
                   else list(range(1, N_SUBJECTS + 1)))
    if not subject_ids:
        raise ValueError("subject_ids cannot be empty")
    if len(set(subject_ids)) != len(subject_ids):
        raise ValueError("subject_ids cannot contain duplicates")
    invalid = [subject for subject in subject_ids if not 1 <= subject <= N_SUBJECTS]
    if invalid:
        raise ValueError(
            f"DEAP subject IDs must be in 1..{N_SUBJECTS}, got {invalid}"
        )
    missing = [s for s in subject_ids if not features_path_for(s, features_dir).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing feature caches for subjects {missing}. "
            "Run `python -m src.benchmark --prepare ...` first."
        )
    feature_configs = [
        load_extraction_config(features_path_for(s, features_dir))
        for s in subject_ids
    ]
    feature_config = feature_configs[0]
    if any(item != feature_config for item in feature_configs[1:]):
        raise ValueError(
            "Participant feature caches use different extraction settings; "
            "re-extract all requested subjects with --overwrite."
        )
    out_dir = split_dir(split_mode, out_base)
    marker = out_dir / "meta.json"
    if marker.exists() and not overwrite:
        # Only reuse a cached build if it covers exactly the requested subjects.
        # Otherwise a partial validation run (say 5 subjects) would silently be
        # reused for a full 32-subject job, and every downstream number would be
        # computed on a fifth of the data without anything saying so.
        try:
            cached_meta = json.loads(marker.read_text(encoding="utf-8"))
            cached = cached_meta.get("subject_ids")
            cached_feature_config = cached_meta.get("feature_config")
        except (json.JSONDecodeError, OSError):
            cached = None
            cached_feature_config = None
        if cached == subject_ids and cached_feature_config == feature_config:
            if verbose:
                print(f"  {split_mode}: already built at {out_dir}, skipping")
            return out_dir
        if verbose:
            n_cached = len(cached) if cached else "?"
            reason = ("participant set changed" if cached != subject_ids
                      else "feature settings changed")
            print(
                f"  {split_mode}: {reason} (cached subjects={n_cached}, "
                f"requested={len(subject_ids)}) -- rebuilding"
            )

    X_tr, y_tr, m_tr = [], [], []
    X_te, y_te, m_te = [], [], []

    for sid in subject_ids:
        with np.load(features_path_for(sid, features_dir)) as d:
            X = d["X"]
            y_bin = d["y_bin"]
            y_cont = d["y_cont"]
            trial_id = d["trial_id"]
            window_id = d["window_id"]
            subject_arr = d["subject_id"]

        idx = np.arange(len(X), dtype=np.int64)
        test = make_test_mask(split_mode, idx, trial_id, subject_arr)
        train = ~test

        # meta columns kept alongside every window: subject, trial, window index,
        # and the four continuous ratings. Needed by the Gap Report stage.
        meta = np.column_stack([
            subject_arr.astype(np.float32),
            trial_id.astype(np.float32),
            window_id.astype(np.float32),
            y_cont.astype(np.float32),
        ]).astype(np.float32)

        if train.any():
            X_tr.append(X[train]); y_tr.append(y_bin[train]); m_tr.append(meta[train])
        if test.any():
            X_te.append(X[test]); y_te.append(y_bin[test]); m_te.append(meta[test])

        if verbose:
            print(f"  s{sid:02d}: train {int(train.sum()):>6,}  test {int(test.sum()):>6,}")

    def _cat(parts, name):
        if not parts:
            raise ValueError(
                f"{name} is empty under split_mode={split_mode!r}. "
                "Check HELD_OUT_SUBJECTS / HELD_OUT_TRIALS against subject_ids."
            )
        return np.concatenate(parts, axis=0)

    X_train = _cat(X_tr, "training set"); y_train = _cat(y_tr, "training labels")
    meta_train = _cat(m_tr, "training meta")
    X_test = _cat(X_te, "test set"); y_test = _cat(y_te, "test labels")
    meta_test = _cat(m_te, "test meta")

    assert_no_leak(
        split_mode,
        meta_train[:, 1], meta_test[:, 1],      # trial ids
        meta_train[:, 0], meta_test[:, 0],      # subject ids
    )
    assert X_train.shape[1] == N_FEATURES, f"expected {N_FEATURES} features, got {X_train.shape[1]}"

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "data_training.npy", X_train)
    np.save(out_dir / "label_training.npy", y_train)
    np.save(out_dir / "meta_training.npy", meta_train)
    np.save(out_dir / "data_testing.npy", X_test)
    np.save(out_dir / "label_testing.npy", y_test)
    np.save(out_dir / "meta_testing.npy", meta_test)

    meta_info = {
        "split_mode": split_mode,
        "subject_ids": subject_ids,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "test_fraction": round(len(X_test) / (len(X_train) + len(X_test)), 4),
        "n_features": int(X_train.shape[1]),
        "feature_config": feature_config,
        "label_columns": ["Valence", "Arousal"],
        "meta_columns": ["subject_id", "trial_id", "window_id",
                         "Valence", "Arousal", "Dominance", "Liking"],
        "held_out_subjects": list(HELD_OUT_SUBJECTS) if split_mode == "subject" else None,
        "held_out_trials": list(HELD_OUT_TRIALS) if split_mode == "trial" else None,
        "positive_rate_train": {
            "Valence": round(float(y_train[:, 0].mean()), 4),
            "Arousal": round(float(y_train[:, 1].mean()), 4),
        },
        "positive_rate_test": {
            "Valence": round(float(y_test[:, 0].mean()), 4),
            "Arousal": round(float(y_test[:, 1].mean()), 4),
        },
        "leakage_warning": (
            "Adjacent windows overlap 93.75%; this split places overlapping "
            "windows from the same trial and subject on both sides. Accuracy "
            "from this split is not a generalisation estimate."
        ) if split_mode == "repo" else None,
    }
    marker.write_text(json.dumps(meta_info, indent=2), encoding="utf-8")

    if verbose:
        nbytes = X_train.nbytes + X_test.nbytes
        print(
            f"\n  [{split_mode}] train {X_train.shape}  test {X_test.shape}  "
            f"({human_bytes(nbytes)} on disk)"
        )
        print(f"  positive rate  train V={meta_info['positive_rate_train']['Valence']:.3f} "
              f"A={meta_info['positive_rate_train']['Arousal']:.3f}  |  "
              f"test V={meta_info['positive_rate_test']['Valence']:.3f} "
              f"A={meta_info['positive_rate_test']['Arousal']:.3f}")
        print(f"  -> {out_dir}")

    return out_dir


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_split(
    split_mode: str = DEFAULT_SPLIT,
    base: Path = DATASET_PATH,
    mmap: bool = True,
    with_meta: bool = False,
) -> Tuple[np.ndarray, ...]:
    """Load an assembled split.

    Parameters
    ----------
    split_mode
        Name of the cached split below ``base``.
    base
        Root directory containing split subdirectories.
    mmap
        Memory-map the two feature matrices read-only. Labels and metadata are
        small enough to load normally.
    with_meta
        Include the provenance matrices in the returned tuple.

    Returns ``(X_train, y_train, X_test, y_test)``, or with ``with_meta=True``
    ``(X_train, y_train, meta_train, X_test, y_test, meta_test)``.

    ``mmap=True`` memory-maps the feature matrices, which keeps peak RAM low --
    the pooled training matrix alone is ~300 MB in float32.

    Raises ``FileNotFoundError`` when the requested split has not been built.
    """
    d = split_dir(split_mode, base)
    if not (d / "meta.json").exists():
        raise FileNotFoundError(
            f"No dataset at {d}. Run `python -m src.benchmark --prepare --split {split_mode}`."
        )
    info = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    if info.get("feature_config") != extraction_config():
        raise ValueError(
            f"Dataset at {d} was built with different preprocessing settings. "
            "Re-run preparation with --overwrite."
        )
    mm = "r" if mmap else None
    out = [
        np.load(d / "data_training.npy", mmap_mode=mm),
        np.load(d / "label_training.npy"),
    ]
    if with_meta:
        out.append(np.load(d / "meta_training.npy"))
    out += [
        np.load(d / "data_testing.npy", mmap_mode=mm),
        np.load(d / "label_testing.npy"),
    ]
    if with_meta:
        out.append(np.load(d / "meta_testing.npy"))
    return tuple(out)


def load_split_info(split_mode: str = DEFAULT_SPLIT, base: Path = DATASET_PATH) -> dict:
    """Load the provenance and class-distribution metadata for one split."""
    return json.loads((split_dir(split_mode, base) / "meta.json").read_text(encoding="utf-8"))
