"""
Stage 1 -- raw DEAP ``.dat`` -> per-window band-power feature vectors.

For every subject we produce one compressed ``.npz`` holding:

    X          (n_windows, 160)  float32  band powers, channel-major
    y_bin      (n_windows, 2)    int8     High/Low for [Valence, Arousal]
    y_cont     (n_windows, 4)    float32  raw 1-9 ratings [V, A, D, L]
    trial_id   (n_windows,)      int16    which of the 40 videos
    window_id  (n_windows,)      int16    index of the window inside its trial
    subject_id (n_windows,)      int16    constant, kept for pooled splits

``y_cont``, ``trial_id`` and ``subject_id`` are not needed to reproduce the
paper -- the reference repo stores only the binarised pair. We keep them
because the downstream Gap Report stage needs continuous ratings and needs to
know which window came from which video and viewer, and recomputing them later
would mean re-reading 3 GB of raw data.

Feature layout is channel-major to match the reference implementation:

    [Fp1_Theta, Fp1_Alpha, Fp1_LowerBeta, Fp1_UpperBeta, Fp1_Gamma,
     AF3_Theta, ...                                              ]
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

from ..config import (
    BAND_EDGES,
    BASELINE_SAMPLES,
    BINARISATION,
    BINARY_THRESHOLD,
    DROP_BASELINE,
    FEATURES_PATH,
    LABEL_INDEX,
    N_BANDS,
    N_SUBJECTS,
    N_CHANNELS,
    N_FEATURES,
    N_TRIALS,
    RAW_DATA_PATH,
    SAMPLE_RATE,
    STEP_SIZE,
    WINDOW_SIZE,
    subject_id_to_filename,
    windows_per_trial,
)
from .bandpower import bin_power


FEATURE_SCHEMA_VERSION = 1


# --------------------------------------------------------------------------- #
# Raw loading
# --------------------------------------------------------------------------- #
def load_raw_subject(subject_id: int, raw_path: Path = RAW_DATA_PATH) -> dict:
    """Load one DEAP ``sNN.dat`` participant dictionary.

    DEAP distributed these files as Python 2 pickles, so ``latin1`` decoding is
    required under Python 3. The returned dictionary contains ``data`` shaped
    ``(40 trials, 40 channels, 8064 samples)`` and ``labels`` shaped ``(40, 4)``.
    """
    if not 1 <= subject_id <= N_SUBJECTS:
        raise ValueError(
            f"DEAP subject_id must be in 1..{N_SUBJECTS}, got {subject_id}"
        )
    path = Path(raw_path) / subject_id_to_filename(subject_id)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Expected the DEAP 'preprocessed_python' files "
            f"(s01.dat .. s32.dat) under {raw_path}."
        )
    with open(path, "rb") as f:
        return pickle.load(f, encoding="latin1")


# --------------------------------------------------------------------------- #
# Windowing
# --------------------------------------------------------------------------- #
def _window_starts(
    n_samples: int,
    window_size: int = WINDOW_SIZE,
    step_size: int = STEP_SIZE,
    drop_baseline: bool = DROP_BASELINE,
) -> np.ndarray:
    """Window start offsets, replicating the reference loop exactly.

    The reference is ``while start + window_size < data.shape[1]`` -- a strict
    ``<``, which discards the single window ending exactly on the last sample.
    With DEAP defaults this gives 488 starts, not 489. Reproducing the off-by-one
    matters: it is what makes the tensor shape match the published 488.
    """
    start = BASELINE_SAMPLES if drop_baseline else 0
    starts = []
    while start + window_size < n_samples:
        starts.append(start)
        start += step_size
    return np.asarray(starts, dtype=np.int64)


def extract_trial_features(
    trial_eeg: np.ndarray,
    window_size: int = WINDOW_SIZE,
    step_size: int = STEP_SIZE,
    band_edges: Iterable[float] = BAND_EDGES,
    sample_rate: int = SAMPLE_RATE,
    drop_baseline: bool = DROP_BASELINE,
) -> np.ndarray:
    """Extract channel-major band features from one EEG trial.

    ``trial_eeg`` is ``(n_channels, n_samples)``; only the 32 EEG channels should
    be passed in (DEAP's channels 32-39 are peripheral and the reference model
    does not use them).

    Returns a ``float32`` matrix shaped ``(n_windows, 160)`` under the default
    DEAP configuration. All windows and channels go through a single batched FFT rather than the
    reference's per-channel-per-window Python loop. Same numbers, far fewer
    round trips -- this is the difference between minutes and hours over 32
    subjects.
    """
    trial_eeg = np.asarray(trial_eeg, dtype=np.float64)
    n_channels, n_samples = trial_eeg.shape

    starts = _window_starts(n_samples, window_size, step_size, drop_baseline)
    n_windows = len(starts)

    # (n_channels, n_valid_starts, window_size) as a strided view -- no copy.
    view = np.lib.stride_tricks.sliding_window_view(trial_eeg, window_size, axis=-1)
    windows = view[:, starts, :]  # (n_channels, n_windows, window_size)

    powers = bin_power(windows, band_edges, sample_rate)  # (n_ch, n_win, n_bands)

    # -> (n_windows, n_channels, n_bands) -> flatten channel-major
    powers = np.transpose(powers, (1, 0, 2))
    return powers.reshape(n_windows, n_channels * powers.shape[-1]).astype(np.float32)


# --------------------------------------------------------------------------- #
# Labels
# --------------------------------------------------------------------------- #
def binarise(
    ratings: np.ndarray,
    mode: str = BINARISATION,
    threshold: float = BINARY_THRESHOLD,
) -> np.ndarray:
    """Continuous 1-9 ratings -> High(1)/Low(0).

    ``ratings`` is ``(n_trials, n_axes)``; the threshold is applied per axis.

    mode="fixed"          rating >= threshold          (reference behaviour)
    mode="subject_median" rating >= this subject's own per-axis median

    The reference uses "fixed". On DEAP that produces badly imbalanced classes
    because participants differ systematically in how they use the scale -- one
    subject may never rate valence above 6. "subject_median" removes that
    offset and lands near 50/50, but it is a deviation from the paper, so it is
    off by default.

    Returns
    -------
    numpy.ndarray
        ``int8`` labels with the same shape as ``ratings``.

    Raises
    ------
    ValueError
        If ``mode`` is neither ``fixed`` nor ``subject_median``.
    """
    ratings = np.asarray(ratings, dtype=np.float64)
    if mode == "fixed":
        return (ratings >= threshold).astype(np.int8)
    if mode == "subject_median":
        return (ratings >= np.median(ratings, axis=0, keepdims=True)).astype(np.int8)
    raise ValueError(
        f"Unknown binarisation mode {mode!r}; use 'fixed' or 'subject_median'."
    )


# --------------------------------------------------------------------------- #
# Per-subject driver
# --------------------------------------------------------------------------- #
def features_path_for(subject_id: int, out_dir: Path = FEATURES_PATH) -> Path:
    """Return the canonical feature-cache path for one participant."""
    return Path(out_dir) / f"Participant_{subject_id:02d}.npz"


def extraction_config(
    window_size: int = WINDOW_SIZE,
    step_size: int = STEP_SIZE,
    drop_baseline: bool = DROP_BASELINE,
    binarisation: str = BINARISATION,
    threshold: float = BINARY_THRESHOLD,
) -> dict:
    """Return the settings that determine cached feature and label values."""
    return {
        "schema_version": FEATURE_SCHEMA_VERSION,
        "window_size": window_size,
        "step_size": step_size,
        "drop_baseline": drop_baseline,
        "binarisation": binarisation,
        "threshold": threshold,
        "band_edges": list(BAND_EDGES),
        "sample_rate": SAMPLE_RATE,
        "n_channels": N_CHANNELS,
    }


def load_extraction_config(path: Path) -> dict:
    """Read and decode the feature signature embedded in a participant cache."""
    with np.load(path) as cache:
        if "extraction_config" not in cache:
            raise ValueError(
                f"Feature cache {path} predates configuration signatures; "
                "re-extract it with --overwrite."
            )
        return json.loads(str(cache["extraction_config"].item()))


def extract_subject(
    subject_id: int,
    raw_path: Path = RAW_DATA_PATH,
    out_dir: Path = FEATURES_PATH,
    overwrite: bool = False,
    window_size: int = WINDOW_SIZE,
    step_size: int = STEP_SIZE,
    drop_baseline: bool = DROP_BASELINE,
    binarisation: str = BINARISATION,
    threshold: float = BINARY_THRESHOLD,
    verbose: bool = True,
) -> Path:
    """Extract every trial for one participant and write a compressed cache.

    The cache schema is documented in the module header. Existing caches are
    reused unless ``overwrite`` is true. Binarization happens once per
    participant so ``subject_median`` sees all of that participant's trials.

    Returns the written or reused ``Participant_NN.npz`` path.
    """
    out_path = features_path_for(subject_id, out_dir)
    expected_config = extraction_config(
        window_size, step_size, drop_baseline, binarisation, threshold
    )
    if out_path.exists() and not overwrite:
        try:
            cached_config = load_extraction_config(out_path)
        except (ValueError, json.JSONDecodeError, OSError):
            cached_config = None
        if cached_config == expected_config:
            if verbose:
                print(f"  s{subject_id:02d}: compatible cache, skipping")
            return out_path
        if verbose:
            print(f"  s{subject_id:02d}: cache settings changed, rebuilding")

    subject = load_raw_subject(subject_id, raw_path)
    data = subject["data"]  # (40, 40, 8064)
    labels = subject["labels"]  # (40, 4)  [Valence, Arousal, Dominance, Liking]

    n_trials = min(N_TRIALS, data.shape[0])
    n_windows = windows_per_trial(window_size, step_size, data.shape[2], drop_baseline)

    # Binarise once per subject so "subject_median" sees all 40 trials.
    va_cont = labels[:, [LABEL_INDEX["Valence"], LABEL_INDEX["Arousal"]]]
    va_bin = binarise(va_cont, binarisation, threshold)

    X = np.empty((n_trials * n_windows, N_FEATURES), dtype=np.float32)
    y_bin = np.empty((n_trials * n_windows, 2), dtype=np.int8)
    y_cont = np.empty((n_trials * n_windows, 4), dtype=np.float32)
    trial_id = np.empty(n_trials * n_windows, dtype=np.int16)
    window_id = np.empty(n_trials * n_windows, dtype=np.int16)

    for trial in range(n_trials):
        lo, hi = trial * n_windows, (trial + 1) * n_windows
        X[lo:hi] = extract_trial_features(
            data[trial, :N_CHANNELS],
            window_size,
            step_size,
            BAND_EDGES,
            SAMPLE_RATE,
            drop_baseline,
        )
        y_bin[lo:hi] = va_bin[trial]
        y_cont[lo:hi] = labels[trial]
        trial_id[lo:hi] = trial
        window_id[lo:hi] = np.arange(n_windows, dtype=np.int16)

    subject_arr = np.full(len(X), subject_id, dtype=np.int16)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        X=X,
        y_bin=y_bin,
        y_cont=y_cont,
        trial_id=trial_id,
        window_id=window_id,
        subject_id=subject_arr,
        extraction_config=np.asarray(json.dumps(expected_config, sort_keys=True)),
    )

    if verbose:
        pos_v = 100 * y_bin[:, 0].mean()
        pos_a = 100 * y_bin[:, 1].mean()
        print(
            f"  s{subject_id:02d}: {X.shape[0]:>6,} windows x {X.shape[1]} feats  "
            f"| High-Valence {pos_v:4.1f}%  High-Arousal {pos_a:4.1f}%"
        )
    return out_path


def extract_all(
    subject_ids: Optional[Iterable[int]] = None,
    overwrite: bool = False,
    **kwargs,
) -> list[Path]:
    """Extract an explicit participant collection or all 32 participants.

    The operation is restartable: completed caches are reused unless
    ``overwrite`` is true. Additional keyword arguments are forwarded to
    :func:`extract_subject`.
    """
    subject_ids = (
        list(subject_ids) if subject_ids is not None else list(range(1, N_SUBJECTS + 1))
    )
    return [extract_subject(sid, overwrite=overwrite, **kwargs) for sid in subject_ids]


def load_subject_features(subject_id: int, out_dir: Path = FEATURES_PATH) -> dict:
    """Load one feature cache into a plain dictionary of NumPy arrays."""
    path = features_path_for(subject_id, out_dir)
    if not path.exists():
        raise FileNotFoundError(
            f"No cached features at {path}. Run `python -m src.benchmark --prepare ...` first."
        )
    with np.load(path) as d:
        return {k: d[k] for k in d.files}
