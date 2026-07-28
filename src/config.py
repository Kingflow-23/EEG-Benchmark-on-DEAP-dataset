"""Authoritative constants for DEAP preprocessing and benchmarking.

Paths, signal geometry, feature ordering, label semantics, split membership,
and reproducibility defaults live here so experiments cannot silently disagree.
Modules may accept explicit overrides for testing, but their defaults must come
from this file. Methodological rationale belongs in ``METHODOLOGY.md`` rather
than being repeated beside every constant.
"""

from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DATA_PATH = PROJECT_ROOT / "data" / "DEAP"
DATA_DIR = PROJECT_ROOT / "output"
FEATURES_PATH = DATA_DIR / "FEATURES"          # per-participant .npz
DATASET_PATH = DATA_DIR / "DATASET"            # assembled train/test splits
BENCHMARK_PATH = DATA_DIR / "BENCHMARKS"

_ALL_OUTPUT_DIRS = (
    DATA_DIR, FEATURES_PATH, DATASET_PATH, BENCHMARK_PATH,
)


def ensure_dirs() -> None:
    """Create configured output directories without writing any data files.

    Directory creation is explicit rather than an import side effect. Benchmark
    runs normally store their complete artifacts below :data:`BENCHMARK_PATH`;
    the other paths support the reusable preprocessing stages.
    """
    for d in _ALL_OUTPUT_DIRS:
        d.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Signal constants
# --------------------------------------------------------------------------- #
SAMPLE_RATE = 128            # Hz, DEAP "preprocessed_python" is downsampled to 128
N_SUBJECTS = 32
N_TRIALS = 40
N_SAMPLES_PER_TRIAL = 8064   # 63 s @ 128 Hz = 3 s pre-stimulus baseline + 60 s stimulus

BASELINE_SEC = 3
BASELINE_SAMPLES = BASELINE_SEC * SAMPLE_RATE          # 384

WINDOW_SIZE = 256            # samples = 2.00 s   (repo value)
STEP_SIZE = 16               # samples = 0.125 s  (repo value) -> 488 windows/trial

# Set True to skip the 3 s pre-stimulus baseline. The reference repo does NOT do
# this: it starts windowing at sample 0, so roughly the first 24 windows of every
# trial contain only pre-stimulus signal while still carrying that trial's
# post-hoc Valence/Arousal rating. Keep False to reproduce the paper.
DROP_BASELINE = False

# --------------------------------------------------------------------------- #
# Electrodes and frequency bands
# --------------------------------------------------------------------------- #
DEAP_ELECTRODES = [
    "Fp1", "AF3", "F3", "F7", "FC5", "FC1", "C3", "T7", "CP5", "CP1", "P3", "P7",
    "PO3", "O1", "Oz", "Pz", "Fp2", "AF4", "Fz", "F4", "F8", "FC6", "FC2", "Cz",
    "C4", "T8", "CP6", "CP2", "P4", "P8", "PO4", "O2",
]
N_CHANNELS = len(DEAP_ELECTRODES)               # 32

# Band edges, as passed to pyeeg.bin_power in the reference repo:
#   band=[4, 8, 12, 16, 25, 45]  ->  5 bands
#
# NOTE: the repo's README advertises Delta 1-4 / Theta 4-8 / Alpha 8-14 /
# Beta 14-31 / Gamma 31-50. The *code* does not do that. There is no Delta band
# and the edges differ. Utils/Constants.py in the repo agrees with the code
# (FREQUENCIES = Theta, Alpha, LowerBeta, UpperBeta, Gamma), so the README is
# simply wrong. We follow the code.
BAND_EDGES = [4, 8, 12, 16, 25, 45]
BAND_NAMES = ["Theta", "Alpha", "LowerBeta", "UpperBeta", "Gamma"]
N_BANDS = len(BAND_NAMES)                       # 5

N_FEATURES = N_CHANNELS * N_BANDS               # 160

# Feature vector layout is channel-major: for each channel, its 5 bands in order.
FEATURE_NAMES = [
    f"{electrode}_{band}"
    for electrode in DEAP_ELECTRODES
    for band in BAND_NAMES
]
assert len(FEATURE_NAMES) == N_FEATURES

# --------------------------------------------------------------------------- #
# Labels
# --------------------------------------------------------------------------- #
# DEAP's labels array is (40, 4) with this column order. This is the single most
# important constant in the file: the reference repo stores columns [0, 1] =
# [Valence, Arousal] during feature extraction, then reads them back in
# LSTMModel/PrepareDataset.py as arousal=col0, valence=col1 -- swapped. Their
# published "92.17% Arousal / 94.46% Valence" therefore have the two axes
# exchanged. We fix it here and address the axes by name everywhere.
LABEL_NAMES = ["Valence", "Arousal", "Dominance", "Liking"]
LABEL_INDEX = {name: i for i, name in enumerate(LABEL_NAMES)}

VALENCE = "Valence"
AROUSAL = "Arousal"
TARGETS = (VALENCE, AROUSAL)                    # the two models we train

# Binarisation of the 1-9 self-report scale into High/Low.
#   "fixed"       -> rating >= BINARY_THRESHOLD  (repo behaviour)
#   "subject_median" -> rating >= that subject's own median for that axis
#
# "fixed" is what the paper uses and is required to reproduce it. Be aware it is
# heavily imbalanced on DEAP: subject-dependent rating bias means the positive
# rate is far from 50%, so a majority-class predictor already scores well.
BINARY_THRESHOLD = 5.0
BINARISATION = "fixed"

# --------------------------------------------------------------------------- #
# Splits
# --------------------------------------------------------------------------- #
# "repo"    : every 4th window -> test. This is BuildDataset.py's `i % 4 == 0`.
#             Adjacent windows overlap 93.75% (2 s window, 0.125 s step), so each
#             test window shares ~75% of its samples with training windows from
#             the same trial and subject. This is the split behind the paper's
#             92-94%. Reproduced for comparison, not to be reported as honest.
# "trial"   : whole trials (videos) held out. No window from a test trial ever
#             appears in training. Same subjects on both sides.
# "subject" : whole subjects held out. Generalisation to an unseen viewer -- the
#             realistic deployment condition for ad testing.
SPLIT_MODES = ("repo", "trial", "subject")
# Default to the deployment-oriented estimate. ``repo`` remains available only
# for reproducing the reference's overlapping-window result.
DEFAULT_SPLIT = "subject"

RANDOM_SEED = 42

# Subjects reserved for the test set under SPLIT_MODE="subject" (8/32 = 25%).
# Fixed rather than random so results are comparable across runs.
HELD_OUT_SUBJECTS = (3, 8, 13, 18, 22, 26, 29, 32)

# Trials reserved for the test set under SPLIT_MODE="trial" (10/40 = 25%).
HELD_OUT_TRIALS = (2, 6, 10, 14, 18, 21, 25, 29, 33, 37)

# --------------------------------------------------------------------------- #
# Shared neural training
# --------------------------------------------------------------------------- #
NEURAL_EPOCHS = 50
NEURAL_BATCH_SIZE = 256
NEURAL_LEARNING_RATE = 1e-3
NEURAL_WEIGHT_DECAY = 1e-4
NEURAL_PATIENCE = 7
NEURAL_VALIDATION_FRACTION = 0.15

# --------------------------------------------------------------------------- #
# Derived helpers
# --------------------------------------------------------------------------- #
def windows_per_trial(
    window_size: int = WINDOW_SIZE,
    step_size: int = STEP_SIZE,
    n_samples: int = N_SAMPLES_PER_TRIAL,
    drop_baseline: bool = DROP_BASELINE,
) -> int:
    """Compute the number of windows emitted for one trial.

    Parameters
    ----------
    window_size, step_size, n_samples
        Window length, stride, and trial length in samples.
    drop_baseline
        Start after the configured three-second baseline when true.

    Returns
    -------
    int
        Count produced by the reference loop.

    Replicates the reference loop `while start + window_size < data.shape[1]`
    exactly -- note the strict `<`, which drops the one window that would end
    precisely on the last sample. With the repo defaults this yields 488.
    """
    start = BASELINE_SAMPLES if drop_baseline else 0
    count = 0
    while start + window_size < n_samples:
        count += 1
        start += step_size
    return count


WINDOWS_PER_TRIAL = windows_per_trial()          # 488 with repo defaults


def subject_id_to_filename(subject_id: int) -> str:
    """Convert a numeric DEAP participant ID to its raw filename.

    For example, ``1`` becomes ``s01.dat`` and ``32`` becomes ``s32.dat``.
    """
    return f"s{subject_id:02d}.dat"


def describe() -> str:
    """Return a compact, human-readable summary of active signal settings."""
    total = N_SUBJECTS * N_TRIALS * WINDOWS_PER_TRIAL
    return "\n".join([
        "=" * 62,
        " PIPELINE CONFIGURATION",
        "=" * 62,
        f" raw data          : {RAW_DATA_PATH}",
        f" sample rate       : {SAMPLE_RATE} Hz",
        f" window            : {WINDOW_SIZE} samples ({WINDOW_SIZE / SAMPLE_RATE:.2f} s)",
        f" step              : {STEP_SIZE} samples ({STEP_SIZE / SAMPLE_RATE:.3f} s)",
        f" overlap           : {100 * (1 - STEP_SIZE / WINDOW_SIZE):.2f}%",
        f" drop baseline     : {DROP_BASELINE}",
        f" windows / trial   : {WINDOWS_PER_TRIAL}",
        f" channels x bands  : {N_CHANNELS} x {N_BANDS} = {N_FEATURES} features",
        f" bands             : {', '.join(f'{n}({BAND_EDGES[i]}-{BAND_EDGES[i + 1]}Hz)' for i, n in enumerate(BAND_NAMES))}",
        f" binarisation      : {BINARISATION} (threshold {BINARY_THRESHOLD})",
        f" total windows     : {total:,}",
        "=" * 62,
    ])


if __name__ == "__main__":
    print(describe())
