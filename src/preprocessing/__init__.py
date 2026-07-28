"""Public preprocessing API for the two deterministic data stages.

Stage 1 converts raw DEAP participant files into provenance-rich feature caches.
Stage 2 pools those caches into one configured train/test split. The benchmark
imports this package instead of maintaining a second data path.
"""

from .bandpower import bin_power, band_bin_indices
from .fft_features import (
    extract_all, extract_subject, extract_trial_features, load_raw_subject,
    load_subject_features, binarise, extraction_config, load_extraction_config,
)
from .build_dataset import build_dataset, make_test_mask, load_split

__all__ = [
    "bin_power", "band_bin_indices",
    "extract_all", "extract_subject", "extract_trial_features",
    "load_raw_subject", "load_subject_features", "binarise",
    "extraction_config", "load_extraction_config",
    "build_dataset", "make_test_mask", "load_split",
]
