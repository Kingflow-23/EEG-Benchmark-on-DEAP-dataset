"""Factories and metadata for every benchmark candidate.

All returned estimators implement scikit-learn's ``fit``, ``predict``, and
``predict_proba`` interface. The registry is closed and required: every model
listed here is part of the benchmark contract and must be available for a run.
This file defines availability, not training or evaluation behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..config import TABICL_USE_CUDA


@dataclass(frozen=True)
class ModelSpec:
    """Describe one registered candidate without instantiating it.

    ``factory`` accepts the experiment seed. ``notes`` records a concise
    methodological or operational caveat for generated reports.
    """

    name: str
    family: str
    factory: Callable[[int], Any]
    notes: str = ""


def _scaled(estimator: Any) -> Pipeline:
    """Place a scale-sensitive estimator behind train-fitted standardization."""
    return Pipeline([("scale", StandardScaler()), ("model", estimator)])


def _specs() -> dict[str, ModelSpec]:
    """Construct the registry mapping for the full required benchmark suite."""
    return {
        "logistic_regression": ModelSpec(
            "logistic_regression",
            "linear",
            lambda seed: _scaled(LogisticRegression(max_iter=2000, random_state=seed)),
        ),
        "extra_trees": ModelSpec(
            "extra_trees",
            "tree",
            lambda seed: ExtraTreesClassifier(
                n_estimators=300, n_jobs=-1, class_weight="balanced", random_state=seed
            ),
        ),
        "xgboost": ModelSpec("xgboost", "boosting", _xgboost),
        "feature_mlp": ModelSpec(
            "feature_mlp",
            "feedforward",
            lambda seed: _torch("feature_mlp", seed),
            "Three-layer MLP over the flat 160-feature vector.",
        ),
        "band_electrode_cnn": ModelSpec(
            "band_electrode_cnn",
            "structured_cnn",
            lambda seed: _torch("band_electrode_cnn", seed),
            "CNN over five band channels and the 32-position electrode axis.",
        ),
        "fft_lstm": ModelSpec(
            "fft_lstm",
            "feature_recurrent",
            lambda seed: _torch("fft_lstm", seed),
            "Published FFT-feature LSTM over the artificial 160-step feature sequence.",
        ),
        "ft_transformer": ModelSpec(
            "ft_transformer",
            "tabular_deep",
            lambda seed: _torch("ft_transformer", seed),
            "Each scalar feature is tokenized independently; no artificial temporal ordering.",
        ),
        "tabicl": ModelSpec(
            "tabicl",
            "in_context",
            lambda seed: _tabicl(seed),
            "In-context learner over tabular features using the shared GPU/CPU policy.",
        ),
    }


def _torch(architecture: str, seed: int) -> Any:
    """Create one neural architecture through the shared PyTorch adapter."""
    from .deep_architecture import TorchTabularClassifier

    return TorchTabularClassifier(architecture=architecture, random_state=seed)


def _xgboost(seed: int) -> Any:
    """Create the required gradient-boosting baseline."""
    from xgboost import XGBClassifier

    return XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        n_jobs=-1,
        eval_metric="logloss",
        random_state=seed,
    )


def _tabicl(seed: int) -> Any:
    """Create TabICL on CUDA when requested and available, otherwise CPU."""
    import torch
    from tabicl import TabICLClassifier

    return TabICLClassifier(
        device="cuda" if TABICL_USE_CUDA and torch.cuda.is_available() else "cpu",
        use_amp="auto",
        offload_mode="auto",
        kv_cache=True,
        batch_size=4,
        random_state=seed,
    )


def available_models() -> list[str]:
    """Return the ordered required benchmark suite.

    The order controls execution only; it has no influence on ranking.
    """
    core = [
        "logistic_regression",
        "extra_trees",
        "xgboost",
        "feature_mlp",
        "band_electrode_cnn",
        "fft_lstm",
        "ft_transformer",
        "tabicl",
    ]
    return core


def create_model(name: str, seed: int) -> tuple[Any, ModelSpec]:
    """Instantiate one candidate and return it with its immutable metadata.

    Raises ``KeyError`` for unknown registry names.
    """
    specs = _specs()
    if name not in specs:
        raise KeyError(f"Unknown model {name!r}. Choices: {', '.join(specs)}")
    spec = specs[name]
    return spec.factory(seed), spec
