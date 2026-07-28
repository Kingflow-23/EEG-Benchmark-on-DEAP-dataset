"""Factories and metadata for every benchmark candidate.

All returned estimators implement scikit-learn's ``fit``, ``predict``, and
``predict_proba`` interface. Imports for optional libraries and PyTorch models
are delayed until model creation so missing extras do not break classical runs.
This file defines availability, not training or evaluation behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class ModelSpec:
    """Describe one registered candidate without instantiating it.

    ``factory`` accepts the experiment seed. ``dependency`` is populated for
    candidates whose implementation may not be installed. ``notes`` records a
    concise methodological or operational caveat for generated reports.
    """

    name: str
    family: str
    factory: Callable[[int], Any]
    dependency: str | None = None
    notes: str = ""


def _scaled(estimator: Any) -> Pipeline:
    """Place a scale-sensitive estimator behind train-fitted standardization."""
    return Pipeline([("scale", StandardScaler()), ("model", estimator)])


def _specs() -> dict[str, ModelSpec]:
    """Construct the registry mapping without importing optional packages."""
    return {
        "logistic_regression": ModelSpec(
            "logistic_regression", "linear",
            lambda seed: _scaled(LogisticRegression(max_iter=2000, random_state=seed)),
        ),
        "extra_trees": ModelSpec(
            "extra_trees", "tree",
            lambda seed: ExtraTreesClassifier(
                n_estimators=300, n_jobs=-1, class_weight="balanced", random_state=seed
            ),
        ),
        "xgboost": ModelSpec("xgboost", "boosting", _xgboost, "xgboost"),
        "feature_mlp": ModelSpec(
            "feature_mlp", "feedforward",
            lambda seed: _torch("feature_mlp", seed), "torch",
            "Three-layer MLP over the flat 160-feature vector.",
        ),
        "band_electrode_cnn": ModelSpec(
            "band_electrode_cnn", "structured_cnn",
            lambda seed: _torch("band_electrode_cnn", seed), "torch",
            "CNN over five band channels and the 32-position electrode axis.",
        ),
        "fft_lstm": ModelSpec(
            "fft_lstm", "feature_recurrent",
            lambda seed: _torch("fft_lstm", seed), "torch",
            "Published FFT-feature LSTM over the artificial 160-step feature sequence.",
        ),
        "ft_transformer": ModelSpec(
            "ft_transformer", "tabular_deep", lambda seed: _torch("ft_transformer", seed), "torch",
            "Each scalar feature is tokenized independently; no artificial temporal ordering.",
        ),
        "tabpfn": ModelSpec(
            "tabpfn", "foundation", _tabpfn, "tabpfn",
            "May be GPU/memory limited; the installed TabPFN version determines sample limits.",
        ),
    }


def _xgboost(seed: int) -> Any:
    """Create the optional XGBoost candidate."""
    from xgboost import XGBClassifier
    return XGBClassifier(n_estimators=500, max_depth=6, learning_rate=.05, n_jobs=-1,
                         eval_metric="logloss", random_state=seed)


def _tabpfn(seed: int) -> Any:
    """Create the optional TabPFN candidate using its native estimator API."""
    from tabpfn import TabPFNClassifier
    return TabPFNClassifier(random_state=seed)


def _torch(architecture: str, seed: int) -> Any:
    """Create one neural architecture through the shared PyTorch adapter."""
    from .deep_architecture import TorchTabularClassifier
    return TorchTabularClassifier(architecture=architecture, random_state=seed)


def available_models(include_optional: bool = False) -> list[str]:
    """Return the ordered standard suite, optionally including extra packages.

    The order controls execution only; it has no influence on ranking.
    """
    core = ["logistic_regression", "extra_trees", "feature_mlp",
            "band_electrode_cnn", "fft_lstm", "ft_transformer"]
    return core + (["xgboost", "tabpfn"] if include_optional else [])


def create_model(name: str, seed: int) -> tuple[Any, ModelSpec]:
    """Instantiate one candidate and return it with its immutable metadata.

    Raises ``KeyError`` for unknown registry names and a contextual
    ``ImportError`` when the candidate's optional package is unavailable.
    """
    specs = _specs()
    if name not in specs:
        raise KeyError(f"Unknown model {name!r}. Choices: {', '.join(specs)}")
    spec = specs[name]
    try:
        return spec.factory(seed), spec
    except ImportError as exc:
        raise ImportError(f"{name} requires optional package {spec.dependency!r}") from exc
