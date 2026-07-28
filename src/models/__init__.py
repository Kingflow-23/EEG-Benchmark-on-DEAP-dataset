"""Public model-registry API.

Call :func:`available_models` to discover registry keys and
:func:`create_model` to obtain a fresh probabilistic estimator plus metadata.
"""

from .registry import ModelSpec, available_models, create_model

__all__ = ["ModelSpec", "available_models", "create_model"]
