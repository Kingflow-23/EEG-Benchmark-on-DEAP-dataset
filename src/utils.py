"""Dependency-light helpers shared across pipeline stages.

This module deliberately contains no DEAP-specific constants or model logic.
Keeping these operations here avoids duplicate seeding, path, and display code.
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np


def human_bytes(value: int) -> str:
    """Format a byte count using binary units, for example ``1.0 MiB``."""
    size = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024 or unit == "TiB":
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TiB"


def seed_everything(seed: int) -> None:
    """Seed supported random generators and request deterministic PyTorch.

    PyTorch is imported lazily so classical-only runs do not require it at
    import time. Deterministic kernels improve repeatability but exact equality
    can still depend on hardware, driver, and library versions.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def safe_name(value: str) -> str:
    """Replace unsafe characters in one experiment path component."""
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in value)


def model_size_bytes(path: Path) -> int:
    """Return the serialized size of a checkpoint file or directory tree."""
    if path.is_file():
        return path.stat().st_size
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())
