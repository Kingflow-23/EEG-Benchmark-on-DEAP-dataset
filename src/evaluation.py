"""Model-independent evaluation of binary class probabilities.

Every candidate reaches this module through the same ``(n_samples, 2)``
probability contract. Keeping metric calculation here prevents architecture
adapters from choosing favorable definitions. Class ``0`` means Low and class
``1`` means High for whichever target is currently being evaluated.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


def evaluate_probabilities(
    y_true: np.ndarray, probabilities: np.ndarray
) -> dict[str, Any]:
    """Validate and score binary predictions.

    Parameters
    ----------
    y_true
        One-dimensional binary labels in ``{0, 1}``.
    probabilities
        Matrix shaped ``(n_samples, 2)``. Columns are ordered ``[P(Low),
        P(High)]``. Each finite row must sum to one within ``1e-4``; accepted
        floating-point drift is normalized before metrics are calculated.

    Returns
    -------
    dict
        JSON-compatible scalar metrics, a 2-by-2 confusion matrix, and a
        ``collapsed`` flag indicating that only one class was predicted.

    Raises
    ------
    ValueError
        If the probability shape or values violate the shared contract.
    """
    y_true = np.asarray(y_true)
    proba = normalize_probabilities(probabilities)
    if y_true.ndim != 1 or len(y_true) != len(proba):
        raise ValueError("y_true must be one-dimensional and match probability rows")
    classes = np.unique(y_true)
    if not np.array_equal(classes, np.array([0, 1])):
        raise ValueError("evaluation requires test labels containing both 0 and 1")
    pred = proba.argmax(axis=1)
    majority = max(float(np.mean(y_true == 0)), float(np.mean(y_true == 1)))
    matrix = confusion_matrix(y_true, pred, labels=[0, 1])
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "majority_accuracy": majority,
        "accuracy_lift": float(accuracy_score(y_true, pred) - majority),
        "macro_f1": float(f1_score(y_true, pred, average="macro", zero_division=0)),
        "f1_low": float(
            f1_score(y_true, pred, labels=[0], average=None, zero_division=0)[0]
        ),
        "f1_high": float(
            f1_score(y_true, pred, labels=[1], average=None, zero_division=0)[0]
        ),
        "roc_auc": float(roc_auc_score(y_true, proba[:, 1])),
        "pr_auc": float(average_precision_score(y_true, proba[:, 1])),
        "brier_score": float(brier_score_loss(y_true, proba[:, 1])),
        "log_loss": float(log_loss(y_true, proba, labels=[0, 1])),
        "confusion_matrix": matrix.tolist(),
        "collapsed": bool(np.unique(pred).size < 2),
    }


def normalize_probabilities(probabilities: np.ndarray) -> np.ndarray:
    """Validate and normalize a binary probability matrix.

    Some estimators, including TabICL, return float32 rows with harmless
    rounding drift. Accepted rows are divided by their sums so strict sklearn
    probability checks and persisted artifacts receive an exact distribution.
    Materially malformed rows remain errors rather than being silently fixed.
    """
    proba = np.asarray(probabilities, dtype=np.float64)
    if proba.ndim != 2 or proba.shape[1] != 2:
        raise ValueError(f"Expected (n, 2) probabilities, got {proba.shape}")
    row_sums = proba.sum(axis=1)
    if (
        not np.all(np.isfinite(proba))
        or np.any(proba < 0)
        or np.any(proba > 1)
        or np.any(row_sums <= 0)
        or not np.allclose(row_sums, 1, atol=1e-4, rtol=1e-5)
    ):
        raise ValueError("Model returned invalid class probabilities")
    return proba / row_sums[:, None]


def save_evaluation_artifacts(
    directory: Path,
    y_true: np.ndarray,
    probabilities: np.ndarray,
    metrics: dict[str, Any],
    metadata: np.ndarray | None = None,
) -> None:
    """Write the complete per-model evaluation bundle.

    ``predictions.npz`` is the downstream machine interface and preserves full
    two-class probabilities plus window metadata when supplied. ``metrics.json``
    is always written. When Matplotlib is installed, four PNG diagnostics are
    added; otherwise a short marker file explains why plots are absent.
    """
    if metadata is not None and len(metadata) != len(y_true):
        raise ValueError("metadata rows must match prediction rows")
    directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "y_true": y_true,
        "probabilities": probabilities,
        "y_pred": probabilities.argmax(axis=1),
    }
    if metadata is not None:
        payload["metadata"] = metadata
    np.savez_compressed(directory / "predictions.npz", **payload)
    (directory / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import ConfusionMatrixDisplay

        fig, ax = plt.subplots(figsize=(4, 4))
        ConfusionMatrixDisplay(
            np.asarray(metrics["confusion_matrix"]), display_labels=["Low", "High"]
        ).plot(ax=ax)
        fig.tight_layout()
        fig.savefig(directory / "confusion_matrix.png", dpi=160)
        plt.close(fig)

        roc_x, roc_y, _ = roc_curve(y_true, probabilities[:, 1])
        precision, recall, _ = precision_recall_curve(y_true, probabilities[:, 1])
        frac, mean = calibration_curve(y_true, probabilities[:, 1], n_bins=10)
        for name, x, y, xlabel, ylabel in (
            ("roc_curve", roc_x, roc_y, "False-positive rate", "True-positive rate"),
            ("precision_recall_curve", recall, precision, "Recall", "Precision"),
            (
                "calibration_curve",
                mean,
                frac,
                "Mean predicted probability",
                "Observed frequency",
            ),
        ):
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.plot(x, y, marker=".")
            if name != "precision_recall_curve":
                ax.plot([0, 1], [0, 1], "--", color="grey")
            ax.set(xlabel=xlabel, ylabel=ylabel, title=name.replace("_", " ").title())
            fig.tight_layout()
            fig.savefig(directory / f"{name}.png", dpi=160)
            plt.close(fig)
    except ImportError:
        (directory / "PLOTS_SKIPPED.txt").write_text(
            "Install matplotlib to generate plots.\n", encoding="utf-8"
        )
