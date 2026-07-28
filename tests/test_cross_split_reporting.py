"""Tests for leak-free cross-split robustness selection."""

import json

from src.reporting import write_cross_split_reports


def _summary(split_scores):
    """Build a minimal per-split benchmark summary for two targets."""
    experiments = []
    for target in ("Valence", "Arousal"):
        for model, score in split_scores.items():
            experiments.append(
                {
                    "target": target,
                    "model": model,
                    "status": "ok",
                    "macro_f1": score,
                    "roc_auc": score,
                }
            )
    return {"experiments": experiments}


def test_robust_ranking_excludes_repo_split(tmp_path):
    summaries = {
        "subject": _summary({"stable": 0.70, "repo_star": 0.65}),
        "trial": _summary({"stable": 0.68, "repo_star": 0.64}),
        "repo": _summary({"stable": 0.60, "repo_star": 0.99}),
    }
    result = write_cross_split_reports(tmp_path, summaries)
    assert result["best_robust_model"] == {"Valence": "stable", "Arousal": "stable"}
    assert result["ranking_rule"]["excluded_from_score"] == ["repo"]
    assert (tmp_path / "CROSS_SPLIT_REPORT.md").exists()
    persisted = json.loads((tmp_path / "cross_split_summary.json").read_text())
    assert persisted["best_robust_model"]["Valence"] == "stable"


def test_model_must_complete_both_generalization_splits(tmp_path):
    summaries = {
        "subject": _summary({"complete": 0.60, "subject_only": 0.95}),
        "trial": _summary({"complete": 0.61}),
    }
    result = write_cross_split_reports(tmp_path, summaries)
    incomplete = next(
        row
        for row in result["models"]
        if row["model"] == "subject_only" and row["target"] == "Valence"
    )
    assert incomplete["eligible"] is False
    assert "robustness_rank" not in incomplete
