"""Aggregate experiment records and rank models independently by target.

Ranking is centralized here so JSON, CSV, Markdown, and plots cannot disagree.
The ordered keys in :data:`RANK_KEYS` define the primary score and tie-breakers.
"""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path
from typing import Any


RANK_KEYS = ("macro_f1", "roc_auc", "accuracy_lift")


def write_reports(output: Path, records: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    """Rank completed experiments and rewrite all aggregate reports.

    The function is intentionally safe to call after every experiment, making
    partial results durable during a long run. Failed and dependency-skipped
    records remain visible but never enter the ranking.

    Returns the JSON-compatible object written to ``summary.json``.
    """
    output.mkdir(parents=True, exist_ok=True)
    successful = [r for r in records if r.get("status") == "ok"]
    winners = {}
    for target in ("Valence", "Arousal"):
        candidates = [r for r in successful if r["target"] == target]
        ranked = sorted(candidates, key=lambda r: tuple(r.get(k, float("-inf")) for k in RANK_KEYS), reverse=True)
        for index, row in enumerate(ranked, 1): row["rank"] = index
        winners[target] = ranked[0]["model"] if ranked else None
    summary = {"configuration": config, "ranking_metric": list(RANK_KEYS),
               "best_model": winners, "experiments": records}
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    columns = sorted({key for row in records for key in row if key != "confusion_matrix"})
    with (output / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns); writer.writeheader()
        writer.writerows({k: v for k, v in row.items() if k != "confusion_matrix"} for row in records)
    lines = ["# DEAP architecture benchmark", "", f"Primary ranking: `{RANK_KEYS[0]}`; ties: `{RANK_KEYS[1]}`, then `{RANK_KEYS[2]}`.", "",
             f"- Best Valence model: **{winners['Valence'] or 'none'}**",
             f"- Best Arousal model: **{winners['Arousal'] or 'none'}**", "",
             "| Target | Rank | Model | Macro F1 | ROC-AUC | Accuracy | Lift | Train s | Infer s |", "|---|---:|---|---:|---:|---:|---:|---:|---:|"]
    for row in sorted(successful, key=lambda r: (r["target"], r.get("rank", 999))):
        lines.append(f"| {row['target']} | {row.get('rank','')} | {row['model']} | {row['macro_f1']:.4f} | {row['roc_auc']:.4f} | {row['accuracy']:.4f} | {row['accuracy_lift']:.4f} | {row['train_seconds']:.2f} | {row['inference_seconds']:.2f} |")
    failed = [r for r in records if r.get("status") != "ok"]
    if failed:
        lines += ["", "## Skipped or failed", ""] + [f"- `{r['target']}/{r['model']}`: {r.get('error', r['status'])}" for r in failed]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    _write_comparison_plot(output, successful)
    return summary


def _write_comparison_plot(output: Path, records: list[dict[str, Any]]) -> None:
    """Plot target-wise macro-F1; silently skip when Matplotlib is unavailable."""
    if not records:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    targets = ("Valence", "Arousal")
    fig, axes = plt.subplots(1, 2, figsize=(12, max(4, len(records) * .22)))
    for ax, target in zip(axes, targets):
        rows = sorted((row for row in records if row["target"] == target),
                      key=lambda row: row["macro_f1"])
        ax.barh([row["model"] for row in rows], [row["macro_f1"] for row in rows])
        ax.axvline(.5, color="grey", linestyle="--", linewidth=.8)
        ax.set(title=target, xlabel="Macro F1", xlim=(0, 1))
    fig.suptitle("DEAP architecture comparison")
    fig.tight_layout(); fig.savefig(output / "model_comparison.png", dpi=160); plt.close(fig)


def write_cross_split_reports(output: Path, summaries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Rank models for robust generalization across leak-free split strategies.

    Only ``subject`` and ``trial`` contribute to the robustness score. A model
    must complete both splits for a target to qualify. Mean macro-F1 is primary;
    worst-split macro-F1 breaks ties and penalizes brittle architectures. The
    ``repo`` result is retained in the report as a reproduction diagnostic.
    """
    generalization_splits = ("subject", "trial")
    rows = []
    for target in ("Valence", "Arousal"):
        by_model: dict[str, dict[str, dict[str, Any]]] = {}
        for split, summary in summaries.items():
            for record in summary["experiments"]:
                if record.get("status") == "ok" and record["target"] == target:
                    by_model.setdefault(record["model"], {})[split] = record
        for model, split_records in by_model.items():
            eligible = all(split in split_records for split in generalization_splits)
            row: dict[str, Any] = {"target": target, "model": model,
                                   "eligible": eligible}
            for split in ("subject", "trial", "repo"):
                record = split_records.get(split)
                row[f"{split}_macro_f1"] = record.get("macro_f1") if record else None
                row[f"{split}_roc_auc"] = record.get("roc_auc") if record else None
            if eligible:
                scores = [split_records[split]["macro_f1"]
                          for split in generalization_splits]
                row["generalization_mean_macro_f1"] = statistics.fmean(scores)
                row["generalization_worst_macro_f1"] = min(scores)
            rows.append(row)
    for target in ("Valence", "Arousal"):
        ranked = sorted(
            (row for row in rows if row["target"] == target and row["eligible"]),
            key=lambda row: (row["generalization_mean_macro_f1"],
                             row["generalization_worst_macro_f1"]),
            reverse=True,
        )
        for rank, row in enumerate(ranked, 1):
            row["robustness_rank"] = rank
    winners = {
        target: next((row["model"] for row in rows
                      if row["target"] == target and row.get("robustness_rank") == 1), None)
        for target in ("Valence", "Arousal")
    }
    result = {"ranking_rule": {
                  "included_splits": list(generalization_splits),
                  "excluded_from_score": ["repo"],
                  "primary": "mean_macro_f1",
                  "tie_breaker": "worst_split_macro_f1",
                  "eligibility": "successful on both included splits",
              }, "best_robust_model": winners, "models": rows}
    output.mkdir(parents=True, exist_ok=True)
    (output / "cross_split_summary.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    columns = sorted({key for row in rows for key in row})
    with (output / "cross_split_comparison.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# Cross-split robustness report", "",
             "The robustness score uses only `subject` and `trial`; the leaky "
             "`repo` split is diagnostic and cannot select a winner.", "",
             f"- Robust Valence model: **{winners['Valence'] or 'none'}**",
             f"- Robust Arousal model: **{winners['Arousal'] or 'none'}**", "",
             "| Target | Rank | Model | Subject F1 | Trial F1 | Mean | Worst | Repo F1 |",
             "|---|---:|---|---:|---:|---:|---:|---:|"]
    for row in sorted((item for item in rows if item.get("robustness_rank")),
                      key=lambda item: (item["target"], item["robustness_rank"])):
        repo = row["repo_macro_f1"]
        repo_text = f"{repo:.4f}" if repo is not None else "n/a"
        lines.append(
            f"| {row['target']} | {row['robustness_rank']} | {row['model']} | "
            f"{row['subject_macro_f1']:.4f} | {row['trial_macro_f1']:.4f} | "
            f"{row['generalization_mean_macro_f1']:.4f} | "
            f"{row['generalization_worst_macro_f1']:.4f} | "
            f"{repo_text} |"
        )
    (output / "CROSS_SPLIT_REPORT.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return result
