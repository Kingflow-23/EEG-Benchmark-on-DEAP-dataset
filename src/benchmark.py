"""Command-line orchestration for the DEAP architecture benchmark.

This module connects the feature/split pipeline to the model registry, shared
evaluator, workflow tracker, and report writer. A benchmark run trains each
requested architecture twice: once for binary Valence and once for binary
Arousal. It also streams structured progress updates and progress bars for the
per-model loop, but it does not contain preprocessing, model-specific metrics,
or target-specific tuning.

The primary entry point is ``python -m src.benchmark``. Programmatic callers can
use :func:`run_benchmark` after preparing a split with
``src.preprocessing.build_dataset``.
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - fallback for minimal environments

    def tqdm(iterable, **kwargs):
        return iterable


if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src import config
    from src.evaluation import (
        evaluate_probabilities,
        normalize_probabilities,
        save_evaluation_artifacts,
    )
    from src.models import available_models, create_model
    from src.preprocessing import build_dataset, extract_all, load_split
    from src.reporting import write_cross_split_reports, write_reports
    from src.utils import model_size_bytes, safe_name, seed_everything
    from src.workflow import WorkflowTracker, setup_logging
else:
    from . import config
    from .evaluation import (
        evaluate_probabilities,
        normalize_probabilities,
        save_evaluation_artifacts,
    )
    from .models import available_models, create_model
    from .preprocessing import build_dataset, extract_all, load_split
    from .reporting import write_cross_split_reports, write_reports
    from .utils import model_size_bytes, safe_name, seed_everything
    from .workflow import WorkflowTracker, setup_logging


def run_benchmark(
    split: str,
    model_names: list[str],
    output: Path,
    seed: int,
    max_train_samples: int | None = None,
    max_test_samples: int | None = None,
    fail_fast: bool = False,
    tracker: WorkflowTracker | None = None,
    tabicl_max_train_samples: int | None = config.TABICL_MAX_TRAIN_SAMPLES,
    tabicl_max_test_samples: int | None = config.TABICL_MAX_TEST_SAMPLES,
) -> dict:
    """Train and evaluate each architecture for both affective targets.

    Structured workflow events are emitted for experiment start, training
    completion, evaluation completion, failure paths, and cumulative
    progress. Neural models also show epoch-level progress via ``tqdm``.

    Parameters
    ----------
    split
        Name of an already assembled split: ``repo``, ``trial``, or ``subject``.
    model_names
        Registry keys returned by :func:`src.models.available_models`.
    output
        Run directory. Each ``<target>/<model>`` child receives a checkpoint,
        probabilities, metrics, and plots; aggregate reports live at the root.
    seed
        Seed applied before sampling and again before every model fit.
    max_train_samples, max_test_samples
        Optional deterministic limits for smoke tests or constrained hardware.
        Sampling preserves the joint Valence/Arousal label distribution.
    tabicl_max_train_samples, tabicl_max_test_samples
        TabICL-specific ceilings applied after the global limits. They prevent
        its in-context output tensors from scaling to hundreds of gigabytes on
        the full window-level split. Set either to ``None`` to disable it.
    fail_fast
        Re-raise the first training/runtime error. By default, failures are
        recorded and the remaining experiments continue.

    Returns
    -------
    dict
        The same serializable summary written to ``summary.json``.

    Notes
    -----
    Model errors produce ``status='failed'`` unless ``fail_fast`` is true.
    """
    seed_everything(seed)
    X_train, y_train, _, X_test, y_test, meta_test = load_split(
        split, mmap=True, with_meta=True
    )
    rng = np.random.default_rng(seed)
    train_idx = _stratified_limit(y_train, max_train_samples, rng)
    test_idx = _stratified_limit(y_test, max_test_samples, rng)
    Xtr, Xte = np.asarray(X_train[train_idx]), np.asarray(X_test[test_idx])
    ytr_all, yte_all = y_train[train_idx], y_test[test_idx]
    selected_meta_test = meta_test[test_idx]
    run_config = _run_config(
        split,
        model_names,
        seed,
        len(train_idx),
        len(test_idx),
        tabicl_max_train_samples,
        tabicl_max_test_samples,
    )
    records = []
    if tracker is not None:
        tracker.log(
            logging.INFO,
            "split_loaded",
            split=split,
            n_train=len(train_idx),
            n_test=len(test_idx),
            models=model_names,
        )
    total_experiments = len(config.TARGETS) * len(model_names)
    for target_index, target in enumerate(config.TARGETS):
        iterator = tqdm(
            model_names,
            total=len(model_names),
            desc=f"{split}:{target}",
            leave=False,
            dynamic_ncols=True,
        )
        for model_name in iterator:
            run_dir = output / target / safe_name(model_name)
            record = {"target": target, "model": model_name, "status": "failed"}
            try:
                model_train_idx, model_test_idx = _model_sample_indices(
                    model_name,
                    ytr_all,
                    yte_all,
                    seed,
                    tabicl_max_train_samples,
                    tabicl_max_test_samples,
                )
                model_Xtr = Xtr[model_train_idx]
                model_ytr = ytr_all[model_train_idx]
                model_Xte = Xte[model_test_idx]
                model_yte = yte_all[model_test_idx]
                model_meta_test = selected_meta_test[model_test_idx]
                record.update(
                    n_train_samples=len(model_train_idx),
                    n_test_samples=len(model_test_idx),
                    sampling_protocol=(
                        "tabicl_stratified_cap"
                        if model_name == "tabicl"
                        and (
                            len(model_train_idx) < len(Xtr)
                            or len(model_test_idx) < len(Xte)
                        )
                        else "run_default"
                    ),
                )
                if tracker is not None:
                    tracker.log(
                        logging.INFO,
                        "experiment_started",
                        split=split,
                        target=target,
                        model=model_name,
                        n_train=len(model_train_idx),
                        n_test=len(model_test_idx),
                    )
                if hasattr(iterator, "set_postfix_str"):
                    iterator.set_postfix_str("training", refresh=False)
                seed_everything(seed)
                model, spec = create_model(model_name, seed)
                record.update(family=spec.family, notes=spec.notes)
                started = time.perf_counter()
                model.fit(model_Xtr, model_ytr[:, target_index])
                record["train_seconds"] = time.perf_counter() - started
                if tracker is not None:
                    tracker.log(
                        logging.INFO,
                        "model_trained",
                        split=split,
                        target=target,
                        model=model_name,
                        train_seconds=record["train_seconds"],
                    )
                if hasattr(iterator, "set_postfix_str"):
                    iterator.set_postfix_str("evaluating", refresh=False)
                started = time.perf_counter()
                probabilities = normalize_probabilities(model.predict_proba(model_Xte))
                record["inference_seconds"] = time.perf_counter() - started
                if tracker is not None:
                    tracker.log(
                        logging.INFO,
                        "model_evaluation_completed",
                        split=split,
                        target=target,
                        model=model_name,
                        inference_seconds=record["inference_seconds"],
                    )
                metrics = evaluate_probabilities(
                    model_yte[:, target_index], probabilities
                )
                record.update(metrics)
                run_dir.mkdir(parents=True, exist_ok=True)
                if hasattr(model, "save"):
                    model_path = run_dir / "model.pt"
                    model.save(model_path)
                else:
                    model_path = run_dir / "model.joblib"
                    joblib.dump(model, model_path)
                record["model_size_bytes"] = model_size_bytes(model_path)
                record["trainable_parameters"] = getattr(model, "n_parameters_", None)
                save_evaluation_artifacts(
                    run_dir,
                    model_yte[:, target_index],
                    probabilities,
                    metrics,
                    metadata=model_meta_test,
                )
                history = getattr(model, "history_", None)
                if history is not None:
                    (run_dir / "training_history.json").write_text(
                        json.dumps(history, indent=2), encoding="utf-8"
                    )
                record["status"] = "ok"
                if tracker is not None:
                    tracker.log(
                        logging.INFO,
                        "experiment_completed",
                        split=split,
                        target=target,
                        model=model_name,
                        macro_f1=record.get("macro_f1"),
                        roc_auc=record.get("roc_auc"),
                    )
            except Exception as exc:
                record["status"] = "failed"
                record["error"] = f"{type(exc).__name__}: {exc}"
                if tracker is not None:
                    tracker.log(
                        logging.ERROR,
                        "experiment_failed",
                        split=split,
                        target=target,
                        model=model_name,
                        error=record["error"],
                    )
                if fail_fast:
                    raise
            records.append(record)
            write_reports(output, records, run_config, tracker=tracker)
            if tracker is not None:
                tracker.log(
                    logging.INFO,
                    "workflow_progress",
                    completed=len(records),
                    total=total_experiments,
                    split=split,
                    target=target,
                    model=model_name,
                )
    return write_reports(output, records, run_config, tracker=tracker)


def _model_sample_indices(
    model_name: str,
    y_train: np.ndarray,
    y_test: np.ndarray,
    seed: int,
    tabicl_max_train_samples: int | None,
    tabicl_max_test_samples: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return deterministic model-specific row selections.

    Only TabICL receives an additional cap. Separate generators with stable
    seeds make the subsets identical across Valence and Arousal experiments
    and independent of model execution order.
    """
    if model_name != "tabicl":
        return np.arange(len(y_train)), np.arange(len(y_test))
    train_rng = np.random.default_rng(seed + 10_001)
    test_rng = np.random.default_rng(seed + 20_001)
    return (
        _stratified_limit(y_train, tabicl_max_train_samples, train_rng),
        _stratified_limit(y_test, tabicl_max_test_samples, test_rng),
    )


def _stratified_limit(
    labels: np.ndarray, limit: int | None, rng: np.random.Generator
) -> np.ndarray:
    """Select row indices while approximately preserving four joint labels.

    ``labels`` has columns ``[Valence, Arousal]``. Their binary combination
    forms four strata, preventing a small smoke sample from accidentally
    discarding a target/class combination.
    """
    if limit is None or limit >= len(labels):
        return np.arange(len(labels))
    if limit <= 0:
        raise ValueError("sample limits must be positive")
    joint = labels[:, 0] * 2 + labels[:, 1]
    groups = [np.flatnonzero(joint == value) for value in np.unique(joint)]
    ideal = np.asarray([limit * len(group) / len(labels) for group in groups])
    quotas = np.floor(ideal).astype(int)
    if limit >= len(groups):
        quotas = np.maximum(quotas, 1)
    quotas = np.minimum(quotas, np.asarray([len(group) for group in groups]))
    while quotas.sum() < limit:
        candidates = [i for i, group in enumerate(groups) if quotas[i] < len(group)]
        index = max(candidates, key=lambda i: (ideal[i] - quotas[i], len(groups[i])))
        quotas[index] += 1
    while quotas.sum() > limit:
        candidates = [i for i in range(len(groups)) if quotas[i] > 1]
        index = min(candidates, key=lambda i: (ideal[i] - quotas[i], -len(groups[i])))
        quotas[index] -= 1
    chosen = []
    for group, quota in zip(groups, quotas):
        chosen.extend(rng.choice(group, quota, replace=False))
    chosen = np.asarray(chosen, dtype=int)
    rng.shuffle(chosen)
    return np.sort(chosen[:limit])


def _run_config(
    split: str,
    models: list[str],
    seed: int,
    n_train: int,
    n_test: int,
    tabicl_max_train_samples: int | None,
    tabicl_max_test_samples: int | None,
) -> dict:
    """Build the provenance block embedded in every aggregate report."""
    versions = {"python": platform.python_version(), "numpy": np.__version__}
    try:
        import sklearn

        versions["scikit_learn"] = sklearn.__version__
    except ImportError:
        versions["scikit_learn"] = None
    try:
        import torch

        versions["torch"] = torch.__version__
        versions["compute_device"] = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        versions.update(torch=None, compute_device=None)
    return {
        "dataset": "DEAP",
        "split": split,
        "models": models,
        "seed": seed,
        "n_train": n_train,
        "n_test": n_test,
        "tabicl_max_train_samples": tabicl_max_train_samples,
        "tabicl_max_test_samples": tabicl_max_test_samples,
        "features": config.N_FEATURES,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "versions": versions,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments, or an explicit argument list in tests/tools."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split", choices=(*config.SPLIT_MODES, "all"), default=config.DEFAULT_SPLIT
    )
    parser.add_argument(
        "--models", nargs="+", default=None, help="Registry names (default: core suite)"
    )
    parser.add_argument(
        "--prepare",
        action="store_true",
        help="Extract features and build the split first",
    )
    parser.add_argument("--subjects", nargs="+", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-test-samples", type=int)
    parser.add_argument(
        "--tabicl-max-train-samples",
        type=int,
        default=config.TABICL_MAX_TRAIN_SAMPLES,
        help="TabICL-only stratified training cap (default: 10000)",
    )
    parser.add_argument(
        "--tabicl-max-test-samples",
        type=int,
        default=config.TABICL_MAX_TEST_SAMPLES,
        help="TabICL-only stratified test cap (default: 2000)",
    )
    parser.add_argument(
        "--smoke", action="store_true", help="Use 2,000/1,000 samples and fast models"
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--list-models", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Prepare data when requested, execute a run, and print its winners.

    The run also writes ``workflow.json``, ``workflow.jsonl``, and
    ``workflow.log`` in the output directory.
    """
    args = parse_args(argv)
    if args.list_models:
        print("\n".join(available_models()))
        return 0
    config.ensure_dirs()
    models = args.models or available_models()
    if args.smoke:
        models = args.models or ["logistic_regression", "extra_trees", "feature_mlp"]
        args.max_train_samples = args.max_train_samples or 2000
        args.max_test_samples = args.max_test_samples or 1000
    # Run the scientifically useful estimates first so an interrupted long job
    # still leaves subject/trial results; the leaky reference split is last.
    splits = ["subject", "trial", "repo"] if args.split == "all" else [args.split]
    tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = args.output or config.BENCHMARK_PATH / f"{args.split}_{tag}"
    logger = setup_logging(output)
    tracker = WorkflowTracker(output, run_name=f"{args.split}_{tag}", logger=logger)
    tracker.log(
        logging.INFO,
        "workflow_started",
        split=args.split,
        models=models,
        seed=args.seed,
        prepare=args.prepare,
    )
    if args.prepare:
        with tracker.stage(
            "feature_extraction", subjects=args.subjects, overwrite=args.overwrite
        ):
            extract_all(args.subjects, overwrite=args.overwrite)
        with tracker.stage(
            "dataset_build", split_modes=splits, overwrite=args.overwrite
        ):
            for split in splits:
                build_dataset(args.subjects, split, overwrite=args.overwrite)
    summaries = {}
    cross_split = None
    for split in splits:
        split_output = output / split if args.split == "all" else output
        with tracker.stage("benchmark", split=split, output=str(split_output)):
            summaries[split] = run_benchmark(
                split=split,
                model_names=models,
                output=split_output,
                seed=args.seed,
                max_train_samples=args.max_train_samples,
                max_test_samples=args.max_test_samples,
                fail_fast=args.fail_fast,
                tracker=tracker,
                tabicl_max_train_samples=args.tabicl_max_train_samples,
                tabicl_max_test_samples=args.tabicl_max_test_samples,
            )
    if args.split == "all":
        with tracker.stage("cross_split_report"):
            cross_split = write_cross_split_reports(output, summaries, tracker=tracker)
        print(json.dumps(cross_split["best_robust_model"], indent=2))
        print(f"Report: {output / 'CROSS_SPLIT_REPORT.md'}")
    else:
        print(json.dumps(summaries[args.split]["best_model"], indent=2))
        print(f"Report: {output / 'REPORT.md'}")
    best_model = (
        cross_split["best_robust_model"]
        if args.split == "all"
        else summaries[args.split]["best_model"]
    )
    tracker.finish(best_model=best_model)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
