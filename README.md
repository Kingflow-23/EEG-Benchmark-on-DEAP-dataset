# Reproducible DEAP EEG Architecture Benchmark

This repository compares model architectures for binary **Valence** and
**Arousal** recognition using one authoritative DEAP pipeline. Every candidate
receives the same 160 channel-major band features, labels, normalization rules,
split, seed, and probability-based evaluation. Winners are selected separately
for the two targets.

The benchmark predicts Low/High independently for each axis; it is not a joint
four-quadrant classifier. The default threshold is a DEAP rating of 5.

## Pipeline at a glance

1. Read DEAP's preprocessed 128 Hz participant files.
2. Reproduce the reference 2-second overlapping windows and five FFT-magnitude
   bands for the first 32 EEG channels.
3. Cache each participant as 160 channel-major features per window.
4. Assemble one fixed `subject`, `trial`, or reference `repo` split.
5. Train every requested model separately for Valence and Arousal.
6. Validate the common two-class probability output, evaluate it once, and
   rank successful models independently by target.

## Quick start

Place the 32 DEAP `preprocessed_python` files at `data/DEAP/s01.dat` through
`s32.dat`. Python 3.10 or newer is recommended. Install the dependencies, then
run:

```powershell
python -m pip install -r requirements.txt
python -m src.benchmark --prepare --split subject
```

The command extracts/caches features, builds the requested split, trains the six
core architectures twice (once per target), and writes checkpoints,
probabilities, metrics, plots, CSV/JSON comparisons, and `REPORT.md` under
`output/BENCHMARKS/`.

Useful alternatives:

```powershell
# Fast end-to-end validation
python -m src.benchmark --prepare --split subject --subjects 1 2 3 8 --smoke

# Explicit suite, reusing prepared data
python -m src.benchmark --split subject --models logistic_regression feature_mlp band_electrode_cnn

# Try optional installed libraries; unavailable ones are reported as skipped
python -m src.benchmark --split subject --include-optional

# Run every split and produce a leak-free cross-split robustness ranking
python -m src.benchmark --prepare --split all

python -m src.benchmark --list-models
```

`subject` is the deployment-oriented split, `trial` holds videos out, and
`repo` reproduces the reference window split but leaks overlapping signal and
must not be interpreted as generalization performance. `subject` is therefore
the command-line default when `--split` is omitted.

When using `--subjects` with the `subject` split, include at least one configured
held-out participant and one non-held-out participant; otherwise one side of the
split is empty. `--smoke` limits rows for integration testing and is not a
publication-quality result.

## Model groups

The default suite intentionally contains one representative per useful modeling
hypothesis:

- `logistic_regression`: scaled linear baseline.
- `extra_trees`: nonlinear bagged-tree baseline.
- `feature_mlp`: `160→256→128→64→2` feedforward feature network.
- `band_electrode_cnn`: CNN using five bands as input channels
  and 32 electrodes as the convolution axis.
- `fft_lstm`: published recurrent stack over FFT features, retained for
  reproducibility despite its artificial
  interpretation of 160 features as 160 steps.
- `ft_transformer`: modern attention-based tabular representative.

`xgboost` and `tabpfn` are optional representatives for gradient boosting and
tabular foundation models. Missing optional packages are reported as skipped.
See the methodology for layer details, limitations, and the redundancy audit.

## Multi-split robustness

`--split all` runs identical models and seeds on `subject`, `trial`, and `repo`.
Each split keeps its own report and winners. The root `CROSS_SPLIT_REPORT.md`
ranks models that completed both leak-free splits using mean macro-F1, with
worst-split macro-F1 as the tie-breaker. The `repo` score is displayed for
reference reproduction but is explicitly excluded from robust model selection.

## Outputs and ranking

Each `<target>/<model>/` directory contains a `model.joblib` or `model.pt`
checkpoint, `metrics.json`, `predictions.npz`, confusion/ROC/PR/calibration
plots, and neural training history when applicable. The run root contains
`comparison.csv`, `summary.json`, and `REPORT.md`. Ranking uses macro-F1, then
ROC-AUC and accuracy lift as deterministic tie-breakers.

## Code map

- `src/config.py`: authoritative paths, signal geometry, labels, and split IDs.
- `src/preprocessing/`: raw loading, feature extraction, and split assembly.
- `src/models/`: lazy registry and the common PyTorch estimator.
- `src/evaluation.py`: the sole probability/metric contract.
- `src/reporting.py`: ranking and aggregate artifacts.
- `src/benchmark.py`: CLI orchestration only.

See [METHODOLOGY.md](METHODOLOGY.md) for data contracts, model rationale,
limitations, reproducibility, and exclusions.
