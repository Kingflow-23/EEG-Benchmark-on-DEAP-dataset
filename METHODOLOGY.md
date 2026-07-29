# Benchmark methodology

## Authoritative data contract

The benchmark does not preprocess data itself. `src/preprocessing/` is the only
route from DEAP's Python files to model input. It reproduces the reference
windowing rule (256 samples, step 16, strict end bound), uses the first 32 EEG
channels, and computes five FFT magnitude bands (4-8, 8-12, 12-16, 16-25,
25-45 Hz). The result is a 160-value, channel-major vector. Continuous DEAP
labels remain in `[Valence, Arousal, Dominance, Liking]` order; the first two
are binarized independently at 5 by default.

The extractor intentionally matches the reference implementation where
scientifically unusual behavior affects comparability: the three-second
pre-stimulus baseline is retained by default; the final exactly aligned window
is omitted by a strict `<` bound; and "band power" is a sum of FFT magnitudes,
not squared physical power. These choices are centralized in `src/config.py`
and `src/preprocessing/bandpower.py` rather than reimplemented by models.

### Persisted schemas

Per-participant `Participant_NN.npz` caches contain:

| Key | Shape | Meaning |
|---|---|---|
| `X` | `(windows, 160)` | `float32` channel-major band features |
| `y_bin` | `(windows, 2)` | binary `[Valence, Arousal]` labels |
| `y_cont` | `(windows, 4)` | original `[Valence, Arousal, Dominance, Liking]` ratings |
| `subject_id`, `trial_id`, `window_id` | `(windows,)` | provenance for splitting and aggregation |
| `extraction_config` | scalar JSON string | schema version and every setting that determines cached values |

Assembled splits persist training/testing feature, label, and metadata arrays as
separate `.npy` files so the large feature matrices can be memory-mapped.
`meta.json` records subjects, dimensions, held-out groups, class rates, the
feature configuration signature, and the reference-split leakage warning.
Feature and assembled-split caches are reused only when these signatures match
the active configuration.

The selected split is built once and loaded once. All architectures receive the
same rows. Scaling is fitted only on training data, either inside a sklearn
pipeline or the neural adapter, and never on test data.

## Splits

- `subject`: eight fixed participants are held out. This measures transfer to a
  new viewer and is the recommended primary result.
- `trial`: ten complete trial identifiers are held out for every participant.
- `repo`: every fourth window is held out to reproduce the source project. As
  adjacent windows overlap by 93.75%, this is explicitly marked as leaky.

The same seed controls Python, NumPy, PyTorch, CUDA, sampling, model factories,
and the zero-worker neural DataLoader. CUDA deterministic mode is enabled.

### Cross-split selection

`--split all` applies the same model configuration to all three strategies and
keeps their metrics separate. A model qualifies for the robustness ranking only
when it succeeds on both `subject` and `trial`. Its primary robustness score is
the mean of those two macro-F1 values; the lower of the two breaks ties,
favoring consistent rather than split-specialized performance. The `repo`
result is shown as a reproduction diagnostic but never contributes to selection
because its overlapping windows leak signal across train and test.

## Fair-training contract

Each model sees identical selected rows for a given run. Scale-sensitive
classical estimators use a `StandardScaler` inside their fitted pipeline. The
common neural estimator fits its scaler on benchmark training rows, then makes
a fixed stratified 15% validation subset from those rows for early stopping.
Test rows never affect scaling, optimization, or epoch selection.

Neural architectures share the production trainer's Adam optimizer (learning
rate `1e-3`, weight decay `1e-4`), cross-entropy loss, a configured batch-size
ceiling of 256, and
ReduceLROnPlateau schedule. They also share 50 maximum epochs, patience 7, a
15% train-only validation fraction, and best-validation-loss restoration.
These values come from `src/config.py`. This is a controlled architecture
comparison, not per-model hyperparameter optimization.

Training, validation, and prediction are all batched. The attention-heavy
FT-Transformer is capped at 32 rows per batch and the five-layer FFT-LSTM at
16 rows per batch to fit consumer GPUs; other neural models may use the
configured ceiling of 256. These caps change execution granularity, not the
selected data or loss definition.

## Candidate rationale

The suite is curated by distinct inductive bias. More implementations of the
same hypothesis do not make the comparison more informative.

| Registry key | Architecture | Why retained | Main limitation |
|---|---|---|---|
| `logistic_regression` | StandardScaler -> linear two-class logistic head | Interpretable capacity floor and probability-native linear baseline. | Cannot learn nonlinear feature interactions. |
| `extra_trees` | 300 fully randomized trees with balanced class weights | One strong nonlinear bagging representative for engineered EEG features. | Large artifact; probabilities from vote fractions can be poorly calibrated. |
| `xgboost` | 500 depth-6 boosted trees, learning rate 0.05 | One mature gradient-boosting representative, often strong on tabular features. | Heavier CPU cost than the tree baseline. |
| `feature_mlp` | `160->256->128->64->2`; BatchNorm, ReLU, dropout 0.3 after each hidden layer | Natural neural model for a flat feature vector; topology matches the active downstream pipeline. | No explicit channel/band structure. |
| `band_electrode_cnn` | Reshape to `(5 bands, 32 electrodes)`; two kernel-3 Conv1D blocks `5->32->64`; global average pool; two-logit head | Tests whether local patterns along the declared electrode order help; topology matches the active downstream pipeline. | Electrode list order is not physical scalp distance, so convolutional locality is only an ordering prior. |
| `fft_lstm` | Reshape to `(160 steps, 1)`; bidirectional LSTM-128, then LSTM `256->256->64->64->32`; dropout `.6/.6/.6/.4/.4`; `32->16->2` head | Published FFT-feature recurrent architecture, necessary to reproduce and compare the incumbent model. | The 160 simultaneous channel-band values are not time; it is slow, over-parameterized, and semantically awkward. |
| `ft_transformer` | Per-feature scalar tokenization; two 16-wide, four-head Transformer encoder layers; mean pooling; two-logit head | One modern tabular-attention hypothesis that preserves feature identity without invented temporal or spatial geometry. | Quadratic attention over 160 tokens and higher compute than the MLP. |
| `tabicl` | TabICL classifier with shared CUDA/CPU policy | In-context learner for tabular features with a qualitatively different inference strategy. | Sensitive to memory and version constraints of the installed package. |

TabICL uses a deterministic, jointly stratified cap of 10,000 training windows
and 2,000 test windows by default. Its in-context output tensors otherwise grow
to impractical sizes on the full window-level DEAP split. The same selected
rows are used for Valence and Arousal, and artifacts record `n_train_samples`,
`n_test_samples`, and `sampling_protocol`. Therefore TabICL scores describe the
reduced-data protocol and must not be treated as a like-for-like full-data
comparison. Use `--tabicl-max-train-samples` and
`--tabicl-max-test-samples` to tune the ceilings for available memory.

This benchmark suite is closed and required. Every listed candidate must be
available in the benchmark environment and is expected to run.

### Redundancy audit

The following candidates were removed from the standard suite:

- linear SVM and SGD: alternative linear objectives/optimizers, but little new
  architectural information beyond probability-native logistic regression;
- random forest: overlaps with Extra Trees as a bagged decision-tree ensemble;
- shallow, medium, and residual benchmark MLPs: superseded by the exact
  production MLP;
- spectral LSTM/GRU/TCN and channel CNN-Attention/CNN-LSTM ports: speculative
  reinterpretations superseded by the actual production CNN and reference LSTM;
- LightGBM and CatBoost: duplicate the boosted-tree hypothesis already
  represented by XGBoost;
- TabNet: another tabular deep model, less distinct once FT-Transformer and the
  production MLP are present.

These removals reduce compute and multiple-comparison noise without dropping a
major learning paradigm.

## Methodological exclusions

Braindecode EEGNet, Deep4Net, ShallowFBCSPNet, and EEGConformer are excluded.
They require densely sampled raw EEG time series and their temporal kernels,
pooling, sampling frequency, and receptive fields are not meaningful on five
band summaries. Adding them would compare representation changes rather than
architectures. A future raw-signal benchmark should be a separately declared
experiment, not mixed into this feature benchmark.

## Unified prediction and evaluation

Every adapter implements `fit`, `predict`, and `predict_proba`; probabilities
must be finite `(n, 2)` rows summing to one. One evaluation function computes:

- accuracy, majority accuracy, and accuracy lift;
- macro and per-class F1;
- ROC-AUC and PR-AUC;
- Brier score and log loss for probability quality;
- confusion matrix and class-collapse status;
- wall-clock fit/inference time, serialized model size, and neural parameter
  count when available.

Each run saves raw probabilities plus confusion, ROC, precision-recall, and
calibration plots. Neural early stopping restores the lowest-validation-loss
epoch and exports the full loss history.

### Artifact contract

Each successful `<run>/<target>/<model>/` directory contains:

- `model.joblib` for sklearn-compatible estimators or `model.pt` for bundled
  PyTorch networks;
- `predictions.npz` with labels, hard predictions, both class probabilities,
  and the corresponding test-window metadata;
- `metrics.json` and the four diagnostic plots;
- `training_history.json` for bundled neural models.

The run root contains `comparison.csv`, `summary.json`, `REPORT.md`, and
`model_comparison.png`. Aggregate files are rewritten after each experiment so
a long interrupted run still has a readable partial report. PyTorch checkpoints
include architecture settings, CPU weights, scaler statistics, class order, and
history; restore them with `TorchTabularClassifier.load`.
The run configuration also records Python, NumPy, scikit-learn, PyTorch,
platform, and CPU/CUDA selection to make environment differences visible.

## Ranking and limitations

Valence and Arousal are separate experiments and have separate winners. The
primary rank is macro-F1 (robust to DEAP imbalance), with ROC-AUC then accuracy
lift as deterministic tie-breakers. `summary.json` stores the rule.

DEAP provides one rating for an entire trial, so every window inherits a noisy,
constant label. Results depend strongly on split choice. A one-off run measures
architecture performance for one seed, not uncertainty; publication-quality
work should repeat several seeds and report dispersion.

## Extension checklist

A new candidate belongs in `src/models/model_registry.py` only when it:

1. consumes the existing `(n, 160)` feature matrix without alternate preprocessing;
2. implements `fit` and returns finite `(n, 2)` rows from `predict_proba`;
3. fits every learned transformation exclusively on training rows;
4. accepts the run seed or documents why determinism is unavailable;
5. can be serialized with enough state for later probability inference;
6. records computational, sample-size, and semantic limitations.
