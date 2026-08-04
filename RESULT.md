# Benchmark Results

This document summarizes the final DEAP benchmark run stored under
`output/BENCHMARKS/all_20260729_220512_subject_median/`.

The benchmark compares eight model families on the binary **Valence** and
**Arousal** tasks using the shared 160-feature EEG representation described in
`README.md` and `METHODOLOGY.md`. Metrics are reported per target, and the
final ranking uses `macro_f1` first, then `roc_auc`, then `accuracy_lift`.

## 1. Experimental setup

- Dataset: DEAP preprocessed Python files.
- Feature representation: 160 FFT band features per window, derived from the
  first 32 EEG channels.
- Targets:
  - `Valence` = low/high split at rating 5.
  - `Arousal` = low/high split at rating 5.
- Main split reported here: `subject`.
- Robustness summary also available from `--split all` across:
  - `subject`
  - `trial`
  - `repo`

The `repo` split is included only as a reproduction diagnostic. Because its
windows overlap heavily, it is not treated as a valid generalization estimate.

## 2. Main Results on the Subject Split

The `subject` split is the most deployment-relevant result because it tests
transfer to unseen participants.

### Valence

| Rank | Model | Macro F1 | ROC-AUC | Accuracy | Lift |
|---:|---|---:|---:|---:|---:|
| 1 | `band_electrode_cnn` | 0.5363 | 0.5776 | 0.5574 | 0.0512 |
| 2 | `logistic_regression` | 0.5346 | 0.5402 | 0.5357 | 0.0294 |
| 3 | `tabicl` | 0.5203 | 0.5431 | 0.5270 | 0.0210 |
| 4 | `xgboost` | 0.5140 | 0.5125 | 0.5159 | 0.0097 |
| 5 | `feature_mlp` | 0.5114 | 0.5185 | 0.5117 | 0.0054 |
| 6 | `extra_trees` | 0.5022 | 0.5134 | 0.5024 | -0.0039 |
| 7 | `ft_transformer` | 0.5012 | 0.4989 | 0.5028 | -0.0034 |
| 8 | `fft_lstm` | 0.3361 | 0.5000 | 0.5063 | 0.0000 |

Best Valence model on the subject split:

- `band_electrode_cnn`

### Arousal

| Rank | Model | Macro F1 | ROC-AUC | Accuracy | Lift |
|---:|---|---:|---:|---:|---:|
| 1 | `feature_mlp` | 0.5099 | 0.5117 | 0.5163 | 0.0100 |
| 2 | `band_electrode_cnn` | 0.5078 | 0.5100 | 0.5097 | 0.0035 |
| 3 | `extra_trees` | 0.4980 | 0.4962 | 0.4986 | -0.0076 |
| 4 | `ft_transformer` | 0.4959 | 0.5009 | 0.4962 | -0.0101 |
| 5 | `xgboost` | 0.4939 | 0.4971 | 0.5022 | -0.0040 |
| 6 | `logistic_regression` | 0.4713 | 0.5021 | 0.4970 | -0.0093 |
| 7 | `tabicl` | 0.4588 | 0.4750 | 0.4900 | -0.0165 |
| 8 | `fft_lstm` | 0.3361 | 0.5000 | 0.5063 | 0.0000 |

Best Arousal model on the subject split:

- `feature_mlp`

## 3. Cross-Split Robustness Result

The cross-split summary ranks models by performance on the leak-free
`subject` and `trial` splits only, using mean macro F1 with the worst split as
the tie-breaker.

### Robustness winners

- Valence: `tabicl`
- Arousal: `feature_mlp`

### Cross-split ranking

| Target | Model | Subject Macro F1 | Trial Macro F1 | Mean Macro F1 | Worst Split |
|---|---|---:|---:|---:|---:|
| Valence | `tabicl` | 0.5203 | 0.5780 | 0.5491 | 0.5203 |
| Valence | `band_electrode_cnn` | 0.5363 | 0.5592 | 0.5477 | 0.5363 |
| Valence | `logistic_regression` | 0.5346 | 0.5484 | 0.5415 | 0.5346 |
| Valence | `feature_mlp` | 0.5114 | 0.5715 | 0.5414 | 0.5114 |
| Valence | `xgboost` | 0.5140 | 0.5681 | 0.5410 | 0.5140 |
| Valence | `extra_trees` | 0.5022 | 0.5645 | 0.5334 | 0.5022 |
| Valence | `ft_transformer` | 0.5012 | 0.5642 | 0.5327 | 0.5012 |
| Valence | `fft_lstm` | 0.3361 | 0.3443 | 0.3402 | 0.3361 |
| Arousal | `feature_mlp` | 0.5099 | 0.5356 | 0.5227 | 0.5099 |
| Arousal | `extra_trees` | 0.4980 | 0.5347 | 0.5164 | 0.4980 |
| Arousal | `xgboost` | 0.4939 | 0.5354 | 0.5147 | 0.4939 |
| Arousal | `band_electrode_cnn` | 0.5078 | 0.5153 | 0.5115 | 0.5078 |
| Arousal | `ft_transformer` | 0.4959 | 0.5268 | 0.5114 | 0.4959 |
| Arousal | `tabicl` | 0.4588 | 0.5317 | 0.4953 | 0.4588 |
| Arousal | `logistic_regression` | 0.4713 | 0.4980 | 0.4847 | 0.4713 |
| Arousal | `fft_lstm` | 0.3361 | 0.3416 | 0.3388 | 0.3361 |

## 4. Interpretation

### 4.1 Valence is easier than Arousal

The best Valence scores are consistently above the best Arousal scores.
Valence reaches a subject-split macro F1 of `0.5363` and a cross-split mean of
`0.5491`, while Arousal peaks at `0.5099` on the subject split and `0.5227`
cross-split mean.

This suggests the feature representation captures Valence-related structure
slightly better than Arousal-related structure in this benchmark setting.

### 4.2 The strongest models are not the heaviest ones

The top results come from:

- `band_electrode_cnn` for subject-split Valence
- `feature_mlp` for subject-split Arousal
- `tabicl` for cross-split Valence

These are not the slowest or largest models. In particular:

- `band_electrode_cnn` is small and efficient, yet leads Valence on the
  subject split.
- `feature_mlp` is competitive on both targets and remains the best Arousal
  model under the robustness ranking.
- `tabicl` performs well on Valence in the robustness ranking, but its score
  reflects the capped sampling protocol, not the full split.

This indicates that architectural fit matters more than raw parameter count on
this feature-level EEG benchmark.

### 4.3 Tree ensembles are competitive but not dominant

`xgboost` and `extra_trees` remain strong middle-of-the-pack baselines. They
are especially useful as sanity checks because they are fast enough to train
and often strong on tabular data.

However, on this run they do not beat the best neural or structured models on
the primary subject split.

### 4.4 The FFT-LSTM is not a good fit here

`fft_lstm` is consistently the worst model:

- macro F1 stays near the majority baseline,
- ROC-AUC is exactly `0.5`,
- the confusion matrix shows class collapse on the subject split,
- training and inference are much slower than the other models.

This is the clearest negative result in the benchmark. The model treats the
160 feature values as a temporal sequence even though they are not time steps,
which likely explains the poor behavior.

### 4.5 Ranking depends on the split

One important result is that the best model changes with the evaluation split:

- subject-split Valence winner: `band_electrode_cnn`
- cross-split Valence robustness winner: `tabicl`
- subject-split Arousal winner: `feature_mlp`
- cross-split Arousal robustness winner: `feature_mlp`

This means the benchmark is sensitive to how the held-out data are defined.
That is expected for DEAP, where a trial-level label is reused across many
windows and the split strategy strongly affects the difficulty of the task.

## 5. Practical Conclusions

1. The shared 160-feature EEG representation is sufficient to produce
   above-baseline performance, but the gains are modest.
2. `Valence` is the easier target in this setup.
3. For the `subject` split, the most useful single models are:
   - `band_electrode_cnn` for Valence
   - `feature_mlp` for Arousal
4. For robustness across leak-free splits:
   - `tabicl` is the best Valence model by the cross-split ranking
   - `feature_mlp` remains the best Arousal model
5. `fft_lstm` should not be used as a preferred model for this benchmark
   representation.

## 6. Limitations

- The benchmark uses window-level labels inherited from trial ratings, so the
  supervision signal is noisy.
- The `repo` split is not a valid generalization estimate because of overlapping
  windows.
- `tabicl` uses a fixed capped sample protocol, so its scores are not directly
  comparable to full-data models.
- These are single-seed results; repeated runs would be needed for uncertainty
  estimates and significance testing.

## 7. Final Conclusion

The benchmark shows that modest but real performance is achievable on DEAP with
carefully shared preprocessing and fair split handling. The strongest
deployment-oriented result on the subject split comes from a structured CNN for
Valence and a feedforward MLP for Arousal. When robustness across leak-free
splits is emphasized, `tabicl` becomes the most stable Valence model, while the
MLP remains the best Arousal model.

Overall, the results support the conclusion that:

- architecture choice matters,
- split design matters even more,
- and overly sequential models such as `fft_lstm` are a poor match for this
  feature-based EEG benchmark.

For a future iteration, the most valuable next step would be repeated-seed
evaluation with confidence intervals and a clearer publication-ready comparison
table.
