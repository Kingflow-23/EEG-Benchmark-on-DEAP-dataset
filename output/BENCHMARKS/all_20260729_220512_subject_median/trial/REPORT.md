# DEAP architecture benchmark

Primary ranking: `macro_f1`; ties: `roc_auc`, then `accuracy_lift`.

- Best Valence model: **tabicl**
- Best Arousal model: **feature_mlp**

| Target | Rank | Model | Macro F1 | ROC-AUC | Accuracy | Lift | Train s | Infer s |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| Arousal | 1 | feature_mlp | 0.5356 | 0.5494 | 0.5356 | 0.0169 | 559.59 | 0.53 |
| Arousal | 2 | xgboost | 0.5354 | 0.5501 | 0.5362 | 0.0175 | 30.92 | 0.22 |
| Arousal | 3 | extra_trees | 0.5347 | 0.5451 | 0.5352 | 0.0165 | 82.93 | 1.00 |
| Arousal | 4 | tabicl | 0.5317 | 0.5385 | 0.5325 | 0.0140 | 14.87 | 2.10 |
| Arousal | 5 | ft_transformer | 0.5268 | 0.5383 | 0.5269 | 0.0081 | 8279.83 | 7.13 |
| Arousal | 6 | band_electrode_cnn | 0.5153 | 0.5244 | 0.5172 | -0.0015 | 523.98 | 0.57 |
| Arousal | 7 | logistic_regression | 0.4980 | 0.5056 | 0.4995 | -0.0193 | 12.41 | 0.10 |
| Arousal | 8 | fft_lstm | 0.3416 | 0.5000 | 0.5188 | 0.0000 | 4528.40 | 52.37 |
| Valence | 1 | tabicl | 0.5780 | 0.6089 | 0.5780 | 0.0530 | 14.82 | 2.00 |
| Valence | 2 | feature_mlp | 0.5715 | 0.6047 | 0.5715 | 0.0465 | 564.65 | 0.59 |
| Valence | 3 | xgboost | 0.5681 | 0.6036 | 0.5682 | 0.0432 | 30.20 | 0.20 |
| Valence | 4 | extra_trees | 0.5645 | 0.5991 | 0.5648 | 0.0398 | 81.84 | 0.99 |
| Valence | 5 | ft_transformer | 0.5642 | 0.5978 | 0.5644 | 0.0394 | 7058.22 | 6.68 |
| Valence | 6 | band_electrode_cnn | 0.5592 | 0.5951 | 0.5602 | 0.0352 | 537.99 | 0.69 |
| Valence | 7 | logistic_regression | 0.5484 | 0.5721 | 0.5484 | 0.0234 | 17.28 | 0.10 |
| Valence | 8 | fft_lstm | 0.3443 | 0.5000 | 0.5250 | 0.0000 | 4514.78 | 57.49 |
