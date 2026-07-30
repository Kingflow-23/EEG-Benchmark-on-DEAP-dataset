# DEAP architecture benchmark

Primary ranking: `macro_f1`; ties: `roc_auc`, then `accuracy_lift`.

- Best Valence model: **band_electrode_cnn**
- Best Arousal model: **tabicl**

| Target | Rank | Model | Macro F1 | ROC-AUC | Accuracy | Lift | Train s | Infer s |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| Arousal | 1 | tabicl | 0.6291 | 0.6658 | 0.6480 | 0.0540 | 9.06 | 1.24 |
| Arousal | 2 | xgboost | 0.6179 | 0.6596 | 0.6413 | 0.0475 | 10.82 | 0.13 |
| Arousal | 3 | extra_trees | 0.6055 | 0.6556 | 0.6352 | 0.0415 | 38.88 | 0.51 |
| Arousal | 4 | ft_transformer | 0.6000 | 0.6359 | 0.6208 | 0.0271 | 3793.58 | 4.66 |
| Arousal | 5 | feature_mlp | 0.5915 | 0.6293 | 0.6140 | 0.0202 | 340.16 | 0.45 |
| Arousal | 6 | logistic_regression | 0.5888 | 0.6390 | 0.6268 | 0.0330 | 27.62 | 0.16 |
| Arousal | 7 | band_electrode_cnn | 0.5702 | 0.6338 | 0.6176 | 0.0238 | 285.93 | 0.50 |
| Arousal | 8 | fft_lstm | 0.3725 | 0.5000 | 0.5938 | 0.0000 | 3535.85 | 37.83 |
| Valence | 1 | band_electrode_cnn | 0.5975 | 0.6343 | 0.6138 | 0.0419 | 355.37 | 0.47 |
| Valence | 2 | extra_trees | 0.5839 | 0.6211 | 0.6002 | 0.0283 | 44.09 | 0.53 |
| Valence | 3 | xgboost | 0.5778 | 0.6092 | 0.5909 | 0.0190 | 10.69 | 0.13 |
| Valence | 4 | tabicl | 0.5731 | 0.5979 | 0.5880 | 0.0160 | 10.25 | 1.28 |
| Valence | 5 | ft_transformer | 0.5683 | 0.5961 | 0.5717 | -0.0001 | 3662.74 | 4.48 |
| Valence | 6 | logistic_regression | 0.5677 | 0.6128 | 0.5939 | 0.0220 | 26.34 | 0.17 |
| Valence | 7 | feature_mlp | 0.5650 | 0.5964 | 0.5731 | 0.0012 | 377.94 | 0.47 |
| Valence | 8 | fft_lstm | 0.3638 | 0.5000 | 0.5719 | 0.0000 | 2590.85 | 38.00 |
