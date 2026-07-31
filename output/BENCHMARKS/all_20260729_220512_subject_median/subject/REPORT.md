# DEAP architecture benchmark

Primary ranking: `macro_f1`; ties: `roc_auc`, then `accuracy_lift`.

- Best Valence model: **band_electrode_cnn**
- Best Arousal model: **feature_mlp**

| Target | Rank | Model | Macro F1 | ROC-AUC | Accuracy | Lift | Train s | Infer s |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| Arousal | 1 | feature_mlp | 0.5099 | 0.5117 | 0.5163 | 0.0100 | 640.03 | 0.68 |
| Arousal | 2 | band_electrode_cnn | 0.5078 | 0.5100 | 0.5097 | 0.0035 | 644.88 | 0.69 |
| Arousal | 3 | extra_trees | 0.4980 | 0.4962 | 0.4986 | -0.0076 | 86.46 | 0.69 |
| Arousal | 4 | ft_transformer | 0.4959 | 0.5009 | 0.4962 | -0.0101 | 3252.28 | 6.80 |
| Arousal | 5 | xgboost | 0.4939 | 0.4971 | 0.5022 | -0.0040 | 33.38 | 0.21 |
| Arousal | 6 | logistic_regression | 0.4713 | 0.5021 | 0.4970 | -0.0093 | 13.11 | 0.07 |
| Arousal | 7 | tabicl | 0.4588 | 0.4750 | 0.4900 | -0.0165 | 15.90 | 2.14 |
| Arousal | 8 | fft_lstm | 0.3361 | 0.5000 | 0.5062 | 0.0000 | 5257.20 | 57.65 |
| Valence | 1 | band_electrode_cnn | 0.5363 | 0.5776 | 0.5574 | 0.0512 | 551.22 | 0.71 |
| Valence | 2 | logistic_regression | 0.5346 | 0.5402 | 0.5357 | 0.0294 | 13.43 | 0.08 |
| Valence | 3 | tabicl | 0.5203 | 0.5431 | 0.5270 | 0.0210 | 20.19 | 2.03 |
| Valence | 4 | xgboost | 0.5140 | 0.5125 | 0.5159 | 0.0097 | 31.43 | 0.22 |
| Valence | 5 | feature_mlp | 0.5114 | 0.5185 | 0.5117 | 0.0054 | 643.77 | 0.58 |
| Valence | 6 | extra_trees | 0.5022 | 0.5134 | 0.5024 | -0.0039 | 84.57 | 0.70 |
| Valence | 7 | ft_transformer | 0.5012 | 0.4989 | 0.5028 | -0.0034 | 8218.87 | 8.02 |
| Valence | 8 | fft_lstm | 0.3361 | 0.5000 | 0.5062 | 0.0000 | 4834.84 | 57.86 |
