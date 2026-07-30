# DEAP architecture benchmark

Primary ranking: `macro_f1`; ties: `roc_auc`, then `accuracy_lift`.

- Best Valence model: **band_electrode_cnn**
- Best Arousal model: **ft_transformer**

| Target | Rank | Model | Macro F1 | ROC-AUC | Accuracy | Lift | Train s | Infer s |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| Arousal | 1 | ft_transformer | 0.5689 | 0.5839 | 0.5769 | -0.0106 | 3696.45 | 4.66 |
| Arousal | 2 | xgboost | 0.5564 | 0.5967 | 0.5866 | -0.0009 | 10.72 | 0.13 |
| Arousal | 3 | logistic_regression | 0.5203 | 0.5810 | 0.5487 | -0.0388 | 21.05 | 0.16 |
| Arousal | 4 | band_electrode_cnn | 0.5083 | 0.5378 | 0.5384 | -0.0491 | 301.51 | 0.41 |
| Arousal | 5 | extra_trees | 0.4658 | 0.5724 | 0.5617 | -0.0258 | 40.10 | 0.33 |
| Arousal | 6 | tabicl | 0.4516 | 0.5937 | 0.5860 | -0.0015 | 10.71 | 1.31 |
| Arousal | 7 | feature_mlp | 0.4351 | 0.4532 | 0.4864 | -0.1011 | 334.78 | 0.46 |
| Arousal | 8 | fft_lstm | 0.3701 | 0.5000 | 0.5875 | 0.0000 | 6445.11 | 37.37 |
| Valence | 1 | band_electrode_cnn | 0.5292 | 0.5631 | 0.5359 | -0.0172 | 320.88 | 0.45 |
| Valence | 2 | ft_transformer | 0.5087 | 0.5162 | 0.5171 | -0.0360 | 2682.35 | 4.57 |
| Valence | 3 | tabicl | 0.4881 | 0.5053 | 0.5245 | -0.0290 | 11.65 | 1.28 |
| Valence | 4 | extra_trees | 0.4745 | 0.5303 | 0.5548 | 0.0016 | 41.42 | 0.35 |
| Valence | 5 | logistic_regression | 0.4685 | 0.4681 | 0.4858 | -0.0673 | 22.45 | 0.16 |
| Valence | 6 | xgboost | 0.4675 | 0.5026 | 0.5396 | -0.0135 | 11.03 | 0.12 |
| Valence | 7 | feature_mlp | 0.4550 | 0.4396 | 0.5026 | -0.0505 | 327.06 | 0.45 |
| Valence | 8 | fft_lstm | 0.3561 | 0.5000 | 0.5531 | 0.0000 | 2898.33 | 37.11 |
