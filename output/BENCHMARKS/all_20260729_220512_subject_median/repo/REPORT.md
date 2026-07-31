# DEAP architecture benchmark

Primary ranking: `macro_f1`; ties: `roc_auc`, then `accuracy_lift`.

- Best Valence model: **extra_trees**
- Best Arousal model: **extra_trees**

| Target | Rank | Model | Macro F1 | ROC-AUC | Accuracy | Lift | Train s | Infer s |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| Arousal | 1 | extra_trees | 0.9798 | 0.9983 | 0.9798 | 0.4681 | 91.27 | 1.20 |
| Arousal | 2 | xgboost | 0.7769 | 0.8634 | 0.7772 | 0.2655 | 32.43 | 0.21 |
| Arousal | 3 | tabicl | 0.7389 | 0.8299 | 0.7390 | 0.2275 | 16.77 | 2.13 |
| Arousal | 4 | feature_mlp | 0.7329 | 0.8212 | 0.7330 | 0.2212 | 603.13 | 0.67 |
| Arousal | 5 | band_electrode_cnn | 0.6041 | 0.6515 | 0.6081 | 0.0963 | 1294.09 | 2.51 |
| Arousal | 6 | ft_transformer | 0.6039 | 0.6642 | 0.6086 | 0.0969 | 9267.03 | 6.86 |
| Arousal | 7 | logistic_regression | 0.5631 | 0.5927 | 0.5656 | 0.0539 | 13.58 | 0.11 |
| Arousal | 8 | fft_lstm | 0.3385 | 0.5000 | 0.5117 | 0.0000 | 10607.47 | 187.16 |
| Valence | 1 | extra_trees | 0.9803 | 0.9981 | 0.9803 | 0.4717 | 84.51 | 1.22 |
| Valence | 2 | xgboost | 0.7909 | 0.8783 | 0.7910 | 0.2824 | 32.21 | 0.20 |
| Valence | 3 | tabicl | 0.7708 | 0.8599 | 0.7710 | 0.2625 | 15.67 | 2.14 |
| Valence | 4 | feature_mlp | 0.7676 | 0.8563 | 0.7676 | 0.2591 | 566.61 | 0.51 |
| Valence | 5 | ft_transformer | 0.7005 | 0.7916 | 0.7033 | 0.1947 | 10740.80 | 7.11 |
| Valence | 6 | band_electrode_cnn | 0.6468 | 0.7053 | 0.6469 | 0.1383 | 1055.04 | 2.71 |
| Valence | 7 | logistic_regression | 0.5922 | 0.6350 | 0.5922 | 0.0836 | 17.38 | 0.08 |
| Valence | 8 | fft_lstm | 0.3371 | 0.5000 | 0.5086 | 0.0000 | 5004.67 | 55.36 |
