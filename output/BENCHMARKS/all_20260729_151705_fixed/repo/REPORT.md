# DEAP architecture benchmark

Primary ranking: `macro_f1`; ties: `roc_auc`, then `accuracy_lift`.

- Best Valence model: **extra_trees**
- Best Arousal model: **extra_trees**

| Target | Rank | Model | Macro F1 | ROC-AUC | Accuracy | Lift | Train s | Infer s |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| Arousal | 1 | extra_trees | 0.9761 | 0.9977 | 0.9770 | 0.3879 | 40.49 | 0.61 |
| Arousal | 2 | xgboost | 0.7872 | 0.8818 | 0.7998 | 0.2108 | 10.90 | 0.12 |
| Arousal | 3 | feature_mlp | 0.7705 | 0.8692 | 0.7842 | 0.1951 | 363.19 | 0.61 |
| Arousal | 4 | tabicl | 0.7572 | 0.8472 | 0.7685 | 0.1795 | 8.83 | 1.26 |
| Arousal | 5 | ft_transformer | 0.7194 | 0.8093 | 0.7364 | 0.1474 | 4931.31 | 4.56 |
| Arousal | 6 | band_electrode_cnn | 0.6423 | 0.7287 | 0.6760 | 0.0870 | 358.18 | 0.61 |
| Arousal | 7 | logistic_regression | 0.6102 | 0.6861 | 0.6504 | 0.0614 | 28.99 | 0.17 |
| Arousal | 8 | fft_lstm | 0.3707 | 0.5000 | 0.5891 | 0.0000 | 4372.40 | 46.63 |
| Valence | 1 | extra_trees | 0.9783 | 0.9980 | 0.9787 | 0.4131 | 48.38 | 0.69 |
| Valence | 2 | xgboost | 0.7836 | 0.8763 | 0.7901 | 0.2245 | 11.78 | 0.12 |
| Valence | 3 | tabicl | 0.7764 | 0.8626 | 0.7810 | 0.2150 | 10.85 | 1.29 |
| Valence | 4 | feature_mlp | 0.7444 | 0.8362 | 0.7502 | 0.1846 | 378.20 | 0.49 |
| Valence | 5 | ft_transformer | 0.7338 | 0.8202 | 0.7370 | 0.1713 | 5365.76 | 5.32 |
| Valence | 6 | band_electrode_cnn | 0.6531 | 0.7181 | 0.6652 | 0.0996 | 273.55 | 0.57 |
| Valence | 7 | logistic_regression | 0.5880 | 0.6585 | 0.6194 | 0.0537 | 27.61 | 0.17 |
| Valence | 8 | fft_lstm | 0.3613 | 0.5000 | 0.5656 | 0.0000 | 12553.53 | 43.15 |
