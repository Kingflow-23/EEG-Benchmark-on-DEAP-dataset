# Cross-split robustness report

The robustness score uses only `subject` and `trial`; the leaky `repo` split is diagnostic and cannot select a winner.

- Robust Valence model: **band_electrode_cnn**
- Robust Arousal model: **xgboost**

| Target | Rank | Model | Subject F1 | Trial F1 | Mean | Worst | Repo F1 |
|---|---:|---|---:|---:|---:|---:|---:|
| Arousal | 1 | xgboost | 0.5564 | 0.6179 | 0.5872 | 0.5564 | 0.7872 |
| Arousal | 2 | ft_transformer | 0.5689 | 0.6000 | 0.5845 | 0.5689 | 0.7194 |
| Arousal | 3 | logistic_regression | 0.5203 | 0.5888 | 0.5546 | 0.5203 | 0.6102 |
| Arousal | 4 | tabicl | 0.4516 | 0.6291 | 0.5403 | 0.4516 | 0.7572 |
| Arousal | 5 | band_electrode_cnn | 0.5083 | 0.5702 | 0.5392 | 0.5083 | 0.6423 |
| Arousal | 6 | extra_trees | 0.4658 | 0.6055 | 0.5357 | 0.4658 | 0.9761 |
| Arousal | 7 | feature_mlp | 0.4351 | 0.5915 | 0.5133 | 0.4351 | 0.7705 |
| Arousal | 8 | fft_lstm | 0.3701 | 0.3725 | 0.3713 | 0.3701 | 0.3707 |
| Valence | 1 | band_electrode_cnn | 0.5292 | 0.5975 | 0.5634 | 0.5292 | 0.6531 |
| Valence | 2 | ft_transformer | 0.5087 | 0.5683 | 0.5385 | 0.5087 | 0.7338 |
| Valence | 3 | tabicl | 0.4881 | 0.5731 | 0.5306 | 0.4881 | 0.7764 |
| Valence | 4 | extra_trees | 0.4745 | 0.5839 | 0.5292 | 0.4745 | 0.9783 |
| Valence | 5 | xgboost | 0.4675 | 0.5778 | 0.5226 | 0.4675 | 0.7836 |
| Valence | 6 | logistic_regression | 0.4685 | 0.5677 | 0.5181 | 0.4685 | 0.5880 |
| Valence | 7 | feature_mlp | 0.4550 | 0.5650 | 0.5100 | 0.4550 | 0.7444 |
| Valence | 8 | fft_lstm | 0.3561 | 0.3638 | 0.3600 | 0.3561 | 0.3613 |
