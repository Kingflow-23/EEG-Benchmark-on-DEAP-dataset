# Cross-split robustness report

The robustness score uses only `subject` and `trial`; the leaky `repo` split is diagnostic and cannot select a winner.

- Robust Valence model: **tabicl**
- Robust Arousal model: **feature_mlp**

| Target | Rank | Model | Subject F1 | Trial F1 | Mean | Worst | Repo F1 |
|---|---:|---|---:|---:|---:|---:|---:|
| Arousal | 1 | feature_mlp | 0.5099 | 0.5356 | 0.5227 | 0.5099 | 0.7329 |
| Arousal | 2 | extra_trees | 0.4980 | 0.5347 | 0.5164 | 0.4980 | 0.9798 |
| Arousal | 3 | xgboost | 0.4939 | 0.5354 | 0.5147 | 0.4939 | 0.7769 |
| Arousal | 4 | band_electrode_cnn | 0.5078 | 0.5153 | 0.5115 | 0.5078 | 0.6041 |
| Arousal | 5 | ft_transformer | 0.4959 | 0.5268 | 0.5114 | 0.4959 | 0.6039 |
| Arousal | 6 | tabicl | 0.4588 | 0.5317 | 0.4953 | 0.4588 | 0.7389 |
| Arousal | 7 | logistic_regression | 0.4713 | 0.4980 | 0.4847 | 0.4713 | 0.5631 |
| Arousal | 8 | fft_lstm | 0.3361 | 0.3416 | 0.3388 | 0.3361 | 0.3385 |
| Valence | 1 | tabicl | 0.5203 | 0.5780 | 0.5491 | 0.5203 | 0.7708 |
| Valence | 2 | band_electrode_cnn | 0.5363 | 0.5592 | 0.5477 | 0.5363 | 0.6468 |
| Valence | 3 | logistic_regression | 0.5346 | 0.5484 | 0.5415 | 0.5346 | 0.5922 |
| Valence | 4 | feature_mlp | 0.5114 | 0.5715 | 0.5414 | 0.5114 | 0.7676 |
| Valence | 5 | xgboost | 0.5140 | 0.5681 | 0.5410 | 0.5140 | 0.7909 |
| Valence | 6 | extra_trees | 0.5022 | 0.5645 | 0.5334 | 0.5022 | 0.9803 |
| Valence | 7 | ft_transformer | 0.5012 | 0.5642 | 0.5327 | 0.5012 | 0.7005 |
| Valence | 8 | fft_lstm | 0.3361 | 0.3443 | 0.3402 | 0.3361 | 0.3371 |
