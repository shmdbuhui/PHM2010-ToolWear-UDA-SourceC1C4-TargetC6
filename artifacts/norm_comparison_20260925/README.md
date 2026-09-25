# PHM2010 input normalization comparison

All 120 trainings completed: 2 normalizations × 6 directions × 5 seeds × 2 model methods.
Values are five-seed mean ± sample SD (ddof=1). Δ is Min–Max minus Z-score, paired by seed.
DARE-GRAM used unlabeled target cuts 1–315. All metrics and prediction plots use cuts 1–315, including target C6.
Target labels were read after both model checkpoints and predictions were fixed.

The C6 full-lifecycle request arrived after the Z-score training run. Its target-C6 final checkpoints
were kept byte-for-byte; the existing prediction and metric functions were rerun on cuts 1–315.
Min–Max used cuts 1–315 during its original evaluation. No C6 model was retrained for this cut change.

| Direction | Method | Metric | Z-score | Min–Max | Δ Min–Max − Z-score |
|---|---|---|---:|---:|---:|
| C1→C4 | source_only | R2 | 0.8312 ± 0.0241 | 0.8198 ± 0.0051 | -0.0114 ± 0.0192 |
| C1→C4 | source_only | MAE | 11.3677 ± 1.4297 | 12.5829 ± 0.6422 | +1.2152 ± 0.8133 |
| C1→C4 | source_only | RMSE | 15.5490 ± 1.1205 | 16.0973 ± 0.2261 | +0.5484 ± 0.8965 |
| C1→C4 | daregram | R2 | 0.8241 ± 0.0186 | 0.8079 ± 0.0130 | -0.0163 ± 0.0172 |
| C1→C4 | daregram | MAE | 12.2958 ± 0.9814 | 13.6968 ± 0.6845 | +1.4010 ± 1.3115 |
| C1→C4 | daregram | RMSE | 15.8873 ± 0.8329 | 16.6167 ± 0.5725 | +0.7295 ± 0.7662 |
| C1→C6 | source_only | R2 | 0.6435 ± 0.0385 | 0.7487 ± 0.1251 | +0.1052 ± 0.1165 |
| C1→C6 | source_only | MAE | 14.7781 ± 0.8938 | 13.0250 ± 3.3806 | -1.7531 ± 3.0735 |
| C1→C6 | source_only | RMSE | 23.9046 ± 1.3214 | 19.4820 ± 5.4990 | -4.4226 ± 5.0648 |
| C1→C6 | daregram | R2 | 0.7436 ± 0.0858 | 0.7459 ± 0.0853 | +0.0024 ± 0.1083 |
| C1→C6 | daregram | MAE | 15.1593 ± 3.0404 | 14.7543 ± 2.5660 | -0.4050 ± 3.6404 |
| C1→C6 | daregram | RMSE | 20.0728 ± 3.3667 | 19.9918 ± 3.2629 | -0.0811 ± 4.2395 |
| C4→C1 | source_only | R2 | 0.5888 ± 0.1047 | 0.5468 ± 0.1690 | -0.0420 ± 0.1081 |
| C4→C1 | source_only | MAE | 14.3205 ± 1.4434 | 14.4129 ± 1.7759 | +0.0924 ± 1.0158 |
| C4→C1 | source_only | RMSE | 17.4007 ± 2.1909 | 18.1303 ± 3.3927 | +0.7297 ± 2.1137 |
| C4→C1 | daregram | R2 | 0.5377 ± 0.1690 | 0.6191 ± 0.1856 | +0.0813 ± 0.1097 |
| C4→C1 | daregram | MAE | 14.2622 ± 1.8797 | 13.7096 ± 2.4501 | -0.5526 ± 0.6513 |
| C4→C1 | daregram | RMSE | 18.3243 ± 3.3339 | 16.5217 ± 3.7183 | -1.8026 ± 2.3741 |
| C4→C6 | source_only | R2 | 0.7896 ± 0.0887 | 0.7186 ± 0.1955 | -0.0710 ± 0.1094 |
| C4→C6 | source_only | MAE | 15.2275 ± 2.6959 | 17.5337 ± 5.1650 | +2.3062 ± 3.2135 |
| C4→C6 | source_only | RMSE | 18.1071 ± 3.5624 | 20.4146 ± 6.6508 | +2.3075 ± 3.3148 |
| C4→C6 | daregram | R2 | 0.7519 ± 0.1024 | 0.8008 ± 0.0433 | +0.0489 ± 0.1330 |
| C4→C6 | daregram | MAE | 17.4657 ± 4.3553 | 15.6786 ± 1.5521 | -1.7871 ± 5.3454 |
| C4→C6 | daregram | RMSE | 19.6510 ± 3.9475 | 17.7996 ± 1.9886 | -1.8514 ± 5.4082 |
| C6→C1 | source_only | R2 | 0.3239 ± 0.2197 | 0.3627 ± 0.5075 | +0.0388 ± 0.6089 |
| C6→C1 | source_only | MAE | 20.5011 ± 4.7314 | 17.0761 ± 7.4851 | -3.4250 ± 9.0402 |
| C6→C1 | source_only | RMSE | 22.2165 ± 3.6355 | 20.5736 ± 8.0577 | -1.6429 ± 9.7105 |
| C6→C1 | daregram | R2 | 0.8683 ± 0.0785 | 0.6918 ± 0.1495 | -0.1766 ± 0.2106 |
| C6→C1 | daregram | MAE | 7.5374 ± 2.5415 | 11.3079 ± 2.7457 | +3.7705 ± 4.7145 |
| C6→C1 | daregram | RMSE | 9.5311 ± 3.0253 | 14.7880 ± 3.7312 | +5.2569 ± 6.1402 |
| C6→C4 | source_only | R2 | 0.5307 ± 0.1071 | 0.5035 ± 0.1843 | -0.0272 ± 0.2738 |
| C6→C4 | source_only | MAE | 18.2347 ± 4.2965 | 18.0376 ± 4.8257 | -0.1971 ± 6.2121 |
| C6→C4 | source_only | RMSE | 25.8281 ± 3.1458 | 26.3994 ± 4.6448 | +0.5713 ± 7.3095 |
| C6→C4 | daregram | R2 | 0.8391 ± 0.0506 | 0.7714 ± 0.0650 | -0.0677 ± 0.0957 |
| C6→C4 | daregram | MAE | 12.8584 ± 3.3074 | 16.8255 ± 2.7481 | +3.9671 ± 4.9757 |
| C6→C4 | daregram | RMSE | 15.0497 ± 2.4976 | 17.9848 ± 2.6082 | +2.9351 ± 4.2260 |

## Execution

Run in `upstream-reproduction`:

```powershell
python run_norm_comparison.py --preflight
python run_norm_comparison.py --norm-method zscore
python reevaluate_c6_full_lifecycle.py --norm-method zscore
python run_norm_comparison.py --norm-method minmax
python audit_norm_comparison.py
```

The preflight JSON records the C1→C4 seed 42 conv1 input check.
`seed_metrics.csv`, `summary.csv`, and `deltas.csv` contain all numerical results.
`protocol_audit.csv` records the 60 configuration and checkpoint comparisons.
Each experiment is under `<norm_method>/<source>_to_<target>/seed_<seed>/<method>/`.
Its config records source fitted parameters, source and target model input ranges,
and the fraction of target values outside [0,1]. The Min–Max target input is not clipped.
The compared configurations, initial weights, source batch orders, training cuts,
and evaluation cuts match; the final checkpoint hashes differ across normalizations.
