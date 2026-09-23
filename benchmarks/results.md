Mean +- std over 5 seeds (20,000 accounts, camouflage 1.5). Threshold chosen on validation, metrics on the test split.

| Model | AUC-ROC | PR-AUC | Precision | Recall | F1 | FPR |
|---|---|---|---|---|---|---|
| **GNN (GATV2)** | 0.979 +- 0.008 | 0.905 +- 0.026 | 0.884 +- 0.025 | 0.821 +- 0.042 | 0.851 +- 0.031 | 0.009 +- 0.002 |
| Logistic Regression | 0.861 +- 0.014 | 0.486 +- 0.028 | 0.484 +- 0.040 | 0.508 +- 0.050 | 0.493 +- 0.021 | 0.047 +- 0.012 |
| HistGradientBoosting (XGBoost not installed) | 0.898 +- 0.012 | 0.632 +- 0.021 | 0.583 +- 0.086 | 0.560 +- 0.051 | 0.564 +- 0.028 | 0.036 +- 0.016 |
| Isolation Forest (unsupervised score) | 0.677 +- 0.023 | 0.167 +- 0.007 | 0.190 +- 0.025 | 0.361 +- 0.063 | 0.244 +- 0.010 | 0.134 +- 0.044 |
| HistGradientBoosting (XGBoost not installed) + neighbour averages (strong, uses the graph) | 0.964 +- 0.009 | 0.840 +- 0.032 | 0.799 +- 0.036 | 0.738 +- 0.041 | 0.767 +- 0.034 | 0.016 +- 0.003 |

GNN (GATV2) minus strongest baseline (HistGradientBoosting (XGBoost not installed) + neighbour averages (strong, uses the graph)), paired over seeds:
- AUC-ROC: +0.015 +- 0.003, better in 5 of 5 seeds
- PR-AUC: +0.064 +- 0.020, better in 5 of 5 seeds
- F1: +0.084 +- 0.022, better in 5 of 5 seeds
- all five brief targets (AUC>=0.87, precision>=0.89, recall>=0.82, F1>=0.85, FPR<=0.02) met in 2 of 5 seeds
