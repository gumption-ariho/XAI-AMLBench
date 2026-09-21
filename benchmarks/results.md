Mean +- std over 5 seeds (5,000 accounts, camouflage 1.5). Threshold chosen on validation, metrics on the test split.

| Model | AUC-ROC | PR-AUC | Precision | Recall | F1 | FPR |
|---|---|---|---|---|---|---|
| **GNN (GATV2)** | 0.981 +- 0.007 | 0.903 +- 0.022 | 0.871 +- 0.085 | 0.819 +- 0.050 | 0.840 +- 0.034 | 0.011 +- 0.008 |
| Logistic Regression | 0.869 +- 0.034 | 0.488 +- 0.114 | 0.493 +- 0.070 | 0.480 +- 0.150 | 0.476 +- 0.103 | 0.040 +- 0.012 |
| HistGradientBoosting (XGBoost not installed) | 0.882 +- 0.034 | 0.596 +- 0.080 | 0.662 +- 0.124 | 0.477 +- 0.066 | 0.550 +- 0.072 | 0.021 +- 0.009 |
| Isolation Forest (unsupervised score) | 0.701 +- 0.038 | 0.195 +- 0.028 | 0.211 +- 0.021 | 0.325 +- 0.100 | 0.248 +- 0.029 | 0.100 +- 0.031 |
| HistGradientBoosting (XGBoost not installed) + neighbour averages (strong, uses the graph) | 0.959 +- 0.013 | 0.817 +- 0.047 | 0.862 +- 0.053 | 0.661 +- 0.069 | 0.747 +- 0.056 | 0.009 +- 0.004 |

GNN (GATV2) minus strongest baseline (HistGradientBoosting (XGBoost not installed) + neighbour averages (strong, uses the graph)), paired over seeds:
- AUC-ROC: +0.022 +- 0.009, better in 5 of 5 seeds
- PR-AUC: +0.085 +- 0.032, better in 5 of 5 seeds
- F1: +0.093 +- 0.033, better in 5 of 5 seeds
- all five brief targets (AUC>=0.87, precision>=0.89, recall>=0.82, F1>=0.85, FPR<=0.02) met in 1 of 5 seeds
