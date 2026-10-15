# gnn_aml_core: graph neural networks for money-laundering detection

GATv2 and RGCN node classifiers (PyTorch Geometric) trained on graphs from `aml_synth`, a FastAPI service that scores and
explains accounts, and the baselines the GNN has to beat.

```bash
python -m aml_synth.graph_generator --out data --to csv
python -m gnn_aml_core.train --data data --model gatv2 --epochs 150 --out models     # also: --model rgcn, --pos-weight 19.0
```
`train` prints a test-set table (GNN and baselines, same split, threshold chosen on validation) and writes
`models/metrics.json`, `models/gnn_model.pt`, `models/graph.pt`. The API (`gnn_aml_core.main`) serves `/predict`,
`/explain` (GNNExplainer) and `/model/info`.

## Design
* **Bidirectional graph.** Every transaction also gets a reversed copy flagged as reversed, so an account hears from the
  accounts it sends money to (a mule paying a central account could otherwise not see it). The direction flag and the relation
  id keep the direction of the money.
* **27 node features** (degrees, amounts, cross-border and near-threshold shares, burstiness, age, KYC risk, account type) and
  5 edge features. The brief mentions 400+ features; extending `features.py` is the intended way to grow this.
* **Class-weighted BCE.** Default weight #negatives / #positives; `--pos-weight 19.0` reproduces the brief's fixed weight.
* **Honest evaluation.** Stratified 60/20/20 split; the decision threshold (best F1) is chosen on validation and reported on test.

## Baselines (`baselines.py`)
Logistic Regression, XGBoost (falls back to scikit-learn's histogram boosting if `xgboost` is not installed, and says so),
Isolation Forest, and a strong extra baseline: boosting on each account's own features plus its 1- and 2-hop neighbourhood averages.
That last one is the fair competitor for a graph model; a GNN that cannot beat it has not shown that message passing helps.

## Metrics reported
AUC-ROC, PR-AUC, precision, recall, F1 and false-positive rate on the test split.
