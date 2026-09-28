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

## Real-world validation: the Elliptic Bitcoin dataset (`elliptic.py`)

Everything above trains and evaluates only on `aml_synth`'s synthetic graphs -- necessarily so, since real bank
transaction data is confidential and cannot be shared for a public benchmark. That means nothing so far
demonstrates real-world detection performance, only that the architecture beats simpler models on data we
generated ourselves. `elliptic.py` closes part of that gap using the [Elliptic dataset](
https://www.kaggle.com/datasets/ellipticco/elliptic-data-set) (Weber et al., 2019, KDD): 203,769 real Bitcoin
transactions with labels derived from actual law-enforcement-linked entities (ransomware, darknet markets,
scams, Ponzi schemes), not another round of synthetic tuning.

**Download the three raw files yourself first** (this module never fetches anything over the network) from the
Kaggle link above, and place them in one folder, unrenamed:
`elliptic_txs_features.csv`, `elliptic_txs_classes.csv`, `elliptic_txs_edgelist.csv`.

```bash
python -m gnn_aml_core.elliptic --data data/elliptic                     # GATv2, temporal split (matches the paper)
python -m gnn_aml_core.elliptic --data data/elliptic --split random --seed 1
```

The same `fit_gnn`, `evaluate_baselines` and calibration code built for `aml_synth` runs unchanged -- only the
data loader and edge features differ, which is itself a real test of how reusable the core pipeline is. Two
structural notes worth knowing: in `aml_synth`, nodes are accounts and edges are transactions; in Elliptic,
nodes are transactions themselves and edges are fund-flow links between them, so edges carry no "amount" of
their own -- the edge feature here is how many time steps apart two linked transactions are, a real, honestly
labelled structural signal rather than a repurposed amount field. And only about 23% of transactions carry a
real label (the rest are "unknown"); training and evaluation use the labelled subset only, with unlabelled
nodes still present in the graph for message-passing context -- the standard way this dataset is used in
published research.

**What this validates, and what it does not.** A result here is a genuinely different kind of evidence than the
synthetic benchmark: real transactions, real law-enforcement-derived labels, no tuning of "how hard the data is"
on our part. It is not equivalent to a pilot on a real bank's own wire-transfer data -- Bitcoin and bank
transfers are different domains, and roughly three-quarters of Elliptic's transactions have no label at all
(not "confirmed clean", just unknown). Report both benchmarks together, not the Elliptic one alone, and say so.

## Keeping the model current: officer feedback (`feedback.py`)

A model trained once and never revisited goes stale, because real launderers adapt once they learn what gets
flagged. `feedback.py` closes that loop using the officer decisions the backend already records (every
`/alerts/{id}/decision` call -- confirmed or dismissed -- is a real, ground-truth-verified label, specific to
this institution's actual accounts, not borrowed from a different bank or a different domain):

```bash
python -m gnn_aml_core.feedback --check                      # report the current drift only, change nothing
python -m gnn_aml_core.feedback --data data --out models     # retrain if the drift trigger has been met
```

**The drift trigger, not a blind schedule.** Retraining on a fixed timetable (every week, regardless) cannot
tell you whether the model actually needs it -- it could go stale in two weeks or stay accurate for six months.
Instead, `disagreement_rate` measures, from real officer decisions, how often the model's original
flagged/not-flagged call turned out to be wrong; `should_retrain` only fires once both a minimum sample of
decisions has accumulated (default 20) and the disagreement rate crosses a threshold (default 15%). A real,
honest limitation worth naming: this only sees accounts that reached an officer in the first place (flagged, or
force-reviewed) -- it detects the model becoming too trigger-happy (more false positives) far more reliably
than it detects it becoming too lax (false negatives it never surfaced for anyone to catch).

**When it does retrain**, `retrain_with_feedback` blends the real, officer-verified examples into the next
training run -- overwriting whatever label (synthetic, SynthAML-derived, or none) the account previously had
with what the officer actually confirmed, and repeating each one (`--feedback-weight`, default 5x) so a small
number of hard-won real decisions is not drowned out by a much larger volume of warm-start training data. This
reuses `gnn_aml_core.train.fit_gnn` unchanged -- the same training loop used everywhere else in this project.

## Multi-seed benchmark (what a paper needs)
```bash
python -m gnn_aml_core.benchmark --seeds 5                        # fresh graph per seed; GNN + all baselines on identical splits
python -m gnn_aml_core.benchmark --seeds 5 --models gatv2,rgcn    # both architectures
python -m gnn_aml_core.benchmark --seeds 3 --no-gnn               # baselines only, seconds, no PyTorch needed
python -m gnn_aml_core.benchmark --seeds 5 --camouflage 3.0       # another difficulty level
```
Writes `benchmarks/results.md` (paste-ready table, mean +- std), `benchmarks/results.json` (every number of every seed), the
paired GNN-minus-strongest-baseline differences, and in how many seeds all of the brief's targets were met at once. Model
selection uses the validation split only; early stopping (`--patience`) keeps each run to a few minutes.
