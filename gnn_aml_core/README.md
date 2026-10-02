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

## A second real-data check, the right domain this time: SynthAML (`synthaml.py`)

Elliptic is real, but it is crypto, not bank wire transfers -- a different domain from what this whole project
targets. [SynthAML](https://doi.org/10.1038/s41597-023-02569-2) (Jensen et al., 2023, *Nature Scientific Data*)
closes that specific gap: it is built directly from a real Danish bank's (Spar Nord) actual transaction and AML
alert data, in the right domain (card, cash, international and wire activity), with a peer-reviewed claim that
performance on it transfers to the real world.

**Read this before using it: SynthAML has no graph structure at all.** Each row is one client's transaction
*history* leading up to an alert, not a network of transactions between accounts -- there is no src/dst, and
`fit_gnn` (GATv2/RGCN) cannot be run on it. What it genuinely gives us is a real, bank-derived tabular benchmark
for our baseline classifiers specifically, directly comparable to the paper's own reported numbers.

```bash
python -m gnn_aml_core.synthaml --data data/synthaml
```

Download SynthAML's two open-access CSV files from https://doi.org/10.6084/m9.figshare.c.6504421.v1 and place
them in that folder. The loader replicates the paper's own feature engineering exactly (56 hand-engineered
summary statistics per alert) and respects its quarter-only date accuracy (any split must fall on a quarter
boundary, enforced rather than silently accepted).

## The real yardstick for this specific niche: IBM AMLworld (`amlworld.py`)

Elliptic validates against real transactions; SynthAML and company_fraud check tabular generalisation -- but
none of them is the benchmark this project's own specific niche (GNN-based AML detection) is actually judged
against in the published literature. IBM's AMLworld (Altman et al., 2023) is: real papers report real F1 scores
on it, ranging roughly 0.03 to 0.76 depending on model and variant -- a genuinely modest state of the art, not
a wall this project is far behind.

```bash
python -m gnn_aml_core.amlworld --data data/amlworld --file LI-Small_Trans.csv
```

Download the LI-Small (or any other) variant's `_Trans.csv` from
https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml. LI = low illicit
ratio, the realistic setting every published paper found hardest; HI variants are easier.

**A real, stated caveat, not glossed over**: this dataset's real label is per TRANSACTION (edge
classification -- exactly what the published F1 scores above measure). This loader aggregates it to per
ACCOUNT to reuse this project's own account-classification pipeline unchanged, which is a genuinely easier
task than the papers' own setup. Any F1 this produces is directionally informative against the numbers above,
not a strict apples-to-apples ranking.



An externally-sourced dataset (5 CSV files: companies, transactions, event order, time-series window ids, and
fraud labels) -- **its original source or a citable paper could not be identified**, so unlike Elliptic and
SynthAML above, this is not represented as a validated, peer-reviewed benchmark. It is included purely as
another tabular sanity check for the baseline classifiers.

**No graph structure here either**, and the task shape is less obvious than it first looks: the fraud label is
not per-transaction, it is per *(company, time window)* -- confirmed by direct inspection of the real files
before writing this loader, not assumed. A transaction can belong to more than one window (a real, confirmed
many-to-many mapping, consistent with overlapping windows), and the loader aggregates each window's
transactions (mean/std/min/max/count) before joining in that window's company's own static features.

```bash
python -m gnn_aml_core.company_fraud --data data/company_fraud
```

Place `companies_{train,test}.csv`, `transactions_{train,test}.csv`, `event_order_{train,test}.csv`,
`time_series_ids_{train,test}.csv`, and `fraud_labels_{train,test}.csv` in that folder.

## Does detection degrade if launderers adapt? (`aml_synth/adversarial.py`)

The feedback loop below assumes retraining helps once launderers change behaviour in response to being
caught -- this module actually tests that assumption instead of just asserting it. It trains a baseline on one
generation of synthetic data, measures which features it relies on most (the model's own coefficients, not an
external guess), generates a second generation with `camouflage` raised specifically for the typologies those
features implicate, and compares the SAME unretrained model's accuracy on both generations -- the drop is a
real, quantified measure of adaptation pressure, not an assumption.

```bash
python -m aml_synth.adversarial --accounts 5000 --seed 1
```

**Honest finding from actually running this**: `camouflage` (currently the only real "blend in more" knob this
generator exposes) does not reliably move detection difficulty here, even at a 5x increase -- consistent with
what difficulty-calibration work earlier this session already found. The simulation mechanism itself works
correctly; what it reveals is that a genuinely strong adaptation test would need the typology-specific
imperfection parameters (retained fraction, timing spread) exposed as configurable knobs too, which they
currently are not. A real, scoped follow-up, not attempted here.

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
