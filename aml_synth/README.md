# aml_synth: synthetic multi-hop money-laundering graphs

Generates privacy-preserving, directed transaction graphs with five embedded laundering typologies, for benchmarking
AML detection without confidential customer data. Everything is synthetic and reproducible from a seed.

```bash
python -m aml_synth.graph_generator --accounts 5000 --to csv                 # accounts.csv + transactions.csv
python -m aml_synth.graph_generator --accounts 50000 --to parquet,json,cypher --out data
```
```python
from aml_synth import AMLGraphGenerator, GeneratorConfig
graph = AMLGraphGenerator(GeneratorConfig(n_accounts=20_000, days=60, seed=7)).generate()
graph.accounts, graph.transactions      # pandas DataFrames
G = graph.to_networkx()                 # directed multigraph (parallel edges allowed)
print(graph.summary())
```

## Output
| Format | Files | Notes |
|---|---|---|
| CSV | `accounts.csv`, `transactions.csv` | spreadsheet friendly |
| Parquet | `accounts.parquet`, `transactions.parquet` | primary format; needs `pyarrow` |
| JSON | `graph_summary.json` | metadata, schema, summary statistics |
| Cypher | `import.cypher` | `cypher-shell -f import.cypher` |
| Kafka / Neo4j | stream / direct load | used by the Docker stack |

Node attributes: `account_type`, `country`, `opened_ts`, `risk_score` (a weak KYC prior, not the label), `is_suspicious`, `typology`.
Edge attributes: `amount` (USD), `timestamp`, `payment_format`, `cross_border`, `is_laundering`, `typology`, `pattern_id`.

## Typologies
`smurfing` (1,000-2,000 small deposits from a crowd of mules into one account, then rapid disbursement), `scatter_gather`,
`cyclic_loop`, `shell_company` (chain of young, often offshore accounts), `cross_border_velocity`.

## Why this benchmark is not trivial
A benchmark that a linear model solves cannot show that a graph model helps. So the generator adds:
* **Benign look-alikes:** merchants and marketplaces with thousands of payers, payroll, subscriptions, cash-heavy shops, rotating savings
  groups (chamas), escrow chains and cash pooling that move large sums, remittance senders, cross-border traders, start-ups.
* **Camouflage:** laundering accounts carry as much ordinary traffic as an ordinary account, and about 40% of smurfing mules
  are recruited ordinary accounts, many of which deposit only once or twice. Their own statistics look normal; their **neighbourhood** does not.
* **Relative amounts, regional rings, ordinary ages, short-lived accounts** so single-column giveaways are removed.

Measured on 5,000-account graphs (generator seeds 1-8, 27 per-account features, held-out split), mean +- std:

| Model | AUC-ROC | Brief expects |
|---|---|---|
| Logistic Regression | 0.84 +- 0.03 | ~0.75 |
| Gradient boosting (XGBoost stand-in) | 0.86 +- 0.03 | ~0.82 |
| Isolation Forest | 0.68 | ~0.65 |
| Gradient boosting + 1- and 2-hop neighbourhood means | 0.95 | (proxy for message passing; **not** a GNN result) |

Results vary by about +-0.03 between seeds, so **report means over several seeds**. The last row only shows that the graph adds
information; GNN results come from `gnn_aml_core`.

## Limits
Synthetic data simplifies real behaviour; labels are the ground truth of injected patterns (real SAR-derived labels are noisier).
Use at least 1,000 accounts for benchmarking (tiny graphs are dominated by the minimum pattern counts). Timestamps of every
pattern are guaranteed to fall inside the requested window. Code: MIT. Data: CC-BY-4.0.
