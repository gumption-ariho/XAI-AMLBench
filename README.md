# XAI-AMLBench

[![PyPI](https://img.shields.io/pypi/v/xai-amlbench.svg)](https://pypi.org/project/xai-amlbench/)
[![CI](https://github.com/OWNER/xai-amlbench/actions/workflows/ci.yml/badge.svg)](https://github.com/OWNER/xai-amlbench/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Dataset: CC BY 4.0](https://img.shields.io/badge/Dataset-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.PLACEHOLDER.svg)](https://doi.org/10.5281/zenodo.PLACEHOLDER)
[![Hugging Face Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-blue)](https://huggingface.co/datasets/OWNER/xai-amlbench)

*(Badges are placeholders until the project is published: `OWNER` needs the real GitHub org/user, and the Zenodo
DOI is minted on first release. See [.zenodo.json](.zenodo.json) and [CITATION.cff](CITATION.cff).)*

An explainable, open-source benchmark for **multi-hop money-laundering detection**: a synthetic transaction-graph
generator with five embedded laundering typologies, GATv2/RGCN graph neural network detectors, and an
explainable-AI layer that turns model decisions into audit-ready Suspicious Activity Report (SAR) narratives --
with every number, date and account id in the narrative traceable back to the underlying evidence.

## Why this exists

FinCEN's April 2026 proposed AML/CFT rule requires financial institutions to document a human-reviewable
explanation for every SAR filed and moves toward outcome-based evaluation of AML programs -- in effect, ruling
out black-box models used without an audit trail. Proprietary AML platforms that meet this bar cost regional
banks and credit unions $1M-$10M a year, putting them out of reach for most of the roughly 9,400 US institutions
below the largest tier. There was also no open, shared, privacy-preserving benchmark for multi-hop laundering
detection that academic research could reproduce against. XAI-AMLBench is a fully open attempt to close both
gaps: a synthetic dataset (so no institution's real customer data is needed to benchmark against), reference
detectors, and an explanation layer built to satisfy the same audit-trail requirement the rule imposes.

## Quick start

```bash
pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu   # CPU build, ~190 MB, no GPU needed
pip install xai-amlbench[gnn,xai]
```
```python
from aml_synth import AMLGraphGenerator, GeneratorConfig
from gnn_aml_core.train import arrays_from_frames, make_splits, fit_gnn

graph = AMLGraphGenerator(GeneratorConfig(n_accounts=5000, seed=1)).generate()
arrays = arrays_from_frames(graph.accounts, graph.transactions)
tr, va, te = make_splits(arrays["y"], seed=1)
result = fit_gnn(arrays, tr, va, te, model_name="gatv2", epochs=100, patience=6, seed=1)
print(result["report"])   # AUC-ROC, PR-AUC, precision, recall, F1, false-positive rate
```
See [`notebooks/quickstart.ipynb`](notebooks/quickstart.ipynb) for the full walk-through, including baselines and
generating a SAR narrative. Module-level usage: [`aml_synth/README.md`](aml_synth/README.md),
[`gnn_aml_core/README.md`](gnn_aml_core/README.md), [`xai_explainer/README.md`](xai_explainer/README.md).

## Benchmark results

Mean +/- std over 5 seeds, 5,000-account synthetic graphs, threshold chosen on validation, metrics on the held-out
test split (see [`gnn_aml_core/benchmark.py`](gnn_aml_core/benchmark.py) to reproduce):

| Model | AUC-ROC | PR-AUC | Precision | Recall | F1 |
|---|---|---|---|---|---|
| **GNN (GATv2)** | **0.981 +/- 0.007** | **0.903** | 0.871 | 0.819 | 0.840 |
| Boosting + 1-/2-hop neighbour averages (strong, graph-aware) | 0.959 +/- 0.013 | 0.817 | 0.862 | 0.661 | 0.747 |
| Boosting, per-account features only | 0.882 +/- 0.034 | 0.596 | 0.662 | 0.477 | 0.550 |
| Logistic Regression | 0.869 +/- 0.034 | 0.488 | 0.493 | 0.480 | 0.476 |
| Isolation Forest (unsupervised) | 0.701 +/- 0.038 | 0.195 | 0.210 | 0.325 | 0.248 |

The GNN beat the strongest baseline in all 5 seeds (paired AUC-ROC +0.022, PR-AUC +0.085, F1 +0.093), and beat
per-account-only boosting by roughly 0.10 AUC and 0.29 F1 -- the clearest evidence that the graph itself, not just
more features, carries the signal. On average the GNN meets the project's AUC-ROC and false-positive-rate targets;
precision, recall and F1 fall just inside one standard deviation of their targets rather than clearing them on
every seed (all five targets were met simultaneously in 1 of 5 seeds). Run
`python -m gnn_aml_core.benchmark --seeds 5` to reproduce or extend this table; results vary by about +/-0.03 AUC
between seeds at this graph size, so always report a mean over several seeds rather than a single run.

**A note on difficulty.** The synthetic data is deliberately hard: laundering accounts carry ordinary background
traffic, about 40% of smurfing "mules" are recruited ordinary accounts, and realistic look-alike structures
(large merchants, payroll, savings groups, escrow chains) share surface features with real typologies. A
benchmark that a linear model already solves cannot demonstrate that a graph model adds anything -- see
`aml_synth/README.md` for the measured baseline difficulty this was tuned against.

## Citation

```bibtex
@software{xai_amlbench_2026,
  title        = {XAI-AMLBench: An Explainable, Open-Source Benchmark for Multi-Hop Money-Laundering Detection},
  author       = {{XAI-AMLBench contributors}},
  year         = {2026},
  version      = {0.1.0},
  url          = {https://github.com/OWNER/xai-amlbench},
  license      = {MIT}
}
```
See [CITATION.cff](CITATION.cff) for the citation-file format version, and [`.zenodo.json`](.zenodo.json) for the
dataset's own CC-BY-4.0 citation once a DOI has been minted.

## Full application (compliance-officer console)

The sections below run the whole system: a Next.js console, a FastAPI backend, the trained GNN as a scoring
service, and a narrative service, wired together through one gateway -- not just the library.

> **See the frontend only (no Docker, no backend):**
> * (needs Node 18.18+) `cd frontend && npm install && npm run dev:demo` -> http://localhost:3000 (the real Next.js app on built-in sample data)
> * or just open `frontend/preview/demo.html` in a browser (needs internet once, for React from a CDN)

> **Python packages for local work:** one file, `requirements.txt` in the project root (it includes the four per-service
> files). Install everything fast with `bash install_all_requirements_v2.sh`. The per-service files stay because each Docker
> image installs only its own service's packages.

## Run everything on this laptop (no Docker image builds, no downloads)
Best on a slow connection: the infrastructure stays in the Docker containers you already have, and the four application services
run straight from `.venv` and `frontend/node_modules`.
```bash
.venv/bin/python -m gnn_aml_core.train --data data --out models        # once: the model service loads ./models
docker compose -f docker-compose.yml -f docker-compose.local.yml up -d database redis-feature-cache immudb-audit-ledger
python3 run_local.py                                                    # model :8001, narratives :8002, backend :8000, website :3000
python3 check_stack_v1.py --base http://localhost:3000 --gnn http://127.0.0.1:8001 --xai http://127.0.0.1:8002
python3 check_latency_v1.py --gnn http://127.0.0.1:8001                # cold/warm p50/p95/p99 vs the 100ms target
```
Open http://localhost:3000. Ctrl+C stops everything; `python3 run_local.py --stop` cleans up leftovers.

## Or build the Docker images (needs a fast connection: about 2 GB of downloads)
```bash
docker compose build backend && docker compose build xai-narrative-lite && docker compose build gnn-detection-api && docker compose build frontend
docker compose up -d gnn-detection-api backend xai-narrative-lite frontend node_agent
python3 check_stack_v1.py
```
(GPU machine with LLM narratives: use `--profile llm` instead of the lite service, not both.)

## 1. Start the infrastructure
```bash
cp .env.example .env            # edit every change_me value
docker compose config           # validate
docker compose up -d            # traefik, kafka, postgres, redis, neo4j, immudb
```

## 2. Generate data and train the model (in Docker, no local Python needed)
```bash
docker compose --profile ml build
docker compose --profile ml run --rm aml-synth-worker \
    python -m aml_synth.graph_generator --out data --to csv,kafka,neo4j
docker compose --profile ml run --rm gnn-detection-api \
    python -m gnn_aml_core.train --data /app/data --out /app/models --model gatv2
```
(`./data` and `./models` on your machine are mounted into the containers. Add
`- ./data:/app/data` under gnn-detection-api volumes if it is not there yet.)

## 3. Run everything
```bash
docker compose --profile app --profile ml --profile obs up -d --build
# add --profile llm once the GPU / model download is sorted
```

| What                | URL                          |
|---------------------|------------------------------|
| Compliance console  | http://localhost (or :3000)  |
| GNN API docs        | http://localhost/gnn/docs    |
| XAI API docs        | http://localhost/xai/docs    |
| Grafana             | http://localhost/grafana     |
| Neo4j Browser       | http://localhost:7474        |
| immudb console      | http://localhost:8080        |
| Traefik dashboard   | http://localhost:8081        |

## Local (no Docker) quick test of the data + model
```bash
pip install numpy pandas scikit-learn torch torch_geometric
python -m aml_synth.graph_generator --out data --to csv
python -m gnn_aml_core.train --data data --out models
```

## Frontend (TypeScript)
```bash
cd frontend
npm install
npm run typecheck     # tsc --noEmit, strict mode
npm run dev           # http://localhost:3000  (needs the backend for live data)
```
Shared API types live in `frontend/src/types.ts` and mirror the backend JSON.
