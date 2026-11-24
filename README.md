# XAI-AMLBench

**Catch money laundering that hides in the shape of a bank's payments, and get a plain-English reason for every alert.**

[![CI](https://github.com/gumption-ariho/XAI-AMLBench/actions/workflows/ci.yml/badge.svg)](https://github.com/gumption-ariho/XAI-AMLBench/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Dataset: CC BY 4.0](https://img.shields.io/badge/Dataset-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Next.js](https://img.shields.io/badge/Console-Next.js%20%2B%20FastAPI-black)
![Detection AUC-ROC](https://img.shields.io/badge/AUC--ROC-0.98%20on%20the%20benchmark-brightgreen)

If this project is useful or interesting to you, **please press the Star button at the top of the page**. It is the
easiest way to help other people find it.

---

## What is this?

Banks must watch for money laundering: criminals who move dirty money through many accounts so that no single payment
looks odd, but the whole chain is plainly deliberate. Most tools look at one account at a time and miss the chain.

XAI-AMLBench looks at the **whole network of payments**. It learns the shapes that laundering makes (money split up,
fanned out, passed round a circle, run through shell companies) and flags the accounts involved. Then it explains
itself: every alert comes with a written reason a compliance officer can read, check and keep on record.

It is two things in one repository:

1. **An open benchmark and library.** A generator that builds realistic transaction networks with laundering hidden
   inside, graph neural network detectors, and baselines to compare against. No real customer data is needed.
2. **A working platform for many banks.** A web console, a server, two detection models, an owner's control room and a
   tamper-proof audit trail, so you can see the whole idea running, not just the maths.

## Results

Average of five independent runs on generated data, with the laundering hidden among honest look-alikes (large
merchants, payroll runs, savings groups). Higher is better.

| Model (5,000 accounts) | AUC-ROC | PR-AUC | Precision | Recall | F1 |
|---|---|---|---|---|---|
| **Graph neural network (GATv2)** | **0.981 ± 0.007** | **0.903** | 0.871 | 0.819 | 0.840 |
| Boosting + neighbour averages | 0.959 ± 0.013 | 0.817 | 0.862 | 0.661 | 0.747 |
| Boosting, account features only | 0.882 ± 0.034 | 0.596 | 0.662 | 0.477 | 0.550 |
| Logistic regression | 0.869 ± 0.034 | 0.488 | 0.493 | 0.480 | 0.476 |
| Isolation Forest (unsupervised) | 0.701 ± 0.038 | 0.195 | 0.210 | 0.325 | 0.248 |

| Model (20,000 accounts) | AUC-ROC | PR-AUC | Precision | Recall | F1 |
|---|---|---|---|---|---|
| **Graph neural network (GATv2)** | **0.979 ± 0.008** | **0.905** | 0.884 | 0.821 | 0.851 |
| Boosting + neighbour averages | 0.964 ± 0.009 | 0.840 | 0.799 | 0.738 | 0.767 |
| Boosting, account features only | 0.898 ± 0.012 | 0.632 | 0.583 | 0.560 | 0.564 |
| Logistic regression | 0.861 ± 0.014 | 0.486 | 0.484 | 0.508 | 0.493 |
| Isolation Forest (unsupervised) | 0.677 ± 0.023 | 0.167 | 0.190 | 0.361 | 0.244 |

**What this shows.** The graph model came first in every run, at both sizes (the brief asked for an AUC-ROC of 0.87).
It beats the model that sees each account alone by about 0.10 AUC, which is the clearest sign that the *network
shape* carries real information, not just more columns. Its lead over the strongest graph-aware baseline narrows as
the data grows (+0.022 AUC at 5,000 accounts, +0.015 at 20,000), so quote both tables, not just one.

**What it does not show yet.** These are generated networks built from known laundering patterns. Results on a real
bank's own transactions are the job of the pilot stage (see the roadmap below). Reproduce either table with
`python -m gnn_aml_core.benchmark --seeds 5` (add `--accounts 20000` for the larger one).

## What the platform can do

**For a bank's analysts**
- Scan any account, or load a file of transactions and score them all at once.
- Review alerts with a written explanation that names the accounts, amounts and dates involved. Each explanation
  carries a fingerprint (SHA-256), so any later change to the text is detectable.
- Confirm or dismiss each alert with a note. Confirmed cases teach the main model.
- See the network of accounts around any flagged account.
- Get a recommendation for every anomaly (likely normal, likely suspicious, or unsure) with a short reason, and clear
  the obviously normal ones in one click.

**Two models that check each other**
- **Model 1** is a graph attention network (GATv2). It learns from labelled laundering examples.
- **Model 2** is a graph autoencoder trained once on a large synthetic population of *normal* accounts (about thirty
  kinds of ordinary behaviour), then frozen. It flags accounts that Model 1 calls normal but that look unlike
  ordinary accounts. Because it never trains on live accounts, a hidden bad account cannot teach it that bad is normal.

**For many banks at once**
- Banks register themselves and wait for approval. Analysts register and are approved one by one.
- Each bank sees only its own accounts, alerts and records. One shared engine serves everyone.
- Optional two-step sign-in, account lock-out after repeated wrong passwords, invitation and reset links.

**For the platform owner**
- A governing console that shows every bank side by side: open and overdue alerts, decision times, reminders.
- Autopilot reminders: if a bank leaves an alert untouched past a grace period, it gets a reminder automatically.
- The owner's pages are hidden from public tunnel traffic, with a private unlock link for the owner's own phone.

**Built to be audited**
- Every model result, alert and officer decision is written to an append-only ledger (immudb) that can be verified
  but not altered.
- The identity check keeps only a fingerprint and the last four characters of an ID number, never the number itself.

## How it fits together

```
 transactions ──► graph of accounts ──► Model 1 (GATv2) ─┐
 (file, paste,        │                                   ├─► alert ──► explanation ──► analyst decides
  or live feed)       └─► Model 2 (normal-only reference)─┘        │                          │
                                                                    └──────► audit ledger ◄────┘
```

Fourteen containers, each with one job: a gateway (Traefik), a message stream (Kafka), the website (Next.js), the
server (FastAPI), the detection service, the explanation service, Postgres, Redis, Neo4j, immudb, Prometheus,
Grafana, a data generator and a background worker. One part can be restarted without touching the rest.

## Quick start

### A. Use it as a library (no Docker)

```bash
git clone https://github.com/gumption-ariho/XAI-AMLBench.git
cd XAI-AMLBench
python3 -m venv .venv && source .venv/bin/activate
pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu   # CPU build, no GPU needed
pip install -e ".[gnn,xai]"
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
The full walk-through, with baselines and a written explanation, is in
[`notebooks/quickstart.ipynb`](notebooks/quickstart.ipynb).

### B. Run the whole platform on your computer (Docker)

You need Docker with Compose, about 8 GB of free memory and 15 GB of free disk.

```bash
cp .env.example .env          # then open .env and replace every "change_me" value
docker compose --profile app --profile ml --profile obs up -d --build
```

The first start downloads images and builds the second model's reference, so allow 10 to 20 minutes. When it is up,
open **https://xai.localhost** (your browser will warn about the local certificate; that is expected on a test
address).

| Page | Address |
|---|---|
| Bank and analyst console | `https://xai.localhost` |
| A bank asks to join | `/request-access` |
| An analyst registers | `/analyst-signup` |
| Sign in | `/signin` |
| Platform owner (use the `ADMIN_PASSKEY` from your `.env`) | `/owner` |
| Governing console (all banks) | `/?view=governance` |
| Model API docs | `/gnn/docs` and `/xai/docs` |
| Dashboards | `/grafana` |

**A first walk-through:** request access as a made-up bank, approve it at `/owner`, sign in with the passkey it
gives you, register an analyst and approve them, then open the **Intake** tab and load a few transactions.

**Written explanations need a graphics card.** The explanation service runs a language model and requires an NVIDIA
GPU. Without one everything else still works (scanning, alerts, anomalies, the owner tools), but the "why was this
flagged" text is unavailable. On a GPU machine, add `--profile llm`.

### C. Try it with sample data

```bash
python tester-pack/sample-data/make_sample.py        # makes sample_transactions.csv (about 4,000 transactions)
```
Paste or upload `sample_transactions.csv` into the **Intake** tab. It looks like bank activity: salaries, merchants,
cash-heavy shops, savings groups and remittances, with laundering patterns hidden inside. All of it is synthetic.

### D. Just look at the website

```bash
cd frontend && npm install && npm run dev:demo      # needs Node 18.18+, then open http://localhost:3000
```

## Test it with friends

Want to try it and report back? [`TESTING.md`](TESTING.md) explains how. A recording browser measures the system
silently while you use it (speed, accessibility, security, data-protection basics) and builds a report for you to
email. The tool lives in [`tester-pack/recorder`](tester-pack/recorder).

## Project layout

| Folder | What is in it |
|---|---|
| `aml_synth/` | The transaction-network generator with seven laundering patterns and honest look-alikes |
| `gnn_aml_core/` | The detection models, training, benchmark, anomaly model, live graph and feedback loop |
| `xai_explainer/` | Turns a model decision into a written explanation with traceable evidence |
| `backend/` | The FastAPI server: sign-in, banks, alerts, owner and governance routes |
| `frontend/` | The Next.js console |
| `node_agent/` | Background agent for live feeds |
| `grafana/`, `prometheus/`, `traefik/` | Monitoring and the gateway |
| `tests/` | Hundreds of automated tests |
| `tester-pack/` | The test recorder and sample data |
| `notebooks/` | A step-by-step notebook |

Run the tests with `pytest`.

## Roadmap

| Version | Stage |
|---|---|
| **0.1** | Friends test on a laptop. **Done.** |
| **1.0** | Pilot: a rented server, a real domain name, one or two real banks with their own data |
| **1.5** | Written explanations on a rented GPU, reminder emails, bank branding, exportable reports |
| **2.0** | A weekly step that learns new ordinary behaviour across banks, monitoring that alerts the owner, a larger server |

## Contributing

Bug reports, ideas and pull requests are welcome. Start with [`CONTRIBUTING.md`](CONTRIBUTING.md). If you have
laundering patterns you would like added to the generator, open an issue and describe them in plain words.

## Citation

```bibtex
@software{xai_amlbench_2026,
  title   = {XAI-AMLBench: An Explainable, Open-Source Benchmark for Multi-Hop Money-Laundering Detection},
  author  = {{XAI-AMLBench contributors}},
  year    = {2026},
  version = {0.1.0},
  url     = {https://github.com/gumption-ariho/XAI-AMLBench},
  license = {MIT}
}
```
See [`CITATION.cff`](CITATION.cff) for the citation-file format.

## License

Code: [MIT](LICENSE). The generated dataset: CC BY 4.0.
