"""Multi-seed benchmark: the table a paper needs.

For every seed it generates a fresh synthetic graph, trains the GNN(s) and every baseline on IDENTICAL splits, and
reports mean +- std over seeds, the paired difference between the GNN and the strongest baseline, and in how many seeds
all of the brief's targets were met at the same time.

    python -m gnn_aml_core.benchmark --seeds 5                         # GATv2, 5 seeds (about 4-6 min per seed on a laptop CPU)
    python -m gnn_aml_core.benchmark --seeds 5 --epochs 60             # quicker: the GNN has nearly converged after ~25 epochs
    python -m gnn_aml_core.benchmark --seeds 5 --models gatv2,rgcn     # both architectures
    python -m gnn_aml_core.benchmark --seeds 3 --no-gnn                # baselines only (fast, no PyTorch needed)
    python -m gnn_aml_core.benchmark --seeds 5 --camouflage 3.0        # a different difficulty level
    python -m gnn_aml_core.benchmark --accounts 20000 --seeds 5        # a large, slow, overnight-scale run

Writes benchmarks/results.json (every number, every seed) and benchmarks/results.md (paste-ready table) AFTER
EVERY SEED, not just at the end -- a large run can take hours, and a laptop going to sleep, losing power, or the
process being interrupted for any reason should not throw away seeds that already finished. Re-running the exact
same command automatically picks up where it left off (matched by output directory, account count, camouflage,
hard-negative ratio and model list); pass --fresh to ignore any partial results and start over.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np

from aml_synth.graph_generator import AMLGraphGenerator, GeneratorConfig
from gnn_aml_core.baselines import evaluate_baselines
from gnn_aml_core.train import arrays_from_frames, fit_gnn, make_splits

log = logging.getLogger("gnn_aml_core.benchmark")

METRICS = ["auc_roc", "pr_auc", "precision", "recall", "f1", "fpr"]
HEADERS = {"auc_roc": "AUC-ROC", "pr_auc": "PR-AUC", "precision": "Precision", "recall": "Recall", "f1": "F1", "fpr": "FPR"}
# the brief's targets (section 2.1)
TARGETS = {"auc_roc": (">=", 0.87), "precision": (">=", 0.89), "recall": (">=", 0.82), "f1": (">=", 0.85), "fpr": ("<=", 0.02)}


def meets_targets(row: dict) -> bool:
    return all(row[m] >= v if op == ">=" else row[m] <= v for m, (op, v) in TARGETS.items())


def aggregate(per_seed: list) -> dict:
    """per_seed: list of {model: metrics}. Returns {model: {metric: (mean, std)}}."""
    out = {}
    for model in per_seed[0]:
        out[model] = {m: (float(np.mean([r[model][m] for r in per_seed])), float(np.std([r[model][m] for r in per_seed]))) for m in METRICS}
    return out


def strongest_baseline(agg: dict) -> str | None:
    base = [k for k in agg if not k.startswith("GNN")]
    return max(base, key=lambda k: agg[k]["auc_roc"][0]) if base else None


def paired(per_seed: list, a: str, b: str, metric: str) -> dict:
    """Paired difference a - b over seeds (same graph and split in each seed)."""
    d = np.array([r[a][metric] - r[b][metric] for r in per_seed])
    return {"mean": float(d.mean()), "std": float(d.std()), "wins": int((d > 0).sum()), "n": int(len(d))}


def markdown_table(agg: dict, per_seed: list, meta: dict) -> str:
    n = len(per_seed)
    lines = [f"Mean +- std over {n} seeds ({meta['accounts']:,} accounts, camouflage {meta['camouflage']}). "
             "Threshold chosen on validation, metrics on the test split.", "",
             "| Model | " + " | ".join(HEADERS[m] for m in METRICS) + " |", "|---|" + "---|" * len(METRICS)]
    for name, row in agg.items():
        cells = " | ".join(f"{row[m][0]:.3f} +- {row[m][1]:.3f}" for m in METRICS)
        lines.append(f"| **{name}** | {cells} |" if name.startswith("GNN") else f"| {name} | {cells} |")
    best = strongest_baseline(agg)
    for g in [k for k in agg if k.startswith("GNN")]:
        if best:
            lines += ["", f"{g} minus strongest baseline ({best}), paired over seeds:"]
            for m in ("auc_roc", "pr_auc", "f1"):
                p = paired(per_seed, g, best, m)
                lines.append(f"- {HEADERS[m]}: {p['mean']:+.3f} +- {p['std']:.3f}, better in {p['wins']} of {p['n']} seeds")
        ok = sum(meets_targets(r[g]) for r in per_seed)
        lines.append(f"- all five brief targets (AUC>=0.87, precision>=0.89, recall>=0.82, F1>=0.85, FPR<=0.02) met in {ok} of {n} seeds")
    return "\n".join(lines)


def config_fingerprint(a) -> dict:
    """The settings that must match for a saved partial run to be safely resumed -- mixing seeds generated
    under different account counts, difficulty settings or model lists into one aggregate would be meaningless."""
    return {"accounts": a.accounts, "camouflage": a.camouflage, "hard_negatives": a.hard_negatives, "models": a.models, "no_gnn": a.no_gnn}


def load_resumable(out_dir: Path, fingerprint: dict) -> tuple[list, list]:
    """Load a previous run's per-seed results from this output directory, if any, and if its configuration
    matches. Returns (per_seed, seed_info) already completed, or ([], []) if there is nothing to resume from."""
    p = out_dir / "results.json"
    if not p.exists():
        return [], []
    try:
        saved = json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):
        log.warning("could not read %s (corrupted?) -- starting fresh", p)
        return [], []
    if saved.get("fingerprint") != fingerprint:
        log.info("found a previous run in %s with different settings -- starting fresh rather than mixing incompatible seeds", out_dir)
        return [], []
    return saved.get("per_seed", []), saved.get("seed_info", [])


def save_progress(out: Path, per_seed: list, extra: list, fingerprint: dict, t_all: float, final: bool) -> None:
    if not per_seed:
        return
    agg = aggregate(per_seed)
    meta = {"accounts": fingerprint["accounts"], "camouflage": fingerprint["camouflage"],
            "hard_negatives": fingerprint["hard_negatives"], "seeds": [e["seed"] for e in extra]}
    md = markdown_table(agg, per_seed, meta)
    if final:
        print("\n" + md + "\n")
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.md").write_text(md + "\n", encoding="utf-8")
    (out / "results.json").write_text(json.dumps({"fingerprint": fingerprint, "meta": meta, "per_seed": per_seed, "seed_info": extra,
                                                  "aggregate": {k: {m: {"mean": v[0], "std": v[1]} for m, v in row.items()} for k, row in agg.items()},
                                                  "complete": final}, indent=2, default=float), encoding="utf-8")
    log.info("%s %s and %s (%d/%d seeds so far, %.0fs elapsed)", "saved" if final else "checkpointed",
             out / "results.md", out / "results.json", len(per_seed), len(extra), time.time() - t_all)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description="Multi-seed benchmark of the GNN against the baselines")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--seed0", type=int, default=1, help="first seed; seeds are seed0, seed0+1, ...")
    ap.add_argument("--accounts", type=int, default=5000)
    ap.add_argument("--camouflage", type=float, default=1.5)
    ap.add_argument("--hard-negatives", type=float, default=2.0)
    ap.add_argument("--models", default="gatv2", help="comma list: gatv2,rgcn")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=6, help="early stopping, in evaluations of 5 epochs")
    ap.add_argument("--pos-weight", type=float, default=None)
    ap.add_argument("--no-gnn", action="store_true", help="baselines only")
    ap.add_argument("--out", default="benchmarks")
    ap.add_argument("--fresh", action="store_true", help="ignore any previous partial run in --out and start over")
    a = ap.parse_args()

    out = Path(a.out)
    fingerprint = config_fingerprint(a)
    per_seed, extra = ([], []) if a.fresh else load_resumable(out, fingerprint)
    done_seeds = {e["seed"] for e in extra}
    if done_seeds:
        log.info("resuming: %d seed(s) already completed in %s (%s) -- skipping those, continuing with the rest",
                 len(done_seeds), out, sorted(done_seeds))

    t_all = time.time()
    for i in range(a.seeds):
        seed = a.seed0 + i
        if seed in done_seeds:
            continue
        t0 = time.time()
        graph = AMLGraphGenerator(GeneratorConfig(n_accounts=a.accounts, seed=seed, camouflage=a.camouflage,
                                                  hard_negative_ratio=a.hard_negatives)).generate()
        arrays = arrays_from_frames(graph.accounts, graph.transactions)
        tr, va, te = make_splits(arrays["y"], seed)
        rows, info = {}, {"seed": seed, "nodes": int(arrays["x"].shape[0]), "positive_rate": float(arrays["y"].mean())}
        if not a.no_gnn:
            for m in [x.strip() for x in a.models.split(",") if x.strip()]:
                res = fit_gnn(arrays, tr, va, te, model_name=m, epochs=a.epochs, patience=a.patience, pos_weight=a.pos_weight,
                              seed=seed, verbose=False)
                rows[f"GNN ({m.upper()})"] = res["report"]
                info[f"{m}_best_epoch"] = res["best_epoch"]
                info[f"{m}_epochs_run"] = res["epochs_run"]
        rows.update(evaluate_baselines(arrays["x"], arrays["y"], tr, va, te, edges=(arrays["src_tx"], arrays["dst_tx"]), seed=seed))
        per_seed.append(rows)
        extra.append(info)
        summary = "  ".join(f"{k.split(' (')[0][:22]} AUC {v['auc_roc']:.3f}" for k, v in rows.items() if k.startswith("GNN") or "neighbour" in k)
        log.info("seed %d done in %.0fs  |  %s", seed, time.time() - t0, summary)
        save_progress(out, per_seed, extra, fingerprint, t_all, final=False)

    if not any(seed not in done_seeds for seed in range(a.seed0, a.seed0 + a.seeds)):
        log.info("all %d requested seed(s) were already completed in %s -- nothing new to do (pass --fresh to redo them)", a.seeds, out)

    save_progress(out, per_seed, extra, fingerprint, t_all, final=True)


if __name__ == "__main__":
    main()
