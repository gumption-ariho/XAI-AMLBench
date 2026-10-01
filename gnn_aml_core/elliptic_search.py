"""gnn_aml_core.elliptic_search: pushes detection performance on the real Elliptic Bitcoin dataset as far as
this architecture can go, using only real, law-enforcement-derived data -- no synthetic tuning, no aml_synth.

Two things this session's default `python -m gnn_aml_core.elliptic` run does NOT do, which this module adds:

  1. A genuine hyperparameter search for GATv2 and RGCN (hidden size, depth, learning rate), rather than one run
     at default settings. The earlier Elliptic result (GATv2 losing to boosting, F1 0.637 vs 0.835) used only
     the defaults -- this checks whether a better-tuned GNN closes any of that gap on real data.

  2. The GNN-embeddings-into-boosting hybrid: instead of feeding boosting either raw features or hand-engineered
     neighbour averages (the approach that won previously), this extracts the trained GNN's own learned
     representation (via the new Detector.embed() method) and feeds THAT into boosting instead. This directly
     tests whether the GNN's message-passing produces a useful representation even when its own final
     classification layer does not outperform simpler models -- a real, distinct question from "does the GNN
     win outright," and one the earlier Elliptic run never asked.

Everything here trains and evaluates only against Elliptic; nothing here touches aml_synth or SynthAML.

    python -m gnn_aml_core.elliptic_search --data data/elliptic
"""
from __future__ import annotations

import argparse
import itertools
import logging

from gnn_aml_core.elliptic import DEFAULT_TEMPORAL_CUTOFF, elliptic_splits, load_elliptic

log = logging.getLogger("gnn_aml_core.elliptic_search")

# A deliberately modest grid: enough to genuinely test whether tuning helps, not an exhaustive search that
# would take hours per run on a real 200k-transaction graph.
GATV2_GRID = [
    {"hidden": 64, "layers": 2, "lr": 0.005},
    {"hidden": 128, "layers": 2, "lr": 0.005},
    {"hidden": 64, "layers": 3, "lr": 0.005},
    {"hidden": 128, "layers": 3, "lr": 0.002},
    {"hidden": 96, "layers": 2, "lr": 0.01},
]
RGCN_GRID = [
    {"hidden": 64, "layers": 2, "lr": 0.005},
    {"hidden": 128, "layers": 2, "lr": 0.005},
    {"hidden": 64, "layers": 3, "lr": 0.002},
]


def search_gnn(arrays: dict, tr, va, te, model_name: str, grid: list[dict], epochs: int, patience: int, seed: int) -> dict:
    """Trains every configuration in `grid`, returns the result dict (from fit_gnn) of whichever one achieved
    the best VALIDATION AUC -- the test split is still touched only once, by whichever config wins on
    validation, not by picking the config with the best test score (that would be test-set leakage)."""
    from gnn_aml_core.train import fit_gnn

    model_kwargs = {"edge_dim": arrays["edge_attr"].shape[1]} if model_name == "gatv2" else {}
    best = None
    for i, hp in enumerate(grid):
        log.info("[%s %d/%d] hidden=%d layers=%d lr=%s", model_name, i + 1, len(grid), hp["hidden"], hp["layers"], hp["lr"])
        res = fit_gnn(arrays, tr, va, te, model_name=model_name, hidden=hp["hidden"], layers=hp["layers"],
                      lr=hp["lr"], epochs=epochs, patience=patience, seed=seed, verbose=False, **model_kwargs)
        log.info("  -> val AUC %.4f (best epoch %d)", res["best_val_auc"], res["best_epoch"])
        if best is None or res["best_val_auc"] > best["best_val_auc"]:
            best = res
            best["hp"] = hp
    return best


def embeddings_plus_boosting(arrays: dict, best_gnn: dict, model_name: str, tr, va, te, seed: int) -> dict:
    """Extracts the best-found GNN's learned node representation and feeds it into gradient boosting, instead
    of that GNN's own classification head. A genuinely different question from "does the GNN win" -- this
    checks whether its message-passing produces a useful representation even when its own final layer does not
    outperform simpler models on their own features."""
    import numpy as np
    import torch

    from gnn_aml_core.baselines import _boosting
    from gnn_aml_core.evaluation import full_report
    from gnn_aml_core.models import build_model

    hp = best_gnn["hp"]
    model_kwargs = {"edge_dim": arrays["edge_attr"].shape[1]} if model_name == "gatv2" else {}
    model = build_model(model_name, in_dim=arrays["x"].shape[1], hidden=hp["hidden"], num_layers=hp["layers"], **model_kwargs)
    model.load_state_dict(best_gnn["state_dict"])
    model.eval()

    x = torch.from_numpy(arrays["x"]).float()
    edge_index = torch.from_numpy(arrays["edge_index"]).long()
    edge_attr = torch.from_numpy(arrays["edge_attr"]).float() if model_name == "gatv2" else None
    edge_type = torch.from_numpy(arrays["edge_type"]).long() if model_name == "rgcn" else None
    with torch.no_grad():
        emb = model.embed(x, edge_index, edge_attr=edge_attr, edge_type=edge_type).numpy()

    y = np.asarray(arrays["y"])
    gb, gb_name = _boosting(y[tr], seed)
    gb.fit(emb[tr], y[tr])
    p_va = gb.predict_proba(emb[va])[:, 1]
    p_te = gb.predict_proba(emb[te])[:, 1]
    return {f"{gb_name} on {model_name.upper()}'s learned embeddings": full_report(y[va], p_va, y[te], p_te)}


def main() -> None:
    import os

    from gnn_aml_core.baselines import evaluate_baselines
    from gnn_aml_core.train import format_table

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="temporal", choices=["temporal", "random"])
    ap.add_argument("--cutoff", type=int, default=DEFAULT_TEMPORAL_CUTOFF)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-rgcn", action="store_true", help="GATv2 search only, roughly halves total runtime")
    a = ap.parse_args()

    arrays = load_elliptic(a.data)
    tr, va, te = elliptic_splits(arrays, split=a.split, cutoff=a.cutoff, seed=a.seed)
    log.info("searching GATv2 (%d configurations)...", len(GATV2_GRID))
    best_gatv2 = search_gnn(arrays, tr, va, te, "gatv2", GATV2_GRID, a.epochs, a.patience, a.seed)
    log.info("best GATv2 config: %s (val AUC %.4f)", best_gatv2["hp"], best_gatv2["best_val_auc"])

    results = {f"GATv2 (tuned: hidden={best_gatv2['hp']['hidden']}, layers={best_gatv2['hp']['layers']})": best_gatv2["report"]}
    results.update(embeddings_plus_boosting(arrays, best_gatv2, "gatv2", tr, va, te, a.seed))

    if not a.skip_rgcn:
        log.info("searching RGCN (%d configurations)...", len(RGCN_GRID))
        best_rgcn = search_gnn(arrays, tr, va, te, "rgcn", RGCN_GRID, a.epochs, a.patience, a.seed)
        log.info("best RGCN config: %s (val AUC %.4f)", best_rgcn["hp"], best_rgcn["best_val_auc"])
        results[f"RGCN (tuned: hidden={best_rgcn['hp']['hidden']}, layers={best_rgcn['hp']['layers']})"] = best_rgcn["report"]
        results.update(embeddings_plus_boosting(arrays, best_rgcn, "rgcn", tr, va, te, a.seed))

    results.update(evaluate_baselines(arrays["x"], arrays["y"], tr, va, te, edges=(arrays["src_tx"], arrays["dst_tx"]), seed=a.seed))

    best_name = max(results, key=lambda k: results[k]["auc_roc"])
    log.info("REAL-DATA-ONLY SEARCH RESULTS on Elliptic (%s split, hyperparameters tuned on validation only)\n%s",
             a.split, format_table(results))
    log.info("Best result found: '%s' (AUC %.3f, F1 %.3f). This is the ceiling this architecture reaches on "
             "real Elliptic data with tuning -- it does not change what domain (crypto, not bank-wire) this "
             "evidence covers.", best_name, results[best_name]["auc_roc"], results[best_name]["f1"])


if __name__ == "__main__":
    main()
