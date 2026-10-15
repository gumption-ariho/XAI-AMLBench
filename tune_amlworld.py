"""Quick hyperparameter comparison on a fixed AMLworld subsample.

Loads the data and builds the train/val/test split ONCE, then trains several GATv2 configs
against that exact same split so the comparison is fair (not confounded by different splits).

Usage (from the project root, venv active):
    python tune_amlworld.py --data data/amlworld --file HI-Small_Trans_mid.csv
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

from gnn_aml_core.amlworld import load_amlworld
from gnn_aml_core.train import arrays_from_frames, format_table, make_splits, fit_gnn

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("tune_amlworld")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True, help="folder containing the AMLworld transactions CSV")
    ap.add_argument("--file", default="HI-Small_Trans_mid.csv")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    accounts, tx = load_amlworld(a.data / a.file)
    n_susp = int(accounts["is_suspicious"].sum())
    log.info("AMLworld (%s): %d accounts (%d flagged, %.3f%%), %d transactions",
             a.file, len(accounts), n_susp, 100 * n_susp / len(accounts), len(tx))

    arrays = arrays_from_frames(accounts, tx.drop(columns=["is_laundering"]))
    tr, va, te = make_splits(arrays["y"], seed=a.seed)
    log.info("split: train %d / val %d / test %d (SAME split used for every config below)", len(tr), len(va), len(te))

    # Each config trains fresh from the same split -- only the hyperparameters differ.
    configs = {
        "baseline (hidden=64, auto pos_weight)":      dict(hidden=64,  pos_weight=None),
        "wider (hidden=128, auto pos_weight)":         dict(hidden=128, pos_weight=None),
        "moderate pos_weight=50 (hidden=64)":          dict(hidden=64,  pos_weight=50.0),
    }

    results = {}
    for name, hp in configs.items():
        t0 = time.time()
        log.info("=== running: %s ===", name)
        res = fit_gnn(arrays, tr, va, te, model_name="gatv2", epochs=a.epochs, patience=a.patience,
                       seed=a.seed, verbose=True, **hp)
        results[name] = res["report"]
        log.info("--- %s done: best epoch %d, val AUC %.4f, %.0fs ---",
                  name, res["best_epoch"], res["best_val_auc"], time.time() - t0)
        log.info("running comparison so far:\n%s", format_table(results))

    log.info("FINAL COMPARISON (same data/split throughout, %s)\n%s", a.file, format_table(results))


if __name__ == "__main__":
    main()