"""gnn_aml_core.livegraph: the REAL, runtime account population for this project's live-ingestion mode, as
opposed to the offline-trained SynthAML/AMLworld account universe gnn_aml_core.train / gnn_aml_core.amlworld
produce.

The Intake tab's "paste a document of transactions" option calls POST /admin/ingest-live (see
gnn_aml_core/main.py) with new transaction rows. This module is what that endpoint uses to (a) append those
rows to a persistent store on disk (LIVE_DATA_DIR/transactions.csv, growing indefinitely -- never replaced,
only ever appended to) and (b) rebuild graph.pt from the FULL accumulated live population every time,
standardized with the ALREADY-TRAINED model's own stored mean/std (gnn_model.pt's "mean"/"std") rather than
recomputing new statistics from whatever's currently live -- refitting a scaler to a handful of live accounts
would silently invalidate every prediction the frozen weights make. gnn_model.pt itself -- the trained weights,
the decision threshold, the calibration curve -- is never opened for writing anywhere in this module. This is
deliberately NOT a retrain: a brand new account is scored by the SAME frozen model that scores every synthetic
account, just fed through the same feature pipeline with real counterparty structure instead of synthetic.

Two honest gaps, stated plainly rather than hidden:
  * This never updates the model's WEIGHTS from decisions made on live accounts -- gnn_aml_core.feedback's
    existing retrain pipeline is still the only path to that, still gated on the same "deliberately
    investigated, confirmed" trigger it always required. Nothing here changes that threshold or bypasses it.
  * A raw transaction document carries no account_type / country / opened_ts / risk_score -- those fields
    degrade to the same kind of honest placeholder gnn_aml_core.amlworld already uses for the real AMLworld
    dataset (see that module's docstring), not invented here as something new.

    python -m gnn_aml_core.livegraph --reset     # wipe the live store AND write a zero-account graph.pt,
                                                  # keeping gnn_model.pt's trained weights exactly as they are
"""
from __future__ import annotations

import argparse
import logging
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from gnn_aml_core.features import EDGE_DIM, build_edge_features, build_node_features, make_bidirectional, standardize

log = logging.getLogger("gnn_aml_core.livegraph")

LIVE_DIR = Path(os.environ.get("LIVE_DATA_DIR", "data/live"))
TX_COLUMNS = ["tx_id", "src", "dst", "amount", "timestamp", "cross_border"]


def _live_tx_path() -> Path:
    return LIVE_DIR / "transactions.csv"


def read_live() -> pd.DataFrame:
    """Everything ingested so far. Empty (right columns, zero rows) if nothing has been ingested yet."""
    p = _live_tx_path()
    if not p.exists():
        return pd.DataFrame(columns=TX_COLUMNS)
    return pd.read_csv(p)


def ingest(rows: list[dict]) -> pd.DataFrame:
    """Appends `rows` (each needing at least src/dst/amount; timestamp/cross_border default sensibly) to the
    persistent live transaction store and returns the FULL accumulated table -- ingestion is additive, a
    document pasted today adds to whatever was pasted before, it never replaces it. Rows missing src/dst are
    silently dropped (malformed line in a pasted document), not raised, since one bad line should not fail an
    entire document's ingestion."""
    existing = read_live()
    next_id = len(existing)
    now = int(time.time())
    new_rows = []
    for i, r in enumerate(rows):
        src, dst = str(r.get("src", "")).strip(), str(r.get("dst", "")).strip()
        if not src or not dst:
            continue
        amount = float(r["amount"])
        ts_raw = r.get("timestamp")
        ts = int(ts_raw) if ts_raw not in (None, "") else now
        xb_raw = r.get("cross_border")
        xb = int(xb_raw) if xb_raw not in (None, "") else 0
        new_rows.append({"tx_id": f"live_{next_id + i}", "src": src, "dst": dst,
                         "amount": amount, "timestamp": ts, "cross_border": xb})
    if not new_rows:
        return existing
    combined = pd.concat([existing, pd.DataFrame(new_rows)], ignore_index=True)
    LIVE_DIR.mkdir(parents=True, exist_ok=True)
    combined.to_csv(_live_tx_path(), index=False)
    log.info("ingest: +%d transaction(s) (%d total live)", len(new_rows), len(combined))
    return combined


def remove_accounts(account_ids: list[str]) -> pd.DataFrame:
    """The inverse of ingest(): drops every transaction that touches any of `account_ids` (as either src or
    dst) from the persistent live store and returns what remains. Used by POST /admin/remove-live for the
    Intake tab's "clear temporary data" control -- a walk-in/one-off customer's accounts leave the live graph
    entirely (unscannable, gone from the network view) once this runs and rebuild_live_graph() is called
    after it, but nothing here touches backend's Postgres alerts/decision history: that's a separate store
    this module has never had any concept of, and clearing the live graph is deliberately NOT the same
    operation as deleting an alert.

    An account_id supplied here that never appears in the live store (already cleared, or never ingested) is
    silently a no-op for that id -- this mirrors ingest()'s own "a bad/missing row doesn't fail the whole
    call" stance rather than raising on it."""
    existing = read_live()
    if len(existing) == 0 or not account_ids:
        return existing
    drop = set(account_ids)
    keep_mask = ~(existing["src"].isin(drop) | existing["dst"].isin(drop))
    remaining = existing[keep_mask].reset_index(drop=True)
    LIVE_DIR.mkdir(parents=True, exist_ok=True)
    remaining.to_csv(_live_tx_path(), index=False)
    log.info("remove_accounts: dropped %d transaction(s) touching %d account(s) (%d remain)",
             len(existing) - len(remaining), len(drop), len(remaining))
    return remaining


def _accounts_from_tx(tx: pd.DataFrame) -> pd.DataFrame:
    """Builds the accounts table honest placeholders fill in for fields a raw transaction document can never
    carry (account_type, country, opened_ts, risk_score) -- the exact same disclosed-gap pattern
    gnn_aml_core.amlworld already uses for the real AMLworld dataset, not invented for this module."""
    ids = pd.unique(pd.concat([tx["src"], tx["dst"]]))
    return pd.DataFrame({
        "account_id": ids, "account_type": "unknown", "country": "unknown", "opened_ts": 0, "risk_score": 0.0,
    })


def write_blank_graph(ckpt: dict, out_dir: str | Path) -> dict:
    """Writes a graph.pt with ZERO accounts -- correctly shaped (matching ckpt's in_dim/feature_names) but
    empty, so the served account population starts genuinely blank until something is ingested. gnn_model.pt
    (the trained weights) is untouched -- only graph.pt, the served account population, is ever written here."""
    import torch
    in_dim = len(ckpt["feature_names"])
    graph = {
        "x": torch.zeros((0, in_dim), dtype=torch.float32), "edge_index": torch.zeros((2, 0), dtype=torch.int64),
        "edge_attr": torch.zeros((0, EDGE_DIM), dtype=torch.float32), "edge_type": torch.zeros((0,), dtype=torch.int64),
        "y": torch.zeros((0,), dtype=torch.float32), "tx_amount": torch.zeros((0,), dtype=torch.float32),
        "tx_timestamp": torch.zeros((0,), dtype=torch.int64), "tx_cross_border": torch.zeros((0,), dtype=torch.int64),
        "opened_ts": torch.zeros((0,), dtype=torch.int64),
        "account_ids": [], "account_type": [], "country": [], "tx_ids": [],
        "feature_names": ckpt["feature_names"], "n_tx": 0,
    }
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(graph, out / "graph.pt")
    log.info("blank live graph written to %s (0 accounts) -- gnn_model.pt untouched", out)
    return {"n_accounts": 0, "n_transactions": 0}


def rebuild_live_graph(ckpt: dict, out_dir: str | Path, reporting_threshold: float = 10_000.0) -> dict:
    """Rebuilds graph.pt from the FULL accumulated live population (read_live()), standardized with the
    ALREADY-TRAINED model's own mean/std -- never refits a new scaler to whatever's currently live. Writes
    graph.pt only; gnn_model.pt (the weights themselves) is never opened for writing anywhere in this module."""
    import torch

    tx = read_live()
    if len(tx) == 0:
        return write_blank_graph(ckpt, out_dir)

    accounts = _accounts_from_tx(tx)
    x_raw, names = build_node_features(accounts, tx, reporting_threshold)
    if names != ckpt["feature_names"]:
        raise ValueError("live feature set doesn't match the trained model's feature_names -- was the model "
                         "retrained with a different feature list since this live store was last rebuilt?")
    x, _, _ = standardize(x_raw, ckpt["mean"].numpy(), ckpt["std"].numpy())

    idx = {a: i for i, a in enumerate(accounts["account_id"])}
    src = tx["src"].map(idx).to_numpy(dtype=np.int64)
    dst = tx["dst"].map(idx).to_numpy(dtype=np.int64)
    attr, etype = build_edge_features(tx, reporting_threshold)
    src2, dst2, attr2, etype2 = make_bidirectional(src, dst, attr, etype)

    graph = {
        "x": torch.from_numpy(x), "edge_index": torch.from_numpy(np.stack([src2, dst2])),
        "edge_attr": torch.from_numpy(attr2), "edge_type": torch.from_numpy(etype2),
        "y": torch.zeros(len(accounts), dtype=torch.float32),   # unknown ground truth for a live account --
                                                                 # never read at serving time, kept only for
                                                                 # shape-compatibility with graph.pt's own schema
        "tx_amount": torch.from_numpy(tx["amount"].to_numpy(dtype=np.float32)),
        "tx_timestamp": torch.from_numpy(tx["timestamp"].to_numpy(dtype=np.int64)),
        "tx_cross_border": torch.from_numpy(tx["cross_border"].to_numpy(dtype=np.int64)),
        "opened_ts": torch.from_numpy(accounts["opened_ts"].to_numpy(dtype=np.int64)),
        "account_ids": accounts["account_id"].tolist(), "account_type": accounts["account_type"].tolist(),
        "country": accounts["country"].tolist(), "tx_ids": tx["tx_id"].tolist(),
        "feature_names": names, "n_tx": int(len(tx)),
    }
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(graph, out / "graph.pt")
    log.info("live graph rebuilt: %d account(s), %d transaction(s)", len(accounts), len(tx))
    return {"n_accounts": len(accounts), "n_transactions": int(len(tx))}


def main() -> None:
    import torch
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", type=Path, default=Path("models"))
    ap.add_argument("--reset", action="store_true",
                    help="wipe the live transaction store AND write a zero-account graph.pt (gnn_model.pt, "
                         "the trained weights, is left exactly as it is)")
    a = ap.parse_args()

    ckpt_path = a.models / "gnn_model.pt"
    if not ckpt_path.exists():
        raise SystemExit(f"{ckpt_path} not found -- train a model first (gnn_aml_core.train or "
                         f"gnn_aml_core.amlworld --save) before switching to live-ingestion mode.")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    if a.reset:
        if _live_tx_path().exists():
            _live_tx_path().unlink()
        write_blank_graph(ckpt, a.models)
        log.warning("live store reset and graph.pt blanked. Restart/reload gnn-detection-api to serve it:\n"
                    "    docker compose --profile ml up -d --force-recreate gnn-detection-api")
        return

    result = rebuild_live_graph(ckpt, a.models)
    log.info("rebuilt from existing live store: %s", result)


if __name__ == "__main__":
    main()
