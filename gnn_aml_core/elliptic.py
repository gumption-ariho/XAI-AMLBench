"""gnn_aml_core.elliptic: load the real-world Elliptic Bitcoin dataset (Weber et al., 2019, KDD) and adapt it to
the same arrays-shaped dict that `gnn_aml_core.train.fit_gnn` and `gnn_aml_core.baselines.evaluate_baselines`
already expect -- so the exact model, training loop, calibration and baseline code built for aml_synth's
synthetic graphs runs against real transaction data completely unchanged.

Why this matters: aml_synth is necessarily synthetic (real bank data is confidential), so nothing in this project
so far demonstrates real-world detection performance. Elliptic is real, public Bitcoin transaction data with
labels derived from actual law-enforcement-linked entities (ransomware, darknet markets, scams, Ponzi schemes,
etc.) -- a genuinely different kind of evidence than another round of synthetic tuning could ever produce.

Structural note: in aml_synth, NODES are accounts and EDGES are transactions. In Elliptic, NODES are transactions
themselves and EDGES are fund-flow links between them (money out of transaction A funds transaction B). Edges
therefore carry no "amount" of their own -- that information already lives on the node features -- so the edge
features built here are a small, honestly-labelled structural signal (how many time steps apart the two
transactions are), not a repurposed version of aml_synth's amount-based edge features.

Download the three raw files yourself first (this module never fetches anything over the network):
    https://www.kaggle.com/datasets/ellipticco/elliptic-data-set
Expected in --data (no renaming needed):
    elliptic_txs_features.csv   (203,769 rows, NO header: txId, time_step, 165 feature columns)
    elliptic_txs_classes.csv    (header: txId,class -- class is "1" illicit, "2" licit, or "unknown")
    elliptic_txs_edgelist.csv   (header: txId1,txId2 -- directed fund-flow edges)

    python -m gnn_aml_core.elliptic --data data/elliptic                        # train + evaluate, temporal split
    python -m gnn_aml_core.elliptic --data data/elliptic --split random --seed 1
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger("gnn_aml_core.elliptic")

RAW_FILES = ("elliptic_txs_features.csv", "elliptic_txs_classes.csv", "elliptic_txs_edgelist.csv")
# The paper's own train/test boundary (Weber et al., 2019): time steps 1-34 train, 35-49 test. Using the same
# split makes results comparable to the published paper and later work that reuses this convention, rather than
# an arbitrary split of our own choosing.
DEFAULT_TEMPORAL_CUTOFF = 34


def _require_files(data_dir: Path) -> None:
    missing = [f for f in RAW_FILES if not (data_dir / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"missing {missing} in {data_dir}. Download the three raw Elliptic files from "
            "https://www.kaggle.com/datasets/ellipticco/elliptic-data-set and place them in this folder "
            "(no renaming needed)."
        )


def load_elliptic(data_dir: str | Path) -> dict:
    """Read the three raw Elliptic files and return an arrays-shaped dict compatible with
    `gnn_aml_core.train.fit_gnn` / `gnn_aml_core.baselines.evaluate_baselines`.

    Returns a dict with the same keys `arrays_from_frames` produces (x, edge_index, edge_attr, edge_type, y,
    src_tx, dst_tx, feature_names, n_tx, account_ids), plus `time_step` (per node) and `labeled_mask` (True for
    the ~23% of nodes with a real class label; the rest are "unknown" and used only as graph context, never as a
    training or evaluation target -- the standard, published way this dataset is used).
    """
    from gnn_aml_core.features import make_bidirectional

    data_dir = Path(data_dir)
    _require_files(data_dir)

    feat = pd.read_csv(data_dir / "elliptic_txs_features.csv", header=None)
    feat.columns = ["txId", "time_step"] + [f"f{i}" for i in range(feat.shape[1] - 2)]
    feat["txId"] = feat["txId"].astype(np.int64)  # guard against pandas ever inferring this column as float
    classes = pd.read_csv(data_dir / "elliptic_txs_classes.csv")
    classes["txId"] = classes["txId"].astype(np.int64)
    edges = pd.read_csv(data_dir / "elliptic_txs_edgelist.csv")
    edges["txId1"] = edges["txId1"].astype(np.int64)
    edges["txId2"] = edges["txId2"].astype(np.int64)

    # The raw file's own convention is 1=illicit, 2=licit, "unknown"=unlabelled. We remap to this project's own
    # convention (1=suspicious/illicit, 0=benign/licit) used everywhere else (gnn_aml_core, xai_explainer), and
    # keep "unknown" nodes in the graph (for message-passing context) but excluded from labels and from every
    # split, exactly as the original paper and all published follow-on work treat them.
    classes = classes.copy()
    classes["class"] = classes["class"].astype(str)
    label_map = {"1": 1, "2": 0}
    y_raw = classes["class"].map(label_map)
    labeled_mask_by_id = y_raw.notna()

    merged = feat.merge(classes, on="txId", how="left")
    n = len(merged)
    tx_ids = merged["txId"].to_numpy()
    id_to_idx = {tid: i for i, tid in enumerate(tx_ids)}

    feature_cols = [c for c in merged.columns if c.startswith("f")]
    x = merged[feature_cols].to_numpy(dtype=np.float32)
    time_step = merged["time_step"].to_numpy(dtype=np.int64)
    y = merged["class"].astype(str).map(label_map).fillna(-1).to_numpy(dtype=np.float32)
    labeled_mask = y >= 0
    y = np.where(labeled_mask, y, 0.0)  # placeholder for unlabelled rows; never read where labeled_mask is False

    src = edges["txId1"].map(id_to_idx).to_numpy()
    dst = edges["txId2"].map(id_to_idx).to_numpy()
    keep = ~(pd.isna(src) | pd.isna(dst))
    if not keep.all():
        log.warning("dropping %d edge(s) referencing a transaction id not present in the features file", (~keep).sum())
    src, dst = src[keep].astype(np.int64), dst[keep].astype(np.int64)
    n_tx = len(src)

    # A small, honest edge feature: how many time steps apart the two linked transactions are (0 if the same
    # step), rather than inventing an "amount" that does not exist at the edge level in this dataset.
    edge_gap = np.abs(time_step[src] - time_step[dst]).astype(np.float32).reshape(-1, 1)
    edge_type = np.zeros(n_tx, dtype=np.int64)  # a single real relation ("Bitcoin fund flow"); GATv2 ignores this,
                                                # RGCN falls back to a single relation automatically if unused.
    src2, dst2, edge_attr2, edge_type2 = make_bidirectional(src, dst, edge_gap, edge_type)

    return {
        "x": x, "edge_index": np.stack([src2, dst2]), "edge_attr": edge_attr2, "edge_type": edge_type2,
        "y": y, "src_tx": src, "dst_tx": dst, "n_tx": n_tx,
        "feature_names": feature_cols, "account_ids": [str(t) for t in tx_ids],
        "time_step": time_step, "labeled_mask": labeled_mask,
    }


def elliptic_splits(arrays: dict, split: str = "temporal", cutoff: int = DEFAULT_TEMPORAL_CUTOFF, seed: int = 42):
    """Train/validation/test indices, restricted to labelled nodes only (never "unknown").

    "temporal" (default, matches the original paper): train on time steps 1..cutoff, split the remaining labelled
    nodes in later time steps evenly into validation and test -- a fair test of generalising to the future, not
    just to unseen nodes from the same time period. "random": a stratified random 60/20/20 split of labelled
    nodes, matching the convention used elsewhere in this project (aml_synth benchmarks); useful for comparing
    the effect of the split itself, since a random split is usually easier than a genuinely temporal one.
    """
    from sklearn.model_selection import train_test_split

    labeled_idx = np.where(arrays["labeled_mask"])[0]
    y = arrays["y"]
    ts = arrays["time_step"]

    if split == "temporal":
        train_idx = labeled_idx[ts[labeled_idx] <= cutoff]
        rest = labeled_idx[ts[labeled_idx] > cutoff]
        if len(rest) < 2:
            raise ValueError(f"temporal cutoff {cutoff} leaves too few labelled nodes after it to split into validation/test")
        va, te = train_test_split(rest, test_size=0.5, random_state=seed,
                                  stratify=y[rest] if len(set(y[rest])) > 1 else None)
        return train_idx, va, te
    if split == "random":
        tr, tmp = train_test_split(labeled_idx, test_size=0.4, random_state=seed, stratify=y[labeled_idx])
        va, te = train_test_split(tmp, test_size=0.5, random_state=seed, stratify=y[tmp])
        return tr, va, te
    raise ValueError(f"unknown split '{split}' (use 'temporal' or 'random')")


def main() -> None:
    import torch

    from gnn_aml_core.baselines import evaluate_baselines
    from gnn_aml_core.train import fit_gnn, format_table

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="folder containing the three raw elliptic_txs_*.csv files")
    ap.add_argument("--split", default="temporal", choices=["temporal", "random"])
    ap.add_argument("--cutoff", type=int, default=DEFAULT_TEMPORAL_CUTOFF, help="temporal split: train on time steps <= this")
    ap.add_argument("--model", default="gatv2", choices=["gatv2", "rgcn"])
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-baselines", action="store_true")
    a = ap.parse_args()

    arrays = load_elliptic(a.data)
    tr, va, te = elliptic_splits(arrays, split=a.split, cutoff=a.cutoff, seed=a.seed)
    n_labeled = int(arrays["labeled_mask"].sum())
    log.info("Elliptic: %d transactions (%d labelled, %.1f%%), %d fund-flow edges, %d features, %s split "
             "(train %d / val %d / test %d)", arrays["x"].shape[0], n_labeled, 100 * n_labeled / arrays["x"].shape[0],
             arrays["n_tx"], arrays["x"].shape[1], a.split, len(tr), len(va), len(te))

    model_kwargs = {"edge_dim": arrays["edge_attr"].shape[1]} if a.model == "gatv2" else {}
    res = fit_gnn(arrays, tr, va, te, model_name=a.model, epochs=a.epochs, patience=a.patience, seed=a.seed, **model_kwargs)
    table = {f"GNN ({a.model.upper()}, best epoch {res['best_epoch']})": res["report"]}
    if not a.no_baselines:
        table.update(evaluate_baselines(arrays["x"], arrays["y"], tr, va, te, edges=(arrays["src_tx"], arrays["dst_tx"]), seed=a.seed))
    log.info("REAL-WORLD RESULTS on the Elliptic Bitcoin dataset (%s split, threshold chosen on validation)\n%s",
             a.split, format_table(table))
    log.info("Reminder: this is real transaction data with real, law-enforcement-derived labels -- a genuinely "
             "different kind of evidence to the synthetic aml_synth benchmark, not a replacement for a pilot on "
             "a real bank's own wire-transfer data.")


if __name__ == "__main__":
    main()
