"""Train the GNN on the CSVs produced by aml_synth.

    python -m aml_synth.graph_generator --out data --to csv
    python -m gnn_aml_core.train --data data --model gatv2 --epochs 150 --out models
"""
from __future__ import annotations

import argparse
import copy
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.model_selection import train_test_split

from gnn_aml_core.features import build_edge_features, build_node_features, standardize
from gnn_aml_core.models import build_model, class_weighted_bce

log = logging.getLogger("gnn_aml_core.train")
REPORTING_THRESHOLD = 10_000.0


def load_graph(data_dir: str):
    accounts = pd.read_csv(Path(data_dir) / "accounts.csv")
    tx = pd.read_csv(Path(data_dir) / "transactions.csv")
    x_raw, names = build_node_features(accounts, tx, REPORTING_THRESHOLD)
    x, mean, std = standardize(x_raw)
    idx = {a: i for i, a in enumerate(accounts["account_id"])}
    src = tx["src"].map(idx).to_numpy()
    dst = tx["dst"].map(idx).to_numpy()
    edge_attr, edge_type = build_edge_features(tx, REPORTING_THRESHOLD)
    graph = {
        "x": torch.from_numpy(x),
        "edge_index": torch.from_numpy(np.stack([src, dst]).astype(np.int64)),
        "edge_attr": torch.from_numpy(edge_attr),
        "edge_type": torch.from_numpy(edge_type),
        "y": torch.tensor(accounts["is_suspicious"].to_numpy(), dtype=torch.float32),
        "account_ids": accounts["account_id"].tolist(),
        "account_type": accounts["account_type"].tolist(),
        "country": accounts["country"].tolist(),
        "tx_ids": tx["tx_id"].tolist(),
        "tx_amount": torch.tensor(tx["amount"].to_numpy(), dtype=torch.float32),
        "tx_timestamp": torch.tensor(tx["timestamp"].to_numpy(), dtype=torch.long),
        "tx_cross_border": torch.tensor(tx["cross_border"].to_numpy(), dtype=torch.long),
        "feature_names": names,
    }
    return graph, mean, std


@torch.no_grad()
def evaluate(model, g, mask):
    model.eval()
    logits = model(g["x"], g["edge_index"], g["edge_attr"], g["edge_type"])
    y = g["y"][mask].cpu().numpy()
    p = torch.sigmoid(logits[mask]).cpu().numpy()
    return float(roc_auc_score(y, p)), float(average_precision_score(y, p)), y, p


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="models")
    ap.add_argument("--model", default="gatv2", choices=["gatv2", "rgcn"])
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--lr", type=float, default=0.005)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--auc-target", type=float, default=0.87)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    g, mean, std = load_graph(a.data)
    n = g["x"].size(0)
    log.info("graph: %d nodes, %d edges, %.2f%% positive nodes, device=%s",
             n, g["edge_index"].size(1), 100 * g["y"].mean().item(), dev)

    y_np = g["y"].numpy()
    tr, tmp = train_test_split(np.arange(n), test_size=0.4, stratify=y_np, random_state=a.seed)
    va, te = train_test_split(tmp, test_size=0.5, stratify=y_np[tmp], random_state=a.seed)

    def mask(ix):
        m = torch.zeros(n, dtype=torch.bool)
        m[ix] = True
        return m.to(dev)

    tr_m, va_m, te_m = mask(tr), mask(va), mask(te)
    tensors = {k: v.to(dev) for k, v in g.items() if isinstance(v, torch.Tensor)}

    hp = {"hidden": a.hidden, "num_layers": a.layers}
    model = build_model(a.model, g["x"].size(1), **hp).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)

    best_auc, best_state = -1.0, None
    for epoch in range(1, a.epochs + 1):
        model.train()
        opt.zero_grad()
        logits = model(tensors["x"], tensors["edge_index"], tensors["edge_attr"], tensors["edge_type"])
        loss = class_weighted_bce(logits[tr_m], tensors["y"][tr_m])
        loss.backward()
        opt.step()
        if epoch % 5 == 0 or epoch == a.epochs:
            auc, ap_, _, _ = evaluate(model, tensors, va_m)
            log.info("epoch %3d  loss %.4f  val AUC %.4f  val AP %.4f", epoch, loss.item(), auc, ap_)
            if auc > best_auc:
                best_auc, best_state = auc, copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    _, _, yv, pv = evaluate(model, tensors, va_m)
    prec, rec, thr = precision_recall_curve(yv, pv)
    f1 = 2 * prec * rec / np.clip(prec + rec, 1e-9, None)
    threshold = float(thr[int(np.argmax(f1[:-1]))]) if len(thr) else 0.5

    test_auc, test_ap, yt, pt = evaluate(model, tensors, te_m)
    pred = pt >= threshold
    tp = int(((pred == 1) & (yt == 1)).sum())
    fp = int(((pred == 1) & (yt == 0)).sum())
    fn = int(((pred == 0) & (yt == 1)).sum())
    metrics = {
        "val_auc_roc": best_auc, "test_auc_roc": test_auc, "test_pr_auc": test_ap,
        "test_precision": tp / max(tp + fp, 1), "test_recall": tp / max(tp + fn, 1),
        "threshold": threshold, "auc_target": a.auc_target,
        "meets_auc_target": bool(test_auc >= a.auc_target),
    }
    log.info("TEST: %s", {k: round(v, 4) if isinstance(v, float) else v for k, v in metrics.items()})

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_name": a.model, "hparams": hp, "in_dim": g["x"].size(1), "state_dict": best_state,
        "feature_names": g["feature_names"], "mean": torch.from_numpy(mean), "std": torch.from_numpy(std),
        "threshold": threshold, "metrics": metrics,
    }, out / "gnn_model.pt")
    torch.save({k: v for k, v in g.items()}, out / "graph.pt")
    log.info("saved %s and %s", out / "gnn_model.pt", out / "graph.pt")


if __name__ == "__main__":
    main()
