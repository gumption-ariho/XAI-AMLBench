"""Train the GNN on the graph produced by aml_synth, evaluate it, and compare it with the baselines.

    python -m aml_synth.graph_generator --out data --to csv
    python -m gnn_aml_core.train --data data --model gatv2 --epochs 150 --out models

Writes  models/gnn_model.pt, models/graph.pt  (served by the API)  and  models/metrics.json  (test AUC-ROC, PR-AUC,
precision, recall, F1, false-positive rate, plus the same numbers for every baseline).

Design notes
  * The graph is BIDIRECTIONAL: every transaction also gets a reversed copy flagged as reversed, so an account hears
    from the accounts it sends to as well as the ones it receives from.
  * The decision threshold is chosen on the validation split (best F1) and reported on the untouched test split.
  * The loss is class-weighted BCE. By default the weight is #negatives / #positives; pass --pos-weight 19.0 for the
    fixed weight in the project brief.
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from gnn_aml_core.baselines import evaluate_baselines
from gnn_aml_core.evaluation import full_report
from gnn_aml_core.features import build_edge_features, build_node_features, make_bidirectional, standardize

log = logging.getLogger("gnn_aml_core.train")
REPORTING_THRESHOLD = 10_000.0


# ------------------------------------------------------------------------------------------ data (torch-free)
def read_frames(data_dir: str):
    d = Path(data_dir)
    if (d / "accounts.csv").exists():
        return pd.read_csv(d / "accounts.csv"), pd.read_csv(d / "transactions.csv")
    if (d / "accounts.parquet").exists():
        return pd.read_parquet(d / "accounts.parquet"), pd.read_parquet(d / "transactions.parquet")
    raise FileNotFoundError(f"no accounts.csv / accounts.parquet in {d}. Run: python -m aml_synth.graph_generator --out {d} --to csv")


def build_graph_arrays(data_dir: str, reporting_threshold: float = REPORTING_THRESHOLD) -> dict:
    accounts, tx = read_frames(data_dir)
    return arrays_from_frames(accounts, tx, reporting_threshold)


def arrays_from_frames(accounts: pd.DataFrame, tx: pd.DataFrame, reporting_threshold: float = REPORTING_THRESHOLD) -> dict:
    x_raw, names = build_node_features(accounts, tx, reporting_threshold)
    x, mean, std = standardize(x_raw)
    idx = {a: i for i, a in enumerate(accounts["account_id"])}
    src = tx["src"].map(idx).to_numpy(dtype=np.int64)
    dst = tx["dst"].map(idx).to_numpy(dtype=np.int64)
    attr, etype = build_edge_features(tx, reporting_threshold)
    src2, dst2, attr2, etype2 = make_bidirectional(src, dst, attr, etype)
    return {
        "x": x, "mean": mean, "std": std, "feature_names": names,
        "src_tx": src, "dst_tx": dst,                                   # original transactions (for the baselines)
        "edge_index": np.stack([src2, dst2]), "edge_attr": attr2, "edge_type": etype2,
        "y": accounts["is_suspicious"].to_numpy(dtype=np.float32), "n_tx": int(len(tx)),
        "account_ids": accounts["account_id"].tolist(), "account_type": accounts["account_type"].tolist(),
        "country": accounts["country"].tolist(), "tx_ids": tx["tx_id"].tolist(),
        "tx_amount": tx["amount"].to_numpy(dtype=np.float32), "tx_timestamp": tx["timestamp"].to_numpy(dtype=np.int64),
        "tx_cross_border": tx["cross_border"].to_numpy(dtype=np.int64),
    }


def make_splits(y: np.ndarray, seed: int):
    """Stratified 60 / 20 / 20 train / validation / test split of the accounts."""
    n = len(y)
    tr, tmp = train_test_split(np.arange(n), test_size=0.4, stratify=y, random_state=seed)
    va, te = train_test_split(tmp, test_size=0.5, stratify=y[tmp], random_state=seed)
    return tr, va, te


def format_table(rows: dict) -> str:
    head = f"{'model':<62}{'AUC':>7}{'PR-AUC':>8}{'prec':>7}{'recall':>8}{'F1':>7}{'FPR':>7}"
    lines = [head, "-" * len(head)]
    for name, m in rows.items():
        lines.append(f"{name:<62}{m['auc_roc']:>7.3f}{m['pr_auc']:>8.3f}{m['precision']:>7.3f}{m['recall']:>8.3f}{m['f1']:>7.3f}{m['fpr']:>7.3f}")
    return "\n".join(lines)


# ------------------------------------------------------------------------------------------------------ training
def fit_gnn(arrays: dict, tr, va, te, *, model_name: str = "gatv2", hidden: int = 64, layers: int = 2, lr: float = 0.005,
            epochs: int = 150, pos_weight: float | None = None, seed: int = 42, patience: int = 0,
            eval_every: int = 5, verbose: bool = True, **model_kwargs) -> dict:
    """Train one GNN and evaluate it. Model selection uses the validation AUC only; the test split is touched once, at the end.

    patience: stop after this many evaluations (each `eval_every` epochs) without a validation improvement; 0 = train all epochs.
    **model_kwargs: extra keyword arguments forwarded to `build_model` beyond hidden/num_layers -- e.g. `edge_dim`
    for a dataset whose edge features are not aml_synth's own EDGE_DIM (see gnn_aml_core.elliptic, which trains
    on the real Elliptic Bitcoin dataset's 1-dimensional time-gap edge feature rather than aml_synth's 5).
    Returns {"report", "best_val_auc", "best_epoch", "epochs_run", "state_dict", "hparams"}; report = AUC, PR-AUC, precision,
    recall, F1, FPR on the test split with the threshold chosen on validation.
    """
    import torch
    from sklearn.metrics import roc_auc_score
    from gnn_aml_core.calibration import brier_score, expected_calibration_error, fit_calibrator
    from gnn_aml_core.models import build_model, class_weighted_bce

    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n, y_np = arrays["x"].shape[0], arrays["y"]
    T = {k: torch.from_numpy(np.array(arrays[k])).to(dev) for k in ("x", "edge_index", "edge_attr", "edge_type", "y")}
    tr_m = torch.zeros(n, dtype=torch.bool, device=dev)          # only the loss needs a mask; evaluation indexes with tr/va/te
    tr_m[torch.from_numpy(tr).to(dev)] = True

    hp = {"hidden": hidden, "num_layers": layers, **model_kwargs}
    model = build_model(model_name, arrays["x"].shape[1], **hp).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    def forward():
        return model(T["x"], T["edge_index"], T["edge_attr"], T["edge_type"])

    best_auc, best_state, best_epoch, stale, epoch = -1.0, None, 0, 0, 0
    for epoch in range(1, epochs + 1):
        model.train()
        opt.zero_grad()
        loss = class_weighted_bce(forward()[tr_m], T["y"][tr_m], pos_weight=pos_weight)
        loss.backward()
        opt.step()
        if epoch % eval_every == 0 or epoch == epochs:
            model.eval()
            with torch.no_grad():
                probs = torch.sigmoid(forward()).cpu().numpy()
            auc = float(roc_auc_score(y_np[va], probs[va]))      # same index array for scores and labels (a boolean mask would reorder)
            if verbose:
                log.info("epoch %3d  loss %.4f  val AUC %.4f", epoch, loss.item(), auc)
            if auc > best_auc + 1e-4:
                best_auc, best_state, best_epoch, stale = auc, copy.deepcopy(model.state_dict()), epoch, 0
            else:
                stale += 1
                if patience and stale >= patience:
                    if verbose:
                        log.info("early stop at epoch %d (no validation improvement for %d evaluations)", epoch, patience)
                    break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        prob = torch.sigmoid(forward()).cpu().numpy()

    # Calibration: fit on the VALIDATION split only, then measure its effect on the untouched test split. This
    # never changes ranking metrics (AUC-ROC, PR-AUC, precision/recall at the chosen threshold all depend only
    # on score ORDER, not absolute value) -- it only makes the score meaningful as a probability, which matters
    # for anything that displays it (a risk gauge, a narrative's "risk score: X%").
    calibrator = fit_calibrator(prob[va], y_np[va])
    calibration_report = {
        "ece_before": expected_calibration_error(prob[te], y_np[te]),
        "ece_after": expected_calibration_error(calibrator(prob[te]), y_np[te]),
        "brier_before": brier_score(prob[te], y_np[te]),
        "brier_after": brier_score(calibrator(prob[te]), y_np[te]),
    }

    return {"report": full_report(y_np[va], prob[va], y_np[te], prob[te]), "best_val_auc": best_auc, "best_epoch": best_epoch,
            "epochs_run": epoch, "state_dict": {k: v.cpu() for k, v in best_state.items()}, "hparams": hp,
            "calibrator": calibrator, "calibration_report": calibration_report}


def main() -> None:
    import torch

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="models")
    ap.add_argument("--model", default="gatv2", choices=["gatv2", "rgcn"])
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--patience", type=int, default=0, help="early stopping: evaluations (every 5 epochs) without improvement; 0 = off")
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--lr", type=float, default=0.005)
    ap.add_argument("--pos-weight", type=float, default=None, help="weight of positive nodes in the loss, e.g. 19.0 (the brief's value); default: #neg/#pos")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--auc-target", type=float, default=0.87)
    ap.add_argument("--no-baselines", action="store_true", help="skip the baseline comparison")
    a = ap.parse_args()

    arrays = build_graph_arrays(a.data)
    n, y_np = arrays["x"].shape[0], arrays["y"]
    log.info("graph: %d nodes, %d transactions (%d directed edges incl. reversed copies), %.2f%% positive nodes, device=%s",
             n, arrays["n_tx"], arrays["edge_index"].shape[1], 100 * y_np.mean(), "cuda" if torch.cuda.is_available() else "cpu")
    tr, va, te = make_splits(y_np, a.seed)

    res = fit_gnn(arrays, tr, va, te, model_name=a.model, hidden=a.hidden, layers=a.layers, lr=a.lr, epochs=a.epochs,
                  pos_weight=a.pos_weight, seed=a.seed, patience=a.patience)
    rep, hp = res["report"], res["hparams"]
    metrics = {
        "val_auc_roc": res["best_val_auc"], "best_epoch": res["best_epoch"], "test_auc_roc": rep["auc_roc"], "test_pr_auc": rep["pr_auc"],
        "test_precision": rep["precision"], "test_recall": rep["recall"], "test_f1": rep["f1"], "test_fpr": rep["fpr"],
        "threshold": rep["threshold"], "auc_target": a.auc_target, "meets_auc_target": bool(rep["auc_roc"] >= a.auc_target),
        **{f"calibration_{k}": v for k, v in res["calibration_report"].items()},
    }

    table = {f"GNN ({a.model.upper()}, best epoch {res['best_epoch']})": rep}
    baselines = {}
    if not a.no_baselines:
        log.info("running baselines on the same split ...")
        baselines = evaluate_baselines(arrays["x"], y_np, tr, va, te, edges=(arrays["src_tx"], arrays["dst_tx"]), seed=a.seed)
        table.update(baselines)
    log.info("TEST SET RESULTS (threshold chosen on validation)\n%s", format_table(table))
    log.info("GNN meets the AUC target of %.2f: %s", a.auc_target, metrics["meets_auc_target"])
    cr = res["calibration_report"]
    log.info("calibration: ECE %.4f -> %.4f, Brier %.4f -> %.4f (raw score is uncalibrated; use 'calibrated_score' at serving time)",
             cr["ece_before"], cr["ece_after"], cr["brier_before"], cr["brier_after"])

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    calibrated_threshold = float(res["calibrator"](metrics["threshold"]))
    torch.save({
        "model_name": a.model, "hparams": hp, "in_dim": arrays["x"].shape[1], "state_dict": res["state_dict"],
        "feature_names": arrays["feature_names"], "mean": torch.from_numpy(arrays["mean"]), "std": torch.from_numpy(arrays["std"]),
        "threshold": metrics["threshold"], "metrics": metrics,
        "calibration": res["calibrator"].to_dict(), "calibrated_threshold": calibrated_threshold,
    }, out / "gnn_model.pt")
    graph = {k: torch.from_numpy(np.array(arrays[k])) for k in ("x", "edge_index", "edge_attr", "edge_type", "y", "tx_amount", "tx_timestamp", "tx_cross_border")}
    graph.update({k: arrays[k] for k in ("account_ids", "account_type", "country", "tx_ids", "feature_names", "n_tx")})
    torch.save(graph, out / "graph.pt")
    (out / "metrics.json").write_text(json.dumps({
        **metrics, "model": a.model, "hparams": hp, "epochs": a.epochs, "seed": a.seed, "n_nodes": n, "n_transactions": arrays["n_tx"],
        "positive_rate": float(y_np.mean()), "baselines": baselines,
    }, indent=2))
    log.info("saved %s, %s and %s", out / "gnn_model.pt", out / "graph.pt", out / "metrics.json")


if __name__ == "__main__":
    main()
