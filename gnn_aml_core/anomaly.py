"""gnn_aml_core.anomaly: Model 2 -- a deep, continuously-learning anomaly detector that is a DELIBERATELY
different kind of model from Model 1 (gnn_aml_core.main's GATv2/RGCN classifier), built to catch a different
kind of threat.

Model 1 learns "what do the laundering typologies I was trained on look like" -- powerful, but it has a
structural blind spot: it can only recognize patterns it was shown, and because the training data
(aml_synth/SynthAML) is a PUBLIC dataset, anyone can study the exact same typologies it was trained on and
design around them. A pattern engineered specifically to avoid every taught typology will sail straight past
Model 1 -- not because the account isn't a mule, but because it's below Model 1's radar by construction.

Model 2 never learns what laundering looks like at all. Instead it is a graph autoencoder (GraphAutoencoder
below) -- a deep neural network trained ONLY to reconstruct what a NORMAL account looks like (its features and
its structural position), using accounts Model 1 currently scores as low-risk as its training population. An
account that reconstructs badly is behaving unlike almost everything else in the real population right now --
whether or not that specific deviation matches any taught typology. This is what catches "below the radar":
nothing here depends on a public catalog of known tricks, so there is nothing for an adversary to study and
route around. It only knows what normal looks like, continuously, from whatever the live graph actually
contains -- never from any synthetic or public dataset, and never retrained from scratch: `anomaly_pass()`
warm-starts from its own last checkpoint every cycle (see train_autoencoder's init_state_dict), so it keeps
accumulating a deeper notion of "normal" indefinitely rather than restarting cold every pass.

Scoping -- "what Model 1 can already see should not stress Model 2": every pass excludes any account Model 1
already flags from BOTH training (so Model 2's notion of normal is never polluted by an account Model 1 itself
considers suspicious) AND alerting (so the two pipelines never duplicate an alert on the same account). Model 2
raises an alert only for something inside Model 1's blind spot -- normal-according-to-Model-1, but
statistically unlike the rest of that same "normal" population.

Alerts are NOT auto-decided. A raised alert starts as status='new' and is explicitly NOT a verdict -- see
backend/main.py's /api/anomalies/* routes for the investigation workflow (claim -> notes -> resolve). Only when
an analyst/investigator resolves an alert as 'confirmed_mule' -- after whatever real-world investigation that
took (see the module docstring on backend/main.py's anomaly routes) -- does that specific, verified account
become a new training example. `sync_confirmed_to_model1()` below is what then teaches Model 1 that exact
pattern, via the SAME warm-start-fine-tune-and-gate pipeline gnn_aml_core.feedback already built
(retrain_and_maybe_promote) -- reusing its evaluation-before-promotion safety net, its audit trail, and its
hot-reload mechanism unchanged. This is a deliberately different, much rarer trigger than
gnn_aml_core.feedback's own routine officer confirm/dismiss stream on Model 1's own alerts: that routine stream
is explicitly NOT used to train anything (a separate, earlier design decision -- every-day confirm/dismiss
clicks on obvious, already-caught cases carry far less information than a deliberately investigated,
previously-invisible pattern, and using them as a training signal would make the model's behaviour drift based
on officer habits rather than genuinely new information). Only a confirmed Model-2 case -- rare, specifically
investigated, proven novel -- ever reaches Model 1's weights.

    python -m gnn_aml_core.anomaly --serve-metrics --metrics-port 9202   # the continuous worker (its own container)
    python -m gnn_aml_core.anomaly --check                               # one pass, print what it found, exit
"""
from __future__ import annotations

import argparse
import logging
import os
import time
from dataclasses import dataclass

from prometheus_client import Counter, Gauge, start_http_server

log = logging.getLogger("anomaly")

ANOMALY_PASSES = Counter("aml_anomaly_passes_total", "Model 2 anomaly-detection passes run")
ANOMALY_ACCOUNTS_SCORED = Counter("aml_anomaly_accounts_scored_total", "Accounts scored for reconstruction error")
ANOMALY_ALERTS_RAISED = Counter("aml_anomaly_alerts_raised_total", "New Model-2 alerts raised (status='new')")
ANOMALY_TRAIN_LOSS = Gauge("aml_anomaly_autoencoder_train_loss", "Most recent autoencoder training loss (lower is better)")
ANOMALY_NORMAL_POPULATION = Gauge("aml_anomaly_normal_population", "Accounts Model 1 currently scores as normal (Model 2's training population)")
ANOMALY_SYNC_TO_MODEL1_TOTAL = Counter("aml_anomaly_sync_to_model1_total", "Confirmed Model-2 cases fed into Model 1's fine-tune")


def _db_url() -> str:
    return os.environ["DATABASE_URL"]


def build_autoencoder(in_dim: int, hidden: int = 64, heads: int = 4, layers: int = 2, dropout: float = 0.2):
    """A graph autoencoder: GATv2Detector's own encoder (embed(), reused unchanged -- same architecture Model 1
    uses to understand graph structure, so Model 2 benefits from the same structural reasoning) feeding a small
    MLP decoder that tries to reconstruct the account's ORIGINAL feature vector from that learned embedding.
    Trained only on accounts Model 1 scores as normal; reconstruction error on any account is the anomaly
    score. Deliberately NOT given any label at all -- this is self-supervised, not classification."""
    import torch.nn as nn

    from gnn_aml_core.models import GATv2Detector

    class GraphAutoencoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = GATv2Detector(in_dim, hidden=hidden, heads=heads, num_layers=layers, dropout=dropout)
            embed_dim = hidden * heads
            self.decoder = nn.Sequential(nn.Linear(embed_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
                                         nn.Linear(hidden, in_dim))

        def forward(self, x, edge_index, edge_attr=None, edge_type=None):
            z = self.encoder.embed(x, edge_index, edge_attr=edge_attr)
            return self.decoder(z), z

    return GraphAutoencoder()


def train_autoencoder(arrays: dict, normal_idx, *, hidden: int = 64, heads: int = 4, layers: int = 2,
                      dropout: float = 0.2, lr: float = 0.001, epochs: int = 20,
                      init_state_dict: dict | None = None, seed: int = 42) -> dict:
    """Trains (or, if init_state_dict is given, CONTINUES training -- never restarts cold) the autoencoder using
    only `normal_idx` rows as the reconstruction target. Returns {"state_dict", "recon_error": per-node
    reconstruction error for EVERY node in arrays, not just normal_idx -- scoring the whole population is the
    point}, "train_loss", "hparams"}."""
    import numpy as np
    import torch

    torch.manual_seed(seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.from_numpy(np.array(arrays["x"])).float().to(dev)
    ei = torch.from_numpy(np.array(arrays["edge_index"])).to(dev)
    ea = torch.from_numpy(np.array(arrays["edge_attr"])).float().to(dev)
    n_idx = torch.from_numpy(np.array(normal_idx)).long().to(dev)

    model = build_autoencoder(x.shape[1], hidden=hidden, heads=heads, layers=layers, dropout=dropout).to(dev)
    if init_state_dict is not None:
        model.load_state_dict(init_state_dict)   # continuous learning: never restarts from scratch
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    loss_fn = torch.nn.MSELoss()
    final_loss = 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        opt.zero_grad()
        recon, _ = model(x, ei, edge_attr=ea)
        loss = loss_fn(recon[n_idx], x[n_idx])
        loss.backward()
        opt.step()
        final_loss = float(loss.item())
        if epoch % 5 == 0 or epoch == epochs:
            log.info("autoencoder epoch %3d  recon loss (normal population) %.5f", epoch, final_loss)

    model.eval()
    with torch.no_grad():
        recon, _ = model(x, ei, edge_attr=ea)
        per_node_error = ((recon - x) ** 2).mean(dim=1).cpu().numpy()

    return {"state_dict": {k: v.cpu() for k, v in model.state_dict().items()}, "recon_error": per_node_error,
            "train_loss": final_loss, "hparams": {"hidden": hidden, "heads": heads, "layers": layers, "dropout": dropout}}


@dataclass
class AnomalyHit:
    account_id: str
    score: float
    threshold: float


def score_population(recon_error, account_ids: list[str], normal_mask, k_std: float = 3.0) -> tuple[list[AnomalyHit], float]:
    """Decides the anomaly threshold from the NORMAL population's own error distribution. Uses the MEDIAN and
    the median absolute deviation (MAD, scaled by 1.4826 so it estimates the standard deviation under a normal
    distribution) rather than the mean/stdev -- a real bug caught in testing: with mean/stdev, the outliers
    THEMSELVES inflate the very statistic used to detect them (a handful of extreme values pull the mean and
    stdev up, which can push the threshold above the outliers and hide them entirely, especially in a small or
    sparsely-anomalous population). Median and MAD barely move in response to a few extreme points, which is
    exactly the robustness anomaly detection needs. Never a fixed global constant either way -- what counts as
    "a lot of reconstruction error" depends on how hard this particular graph's normal accounts are to
    reconstruct right now, which changes over time. Returns (hits among the normal-but-outlier accounts, the
    threshold used) -- hits never include an account Model 1 already flagged; normal_mask is what encodes that
    exclusion."""
    import numpy as np

    normal_errors = recon_error[normal_mask]
    if len(normal_errors) < 2:
        return [], float("inf")
    median = float(np.median(normal_errors))
    mad = float(np.median(np.abs(normal_errors - median)))
    robust_std = mad * 1.4826
    threshold = median + k_std * robust_std if robust_std > 0 else median
    hits = []
    for i, is_normal in enumerate(normal_mask):
        if not is_normal:
            continue
        if recon_error[i] > threshold:
            hits.append(AnomalyHit(account_id=account_ids[i], score=float(recon_error[i]), threshold=threshold))
    return hits, threshold


def _upsert_alert(conn, hit: AnomalyHit) -> bool:
    """Inserts a new 'new' alert for this account unless one is already open (new/under_investigation) --
    returns True if a NEW row was actually created, so the caller can count real new alerts rather than
    no-op re-detections of something already in the queue."""
    existing = conn.execute(
        "SELECT id FROM anomaly_alerts WHERE account_id = %s AND status IN ('new','under_investigation')",
        (hit.account_id,),
    ).fetchone()
    if existing:
        return False
    conn.execute(
        "INSERT INTO anomaly_alerts (account_id, anomaly_score, threshold, status) VALUES (%s, %s, %s, 'new')",
        (hit.account_id, hit.score, hit.threshold),
    )
    return True


def anomaly_pass(*, data_dir: str = "data", out_dir: str = "models", gnn_api: str,
                 hidden: int = 64, heads: int = 4, layers: int = 2, lr: float = 0.001, epochs: int = 20,
                 k_std: float = 3.0) -> dict:
    """One full cycle: score the live graph with Model 1, train/continue-training the autoencoder on whatever
    Model 1 currently calls normal, score reconstruction error for everyone, and raise alerts for normal-but-
    outlier accounts. Persists the autoencoder's weights to out_dir/anomaly_autoencoder.pt so the NEXT call
    warm-starts from here -- this model is never reset to random weights once it exists."""
    import json as _json
    from pathlib import Path

    import numpy as np
    import psycopg
    import torch
    from psycopg.rows import dict_row

    out = Path(out_dir)
    g = torch.load(out / "graph.pt", map_location="cpu", weights_only=False)
    arrays = {k: (v.numpy() if hasattr(v, "numpy") else v) for k, v in g.items()}
    account_ids = list(arrays["account_ids"])

    # Ask Model 1 what it currently thinks is normal vs flagged -- the live, authoritative source, not a
    # locally-recomputed threshold, so Model 2's exclusion always matches what Model 1 is ACTUALLY alerting on
    # right now, including after a hot-reload promotion changed Model 1's weights.
    import httpx
    resp = httpx.post(f"{gnn_api.rstrip('/')}/predict", json={"account_ids": account_ids}, timeout=120)
    resp.raise_for_status()
    results = {r["account_id"]: r["flagged"] for r in resp.json()["results"]}
    flagged_mask = np.array([results.get(aid, False) for aid in account_ids], dtype=bool)
    normal_mask = ~flagged_mask
    ANOMALY_NORMAL_POPULATION.set(int(normal_mask.sum()))

    ckpt_path = out / "anomaly_autoencoder.pt"
    init_state_dict, saved_hp = None, {}
    if ckpt_path.exists():
        prev = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        init_state_dict, saved_hp = prev["state_dict"], prev.get("hparams", {})
        hidden, heads, layers = saved_hp.get("hidden", hidden), saved_hp.get("heads", heads), saved_hp.get("layers", layers)

    normal_idx = np.where(normal_mask)[0]
    res = train_autoencoder(arrays, normal_idx, hidden=hidden, heads=heads, layers=layers, lr=lr, epochs=epochs,
                            init_state_dict=init_state_dict)
    torch.save({"state_dict": res["state_dict"], "hparams": res["hparams"], "ts": time.time()}, ckpt_path)
    ANOMALY_TRAIN_LOSS.set(res["train_loss"])
    ANOMALY_ACCOUNTS_SCORED.inc(len(account_ids))

    hits, threshold = score_population(res["recon_error"], account_ids, normal_mask, k_std=k_std)

    new_count = 0
    with psycopg.connect(_db_url(), row_factory=dict_row) as conn:
        for hit in hits:
            if _upsert_alert(conn, hit):
                new_count += 1
    ANOMALY_ALERTS_RAISED.inc(new_count)
    log.info("anomaly pass: %d accounts scored, %d in Model 1's blind spot above threshold %.5f, %d NEW alert(s) raised",
             len(account_ids), len(hits), threshold, new_count)
    ANOMALY_PASSES.inc()
    return {"scored": len(account_ids), "hits": len(hits), "new_alerts": new_count, "threshold": threshold,
            "train_loss": res["train_loss"]}


def sync_confirmed_to_model1(*, data_dir: str = "data", out_dir: str = "models", gnn_api: str | None = None,
                             gnn_admin_token: str | None = None, feedback_weight: int = 5,
                             min_improvement: float = 0.0, finetune_lr: float = 0.0005,
                             finetune_epochs: int = 30, finetune_patience: int = 5) -> list[dict]:
    """Finds every anomaly_alerts row resolved as 'confirmed_mule' that has not yet been fed into Model 1
    (trained_at IS NULL), builds them into the SAME DecisionRecord shape gnn_aml_core.feedback already knows how
    to turn into a feedback dataset, and runs them through retrain_and_maybe_promote() unchanged -- same
    fine-tune-from-current-weights behaviour, same evaluate-before-promote gate, same audit trail, same
    hot-reload. This is the ONLY path that trains Model 1 from this module; it never runs on an unresolved or
    'cleared' alert, and it is a one-way street: a 'cleared' alert is logged as evidence this module never acts
    on, never retried, never silently promoted later."""
    import psycopg
    from psycopg.rows import dict_row

    from gnn_aml_core.feedback import DecisionRecord, retrain_and_maybe_promote

    with psycopg.connect(_db_url(), row_factory=dict_row) as conn:
        rows = conn.execute(
            "SELECT * FROM anomaly_alerts WHERE status = 'confirmed_mule' AND trained_at IS NULL"
        ).fetchall()

    if not rows:
        log.info("sync_confirmed_to_model1: nothing new to teach Model 1 (no unsynced confirmed_mule cases)")
        return []

    records = [DecisionRecord(account_id=r["account_id"], predicted_score=r["anomaly_score"],
                              threshold=r["threshold"], decision="confirmed") for r in rows]
    log.info("sync_confirmed_to_model1: %d newly-confirmed Model-2 case(s) ready to teach Model 1 -- %s",
             len(records), [r.account_id for r in records])

    outcome = retrain_and_maybe_promote(records, data_dir=data_dir, out_dir=out_dir,
                                        feedback_weight=feedback_weight, min_improvement=min_improvement,
                                        gnn_api=gnn_api, gnn_admin_token=gnn_admin_token,
                                        finetune_lr=finetune_lr, finetune_epochs=finetune_epochs,
                                        finetune_patience=finetune_patience)
    outcome["source"] = "model2_confirmed_anomalies"
    outcome["account_ids"] = [r["account_id"] for r in rows]

    # Mark these synced regardless of promote/reject -- a REJECTED candidate still means "we tried teaching
    # Model 1 this pattern and it didn't help enough to promote," which should not be retried every cycle on
    # the same already-tried cases; it's logged in retrain_history.jsonl either way, same as every other
    # retrain_and_maybe_promote call in this codebase.
    with psycopg.connect(_db_url(), row_factory=dict_row) as conn:
        for r in rows:
            conn.execute("UPDATE anomaly_alerts SET trained_at = now() WHERE id = %s", (r["id"],))
    ANOMALY_SYNC_TO_MODEL1_TOTAL.inc(len(rows))
    return [outcome]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="models")
    ap.add_argument("--gnn-api", default=os.getenv("GNN_API_URL", "http://gnn-detection-api:8000"))
    ap.add_argument("--gnn-admin-token", default=os.getenv("GNN_ADMIN_TOKEN"))
    ap.add_argument("--check", action="store_true", help="run one pass and exit, instead of looping forever")
    ap.add_argument("--serve-metrics", action="store_true", help="run forever, scraping-ready on --metrics-port")
    ap.add_argument("--metrics-port", type=int, default=9202)
    ap.add_argument("--loop-interval", type=int, default=int(os.getenv("ANOMALY_INTERVAL_S", 300)))
    ap.add_argument("--sync-interval", type=int, default=int(os.getenv("ANOMALY_SYNC_INTERVAL_S", 60)),
                    help="how often (seconds) to check for newly-confirmed Model-2 cases to teach Model 1 -- "
                         "kept short and separate from --loop-interval, since a confirmed case should reach "
                         "Model 1 promptly, not wait for the next slow anomaly-detection pass")
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--k-std", type=float, default=float(os.getenv("ANOMALY_K_STD", 3.0)),
                    help="an account is anomalous if its reconstruction error exceeds the normal "
                         "population's own mean + k_std * stdev -- NOT a fixed global number, since what "
                         "counts as unusual depends on how hard THIS graph's normal accounts are to "
                         "reconstruct right now")
    ap.add_argument("--min-improvement", type=float, default=float(os.getenv("FEEDBACK_MIN_IMPROVEMENT", 0.0)))
    ap.add_argument("--finetune-lr", type=float, default=float(os.getenv("FEEDBACK_FINETUNE_LR", 0.0005)))
    ap.add_argument("--finetune-epochs", type=int, default=int(os.getenv("FEEDBACK_FINETUNE_EPOCHS", 30)))
    ap.add_argument("--finetune-patience", type=int, default=int(os.getenv("FEEDBACK_FINETUNE_PATIENCE", 5)))
    a = ap.parse_args()

    if a.check:
        result = anomaly_pass(data_dir=a.data, out_dir=a.out, gnn_api=a.gnn_api, hidden=a.hidden,
                              epochs=a.epochs, k_std=a.k_std)
        print(result)
        return

    start_http_server(a.metrics_port)
    log.info("Model 2 (anomaly detector) serving metrics on :%d/metrics -- anomaly pass every %ds, "
             "confirmed-case sync check every %ds", a.metrics_port, a.loop_interval, a.sync_interval)
    last_anomaly_pass = 0.0
    while True:
        now = time.time()
        try:
            if now - last_anomaly_pass >= a.loop_interval:
                anomaly_pass(data_dir=a.data, out_dir=a.out, gnn_api=a.gnn_api, hidden=a.hidden,
                            epochs=a.epochs, k_std=a.k_std)
                last_anomaly_pass = now
            sync_confirmed_to_model1(data_dir=a.data, out_dir=a.out, gnn_api=a.gnn_api,
                                     gnn_admin_token=a.gnn_admin_token, min_improvement=a.min_improvement,
                                     finetune_lr=a.finetune_lr, finetune_epochs=a.finetune_epochs,
                                     finetune_patience=a.finetune_patience)
        except Exception as exc:  # noqa: BLE001 -- this is a long-lived process; one bad cycle (DB hiccup,
                                  # gnn-detection-api briefly unreachable, etc.) must never take it down
            log.error("anomaly worker cycle failed (%s: %s) -- will retry next cycle", exc.__class__.__name__, exc)
        time.sleep(a.sync_interval)


if __name__ == "__main__":
    main()
