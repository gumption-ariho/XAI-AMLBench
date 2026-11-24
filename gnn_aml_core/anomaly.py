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
import json
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


def _robust_z(values, normal_mask):
    """Scores each node against the NORMAL population's own median and MAD (scaled to a standard deviation), so
    three differently-scaled neural signals can be compared and combined fairly."""
    import numpy as np

    ref = values[normal_mask]
    med = float(np.median(ref))
    mad = float(np.median(np.abs(ref - med))) * 1.4826
    if mad < 1e-12:
        mad = float(ref.std())
    if mad < 1e-12:
        mad = 1.0
    return (values - med) / mad


def _structure_error(z, edge_index, n_nodes):
    """Per-node STRUCTURE error of a graph autoencoder: how badly the learned embeddings predict this node's own
    real connections (a link-reconstruction score, sigmoid of the embedding dot product). An account whose
    counterparties look nothing like what the network expects for a normal account -- even when its own numbers
    look ordinary -- scores high here."""
    import torch
    import torch.nn.functional as F

    src, dst = edge_index[0], edge_index[1]
    e = F.softplus(-(z[src] * z[dst]).sum(dim=1))          # -log sigmoid(dot) for every real edge
    tot = torch.zeros(n_nodes, device=z.device).index_add_(0, src, e).index_add_(0, dst, e)
    cnt = torch.zeros(n_nodes, device=z.device).index_add_(0, src, torch.ones_like(e)).index_add_(0, dst, torch.ones_like(e))
    return tot / cnt.clamp_min(1.0)


def train_autoencoder(arrays: dict, normal_idx, *, hidden: int = 64, heads: int = 4, layers: int = 2,
                      dropout: float = 0.2, lr: float = 0.001, epochs: int = 20,
                      init_state_dict: dict | None = None, seed: int = 42, mode: str = "classic",
                      trim: float = 0.0) -> dict:
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
    ensemble = mode == "ensemble"
    n_nodes = x.shape[0]
    normal_t = torch.zeros(n_nodes, dtype=torch.bool, device=dev)
    normal_t[n_idx] = True
    # edges where BOTH ends are normal accounts -- the only ones the "what normal connections look like" loss may use
    ok = normal_t[ei[0]] & normal_t[ei[1]]
    train_edges = ei[:, ok] if ensemble else None
    final_loss = 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        opt.zero_grad()
        recon, z = model(x, ei, edge_attr=ea)
        if trim > 0 and epoch > 2:
            # TRIMMED training: the accounts that reconstruct WORST right now (the most suspicious-looking ones
            # inside the "normal" group) are left out of this step's loss, so a hidden mule that Model 1 missed
            # is never taught to the network as normal. It stays unexplained, and is therefore still caught.
            per = ((recon[n_idx] - x[n_idx]) ** 2).mean(dim=1)
            keep = per.detach().argsort()[: max(1, int(len(per) * (1.0 - trim)))]
            loss = per[keep].mean()
        else:
            loss = loss_fn(recon[n_idx], x[n_idx])
        if ensemble:
            # (1) embedding signal: pull normal accounts' embeddings toward their common centre (deep one-class)
            zn = z[n_idx]
            loss = loss + 0.1 * ((zn - zn.mean(dim=0, keepdim=True).detach()) ** 2).mean()
            # (2) structure signal: real normal-normal connections should score high, random pairs low
            if train_edges is not None and train_edges.shape[1] > 0:
                m = train_edges.shape[1]
                pick = torch.randperm(m, device=dev)[:20000]
                es, ed = train_edges[0, pick], train_edges[1, pick]
                pos = (z[es] * z[ed]).sum(dim=1)
                neg = (z[es] * z[n_idx[torch.randint(0, len(n_idx), (len(pick),), device=dev)]]).sum(dim=1)
                loss = loss + 0.1 * (torch.nn.functional.softplus(-pos).mean() + torch.nn.functional.softplus(neg).mean())
        loss.backward()
        opt.step()
        final_loss = float(loss.item())
        if epoch % 5 == 0 or epoch == epochs:
            log.info("autoencoder epoch %3d  recon loss (normal population) %.5f", epoch, final_loss)

    model.eval()
    with torch.no_grad():
        recon, z = model(x, ei, edge_attr=ea)
        sq = (recon - x) ** 2
        per_node_error = sq.mean(dim=1).cpu().numpy()
        components = {"feature": per_node_error}
        if ensemble:
            try:
                # per-feature scaling so no single noisy column dominates the feature signal
                scale = sq[n_idx].mean(dim=0).clamp_min(1e-8)
                feat = (sq / scale).mean(dim=1).cpu().numpy()
                centre = z[n_idx].mean(dim=0, keepdim=True)
                emb = ((z - centre) ** 2).sum(dim=1).cpu().numpy()
                struct = _structure_error(z, ei, n_nodes).cpu().numpy()
                normal_np = normal_t.cpu().numpy()
                zs = np.stack([_robust_z(feat, normal_np), _robust_z(emb, normal_np), _robust_z(struct, normal_np)])
                # one strong signal OR several moderate ones: half the strongest + half the average, kept >= 0
                combined = np.clip(0.5 * zs.max(axis=0) + 0.5 * zs.mean(axis=0), -3.0, None) + 3.0
                components = {"feature": feat, "embedding": emb, "structure": struct}
                per_node_error = combined
            except Exception:   # never let the extra signals take the detector down: fall back to the classic score
                log.exception("ensemble scoring failed; falling back to the classic reconstruction-error score")

    return {"state_dict": {k: v.cpu() for k, v in model.state_dict().items()}, "recon_error": per_node_error,
            "components": components, "mode": mode,
            "train_loss": final_loss, "hparams": {"hidden": hidden, "heads": heads, "layers": layers, "dropout": dropout}}


@dataclass
class AnomalyHit:
    account_id: str
    score: float
    threshold: float
    reason: dict | None = None      # plain-language "why flagged", shown to the analyst


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


_FRIENDLY = {
    "out_deg": "number of payments it sends", "in_deg": "number of payments it receives",
    "out_uniq": "number of different accounts it pays", "in_uniq": "number of different accounts that pay it",
    "out_amt_sum": "total money sent", "in_amt_sum": "total money received",
    "out_amt_mean": "typical amount sent", "in_amt_mean": "typical amount received",
    "out_amt_max": "largest payment sent", "in_amt_max": "largest payment received",
    "out_amt_std": "how much its sent amounts vary", "in_amt_std": "how much its received amounts vary",
    "flow_ratio": "money in versus money out", "retained_frac": "how much money it keeps",
    "xb_out_ratio": "share of payments sent abroad", "xb_in_ratio": "share of payments received from abroad",
    "n_cp_countries": "number of countries it deals with", "near_thr_cnt": "payments just under the reporting limit",
    "near_thr_ratio": "share of payments just under the reporting limit", "burst_6h": "many payments in a few hours",
    "active_span_h": "how long it was active", "age_days": "account age", "is_offshore": "registered offshore",
    "kyc_risk": "customer risk rating", "round_amt_ratio": "share of round amounts",
    "micro_tx_ratio": "share of very small payments", "decimal_precision_ratio": "unusually precise amounts",
    "volume_shift_ratio": "sudden change in volume", "avg_amt_shift_ratio": "sudden change in typical amount",
    "tax_haven_cp_ratio": "dealings with tax-haven accounts", "hour_concentration": "payments all at the same hours",
    "pass_through_match_ratio": "money passed straight through", "high_risk_dest_ratio": "payments to high-risk merchants",
    "shared_memo_ratio": "the same payment reference as unrelated accounts", "dispute_rate": "share of payments disputed",
    "counterparty_registration_cluster_ratio": "paying accounts that were all opened together",
}


def explain_reason(z: dict, top_features: list[str]) -> dict:
    """Turns the three neural signals into one plain sentence an analyst can read in a second."""
    parts = []
    if z.get("feature", 0) >= 3:
        parts.append("its own activity is unlike normal accounts")
    if z.get("embedding", 0) >= 3:
        parts.append("it sits far from where normal accounts cluster")
    if z.get("structure", 0) >= 3:
        parts.append("the accounts it deals with are unusual for a normal account")
    if not parts:
        parts.append("several small differences from normal add up")
    what = ", ".join(_FRIENDLY.get(f, f.replace("_", " ")) for f in top_features)
    text = parts[0].capitalize() + ("; " + "; ".join(parts[1:]) if len(parts) > 1 else "") + "."
    if what:
        text += f" Biggest differences: {what}."
    return {"signals": {k: round(float(v), 2) for k, v in z.items()}, "top_features": top_features, "text": text}


def ai_verdict(z: dict, streak: int, m1_prob: float | None) -> dict:
    """The two AI models' evidence, weighed into ONE recommendation so a person confirms instead of working it out.
    Inputs are what the models themselves produced: Model 2's three signals (robust z-scores), how many passes in
    a row this account has stayed anomalous, and Model 1's calibrated probability for the same account.
    This is a transparent rule over those numbers (not a third trained model), and its confidence is a score from
    how much evidence agrees, not a measured probability. Verdicts: likely_normal | likely_suspicious | unsure."""
    zmax = max(z.values()) if z else 0.0
    hi = sum(1 for v in z.values() if v >= 3.0)
    m1 = 0.0 if m1_prob is None else float(m1_prob)
    sus = (2 if hi >= 2 else 0) + (2 if zmax >= 6 else 0) + (1 if streak >= 3 else 0) + (2 if m1 >= 0.25 else 0) + (1 if m1 >= 0.4 else 0)
    nor = (1 if hi <= 1 else 0) + (2 if zmax < 4.5 else 0) + (2 if m1 < 0.10 else 0) + (1 if streak <= 2 else 0)
    if sus >= 5:
        return {"verdict": "likely_suspicious", "confidence": min(95, 55 + 8 * sus),
                "headline": f"{hi} of 3 anomaly signals agree" + (" and the main model also leans risky" if m1 >= 0.25 else "")
                            + (f"; it has stayed unusual for {streak} checks" if streak >= 3 else "") + ". Worth investigating."}
    if sus <= 1 and nor >= 4:
        return {"verdict": "likely_normal", "confidence": min(95, 50 + 8 * nor),
                "headline": "Only a weak difference from normal" + (" and the main model sees low risk" if m1 < 0.10 else "") + ". Looks like a normal account."}
    return {"verdict": "unsure", "confidence": 50,
            "headline": "The two models do not clearly agree" + (f" (main model risk {m1:.0%})" if m1_prob is not None else "") + ". This one needs your judgement."}


def _verified_labels() -> tuple[set[str], set[str]]:
    """What people have actually VERIFIED, as opposed to what Model 1 merely scored low: returns
    (trusted_normal, known_bad) account ids. trusted_normal = accounts an analyst cleared in the anomaly queue
    or dismissed as a false alarm on Model 1's queue. known_bad = accounts confirmed as mules in either queue.
    Never raises -- if the database cannot be read, the detector simply trains as before."""
    try:
        import psycopg
        from psycopg.rows import dict_row

        with psycopg.connect(_db_url(), row_factory=dict_row) as conn:
            trusted = {r["account_id"] for r in conn.execute(
                "SELECT account_id FROM anomaly_alerts WHERE status = 'cleared' "
                "UNION SELECT account_id FROM alerts WHERE status = 'dismissed'").fetchall()}
            bad = {r["account_id"] for r in conn.execute(
                "SELECT account_id FROM anomaly_alerts WHERE status = 'confirmed_mule' "
                "UNION SELECT account_id FROM alerts WHERE status = 'confirmed'").fetchall()}
        return trusted - bad, bad
    except Exception:
        log.warning("could not read verified labels; training on Model 1's view of normal only", exc_info=True)
        return set(), set()


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
    # Do not hand the analysts the same account again right after they dealt with it: an account they cleared
    # (or already confirmed) inside the cooldown is only re-raised if it now looks clearly WORSE than before
    # (score over 1.5x the earlier one). Without this, clearing an alert would just re-create it next pass.
    cooldown_days = int(os.getenv("ANOMALY_CLEARED_COOLDOWN_DAYS", "30"))
    if cooldown_days > 0:
        prior = conn.execute(
            "SELECT status, anomaly_score FROM anomaly_alerts WHERE account_id = %s "
            "AND status IN ('cleared','confirmed_mule') "
            "AND COALESCE(updated_at, created_at) > now() - (%s * interval '1 day') ORDER BY id DESC LIMIT 1",
            (hit.account_id, cooldown_days),
        ).fetchone()
        if prior:
            if prior["status"] == "confirmed_mule" or hit.score <= float(prior["anomaly_score"]) * 1.5:
                return False
    if hit.reason:
        conn.execute(
            "INSERT INTO anomaly_alerts (account_id, anomaly_score, threshold, status, reason) "
            "VALUES (%s, %s, %s, 'new', %s::jsonb)",
            (hit.account_id, hit.score, hit.threshold, json.dumps(hit.reason)),
        )
    else:
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
    m1_prob = {r["account_id"]: r.get("calibrated_score", r.get("score")) for r in resp.json()["results"]}
    flagged_mask = np.array([results.get(aid, False) for aid in account_ids], dtype=bool)
    normal_mask = ~flagged_mask
    ANOMALY_NORMAL_POPULATION.set(int(normal_mask.sum()))

    # SOURCE OF "NORMAL". "reference" (default): Model 2 is trained ONCE on a synthetic population that is normal by
    # construction (gnn_aml_core.anomaly_reference) and then frozen -- live accounts are only ever SCORED against
    # it, never trained on, so a mule that Model 1 missed can never be taught to the network as normal.
    # "live" keeps the old behaviour (retrain every pass on whatever Model 1 scores low).
    source = os.getenv("ANOMALY_SOURCE", "reference").strip().lower()
    hits, threshold = [], float("inf")
    res = {"train_loss": 0.0}          # the frozen reference does no training in a pass
    scored = None
    if source == "reference":
        from gnn_aml_core import anomaly_reference as ref_mod
        if not (out / "anomaly_reference.pt").exists():
            log.warning("no anomaly_reference.pt yet -- building the normal-only reference model now (one-off)")
            ref_mod.build_reference(str(out), n_accounts=int(os.getenv("ANOMALY_REF_ACCOUNTS", "20000")),
                                    epochs=int(os.getenv("ANOMALY_REF_EPOCHS", "150")))
        scored = ref_mod.score_live(str(out), arrays, k_std)
        if scored is None:
            raise RuntimeError("anomaly reference model is missing and could not be built")
        scores, threshold, extras = scored
        ANOMALY_ACCOUNTS_SCORED.inc(len(account_ids))
        trusted_ids, bad_ids = _verified_labels()          # used only to leave already-confirmed mules out of the queue
        id_index = {a: i for i, a in enumerate(account_ids)}
        for a in bad_ids:
            if a in id_index:
                normal_mask[id_index[a]] = False
        hits = []
        for i in np.where(normal_mask & (scores > threshold))[0]:
            try:
                pf = extras["per_feature"][i]
                top = [extras["feature_names"][j] for j in np.argsort(-pf)[:3]]
                reason = explain_reason({k: extras["z"][k][i] for k in extras["z"]}, top)
            except Exception:
                reason = None
            hits.append(AnomalyHit(account_id=account_ids[i], score=float(scores[i]), threshold=threshold, reason=reason))
        log.info("scored %d live accounts against the frozen normal-only reference; threshold %.3f; %d above it",
                 len(account_ids), threshold, len(hits))
    else:
        ckpt_path = out / "anomaly_autoencoder.pt"
        init_state_dict, saved_hp = None, {}
        if ckpt_path.exists():
            prev = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            init_state_dict, saved_hp = prev["state_dict"], prev.get("hparams", {})
            hidden, heads, layers = saved_hp.get("hidden", hidden), saved_hp.get("heads", heads), saved_hp.get("layers", layers)

        # Teach "normal" from what people VERIFIED, not only from what Model 1 scored low: accounts confirmed as mules
        # never count as normal, and accounts analysts cleared or dismissed are given extra weight (seen several
        # times per training pass), so every legitimate-but-unusual pattern they found widens what counts as normal
        # and stops being flagged. This is how the analysts' work improves the detector instead of being repeated.
        trusted_ids, bad_ids = _verified_labels()
        id_index = {a: i for i, a in enumerate(account_ids)}
        for a in bad_ids:
            if a in id_index:
                normal_mask[id_index[a]] = False
        normal_idx = np.where(normal_mask)[0]
        trusted_idx = np.array([id_index[a] for a in trusted_ids if a in id_index and normal_mask[id_index[a]]], dtype=int)
        weight = max(1, int(os.getenv("ANOMALY_TRUSTED_WEIGHT", "5")))
        if len(trusted_idx) and weight > 1:
            normal_idx = np.concatenate([normal_idx] + [trusted_idx] * (weight - 1))
        log.info("training population: %d accounts (%d analyst-verified normal weighted x%d, %d known-bad excluded)",
                 len(normal_idx), len(trusted_idx), weight, len(bad_ids))
        res = train_autoencoder(arrays, normal_idx, hidden=hidden, heads=heads, layers=layers, lr=lr, epochs=epochs,
                                init_state_dict=init_state_dict,
                                mode=os.getenv("ANOMALY_MODE", "ensemble").strip().lower(),
                                trim=min(0.5, max(0.0, float(os.getenv("ANOMALY_TRIM_FRACTION", "0.05")))))
        torch.save({"state_dict": res["state_dict"], "hparams": res["hparams"], "ts": time.time()}, ckpt_path)
        ANOMALY_TRAIN_LOSS.set(res["train_loss"])
        ANOMALY_ACCOUNTS_SCORED.inc(len(account_ids))

        hits, threshold = score_population(res["recon_error"], account_ids, normal_mask, k_std=k_std)

    # Keep the analysts' queue short and meaningful: (1) persistence -- an account must stay above the threshold
    # for ANOMALY_MIN_PASSES passes in a row, so a one-off blip never reaches a person; (2) strongest first,
    # capped at ANOMALY_MAX_NEW_PER_PASS new alerts per pass, so a bad day cannot bury the team.
    min_passes = max(1, int(os.getenv("ANOMALY_MIN_PASSES", "2")))
    max_new = max(1, int(os.getenv("ANOMALY_MAX_NEW_PER_PASS", "20")))
    streak_path = out / "anomaly_streaks.json"
    try:
        prev_streaks = json.loads(streak_path.read_text()) if streak_path.exists() else {}
    except Exception:
        prev_streaks = {}
    streaks = {h.account_id: int(prev_streaks.get(h.account_id, 0)) + 1 for h in hits}   # misses reset to zero
    try:
        streak_path.write_text(json.dumps(streaks))
    except Exception:
        log.warning("could not save the anomaly streak file; persistence will restart next pass")
    eligible = sorted((h for h in hits if streaks[h.account_id] >= min_passes), key=lambda h: h.score, reverse=True)
    for h in eligible:       # the AI's recommendation, so the person confirms instead of working it out
        if h.reason and h.reason.get("signals"):
            try:
                h.reason["ai"] = ai_verdict(h.reason["signals"], streaks[h.account_id], m1_prob.get(h.account_id))
            except Exception:
                log.warning("could not compute the AI verdict for %s", h.account_id, exc_info=True)

    new_count = 0
    with psycopg.connect(_db_url(), row_factory=dict_row) as conn:
        conn.execute("ALTER TABLE anomaly_alerts ADD COLUMN IF NOT EXISTS reason JSONB")
        for hit in eligible:
            if new_count >= max_new:
                break
            if _upsert_alert(conn, hit):
                new_count += 1
        # Optional hands-off mode: a high-confidence "likely normal" nobody has touched for N hours is cleared by
        # the AI itself (logged as such, never used for training). 0 = off.
        auto_h = int(os.getenv("ANOMALY_AUTO_CLEAR_NORMAL_AFTER_HOURS", "0"))
        if auto_h > 0:
            conn.execute(
                "UPDATE anomaly_alerts SET status='cleared', updated_at=now(), "
                "investigation_log = investigation_log || jsonb_build_array(jsonb_build_object("
                "'ts', extract(epoch from now()), 'by', 'AI', 'status', 'cleared', "
                "'note', 'Auto-cleared: the AI judged this a normal account and nobody objected within the review window')) "
                "WHERE status = 'new' AND reason->'ai'->>'verdict' = 'likely_normal' "
                "AND (reason->'ai'->>'confidence')::int >= 85 AND created_at < now() - (%s * interval '1 hour')",
                (auto_h,))
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
    ap.add_argument("--k-std", type=float, default=float(os.getenv("ANOMALY_K_STD", 3.5)),
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
