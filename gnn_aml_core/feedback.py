"""gnn_aml_core.feedback: turns real officer decisions (already stored in Postgres by backend/main.py's
/alerts/{id}/decision endpoint) into two things a static, one-time-trained model cannot give you on its own:

  1. A measurable drift signal. Real launderers adapt once they learn what gets flagged, so a model trained
     once and never revisited goes stale -- but "stale" is not a fixed timeline, and retraining on a blind
     schedule (every week, regardless) is not a real answer to that. `disagreement_rate` measures, from actual
     officer decisions, how often the model's original flagged/not-flagged call turned out to be wrong. Rising
     disagreement is a concrete, honest trigger: it tells you retraining is actually due, rather than guessing.

  2. A real feedback dataset. Every confirmed alert is a real, officer-verified positive; every dismissed
     alert is a real, officer-verified negative -- ground truth from actual usage at this specific
     institution, not borrowed from a different bank's data (SynthAML) or a different domain (Elliptic) or
     invented from scratch (aml_synth). `retrain_with_feedback` blends these into the next training run.

This module never talks to Postgres directly -- it operates on plain DecisionRecord objects -- so all of its
logic (drift measurement, dataset assembly) is fully testable without a live database. Only `read_decisions_from_db`
needs one, and it is a single, small, swappable function.

    python -m gnn_aml_core.feedback --check                     # just report the current drift, do nothing else
    python -m gnn_aml_core.feedback --data data --out models    # retrain if warranted, using the current graph

Auto-retrain, gated before deploy (--serve-metrics --auto-retrain): closes the loop fully -- when the drift
trigger above fires, retrain_and_maybe_promote() FINE-TUNES the model already in production (warm-starting from
its exact saved weights, a short run at a low learning rate -- see fit_gnn's init_state_dict param and
--finetune-lr/--finetune-epochs/--finetune-patience) rather than training a brand-new model from random weights.
It then EVALUATES that fine-tuned candidate against the exact same held-out test split as the CURRENT production
model (not the candidate's own reported number against some other split, and not trusting the current model's
originally-stored metric, which could be stale or from a different split seed -- see evaluate_checkpoint()). The
candidate is promoted into production ONLY if it beats the current model's test AUC-ROC by at least
--min-improvement (default: any improvement at all, however small, beats nothing). If it does not beat the
current model, NOTHING changes on disk -- the current model keeps serving, this is logged and counted (not
silently discarded), and the next drift-trigger cycle will try again with more accumulated feedback. This
mirrors the reasoning that already kept retraining itself a manual-only action for years: an unattended process
is fine to DECIDE when retraining is warranted and even to DO the retraining work, but letting it skip
validation and replace a working model with a worse one is a materially bigger, riskier decision for a
compliance system -- so that step stays gated on a real comparison, every time, with no exceptions and no
"it's probably fine."

On promotion: the current checkpoint is backed up (gnn_model.pt.bak-<timestamp>, never deleted automatically),
the new one is written to the same path gnn-detection-api reads, and that service is asked to hot-swap it in via
POST /admin/reload-model -- no container restart, no served-traffic gap (see gnn_aml_core/main.py's module
docstring for how the prediction cache is kept correct across that swap). If the live reload call fails (service
unreachable, wrong token), the new checkpoint is still safely on disk and gets picked up on that service's next
restart regardless -- promotion-on-disk and live-reload are two separate, independently-recoverable steps, not
one atomic operation that can half-fail. Every promotion AND every rejection is appended to
models/retrain_history.jsonl -- an auditable record of every time the live model changed (or didn't), and why,
which a compliance system needs to be able to produce on demand.

    python -m gnn_aml_core.feedback --serve-metrics --auto-retrain --gnn-api http://127.0.0.1:8001
"""
from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass

from prometheus_client import Counter, Gauge, start_http_server

log = logging.getLogger("gnn_aml_core.feedback")

# A real gap, found during an observability review: this module's drift signal (officer disagreement rate,
# whether a retrain gets triggered) was only ever visible to whoever happened to read this CLI tool's own
# stdout -- completely invisible to the monitoring stack, even though the document's own architecture promised
# "model data drift values" scraped into Grafana. These three are updated every time should_retrain() runs
# (see main()'s --serve-metrics loop below) and exposed for Prometheus to scrape, matching the same
# prometheus_client pattern already used by gnn_aml_core.main and xai_explainer.main.
DISAGREEMENT_RATE = Gauge("aml_officer_disagreement_rate", "Current officer disagreement rate (None-safe: only "
                         "set once at least min_decisions real decisions exist)")
FEEDBACK_DECISIONS_TOTAL = Gauge("aml_feedback_decisions_total", "Officer decisions available for drift detection")
RETRAIN_TRIGGERED_TOTAL = Counter("aml_retrain_triggered_total", "Times the disagreement-rate threshold has "
                                 "been met or exceeded, since this process started")
# --auto-retrain only: ATTEMPTS counts every automatic retrain this process has run; PROMOTED and REJECTED
# split that by outcome, so "is auto-retrain actually promoting anything, or always rejecting" is visible at a
# glance on a dashboard without reading logs or retrain_history.jsonl.
AUTO_RETRAIN_ATTEMPTS = Counter("aml_auto_retrain_attempts_total", "Automatic retrains run (--auto-retrain only)")
AUTO_RETRAIN_PROMOTED = Counter("aml_auto_retrain_promoted_total", "Automatic retrains that beat the current "
                               "model and were promoted into production")
AUTO_RETRAIN_REJECTED = Counter("aml_auto_retrain_rejected_total", "Automatic retrains that did NOT beat the "
                               "current model and were discarded -- the current model kept serving")


@dataclass
class DecisionRecord:
    """One officer decision on one previously-flagged (or force-reviewed) account."""
    account_id: str
    predicted_score: float   # the raw model score at the time this account was scanned
    threshold: float         # the decision threshold in effect at that time
    decision: str            # "confirmed" (officer agrees it is genuinely suspicious) or "dismissed" (officer disagrees)


def true_label(record: DecisionRecord) -> float:
    """The real, officer-verified ground truth: 1.0 if confirmed suspicious, 0.0 if dismissed as a false positive."""
    if record.decision == "confirmed":
        return 1.0
    if record.decision == "dismissed":
        return 0.0
    raise ValueError(f"unknown decision '{record.decision}' (expected 'confirmed' or 'dismissed')")


def model_agreed(record: DecisionRecord) -> bool:
    """Did the model's original flagged/not-flagged call match what the officer later confirmed?

    A real, honest blind spot worth naming rather than hiding: this only sees accounts that reached an officer
    in the first place (flagged, or force-reviewed). An account the model never flagged and no one ever
    force-reviewed contributes no signal here at all -- so a rising disagreement rate detects the model
    becoming too trigger-happy (more false positives) faster and more reliably than it detects the model
    becoming too lax (more false negatives it never surfaced for anyone to catch)."""
    predicted_positive = record.predicted_score >= record.threshold
    actual_positive = true_label(record) == 1.0
    return predicted_positive == actual_positive


def disagreement_rate(records: list[DecisionRecord]) -> float:
    """Share of decisions where the model's original call did not match the officer's. 0.0 on an empty list
    (nothing to disagree with yet), not an error -- a fresh deployment with no decisions yet is not "wrong"."""
    if not records:
        return 0.0
    return sum(1 for r in records if not model_agreed(r)) / len(records)


def should_retrain(records: list[DecisionRecord], threshold: float = 0.15, min_decisions: int = 20) -> tuple[bool, dict]:
    """Whether accumulated officer decisions justify retraining now. Requires at least `min_decisions` real
    decisions first -- a handful of reviews is too small a sample to justify retraining the whole model on,
    and would make the trigger noisy rather than informative. Returns (should_retrain, reasoning_dict) so the
    reasoning can be logged or shown on a dashboard, not just acted on silently."""
    n = len(records)
    if n < min_decisions:
        return False, {"reason": f"only {n} decision(s) so far, need at least {min_decisions} before this is a meaningful sample",
                       "disagreement_rate": None, "n_decisions": n, "threshold": threshold}
    rate = disagreement_rate(records)
    triggered = rate >= threshold
    return triggered, {"reason": f"disagreement rate {rate:.1%} {'meets or exceeds' if triggered else 'is below'} the {threshold:.0%} retrain trigger",
                       "disagreement_rate": rate, "n_decisions": n, "threshold": threshold}


def build_feedback_dataset(records: list[DecisionRecord], account_ids_in_graph: set[str]) -> tuple[list[str], list[float]]:
    """The real, officer-verified (account_id, true_label) pairs usable for retraining, restricted to accounts
    that still exist in the currently loaded graph. A decision about an account from an older graph snapshot
    cannot be matched to features that no longer exist, so it is skipped (with a warning), not guessed at."""
    ids, labels = [], []
    skipped = 0
    for r in records:
        if r.account_id not in account_ids_in_graph:
            skipped += 1
            continue
        ids.append(r.account_id)
        labels.append(true_label(r))
    if skipped:
        log.warning("skipped %d officer decision(s) for account(s) no longer present in the current graph", skipped)
    return ids, labels


def read_decisions_from_db(dsn: str) -> list[DecisionRecord]:
    """The one function in this module that needs a live database. Reads every resolved (confirmed or
    dismissed) alert from Postgres. Kept deliberately thin and separate from every function above, so all the
    actual decision logic is testable without a live database, and only this one function needs one."""
    import psycopg

    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute("SELECT account_id, risk_score, status FROM alerts WHERE status IN ('confirmed', 'dismissed')")
        rows = cur.fetchall()
    # The alerts table does not currently store the threshold that was in effect at scan time (only the score),
    # so this reads it from the environment as a stand-in for "the threshold now" -- an honest limitation: a
    # threshold that changed between when an old alert was scored and now would make model_agreed() slightly
    # wrong for that alert. Storing the scan-time threshold on the alerts row is a small, worthwhile follow-up.
    import os
    thr = float(os.getenv("FEEDBACK_THRESHOLD_FALLBACK", "0.5"))
    return [DecisionRecord(account_id=row[0], predicted_score=float(row[1]), threshold=thr, decision=row[2]) for row in rows]


def retrain_with_feedback(arrays: dict, tr, va, te, feedback_ids: list[str], feedback_labels: list[float],
                          feedback_weight: int = 5, **fit_kwargs) -> dict:
    """Continue training with real officer-verified examples blended into the training split.

    feedback_weight: how many times each feedback example is repeated in the training split, so a small
    number of hard-won real decisions is not drowned out by a much larger volume of warm-start (SynthAML or
    synthetic) training data -- a simple, standard way to upweight scarce high-value examples.
    """
    import numpy as np

    from gnn_aml_core.train import fit_gnn

    id_to_idx = {aid: i for i, aid in enumerate(arrays["account_ids"])}
    y = arrays["y"].copy()
    feedback_idx = []
    for aid, label in zip(feedback_ids, feedback_labels):
        idx = id_to_idx.get(aid)
        if idx is None:
            continue
        y[idx] = label  # the real, officer-confirmed label overrides whatever label (or lack of one) the account had
        feedback_idx.append(idx)
    arrays = {**arrays, "y": y}
    tr_with_feedback = np.concatenate([tr, np.repeat(np.array(feedback_idx, dtype=tr.dtype), feedback_weight)]) if feedback_idx else tr
    return fit_gnn(arrays, tr_with_feedback, va, te, **fit_kwargs)


def evaluate_checkpoint(ckpt: dict, arrays: dict, te) -> float:
    """Test-set AUC-ROC of an ALREADY-TRAINED checkpoint, measured fresh against a given (numpy) graph + test
    index split. Used to score the CURRENT production model on the exact same split a retrain candidate is
    about to be judged against, rather than trusting the current checkpoint's own stored metric -- which could
    have been computed on a different graph snapshot, a different split seed, or simply be out of date. An
    apples-to-apples comparison requires both models to be evaluated the same way, right now, on the same data."""
    import numpy as np
    import torch
    from sklearn.metrics import roc_auc_score

    from gnn_aml_core.models import build_model

    model = build_model(ckpt["model_name"], arrays["x"].shape[1], **ckpt["hparams"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    x = torch.from_numpy(np.array(arrays["x"]))
    ei = torch.from_numpy(np.array(arrays["edge_index"]))
    ea = torch.from_numpy(np.array(arrays["edge_attr"]))
    et = torch.from_numpy(np.array(arrays["edge_type"]))
    with torch.no_grad():
        prob = torch.sigmoid(model(x, ei, ea, et)).numpy()
    return float(roc_auc_score(arrays["y"][te], prob[te]))


def _append_retrain_history(out_dir, outcome: dict) -> None:
    """Append-only audit trail of every automatic retrain decision (promoted or rejected), one JSON line per
    attempt, never overwritten or rotated by this module. A compliance system needs to be able to answer "when
    did the live model last change, and why" without digging through container logs that may have rolled over."""
    import json
    from pathlib import Path

    p = Path(out_dir) / "retrain_history.jsonl"
    with p.open("a") as f:
        f.write(json.dumps(outcome, default=str) + "\n")


def retrain_and_maybe_promote(records: list[DecisionRecord], *, data_dir: str = "data", out_dir: str = "models",
                              feedback_weight: int = 5, min_improvement: float = 0.0,
                              gnn_api: str | None = None, gnn_admin_token: str | None = None,
                              finetune_lr: float = 0.0005, finetune_epochs: int = 30,
                              finetune_patience: int = 5) -> dict:
    """The automatic half of the feedback loop (see the module docstring's "Auto-retrain, gated before deploy"
    section for the full reasoning). FINE-TUNES the model already in production -- warm-starting from its exact
    weights (current_ckpt["state_dict"], via fit_gnn's new init_state_dict param) and continuing training for a
    short run (finetune_epochs, default 30, vs. a from-scratch training run's 150) at a low learning rate
    (finetune_lr, default 0.0005, 10x lower than train.py's from-scratch default of 0.005) -- rather than
    building a brand-new model from random weights on every cycle. A fresh random init on every drift trigger
    would (a) throw away everything the current model already learned and (b) make the gating comparison noisy
    for the wrong reason: a candidate could lose purely to bad luck in its random seed, not because the
    feedback data didn't help. Fine-tuning changes one thing at a time -- the same model, nudged by the new
    officer-verified examples -- which is also the more defensible compliance story ("the model was adjusted
    using N new real examples" vs. "the model was replaced by an unrelated one trained from scratch").

    Evaluates the candidate against the SAME held-out test split as the CURRENT production model
    (evaluate_checkpoint(), not either model's own possibly-stale stored metric), and promotes into production
    ONLY if the candidate's test AUC-ROC beats the current model's by at least `min_improvement`. Always returns
    a dict describing what happened (promoted, rejected, and why) and always appends it to
    models/retrain_history.jsonl -- a rejection is a normal, expected, logged outcome, not a failure of this
    function."""
    import shutil
    import time as _time
    from pathlib import Path

    import torch

    from gnn_aml_core.train import make_splits

    out = Path(out_dir)
    g = torch.load(out / "graph.pt", map_location="cpu", weights_only=False)
    for key in ("x", "edge_index", "edge_attr", "edge_type", "y"):
        if hasattr(g[key], "numpy"):
            g[key] = g[key].numpy()
    current_ckpt = torch.load(out / "gnn_model.pt", map_location="cpu", weights_only=False)

    ids, labels = build_feedback_dataset(records, set(g["account_ids"]))
    tr, va, te = make_splits(g["y"], seed=current_ckpt["hparams"].get("seed", 42))

    AUTO_RETRAIN_ATTEMPTS.inc()
    current_auc = evaluate_checkpoint(current_ckpt, g, te)
    log.info("auto-retrain: fine-tuning the current production model (not training a new one from scratch) "
            "with %d real officer-verified example(s) blended in, %d epoch(s) max at lr=%.5f "
            "(current production model's test AUC on this split: %.4f)",
            len(ids), finetune_epochs, finetune_lr, current_auc)

    # Pass the FULL saved hparams dict through (not just hidden/num_layers) -- fit_gnn stores every
    # build_model() kwarg it was given (heads, dropout, edge_dim, etc., whatever that model actually uses)
    # under "hparams", and warm-starting via init_state_dict requires the rebuilt model's shapes to match the
    # saved state_dict EXACTLY. Cherry-picking just hidden/num_layers here would silently rebuild the wrong
    # architecture for any model trained with a non-default heads/dropout/etc., and load_state_dict would then
    # raise a shape-mismatch error -- failing loudly, which is the right failure mode for a mismatch like this,
    # rather than quietly training a differently-shaped model that was never actually warm-started.
    saved_hp = dict(current_ckpt["hparams"])
    hidden = saved_hp.pop("hidden")
    layers = saved_hp.pop("num_layers")
    res = retrain_with_feedback(g, tr, va, te, ids, labels, feedback_weight=feedback_weight,
                                model_name=current_ckpt["model_name"], hidden=hidden, layers=layers, **saved_hp,
                                init_state_dict=current_ckpt["state_dict"],
                                lr=finetune_lr, epochs=finetune_epochs, patience=finetune_patience,
                                seed=current_ckpt["hparams"].get("seed", 42))
    candidate_auc = res["report"]["auc_roc"]

    outcome = {
        "ts": _time.time(), "n_feedback_examples": len(ids), "current_test_auc": current_auc,
        "candidate_test_auc": candidate_auc, "min_improvement": min_improvement,
    }

    if candidate_auc <= current_auc + min_improvement:
        outcome["decision"] = "rejected"
        outcome["reason"] = (f"candidate test AUC {candidate_auc:.4f} did not beat current {current_auc:.4f} "
                             f"by the required margin ({min_improvement:.4f}) -- current model stays in production")
        AUTO_RETRAIN_REJECTED.inc()
        log.warning("auto-retrain: %s", outcome["reason"])
        _append_retrain_history(out, outcome)
        return outcome

    # Promote: build the same checkpoint schema gnn_aml_core.train's main() writes, so gnn_aml_core.main's
    # _load_model() (which /admin/reload-model calls) can read it with zero special-casing.
    calibrated_threshold = float(res["calibrator"](res["report"]["threshold"]))
    new_metrics = {
        "val_auc_roc": res["best_val_auc"], "best_epoch": res["best_epoch"], "test_auc_roc": res["report"]["auc_roc"],
        "test_pr_auc": res["report"]["pr_auc"], "test_precision": res["report"]["precision"],
        "test_recall": res["report"]["recall"], "test_f1": res["report"]["f1"], "test_fpr": res["report"]["fpr"],
        "threshold": res["report"]["threshold"], "auc_target": current_ckpt["metrics"].get("auc_target", 0.87),
        "meets_auc_target": bool(res["report"]["auc_roc"] >= current_ckpt["metrics"].get("auc_target", 0.87)),
        **{f"calibration_{k}": v for k, v in res["calibration_report"].items()},
        "retrained_from_feedback": True, "fine_tuned_from_existing_model": True, "n_feedback_examples": len(ids),
        "finetune_lr": finetune_lr, "finetune_epochs_run": res["epochs_run"],
    }
    new_ckpt = {
        "model_name": current_ckpt["model_name"], "hparams": res["hparams"], "in_dim": current_ckpt["in_dim"],
        "state_dict": res["state_dict"], "feature_names": current_ckpt["feature_names"],
        "mean": current_ckpt["mean"], "std": current_ckpt["std"],
        "threshold": res["report"]["threshold"], "metrics": new_metrics,
        "calibration": res["calibrator"].to_dict(), "calibrated_threshold": calibrated_threshold,
    }

    backup_path = out / f"gnn_model.pt.bak-{_time.strftime('%Y%m%d-%H%M%S')}"
    shutil.copy2(out / "gnn_model.pt", backup_path)
    torch.save(new_ckpt, out / "gnn_model.pt")

    outcome["decision"] = "promoted"
    outcome["reason"] = (f"candidate test AUC {candidate_auc:.4f} beat current {current_auc:.4f} "
                         f"by {candidate_auc - current_auc:.4f}")
    outcome["backup_path"] = str(backup_path)
    AUTO_RETRAIN_PROMOTED.inc()
    log.info("auto-retrain: %s -- promoted, old model backed up to %s", outcome["reason"], backup_path)

    if gnn_api:
        import json
        import urllib.error
        import urllib.request

        req = urllib.request.Request(f"{gnn_api.rstrip('/')}/admin/reload-model", method="POST")
        if gnn_admin_token:
            req.add_header("Authorization", f"Bearer {gnn_admin_token}")
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                outcome["reload"] = json.loads(resp.read())
            log.info("auto-retrain: live reload succeeded -- %s", outcome["reload"])
        except Exception as exc:  # noqa: BLE001 -- the new checkpoint is already safely on disk regardless of
                                  # whether this network call succeeds; a failure here is never the reason the
                                  # new model fails to eventually serve, only a delay until the next restart
            outcome["reload_error"] = f"{exc.__class__.__name__}: {exc}"
            log.warning("auto-retrain: promoted on disk, but the live reload call failed (%s) -- the new model "
                       "will be picked up on gnn-detection-api's next restart in the meantime", outcome["reload_error"])
    else:
        outcome["reload"] = "skipped (no --gnn-api given) -- new model written to disk, will be picked up on next restart"
        log.info("auto-retrain: %s", outcome["reload"])

    _append_retrain_history(out, outcome)
    return outcome


def _default_dsn() -> str:
    """The local (host-machine) Postgres connection string, read from this project's own .env file -- the
    same source of truth every other tool here uses (run_local.py, backend/main.py's DATABASE_URL). An earlier
    version of this function hardcoded a guessed default (postgresql://aml:aml@localhost:15432/aml) that does
    not match this project's real credentials (POSTGRES_USER/POSTGRES_PASSWORD/POSTGRES_DB in .env, which
    default to "projectxy"/"change_me_pg"/"projectxy" in .env.example) -- a real bug, not a placeholder that
    happened to be fine to leave: it silently failed authentication against a correctly running Postgres
    rather than erroring at the actually-wrong assumption."""
    import os
    from pathlib import Path

    env = {}
    p = Path(".env")
    if p.exists():
        for line in p.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip().strip('"').strip("'")
    user = os.getenv("POSTGRES_USER", env.get("POSTGRES_USER", "projectxy"))
    password = os.getenv("POSTGRES_PASSWORD", env.get("POSTGRES_PASSWORD", ""))
    db = os.getenv("POSTGRES_DB", env.get("POSTGRES_DB", "projectxy"))
    return f"postgresql://{user}:{password}@127.0.0.1:15432/{db}"


def main() -> None:
    import os

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dsn", default=os.getenv("DATABASE_URL", _default_dsn()),
                    help="Postgres connection string; defaults to DATABASE_URL, or this project's own .env credentials")
    ap.add_argument("--data", default="data", help="folder with the graph data to retrain against")
    ap.add_argument("--out", default="models")
    ap.add_argument("--check", action="store_true", help="report the current drift only; never retrain")
    ap.add_argument("--retrain-threshold", type=float, default=0.15)
    ap.add_argument("--min-decisions", type=int, default=20)
    ap.add_argument("--feedback-weight", type=int, default=5)
    ap.add_argument("--serve-metrics", action="store_true",
                    help="expose the drift signal for Prometheus to scrape (aml_officer_disagreement_rate, "
                         "aml_feedback_decisions_total, aml_retrain_triggered_total) and re-check on a loop "
                         "rather than exiting after one check -- run this as its own long-lived process "
                         "(e.g. a container, or a node_agent scheduled job) so Prometheus has something "
                         "continuously running to scrape, the same way gnn-detection-api and xai-narrative-api "
                         "already do for their own metrics")
    ap.add_argument("--metrics-port", type=int, default=9201)
    ap.add_argument("--loop-interval", type=int, default=300, help="seconds between drift checks in --serve-metrics mode")
    ap.add_argument("--auto-retrain", action="store_true",
                    default=os.getenv("FEEDBACK_AUTO_RETRAIN", "").strip().lower() in ("1", "true", "yes"),
                    help="(--serve-metrics only) when the drift trigger fires, automatically retrain AND "
                         "evaluate the candidate against the current production model -- promoting it only if "
                         "it measurably wins; see the module docstring's 'Auto-retrain, gated before deploy' "
                         "section. Off by default (also settable via FEEDBACK_AUTO_RETRAIN=true)")
    ap.add_argument("--gnn-api", default=os.getenv("GNN_API_URL"),
                    help="base URL of the running gnn-detection-api, used to call POST /admin/reload-model "
                         "after a promotion so the new model goes live without a container restart. Omit to "
                         "promote on disk only (the new model is picked up on that service's next restart)")
    ap.add_argument("--gnn-admin-token", default=os.getenv("GNN_ADMIN_TOKEN"),
                    help="sent as 'Authorization: Bearer <token>' on the reload call; must match "
                         "gnn-detection-api's own GNN_ADMIN_TOKEN if that service has one set")
    ap.add_argument("--min-improvement", type=float, default=float(os.getenv("FEEDBACK_MIN_IMPROVEMENT", 0.0)),
                    help="a retrained candidate must beat the current production model's test AUC-ROC by at "
                         "least this much to be promoted (default 0.0: any real improvement, however small, "
                         "beats none)")
    ap.add_argument("--finetune-lr", type=float, default=float(os.getenv("FEEDBACK_FINETUNE_LR", 0.0005)),
                    help="learning rate for the auto-retrain fine-tune (--auto-retrain warm-starts from the "
                         "CURRENT production model's own weights rather than training a new model from "
                         "scratch -- see retrain_and_maybe_promote's docstring). Kept low (default 0.0005, vs. "
                         "train.py's from-scratch default of 0.005) since this is a short nudge, not a full "
                         "training run")
    ap.add_argument("--finetune-epochs", type=int, default=int(os.getenv("FEEDBACK_FINETUNE_EPOCHS", 30)),
                    help="max epochs for the auto-retrain fine-tune (default 30, vs. train.py's from-scratch "
                         "default of 150) -- a warm-started model needs far fewer epochs to incorporate new "
                         "feedback than a model starting from random weights")
    ap.add_argument("--finetune-patience", type=int, default=int(os.getenv("FEEDBACK_FINETUNE_PATIENCE", 5)),
                    help="early-stop the fine-tune after this many eval checkpoints (every 5 epochs) without a "
                         "validation improvement")
    a = ap.parse_args()

    def _check_once() -> tuple[bool, dict, list]:
        try:
            records = read_decisions_from_db(a.dsn)
        except ImportError:
            print("psycopg is not installed -> pip install 'psycopg[binary]' (already in gnn_aml_core/requirements.txt)")
            raise SystemExit(1)
        except Exception as exc:  # noqa: BLE001 -- deliberately broad; the two branches below cover the specific,
                                  # actionable cases, and the final else covers anything else with a generic hint
            import psycopg.errors

            if isinstance(exc, psycopg.errors.UndefinedTable):
                print("connected to Postgres, but the 'alerts' table does not exist yet\n"
                      "-> the backend service creates it on startup (backend/main.py's lifespan hook); "
                      "start it at least once: python3 run_local.py , or docker compose up -d backend")
            else:
                print(f"could not reach Postgres at the given --dsn ({exc.__class__.__name__}: {exc})\n"
                      "-> is it running? docker compose up -d database , or check DATABASE_URL / --dsn")
            raise SystemExit(1)
        triggered, reasoning = should_retrain(records, threshold=a.retrain_threshold, min_decisions=a.min_decisions)
        log.info("feedback status: %s", reasoning["reason"])
        FEEDBACK_DECISIONS_TOTAL.set(reasoning["n_decisions"])
        if reasoning["disagreement_rate"] is not None:
            DISAGREEMENT_RATE.set(reasoning["disagreement_rate"])
        if triggered:
            RETRAIN_TRIGGERED_TOTAL.inc()
        return triggered, reasoning, records

    if a.serve_metrics:
        start_http_server(a.metrics_port)
        mode = "auto-retrain ENABLED (gated: promotes only if it beats the current model)" if a.auto_retrain \
            else "report-only (pass --auto-retrain, or set FEEDBACK_AUTO_RETRAIN=true, to close the loop)"
        log.info("serving metrics on :%d/metrics, checking every %ds -- %s (Ctrl+C to stop)",
                 a.metrics_port, a.loop_interval, mode)
        while True:
            try:
                triggered, reasoning, records = _check_once()
                if triggered and a.auto_retrain:
                    log.info("auto-retrain: drift trigger met (%s) -- starting an automatic retrain attempt",
                             reasoning["reason"])
                    try:
                        retrain_and_maybe_promote(records, data_dir=a.data, out_dir=a.out,
                                                  feedback_weight=a.feedback_weight, min_improvement=a.min_improvement,
                                                  gnn_api=a.gnn_api, gnn_admin_token=a.gnn_admin_token,
                                                  finetune_lr=a.finetune_lr, finetune_epochs=a.finetune_epochs,
                                                  finetune_patience=a.finetune_patience)
                    except Exception as exc:  # noqa: BLE001 -- a failed retrain attempt must never take this
                                              # long-lived loop down; the current model keeps serving regardless,
                                              # and the next drift cycle will simply try again
                        log.error("auto-retrain: attempt failed with an unexpected error (%s) -- current model "
                                 "stays in production, will retry on the next drift-triggered cycle",
                                 exc.__class__.__name__)
            except SystemExit:
                log.warning("a check failed (see above) -- will retry on the next interval rather than exiting, "
                           "since this process is meant to keep running")
            time.sleep(a.loop_interval)
        return  # pragma: no cover -- unreachable (the loop above only exits via Ctrl+C/SIGTERM), kept for clarity

    triggered, reasoning, records = _check_once()
    if a.check or not triggered:
        return

    import torch

    from gnn_aml_core.train import arrays_from_frames, make_splits

    g = torch.load(f"{a.out}/graph.pt", map_location="cpu", weights_only=False)
    # graph.pt stores tensor fields as torch objects (that is what train.py saves); make_splits/fit_gnn expect
    # the same numpy arrays arrays_from_frames() normally produces, so convert explicitly rather than assume
    # torch tensors behave identically everywhere numpy would.
    for key in ("x", "edge_index", "edge_attr", "edge_type", "y"):
        if hasattr(g[key], "numpy"):
            g[key] = g[key].numpy()
    ckpt = torch.load(f"{a.out}/gnn_model.pt", map_location="cpu", weights_only=False)
    ids, labels = build_feedback_dataset(records, set(g["account_ids"]))
    log.info("retraining with %d real officer-verified example(s) blended into the training split", len(ids))
    tr, va, te = make_splits(g["y"], seed=ckpt["hparams"].get("seed", 42))
    res = retrain_with_feedback(g, tr, va, te, ids, labels, feedback_weight=a.feedback_weight,
                                model_name=ckpt["model_name"], hidden=ckpt["hparams"]["hidden"],
                                layers=ckpt["hparams"]["num_layers"])
    log.info("retrained: val AUC %.4f (best epoch %d), test report %s", res["best_val_auc"], res["best_epoch"], res["report"])


if __name__ == "__main__":
    main()
