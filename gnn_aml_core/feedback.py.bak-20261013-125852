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
        log.info("serving metrics on :%d/metrics, checking every %ds (Ctrl+C to stop)", a.metrics_port, a.loop_interval)
        while True:
            try:
                _check_once()   # loop mode only reports drift status; it never retrains automatically, since an
                                # unattended, periodically-running process silently retraining the live model
                                # with no human in the loop is a materially different, bigger decision than
                                # just reporting a metric -- real retraining stays an explicit, manually-invoked
                                # action (the non-loop path below), not something this loop does on its own
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
