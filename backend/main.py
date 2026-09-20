"""ProjectXY core backend (container: backend).

Alert workflow for compliance officers:
  scan account  ->  alert (Postgres)  ->  generate SAR narrative  ->  officer decision (audited in immudb)

All routes live under /api (Traefik forwards /api/* here without stripping the prefix).
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Literal

import httpx
import psycopg
from fastapi import APIRouter, FastAPI, HTTPException
from psycopg.rows import dict_row
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("backend")

DATABASE_URL = os.environ["DATABASE_URL"]
GNN_API_URL = os.getenv("GNN_API_URL", "http://gnn-detection-api:8000")
XAI_API_URL = os.getenv("XAI_API_URL", "http://xai-narrative-api:8000")

SCHEMA = """
CREATE TABLE IF NOT EXISTS alerts (
    id            SERIAL PRIMARY KEY,
    account_id    TEXT        NOT NULL,
    risk_score    DOUBLE PRECISION NOT NULL,
    status        TEXT        NOT NULL DEFAULT 'open',      -- open | confirmed | dismissed
    typology      TEXT,
    narrative     TEXT,
    narrative_source TEXT,
    narrative_sha256 TEXT,
    reviewed_by   TEXT,
    review_note   TEXT,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    reviewed_at   TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS alerts_status_idx ON alerts (status);
"""


def db():
    return psycopg.connect(DATABASE_URL, row_factory=dict_row)


@asynccontextmanager
async def lifespan(_: FastAPI):
    for attempt in range(20):                       # wait for Postgres on cold start
        try:
            with db() as conn:
                conn.execute(SCHEMA)
            log.info("database ready")
            break
        except psycopg.OperationalError:
            log.warning("database not ready (attempt %d)", attempt + 1)
            time.sleep(2)
    yield


app = FastAPI(title="ProjectXY Backend", version="0.1.0", lifespan=lifespan)
api = APIRouter(prefix="/api")

_immu = None


def audit(kind: str, payload: dict) -> None:
    """Append to the immudb ledger (officer decisions, narratives). Never raises."""
    global _immu
    host = os.getenv("IMMUDB_HOST")
    if not host:
        return
    try:
        if _immu is None:
            from immudb import ImmudbClient
            c = ImmudbClient(f"{host}:{os.getenv('IMMUDB_PORT', '3322')}")
            c.login(os.getenv("IMMUDB_USER", "immudb"), os.getenv("IMMUDB_ADMIN_PASSWORD", "immudb"))
            _immu = c
        key = f"{kind}:{payload.get('alert_id', '-')}:{time.time_ns()}".encode()
        _immu.verifiedSet(key, json.dumps({"kind": kind, "ts": time.time(), **payload}, sort_keys=True).encode())
    except Exception as exc:  # noqa: BLE001
        log.warning("immudb audit failed (%s)", exc.__class__.__name__)
        _immu = None


def _call(method: str, url: str, timeout: float = 30, **kw):
    try:
        r = httpx.request(method, url, timeout=timeout, **kw)
    except httpx.HTTPError as exc:
        raise HTTPException(502, f"upstream unreachable: {exc.__class__.__name__}") from exc
    if r.status_code >= 400:
        raise HTTPException(r.status_code, r.text)
    return r.json()


# ------------------------------------------------------------------ schemas
class ScanRequest(BaseModel):
    account_id: str
    force_alert: bool = False       # create an alert even if the score is below the threshold


class DecisionRequest(BaseModel):
    officer: str = Field(min_length=2, max_length=80)
    decision: Literal["confirmed", "dismissed"]
    note: str = Field("", max_length=2000)


# ------------------------------------------------------------------ routes
@api.get("/health")
def health():
    with db() as conn:
        conn.execute("SELECT 1")
    return {"status": "ok"}


@api.get("/accounts/sample")
def sample_accounts():
    return _call("GET", f"{GNN_API_URL}/accounts/sample", params={"n": 5})


@api.get("/model/info")
def model_info():
    return _call("GET", f"{GNN_API_URL}/model/info")


@api.get("/accounts/{account_id}/explain")
def explain_account(account_id: str):
    """GNNExplainer subgraph + edge importances, drawn as the network view in the UI."""
    return _call("POST", f"{GNN_API_URL}/explain", timeout=120, json={"account_id": account_id})


@api.post("/alerts/scan")
def scan(req: ScanRequest):
    pred = _call("POST", f"{GNN_API_URL}/predict", json={"account_ids": [req.account_id]})
    res = pred["results"][0]
    alert = None
    if res["flagged"] or req.force_alert:
        with db() as conn:
            alert = conn.execute(
                "INSERT INTO alerts (account_id, risk_score) VALUES (%s, %s) RETURNING *",
                (req.account_id, res["score"]),
            ).fetchone()
        audit("alert_created", {"alert_id": alert["id"], "account_id": req.account_id, "score": res["score"]})
    return {"score": res["score"], "flagged": res["flagged"], "threshold": pred["threshold"],
            "latency_ms": pred["latency_ms"], "alert": alert}


@api.get("/alerts")
def list_alerts(status: str | None = None, limit: int = 100):
    q, args = "SELECT * FROM alerts", []
    if status:
        q += " WHERE status = %s"
        args.append(status)
    q += " ORDER BY created_at DESC LIMIT %s"
    args.append(min(limit, 500))
    with db() as conn:
        return conn.execute(q, args).fetchall()


@api.post("/alerts/{alert_id}/narrative")
def generate_narrative(alert_id: int):
    with db() as conn:
        alert = conn.execute("SELECT * FROM alerts WHERE id = %s", (alert_id,)).fetchone()
    if not alert:
        raise HTTPException(404, "alert not found")
    out = _call("POST", f"{XAI_API_URL}/narrative", timeout=180, json={"account_id": alert["account_id"]})
    with db() as conn:
        alert = conn.execute(
            "UPDATE alerts SET narrative=%s, narrative_source=%s, narrative_sha256=%s, typology=%s "
            "WHERE id=%s RETURNING *",
            (out["narrative"], out["source"], out["narrative_sha256"], out["typology"], alert_id),
        ).fetchone()
    return alert


@api.post("/alerts/{alert_id}/decision")
def decide(alert_id: int, req: DecisionRequest):
    with db() as conn:
        alert = conn.execute("SELECT * FROM alerts WHERE id = %s", (alert_id,)).fetchone()
        if not alert:
            raise HTTPException(404, "alert not found")
        if alert["status"] != "open":
            raise HTTPException(409, f"alert already {alert['status']}")
        alert = conn.execute(
            "UPDATE alerts SET status=%s, reviewed_by=%s, review_note=%s, reviewed_at=now() "
            "WHERE id=%s RETURNING *",
            (req.decision, req.officer, req.note, alert_id),
        ).fetchone()
    # Record hash of the exact reviewed content so later tampering with Postgres is detectable.
    record = {"alert_id": alert_id, "account_id": alert["account_id"], "decision": req.decision,
              "officer": req.officer, "note": req.note, "narrative_sha256": alert["narrative_sha256"]}
    record["record_sha256"] = hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()
    audit("officer_decision", record)
    return alert


app.include_router(api)
