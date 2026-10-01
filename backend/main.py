"""ProjectXY core backend (container: backend).

Alert workflow for compliance officers:
  scan account  ->  alert (Postgres)  ->  generate SAR narrative  ->  officer decision (audited in immudb)

All routes live under /api (Traefik forwards /api/* here without stripping the prefix).

Authentication: every route below except /health requires an API key, sent as
"Authorization: Bearer <key>". Keys are managed with manage_keys.py (create/list/revoke), never through an
HTTP endpoint -- an endpoint that creates new credentials would itself need protecting by something, which is
exactly the chicken-and-egg problem a CLI script run with direct database access avoids. This closes two real
findings from a security hygiene pass: the API previously had no authentication at all (anyone with network
access could confirm or dismiss real compliance alerts), and the officer field on a decision was free-text,
self-reported, with no identity verification -- it is now taken from the authenticated key, not request input.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import secrets
import time
from contextlib import asynccontextmanager
from typing import Literal

import httpx
import psycopg
from fastapi import APIRouter, Depends, FastAPI, Header, HTTPException
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

CREATE TABLE IF NOT EXISTS api_keys (
    id            SERIAL PRIMARY KEY,
    officer_name  TEXT        NOT NULL,
    key_hash      TEXT        NOT NULL UNIQUE,   -- sha256 of the actual key; the plaintext key is shown to the
                                                  -- officer exactly once, at creation, and never stored anywhere
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    revoked_at    TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS api_keys_hash_idx ON api_keys (key_hash) WHERE revoked_at IS NULL;
"""


def hash_api_key(key: str) -> str:
    """A straight SHA-256 hash, not bcrypt/scrypt: correct for API keys specifically (unlike passwords, they
    are generated with high entropy via secrets.token_urlsafe, never user-chosen or reused across sites, so the
    slow, salted hashing that defends against offline dictionary attacks on passwords has nothing to defend
    against here) -- the same accepted practice used by, e.g., Stripe and GitHub for their own API keys."""
    return hashlib.sha256(key.encode()).hexdigest()


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


def get_current_officer(authorization: str | None = Header(None)) -> str:
    """Every route but /health depends on this. Returns the officer name tied to a valid, non-revoked API key --
    the real fix for the free-text 'officer' field decide() used to accept as-is: identity now comes from an
    authenticated credential, never from request input a caller could set to anything."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "missing or malformed Authorization header (expected 'Bearer <api-key>')")
    key = authorization.removeprefix("Bearer ").strip()
    if not key:
        raise HTTPException(401, "empty API key")
    with db() as conn:
        row = conn.execute(
            "SELECT officer_name FROM api_keys WHERE key_hash = %s AND revoked_at IS NULL",
            (hash_api_key(key),),
        ).fetchone()
    if not row:
        raise HTTPException(401, "invalid or revoked API key")
    return row["officer_name"]


# ------------------------------------------------------------------ schemas
class ScanRequest(BaseModel):
    account_id: str
    force_alert: bool = False       # create an alert even if the score is below the threshold


class DecisionRequest(BaseModel):
    decision: Literal["confirmed", "dismissed"]
    note: str = Field("", max_length=2000)


# ------------------------------------------------------------------ routes
@api.get("/health")
def health():
    with db() as conn:
        conn.execute("SELECT 1")
    return {"status": "ok"}


@api.get("/whoami")
def whoami(officer: str = Depends(get_current_officer)):
    """Lets the frontend verify a newly-entered API key actually works, and display who is logged in, without
    needing to decode anything client-side -- the officer's name comes from the authenticated key, the same
    way every other route's identity does."""
    return {"officer": officer}


@api.get("/accounts/sample")
def sample_accounts(officer: str = Depends(get_current_officer)):
    return _call("GET", f"{GNN_API_URL}/accounts/sample", params={"n": 5})


@api.get("/model/info")
def model_info(officer: str = Depends(get_current_officer)):
    return _call("GET", f"{GNN_API_URL}/model/info")


@api.get("/accounts/{account_id}/explain")
def explain_account(account_id: str, officer: str = Depends(get_current_officer)):
    """GNNExplainer subgraph + edge importances, drawn as the network view in the UI."""
    return _call("POST", f"{GNN_API_URL}/explain", timeout=120, json={"account_id": account_id})


@api.post("/alerts/scan")
def scan(req: ScanRequest, officer: str = Depends(get_current_officer)):
    pred = _call("POST", f"{GNN_API_URL}/predict", json={"account_ids": [req.account_id]})
    res = pred["results"][0]
    alert = None
    if res["flagged"] or req.force_alert:
        with db() as conn:
            alert = conn.execute(
                "INSERT INTO alerts (account_id, risk_score) VALUES (%s, %s) RETURNING *",
                (req.account_id, res["score"]),
            ).fetchone()
        audit("alert_created", {"alert_id": alert["id"], "account_id": req.account_id, "score": res["score"],
                                "scanned_by": officer})
    # "score"/"threshold" (raw) are what the flagged decision and the stored alert are based on, and stay
    # exactly as before; "calibrated_score"/"calibrated_threshold" are the same pair mapped through isotonic
    # calibration, meaningful as an actual probability -- meant for display (a risk gauge showing "99.9%" for
    # every flagged account is not informative), never used for the flagged decision itself. .get() with a
    # fallback keeps this working against an older gnn-detection-api that predates calibration.
    return {"score": res["score"], "calibrated_score": res.get("calibrated_score", res["score"]),
            "flagged": res["flagged"], "threshold": pred["threshold"],
            "calibrated_threshold": pred.get("calibrated_threshold", pred["threshold"]),
            "latency_ms": pred["latency_ms"], "alert": alert}


@api.get("/alerts")
def list_alerts(status: str | None = None, limit: int = 100, officer: str = Depends(get_current_officer)):
    q, args = "SELECT * FROM alerts", []
    if status:
        q += " WHERE status = %s"
        args.append(status)
    q += " ORDER BY created_at DESC LIMIT %s"
    args.append(min(limit, 500))
    with db() as conn:
        return conn.execute(q, args).fetchall()


@api.post("/alerts/{alert_id}/narrative")
def generate_narrative(alert_id: int, officer: str = Depends(get_current_officer)):
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
def decide(alert_id: int, req: DecisionRequest, officer: str = Depends(get_current_officer)):
    with db() as conn:
        alert = conn.execute("SELECT * FROM alerts WHERE id = %s", (alert_id,)).fetchone()
        if not alert:
            raise HTTPException(404, "alert not found")
        if alert["status"] != "open":
            raise HTTPException(409, f"alert already {alert['status']}")
        alert = conn.execute(
            "UPDATE alerts SET status=%s, reviewed_by=%s, review_note=%s, reviewed_at=now() "
            "WHERE id=%s RETURNING *",
            (req.decision, officer, req.note, alert_id),
        ).fetchone()
    # Record hash of the exact reviewed content so later tampering with Postgres is detectable.
    record = {"alert_id": alert_id, "account_id": alert["account_id"], "decision": req.decision,
              "officer": officer, "note": req.note, "narrative_sha256": alert["narrative_sha256"]}
    record["record_sha256"] = hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()
    audit("officer_decision", record)
    return alert


app.include_router(api)
