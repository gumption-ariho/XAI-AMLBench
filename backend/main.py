"""ProjectXY core backend (container: backend).

Two detection pipelines feed this console now:
  Model 1 (gnn_aml_core.main): known-typology classifier -- scan account -> alert -> SAR narrative -> officer
    decision (audited in immudb). Unchanged from before.
  Model 2 (gnn_aml_core.anomaly): a deep graph autoencoder trained only on whatever Model 1 currently calls
    normal, scoped away from anything Model 1 already flags. It raises an alert into the SEPARATE
    anomaly_alerts table below -- never auto-decided. The /api/anomalies/* routes are the investigation
    workflow those alerts go through: claim -> investigate (notes) -> resolve as confirmed_mule or cleared. A
    resolve to confirmed_mule is the ONLY thing that ever teaches Model 1 a new pattern (via
    gnn_aml_core.anomaly.sync_confirmed_to_model1, running in the anomaly-detector container) -- the routine
    confirm/dismiss clicks on Model 1's own alerts below are deliberately NEVER used to train anything; that
    was an earlier, explicit design decision (see gnn_aml_core/feedback.py's module docstring) since
    everyday rubber-stamp decisions on already-obvious cases carry far less signal than a specifically
    investigated, previously-invisible pattern, and training on them would make the model drift with officer
    habits rather than with genuinely new information.

All routes live under /api (Traefik forwards /api/* here without stripping the prefix).

Authentication: a single simple login -- POST /api/login with one PREDEFINED passkey (ADMIN_PASSKEY /
ANALYST_PASSKEY / one of BANK_PASSKEYS, set in .env) plus the person's own name. This replaces the earlier
per-officer database-managed API key system (manage_keys.py) with something much simpler to operate, while
keeping the one thing that system existed for: every decision in the audit trail is still attributed to a
SPECIFIC named person, not just "an analyst" -- the passkey only proves which ROLE (and, for a bank passkey,
which TENANT) you have; the name you type is what the audit trail actually records against every action you
take, exactly as before. The server signs a session token (name + role + bank + issued-at, HMAC'd with
AUTH_TOKEN_SECRET) and every other route verifies that signature -- there's no database lookup and no
credential to issue, revoke or manage, which is the whole point of this change: a predefined passkey needs no
administration at all. If someone can't be trusted to type their own real name honestly, that's a people
problem no login mechanism solves; what this DOES still guarantee is that the audit trail can't be forged by
request input alone, since the name is bound into a server-signed token, not read fresh off each request body.

Multi-tenant SaaS: a BANK_PASSKEYS login binds a specific bank into the session token (see the comment above
BANK_NAME_TO_PASSKEY below), and every route that reads or writes bank-attributable data -- alerts, anomalies,
activity, intake, scan, explain -- depends on get_current_bank alongside get_current_officer and silently
scopes itself to that one bank. An ADMIN_PASSKEY or the legacy shared ANALYST_PASSKEY still sees every bank's
data, unaffected. Both Model 1 and Model 2 stay ONE shared detection engine across every tenant -- there is no
per-bank model, and nothing here ever forks, retrains-per-tenant, or otherwise splits the ML layer; only the
data access layer is partitioned. See each route's own docstring below for exactly how it's scoped.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import os
import secrets
import time
from contextlib import asynccontextmanager
from typing import Literal

import httpx
import psycopg
import pyotp
from fastapi import APIRouter, Depends, FastAPI, Header, HTTPException, Request
from prometheus_fastapi_instrumentator import Instrumentator, metrics
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("backend")

DATABASE_URL = os.environ["DATABASE_URL"]
GNN_API_URL = os.getenv("GNN_API_URL", "http://gnn-detection-api:8000")
GNN_ADMIN_TOKEN = os.getenv("GNN_ADMIN_TOKEN", "")   # forwarded as Bearer auth to gnn-detection-api's
                                                      # admin-only routes (POST /admin/ingest-live), see
                                                      # the Intake tab's "real accounts" ingestion route below
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
-- Added for multi-bank attribution: every alert should say which bank's data it came from. A plain ADD
-- COLUMN (not inside the CREATE TABLE above) so this lands correctly on a database that already has the
-- alerts table from before this feature existed -- CREATE TABLE IF NOT EXISTS is a no-op there.
ALTER TABLE alerts ADD COLUMN IF NOT EXISTS source_bank TEXT NOT NULL DEFAULT 'unknown';
CREATE INDEX IF NOT EXISTS alerts_status_idx ON alerts (status);
CREATE INDEX IF NOT EXISTS alerts_source_bank_idx ON alerts (source_bank);

-- Which bank an account belongs to, for every account that arrived through Intake or Connect (Options 1/2).
-- Deliberately first-seen-wins (ON CONFLICT DO NOTHING in tag_accounts_bank() below): an account id that
-- later shows up tagged with a different bank keeps its original attribution rather than being silently
-- reassigned, since the same account id being claimed by two different banks is itself worth noticing, not
-- something to paper over. Accounts that predate this feature (SynthAML's training/demo accounts) simply
-- have no row here -- looked up as 'unknown', not fabricated into belonging to some bank they never did.
CREATE TABLE IF NOT EXISTS account_banks (
    account_id TEXT PRIMARY KEY,
    bank       TEXT        NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Model 2's alerts. Deliberately a SEPARATE table from `alerts` above (Model 1's), never merged: the two
-- pipelines have different purposes (known-typology vs. "this is unlike anything normal"), different alert
-- volumes expected (Model 2 should be rare -- it only fires in Model 1's blind spot), and a different
-- lifecycle (Model 1's alerts go straight to a decision; Model 2's alerts require an actual investigation
-- before anyone calls them a verdict).
CREATE TABLE IF NOT EXISTS anomaly_alerts (
    id                SERIAL PRIMARY KEY,
    account_id        TEXT        NOT NULL,
    anomaly_score     DOUBLE PRECISION NOT NULL,
    threshold         DOUBLE PRECISION NOT NULL,
    status            TEXT        NOT NULL DEFAULT 'new',   -- new | under_investigation | confirmed_mule | cleared
    assigned_to       TEXT,
    investigation_log JSONB       NOT NULL DEFAULT '[]',    -- append-only [{ts, by, note, status}, ...]
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at        TIMESTAMPTZ,
    trained_at        TIMESTAMPTZ   -- set once gnn_aml_core.anomaly has fed a confirmed_mule case into Model 1
);
CREATE INDEX IF NOT EXISTS anomaly_alerts_status_idx ON anomaly_alerts (status);

-- DB-backed tenants: this is what lets the owner add/suspend/remove a bank from the /owner backoffice at
-- runtime, no .env edit and no restart -- the BANK_PASSKEYS env var (see its own comment above
-- _BANK_PASSKEYS_RAW below) still works too, as a legacy/static set of tenants, and the two are merged at
-- login time, env first: an env-configured bank name can't be shadowed or suspended by a row here, since the
-- operator who controls the env file already has the stronger form of control.
CREATE TABLE IF NOT EXISTS banks (
    name          TEXT        PRIMARY KEY,
    passkey       TEXT        NOT NULL,
    status        TEXT        NOT NULL DEFAULT 'active',   -- active | suspended
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    logo_url      TEXT,
    accent_color  TEXT,
    contact_email TEXT   -- who notify_email() tells about a new alert for this bank, if set (see POST
                         -- /alerts/scan and /system/live-ingest); null is a silent no-op, same as SMTP_HOST unset
);
-- Full registration detail, captured when the bank itself signs up (see SignupRequestBody/create_signup_request
-- below) or added by the owner by hand (CreateBankRequest/admin_create_bank) -- this is what lets the owner
-- actually tell WHICH bank this is and where: its own registration/license number, which country/region it
-- operates in, a postal address, a public website. All nullable and never guessed -- a bank added before this
-- field existed, or added manually with only a name on hand, simply shows blanks here, not invented values.
ALTER TABLE banks ADD COLUMN IF NOT EXISTS registration_number TEXT;
ALTER TABLE banks ADD COLUMN IF NOT EXISTS country TEXT;
ALTER TABLE banks ADD COLUMN IF NOT EXISTS region TEXT;
ALTER TABLE banks ADD COLUMN IF NOT EXISTS address TEXT;
ALTER TABLE banks ADD COLUMN IF NOT EXISTS website TEXT;

-- A bank that wants in but hasn't been given a passkey yet -- the public, unauthenticated "request access"
-- form posts here (POST /signup-requests); the owner backoffice lists pending rows and turns an approval
-- into a real `banks` row (generating its passkey) in one action. Deliberately its own table, not a status
-- column tacked onto `banks`, since a request isn't a tenant yet and never silently becomes one without the
-- owner acting on it.
CREATE TABLE IF NOT EXISTS signup_requests (
    id            SERIAL PRIMARY KEY,
    bank_name     TEXT        NOT NULL,
    contact_name  TEXT        NOT NULL,
    contact_email TEXT        NOT NULL,
    note          TEXT        NOT NULL DEFAULT '',
    status        TEXT        NOT NULL DEFAULT 'pending',   -- pending | approved | declined
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    decided_at    TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS signup_requests_status_idx ON signup_requests (status);
-- Same registration detail as banks' own new columns (see the ALTER TABLE block above `banks`) -- carried on
-- the REQUEST too, not just the eventual tenant, so approving a request (admin_approve_signup_request) can
-- copy it straight across instead of asking the bank to re-enter it a second time.
ALTER TABLE signup_requests ADD COLUMN IF NOT EXISTS registration_number TEXT;
ALTER TABLE signup_requests ADD COLUMN IF NOT EXISTS country TEXT;
ALTER TABLE signup_requests ADD COLUMN IF NOT EXISTS region TEXT;
ALTER TABLE signup_requests ADD COLUMN IF NOT EXISTS address TEXT;
ALTER TABLE signup_requests ADD COLUMN IF NOT EXISTS website TEXT;

-- Per-person accounts, layered on top of (not replacing) the shared admin/analyst/bank passkeys above. A
-- passkey still works exactly as before -- this exists so one compromised or departing person's access can be
-- revoked without rotating an entire bank's passkey and handing out a new one to everybody else there. `bank`
-- is NULL for a platform-level account (an owner-side admin); a bank-scoped account always has role='analyst'
-- (there is no bank-scoped admin -- see admin_invite_user()'s own check). `password_hash`/`password_salt` are
-- NULL until the invite is accepted (see accept-invite below); `status='invited'` cannot sign in either way.
--
-- 'pending' is the other way a row here gets created: a bank's own analyst self-registering on the public
-- /analyst-signup page (POST /auth/signup) rather than being invited by the owner. That row already HAS a
-- password (the person set it themselves at signup) but stays unable to sign in -- /auth/login only accepts
-- status='active' -- until the owner reviews and approves it from the "People" section of /owner. This is the
-- enforcement point behind "only people whose details are registered get an account, and every account is
-- tied to exactly one specific bank chosen at signup, never guessed, never shared across banks": a passkey
-- still proves a ROLE and a TENANT the way it always did, but a per-person account proves a specific person
-- vouched for, one bank at a time, by the owner.
CREATE TABLE IF NOT EXISTS users (
    id              SERIAL PRIMARY KEY,
    bank            TEXT,
    name            TEXT        NOT NULL,
    email           TEXT        NOT NULL,
    role            TEXT        NOT NULL DEFAULT 'analyst',   -- admin | analyst
    status          TEXT        NOT NULL DEFAULT 'invited',   -- invited | pending | active | disabled
    password_hash   TEXT,
    password_salt   TEXT,
    totp_secret     TEXT,
    totp_enabled    BOOLEAN     NOT NULL DEFAULT false,
    invite_token    TEXT,
    invite_expires  TIMESTAMPTZ,
    reset_token     TEXT,
    reset_expires   TIMESTAMPTZ,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE UNIQUE INDEX IF NOT EXISTS users_email_idx ON users (lower(email));
CREATE INDEX IF NOT EXISTS users_invite_token_idx ON users (invite_token);
CREATE INDEX IF NOT EXISTS users_reset_token_idx ON users (reset_token);

-- One row per login attempt (success or failure), used only to answer "how many recent failures for this
-- identifier" -- see _check_login_rate_limit() below. Pruned lazily (a DELETE of rows older than the lockout
-- window runs inside the same check, so this table never grows unbounded even with no separate cron job).
CREATE TABLE IF NOT EXISTS login_attempts (
    id          SERIAL PRIMARY KEY,
    identifier  TEXT        NOT NULL,   -- IP address, or IP address + the passkey/email tried, see caller
    success     BOOLEAN     NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS login_attempts_identifier_idx ON login_attempts (identifier, created_at);

-- Identity verification checks (see check_identity() below) -- "does this ID number actually belong to a
-- real, living person, and does that person's real name match the name on this account". Deliberately never
-- stores the raw national ID number: only a SHA-256 hash (for "has this exact ID been checked before")
-- plus the last 4 characters for an officer to visually confirm which ID a row refers to. The hash still
-- lets two different accounts that share one stolen ID surface as a pattern without this table itself
-- becoming a second place a real ID number could leak from.
CREATE TABLE IF NOT EXISTS identity_checks (
    id              SERIAL PRIMARY KEY,
    account_id      TEXT        NOT NULL,
    bank            TEXT        NOT NULL DEFAULT 'unknown',
    claimed_name    TEXT        NOT NULL,
    id_number_hash  TEXT        NOT NULL,
    id_number_last4 TEXT        NOT NULL,
    provider        TEXT        NOT NULL,   -- which IDENTITY_PROVIDER ran this check, or 'unconfigured'
    status          TEXT        NOT NULL,   -- unconfigured | match | mismatch | not_found | deceased | provider_error
    detail          TEXT        NOT NULL DEFAULT '',
    raw_response    JSONB,                  -- whatever the provider returned, for an officer who wants the full record
    checked_by      TEXT        NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS identity_checks_account_idx ON identity_checks (account_id);
CREATE INDEX IF NOT EXISTS identity_checks_bank_idx ON identity_checks (bank);
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
# A real gap, found during a later observability review: this service had zero Prometheus metrics at all,
# unlike gnn-detection-api and xai-narrative-api, which both already expose /metrics this same way -- the
# core application layer handling every officer's scan and decision request had no request-rate or latency
# visibility whatsoever. Matches the same pattern exactly for consistency across all three services.
_inst = Instrumentator()
_inst.add(metrics.latency(buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10)))
_inst.add(metrics.requests())
# endpoint="/api/metrics", NOT the library's "/metrics" default: a real bug, found on a real run. Unlike
# gnn-detection-api and xai-narrative-api (both reached through a Traefik PathPrefix route that STRIPS the
# prefix before forwarding, so their own root-level /metrics is what actually gets hit), backend's own Traefik
# rule forwards /api/* WITHOUT stripping it -- matching how every one of its real routes is already registered
# under api = APIRouter(prefix="/api"). Exposing at the bare "/metrics" default put this endpoint at a path
# Traefik never forwards to this container at all, confirmed unreachable (a 404) on a real run before this fix.
_inst.instrument(app).expose(app, endpoint="/api/metrics", include_in_schema=False)
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


# ------------------------------------------------------------------ email notifications
# Off unless SMTP_HOST is set in .env -- same optional-service pattern as audit()/immudb above: nothing here
# ever blocks or fails the request that triggered it. This deployment has no email provider wired up out of
# the box (no SendGrid/Resend/SES account comes bundled, by design -- that's a real external account someone
# has to create), so until SMTP_HOST is set this silently no-ops and the only place a new alert surfaces is
# the console itself. Set SMTP_HOST/SMTP_PORT/SMTP_USER/SMTP_PASSWORD/SMTP_FROM in .env to turn it on.
SMTP_HOST = os.getenv("SMTP_HOST", "")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER", "")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")
SMTP_FROM = os.getenv("SMTP_FROM", "alerts@xai-amlbench.local")


def notify_email(to: list[str], subject: str, body: str) -> None:
    """Best-effort email notification -- never raises, never blocks the caller on a slow/down SMTP server
    beyond its own short timeout. `to` with no addresses is a silent no-op (a bank that hasn't set a contact
    email yet just doesn't get notified, same as SMTP_HOST being unset entirely)."""
    if not SMTP_HOST or not to:
        return
    try:
        import smtplib
        from email.mime.text import MIMEText
        msg = MIMEText(body)
        msg["Subject"] = subject
        msg["From"] = SMTP_FROM
        msg["To"] = ", ".join(to)
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=8) as s:
            s.starttls()
            if SMTP_USER:
                s.login(SMTP_USER, SMTP_PASSWORD)
            s.sendmail(SMTP_FROM, to, msg.as_string())
    except Exception as exc:  # noqa: BLE001 -- a notification failing must never fail the alert/scan it rode in on
        log.warning("email notification failed (%s)", exc.__class__.__name__)


# ------------------------------------------------------------------ identity verification (ID-theft check)
# Answers one question only: does this claimed name/ID number pair match a real identity on file with
# whichever KYC/identity-verification provider you've configured below (e.g. Smile ID, which backs its Kenya
# checks with the government's own IPRS registry)? It does NOT and CANNOT answer "is the real owner aware
# this account exists" -- no API can; that is a manual follow-up step, which is why a mismatch below is
# surfaced as "needs manual follow-up", never as a verdict. It also does NOT reliably know whether the real
# ID holder has died -- some providers expose a deceased flag from the civil registry and some don't; this
# adapter passes through whatever the provider actually returns rather than inventing a field for one that
# doesn't.
#
# Unconfigured by default (IDENTITY_PROVIDER unset) -- every check then returns status="unconfigured" rather
# than silently pretending to have verified something. Once you have a real account with a provider:
#   IDENTITY_PROVIDER=<any name, just a label for the UI/audit trail>
#   IDENTITY_API_URL=<the provider's real endpoint for a single identity/ID-number lookup>
#   IDENTITY_API_KEY=<your key from that provider>
#
# IMPORTANT: the request/response shape below is a generic placeholder (name + id_number in, a few common
# field names read back out) -- it is NOT verified against any specific provider's real API, because no
# provider was named or an account provided. Every real KYC provider (Smile ID, Onfido, Trulioo, ...) has its
# own exact request format, auth header, and response field names, documented once you have an account with
# them. Before this goes live: open that provider's API reference and adjust the request body built below and
# the field names read out of `data` to match it exactly -- this function is the ONLY place that needs to
# change.
IDENTITY_PROVIDER = os.getenv("IDENTITY_PROVIDER", "")
IDENTITY_API_URL = os.getenv("IDENTITY_API_URL", "")
IDENTITY_API_KEY = os.getenv("IDENTITY_API_KEY", "")


def check_identity(claimed_name: str, id_number: str) -> dict:
    """Returns {status, detail, raw}. status is one of:
      unconfigured   -- IDENTITY_PROVIDER/IDENTITY_API_URL/IDENTITY_API_KEY not set
      match          -- provider confirms a real identity with this ID number, name matches
      mismatch       -- provider found a real identity with this ID number, but the name does NOT match
                        (the classic stolen-ID pattern: a real ID number, a different real owner)
      not_found      -- provider has no record of this ID number at all
      deceased       -- provider's own record marks the real ID holder as deceased (ONLY set if the
                        provider actually returns such a flag -- never inferred)
      provider_error -- the provider call itself failed (network, auth, bad response) -- NOT the same as
                        "not_found"; this means we don't actually know yet, re-run once the provider is reachable
    Never raises -- a provider outage must surface as a status an officer can see and act on, not a 500 that
    takes the rest of the check down with it."""
    if not (IDENTITY_PROVIDER and IDENTITY_API_URL and IDENTITY_API_KEY):
        return {"status": "unconfigured",
                "detail": "No identity-verification provider is configured (set IDENTITY_PROVIDER, "
                          "IDENTITY_API_URL, IDENTITY_API_KEY in .env). This account's identity has NOT been "
                          "verified against any external source.",
                "raw": None}
    try:
        r = httpx.post(
            IDENTITY_API_URL,
            headers={"Authorization": f"Bearer {IDENTITY_API_KEY}", "Content-Type": "application/json"},
            json={"name": claimed_name, "id_number": id_number},   # <-- adjust to the real provider's request shape
            timeout=20,
        )
        r.raise_for_status()
        data = r.json()
    except httpx.HTTPError as exc:
        log.warning("identity provider call failed (%s)", exc.__class__.__name__)
        return {"status": "provider_error", "detail": f"{exc.__class__.__name__}: {exc}", "raw": None}

    # <-- adjust these field reads to the real provider's response shape
    found = bool(data.get("found", data.get("exists")))
    name_match = bool(data.get("name_match", data.get("match")))
    deceased = bool(data.get("deceased")) if "deceased" in data else None

    if not found:
        return {"status": "not_found",
                "detail": "No record of this ID number was found with the configured provider.", "raw": data}
    if deceased:
        return {"status": "deceased",
                "detail": "The provider's own record marks this ID's registered owner as deceased.", "raw": data}
    if not name_match:
        return {"status": "mismatch",
                "detail": "This ID number is real, but its registered owner's name does NOT match the name "
                          "on this account. This needs manual follow-up with the real owner of record -- no "
                          "system can confirm on its own whether they are aware of this account.", "raw": data}
    return {"status": "match", "detail": "The claimed name matches this ID number's registered owner.", "raw": data}


def _store_identity_check(account_id: str, bank: str, claimed_name: str, id_number: str, result: dict, by: str) -> None:
    id_hash = hashlib.sha256(id_number.encode()).hexdigest()
    with db() as conn:
        conn.execute(
            "INSERT INTO identity_checks (account_id, bank, claimed_name, id_number_hash, id_number_last4, "
            "provider, status, detail, raw_response, checked_by) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
            (account_id, bank, claimed_name, id_hash, id_number[-4:], IDENTITY_PROVIDER or "unconfigured",
             result["status"], result["detail"], Jsonb(result["raw"]) if result["raw"] is not None else None, by),
        )


def _auto_identity_checks(rows: "list[IntakeIngestRow]", bank: str, by: str) -> None:
    """The automatic half of identity verification: runs check_identity() the moment a row arrives with both
    claimed_name and id_number set, with no officer action at all -- this is what makes identity checking
    "automatical by the model" for any ingestion path that supplies those two fields (today: /intake/ingest
    and /system/live-ingest; see IntakeIngestRow's own docstring on why `src` is the account checked). One
    identity check per distinct account per batch, even if several rows repeat it. Never raises -- a provider
    hiccup on one account must not fail the transaction ingestion it rode in on; see check_identity()'s own
    docstring for why a provider error is its own status, not silently swallowed."""
    seen: set[str] = set()
    for r in rows:
        if not (r.claimed_name and r.id_number) or r.src in seen:
            continue
        seen.add(r.src)
        try:
            result = check_identity(r.claimed_name, r.id_number)
            _store_identity_check(r.src, bank, r.claimed_name, r.id_number, result, by)
        except Exception as exc:  # noqa: BLE001
            log.warning("auto identity check failed for %s (%s)", r.src, exc.__class__.__name__)


def _attach_identity_status(alerts: list[dict]) -> list[dict]:
    """Merges each alert's most recent identity_checks result (if any) in as identity_status/identity_detail,
    so an officer sees a mule account's ID-verification result right on the alert itself, without a separate
    trip to the Identity Check page. Both fields are simply absent (None) on an alert whose account has never
    had an identity check run -- that absence means "nothing checked yet", never "confirmed clean"."""
    if not alerts:
        return alerts
    account_ids = sorted({a["account_id"] for a in alerts})
    with db() as conn:
        rows = conn.execute(
            "SELECT DISTINCT ON (account_id) account_id, status, detail FROM identity_checks "
            "WHERE account_id = ANY(%s) ORDER BY account_id, created_at DESC", (account_ids,),
        ).fetchall()
    by_account = {r["account_id"]: r for r in rows}
    for a in alerts:
        hit = by_account.get(a["account_id"])
        a["identity_status"] = hit["status"] if hit else None
        a["identity_detail"] = hit["detail"] if hit else None
    return alerts


def tag_accounts_bank(account_ids: list[str], bank: str) -> None:
    """Records which bank each of these accounts came from, first-seen-wins (see account_banks' own comment
    in SCHEMA above). Called from both ingestion paths -- /intake/ingest (Option 1, one bank per pasted
    document) and /system/live-ingest (Option 2, per Kafka message or a per-deployment default) -- so every
    account that enters the live graph through either route gets attributed, not just one of them."""
    if not account_ids:
        return
    with db() as conn:
        conn.executemany(
            "INSERT INTO account_banks (account_id, bank) VALUES (%s, %s) ON CONFLICT (account_id) DO NOTHING",
            [(a, bank) for a in account_ids],
        )


def _bank_contact_emails(bank: str) -> list[str]:
    """notify_email()'s recipient list for a given bank -- [] (silent no-op) for 'unknown', an env-configured
    BANK_PASSKEYS tenant (no row in `banks` to hold an email), or a DB-backed tenant that hasn't set one."""
    if bank in ("unknown", ""):
        return []
    with db() as conn:
        row = conn.execute("SELECT contact_email FROM banks WHERE name = %s", (bank,)).fetchone()
    return [row["contact_email"]] if row and row["contact_email"] else []


# ------------------------------------------------------------------ passwords (per-person accounts)
# hashlib.scrypt is stdlib -- no bcrypt/argon2 C-extension dependency to install on a box that may not have a
# compiler handy. scrypt is a well-regarded KDF in its own right (RFC 7914); these cost parameters (N=2**14,
# r=8, p=1) match the values that RFC recommends for an interactive login, where the hash needs to take a
# noticeable fraction of a second, not longer -- a slower setting would make every login sluggish.
_SCRYPT_N, _SCRYPT_R, _SCRYPT_P = 2 ** 14, 8, 1


def _hash_password(password: str, salt_hex: str | None = None) -> tuple[str, str]:
    """Returns (hash_hex, salt_hex). Pass salt_hex back in to verify an existing password against the same
    salt; omit it to generate a new salt when setting a password for the first time."""
    salt = bytes.fromhex(salt_hex) if salt_hex else secrets.token_bytes(16)
    digest = hashlib.scrypt(password.encode(), salt=salt, n=_SCRYPT_N, r=_SCRYPT_R, p=_SCRYPT_P, dklen=32)
    return digest.hex(), salt.hex()


def _verify_password(password: str, hash_hex: str, salt_hex: str) -> bool:
    candidate, _ = _hash_password(password, salt_hex)
    return hmac.compare_digest(candidate, hash_hex)


# ------------------------------------------------------------------ login rate limiting / lockout
# Postgres-backed (not Redis/in-memory) so this works the same whether the backend is one process or several,
# and survives a restart -- see login_attempts in SCHEMA. Deliberately conservative defaults: a genuine user
# who mistypes a passkey a couple of times is barely affected, but repeated automated guessing hits the wall
# fast. Overridable via env for a deployment that wants a different balance.
LOGIN_LOCKOUT_MAX_FAILURES = int(os.getenv("LOGIN_LOCKOUT_MAX_FAILURES", "5"))
LOGIN_LOCKOUT_WINDOW_S = int(os.getenv("LOGIN_LOCKOUT_WINDOW_S", str(15 * 60)))


def _check_login_rate_limit(identifier: str) -> None:
    """Raises 429 if `identifier` (caller IP, optionally combined with the passkey/email tried -- see each
    caller) has LOGIN_LOCKOUT_MAX_FAILURES or more failed attempts within the last LOGIN_LOCKOUT_WINDOW_S
    seconds. Call this BEFORE checking the submitted credential; call _record_login_attempt() AFTER, with the
    real outcome, from every login path (passkey and per-person) so the two can't be guessed around
    separately.

    Fails OPEN on a DB outage (logs and returns, same as the DB-backed-passkey fallback in login()): the DB
    being briefly unreachable is an availability incident, not a reason to make a correct passkey/password
    stop working. The tradeoff is explicit -- during that window, lockout protection is off, not that logins
    themselves fail."""
    try:
        with db() as conn:
            conn.execute("DELETE FROM login_attempts WHERE created_at < now() - interval '1 day'")
            row = conn.execute(
                "SELECT count(*) AS n FROM login_attempts "
                "WHERE identifier = %s AND success = false AND created_at > now() - interval '%s seconds'",
                (identifier, LOGIN_LOCKOUT_WINDOW_S),
            ).fetchone()
    except psycopg.Error:
        log.exception("login rate-limit check failed (DB unreachable) -- failing open, no lockout applied")
        return
    if row["n"] >= LOGIN_LOCKOUT_MAX_FAILURES:
        raise HTTPException(429, "too many failed sign-in attempts -- try again in a few minutes")


def _record_login_attempt(identifier: str, success: bool) -> None:
    try:
        with db() as conn:
            conn.execute(
                "INSERT INTO login_attempts (identifier, success) VALUES (%s, %s)", (identifier, success),
            )
    except psycopg.Error:
        log.exception("could not record login attempt (DB unreachable)")


def _client_ip(request) -> str:
    """Best-effort caller IP for rate-limiting -- X-Forwarded-For first (Traefik/any reverse proxy in front of
    this service sets it), falling back to the direct connection. Not used for anything security-critical
    beyond "which bucket does this attempt count against" -- a spoofed header at worst lets an attacker split
    their own attempts across more buckets, it can never let them see or touch another identifier's data."""
    fwd = request.headers.get("x-forwarded-for")
    if fwd:
        return fwd.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def lookup_bank(account_id: str) -> str:
    """'unknown' for an account that predates bank attribution (e.g. a SynthAML training/demo account) or was
    never tagged for some other reason -- never guessed."""
    with db() as conn:
        row = conn.execute("SELECT bank FROM account_banks WHERE account_id = %s", (account_id,)).fetchone()
    return row["bank"] if row else "unknown"


def accounts_for_bank(bank: str) -> list[str]:
    """Every account_id ever tagged as belonging to `bank` -- the exact set POST /intake/clear removes from
    the live graph. Deliberately reads account_banks rather than anything in the live graph itself, since that
    graph has no concept of banks at all (see gnn_aml_core.livegraph's module docstring); this table is the
    ONLY place that attribution exists."""
    with db() as conn:
        rows = conn.execute("SELECT account_id FROM account_banks WHERE bank = %s", (bank,)).fetchall()
    return [r["account_id"] for r in rows]


def untag_accounts_bank(account_ids: list[str]) -> None:
    """Removes account_banks rows for accounts that have just been cleared out of the live graph. Deliberately
    touches ONLY this table -- `alerts` and `anomaly_alerts` (and the immudb audit trail) are never written to
    here, so a bank's already-raised alerts and their decision history survive a clear exactly as confirmed:
    the account disappears from the live network view and becomes unscannable, but nothing about what was
    already found or decided about it is lost."""
    if not account_ids:
        return
    with db() as conn:
        conn.execute("DELETE FROM account_banks WHERE account_id = ANY(%s)", (account_ids,))


def _call(method: str, url: str, timeout: float = 30, **kw):
    try:
        r = httpx.request(method, url, timeout=timeout, **kw)
    except httpx.HTTPError as exc:
        raise HTTPException(502, f"upstream unreachable: {exc.__class__.__name__}") from exc
    if r.status_code >= 400:
        raise HTTPException(r.status_code, r.text)
    return r.json()


DEMO_LOGIN_BYPASS = os.getenv("DEMO_LOGIN_BYPASS", "").lower() in ("1", "true", "yes")
if DEMO_LOGIN_BYPASS:
    log.warning("=" * 70)
    log.warning("DEMO_LOGIN_BYPASS is ON: every request is accepted with NO real credential check at all.")
    log.warning("This is for a reviewer demo ONLY. Remove DEMO_LOGIN_BYPASS from .env and restart to")
    log.warning("restore real per-officer authentication before this touches anything but a throwaway demo.")
    log.warning("=" * 70)


# ------------------------------------------------------------------ passkey + name auth
ADMIN_PASSKEY = os.getenv("ADMIN_PASSKEY")
ANALYST_PASSKEY = os.getenv("ANALYST_PASSKEY")

# Multi-tenant SaaS login: one passkey per bank, so each institution gets its own door into this console
# without a user-account database to run. BANK_PASSKEYS is a JSON object in .env, {"Bank name": "passkey"},
# e.g. BANK_PASSKEYS={"Equity Bank Kenya":"ek-2026-xyz","Acme Bank":"acme-2026-abc"}. Logging in with one of
# these passkeys gets role="analyst" same as ANALYST_PASSKEY, but with a bank bound into the session token --
# every data-reading and data-writing route below checks that bank and silently scopes to it, so a Bank A
# session can never see, scan, decide on, or tag as its own anything belonging to Bank B. ADMIN_PASSKEY and
# the legacy shared ANALYST_PASSKEY are unaffected by any of this: both still see every bank's data, same as
# before this feature existed -- ADMIN_PASSKEY is meant for the person operating this deployment, ANALYST_PASSKEY
# stays as a global/legacy role for a single-institution deployment that doesn't need tenant isolation at all.
_BANK_PASSKEYS_RAW = os.getenv("BANK_PASSKEYS", "")
BANK_NAME_TO_PASSKEY: dict[str, str] = {}
if _BANK_PASSKEYS_RAW.strip():
    try:
        parsed = json.loads(_BANK_PASSKEYS_RAW)
        if not isinstance(parsed, dict):
            raise ValueError("BANK_PASSKEYS must be a JSON object of {\"bank name\": \"passkey\"}")
        BANK_NAME_TO_PASSKEY = {str(k): str(v) for k, v in parsed.items() if str(k).strip() and str(v).strip()}
    except (json.JSONDecodeError, ValueError) as exc:
        log.warning("BANK_PASSKEYS is set but not valid JSON (%s) -- no per-bank logins will work until this "
                    "is fixed. Expected format: {\"Bank name\": \"passkey\", ...}", exc)
# Built once, not per-login: passkey -> bank name, so checking a submitted passkey against every configured
# bank is an O(1) lookup rather than a loop over every tenant on every login attempt. An empty-string passkey
# is never a valid match (login() never gets this far with one -- see LoginRequest/the empty-passkey check
# elsewhere), but it's excluded here too so a misconfigured BANK_PASSKEYS value can never become one.
PASSKEY_TO_BANK: dict[str, str] = {v: k for k, v in BANK_NAME_TO_PASSKEY.items() if v}

AUTH_TOKEN_SECRET = os.getenv("AUTH_TOKEN_SECRET")
if not AUTH_TOKEN_SECRET:
    AUTH_TOKEN_SECRET = secrets.token_urlsafe(32)
    log.warning("AUTH_TOKEN_SECRET not set -- generated a random one for this process. Every session this "
               "service issues becomes invalid the moment the process restarts (everyone gets signed out). "
               "Set AUTH_TOKEN_SECRET in .env so sessions survive a restart.")
SESSION_MAX_AGE_S = int(os.getenv("SESSION_MAX_AGE_S", 12 * 3600))


def _sign(payload: str) -> str:
    return hmac.new(AUTH_TOKEN_SECRET.encode(), payload.encode(), hashlib.sha256).hexdigest()


def issue_session_token(name: str, role: str, bank: str | None = None, uid: int | None = None) -> str:
    """A stateless, server-signed session token: base64(json payload) + '.' + HMAC-SHA256 of that payload.
    Nothing is stored server-side -- there is no session table to manage, expire or look up, which is the
    whole appeal of a predefined-passkey login over the per-officer database-backed API keys this replaces.
    The signature is what stops a caller from just typing a different name into the request later and having
    it believed: the name is bound into the token at issue time, verified on every later request, never
    re-read from anything the caller sends after login.

    `bank` is None for ADMIN_PASSKEY/ANALYST_PASSKEY logins (sees every bank's data, exactly as before this
    field existed) and a specific bank name for a BANK_PASSKEYS login -- bound into the token the same
    tamper-evident way `name` is, so a tenant session can't be widened by anything the caller sends later.

    `uid` is set only for a per-person account (POST /auth/login) -- it is the `users.id` row this session
    belongs to, and is what lets 2FA setup/enable and "disable this one person" (admin_set_user_status) apply
    to the right account. A legacy shared-passkey session has no uid at all (None), which is also how
    get_current_user_id() tells the two kinds of session apart -- a passkey session can never pass that check,
    by design: there's no individual account behind it to act on."""
    payload = json.dumps({"name": name, "role": role, "bank": bank, "uid": uid, "iat": time.time()}, sort_keys=True)
    encoded = base64.urlsafe_b64encode(payload.encode()).decode()
    return f"{encoded}.{_sign(payload)}"


def _decode_session_token(token: str) -> dict:
    try:
        encoded, sig = token.split(".", 1)
        payload = base64.urlsafe_b64decode(encoded.encode()).decode()
    except Exception as exc:  # noqa: BLE001 -- any malformed token (wrong shape, bad base64, ...) is just an
                              # invalid session from the caller's point of view, not a server error
        raise HTTPException(401, "malformed session token") from exc
    if not hmac.compare_digest(_sign(payload), sig):
        raise HTTPException(401, "invalid session token (bad signature)")
    data = json.loads(payload)
    if time.time() - data["iat"] > SESSION_MAX_AGE_S:
        raise HTTPException(401, "session expired -- please sign in again")
    return data


def get_current_officer(authorization: str | None = Header(None)) -> str:
    """Every route but /health and /login depends on this. Returns the NAME the person typed at login, bound
    into a server-signed session token -- identity for the audit trail still comes from an authenticated
    credential (the signature), never from request input a caller could set to anything, exactly as before
    this module moved from per-officer database-managed API keys to a predefined admin/analyst passkey.

    DEMO_LOGIN_BYPASS (off by default) skips all of that for a reviewer demo: every request is treated as a
    fixed "Reviewer (DEMO MODE)" identity with no credential needed at all. The returned name deliberately says
    "DEMO MODE" rather than a real-looking name, so it is unmistakable in the UI and the audit trail that these
    actions were not actually authenticated -- the whole point of this project's own earlier security work was
    making sure the audit trail's identity is always genuine, so this bypass must never look like it still is.
    """
    if DEMO_LOGIN_BYPASS:
        return "Reviewer (DEMO MODE)"
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "missing or malformed Authorization header (expected 'Bearer <session-token>')")
    token = authorization.removeprefix("Bearer ").strip()
    if not token:
        raise HTTPException(401, "empty session token")
    return _decode_session_token(token)["name"]


def get_current_role(authorization: str | None = Header(None)) -> str:
    """Same token, but returns the ROLE ('admin'/'analyst') instead of the name -- for the handful of routes
    that need to know which passkey was used, not just who's signed in. Not currently used to gate anything
    (both roles can do everything below), but kept available since the login already carries it."""
    if DEMO_LOGIN_BYPASS:
        return "admin"
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "missing or malformed Authorization header")
    token = authorization.removeprefix("Bearer ").strip()
    return _decode_session_token(token)["role"]


def get_current_bank(authorization: str | None = Header(None)) -> str | None:
    """The tenant boundary. None means "see everything" (ADMIN_PASSKEY, the legacy shared ANALYST_PASSKEY, or
    DEMO_LOGIN_BYPASS) -- a specific bank name means this session logged in with that bank's own BANK_PASSKEYS
    passkey and every route below must silently scope its data to that bank alone. Every route that reads or
    writes bank-attributable data (alerts, anomalies, activity, intake, scan, explain) depends on this
    alongside get_current_officer; a handful of genuinely platform-wide routes (system/status, model/info,
    accounts/sample) deliberately do not, since nothing about them is any one bank's data."""
    if DEMO_LOGIN_BYPASS:
        return None
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "missing or malformed Authorization header (expected 'Bearer <session-token>')")
    token = authorization.removeprefix("Bearer ").strip()
    if not token:
        raise HTTPException(401, "empty session token")
    return _decode_session_token(token).get("bank")


def get_current_user_id(authorization: str | None = Header(None)) -> int:
    """Only a per-person session (issued by POST /auth/login) carries a uid -- a shared admin/analyst/bank
    passkey session never does, and is refused here with 403 rather than silently treated as "no account":
    2FA and account-settings routes act on a specific users row, and there is no such row for a passkey
    session to act on. DEMO_LOGIN_BYPASS has no uid either and is refused the same way -- 2FA on a bypass
    identity would be meaningless."""
    if DEMO_LOGIN_BYPASS:
        raise HTTPException(403, "not available under DEMO_LOGIN_BYPASS")
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "missing or malformed Authorization header")
    token = authorization.removeprefix("Bearer ").strip()
    uid = _decode_session_token(token).get("uid")
    if uid is None:
        raise HTTPException(403, "this action requires an individual account, not a shared passkey session")
    return uid


# ------------------------------------------------------------------ schemas
class ScanRequest(BaseModel):
    account_id: str
    force_alert: bool = False       # create an alert even if the score is below the threshold


class DecisionRequest(BaseModel):
    decision: Literal["confirmed", "dismissed"]
    note: str = Field("", max_length=2000)


class LoginRequest(BaseModel):
    passkey: str
    name: str = Field(..., min_length=1, max_length=200)


class CreateBankRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    contact_email: str | None = Field(None, max_length=320)
    registration_number: str | None = Field(None, max_length=200)
    country: str | None = Field(None, max_length=200)
    region: str | None = Field(None, max_length=200)
    address: str | None = Field(None, max_length=500)
    website: str | None = Field(None, max_length=300)


class BankStatusRequest(BaseModel):
    status: Literal["active", "suspended"]


class SignupRequestBody(BaseModel):
    bank_name: str = Field(..., min_length=1, max_length=200)
    contact_name: str = Field(..., min_length=1, max_length=200)
    contact_email: str = Field(..., min_length=3, max_length=320)
    note: str = Field("", max_length=2000)
    # Full registration detail -- see the comment above `banks`' own new columns in SCHEMA for why this exists:
    # so the owner governing several banks can actually tell which bank this is and where, not just a name.
    # All optional: a bank filling this in on a phone, in a hurry, may not have every field on hand right now.
    registration_number: str | None = Field(None, max_length=200)
    country: str | None = Field(None, max_length=200)
    region: str | None = Field(None, max_length=200)
    address: str | None = Field(None, max_length=500)
    website: str | None = Field(None, max_length=300)


class AnalystSignupRequest(BaseModel):
    """The public, self-service counterpart to admin_invite_user() below -- a bank's own analyst registers
    THEMSELVES, picking which already-onboarded bank they belong to, instead of waiting for the owner to type
    their details in first. Deliberately still gated: this creates a status='pending' row (see users' own
    comment in SCHEMA) that cannot sign in (/auth/login requires status='active') until the owner reviews and
    approves it on /owner -- self-registering proves nothing on its own about who someone actually is, only
    the owner's approval does. `password` is set by the person themselves, right here, not issued later the
    way an invited account's is -- there is no separate accept-invite step for this path."""
    name: str = Field(..., min_length=1, max_length=200)
    email: str = Field(..., min_length=3, max_length=320)
    password: str = Field(..., min_length=10, max_length=200)
    bank: str = Field(..., min_length=1, max_length=200)


# ------------------------------------------------------------------ schemas: per-person accounts
class InviteUserRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    email: str = Field(..., min_length=3, max_length=320)
    # Omitted/None => a platform-level admin (owner backoffice). Set => a bank-scoped analyst for that bank.
    # There is no "bank-scoped admin": a person managing one bank's own people uses the admin role at the
    # platform level, or just shares that bank's passkey -- see admin_invite_user()'s own check.
    bank: str | None = Field(None, max_length=200)
    role: Literal["admin", "analyst"] = "analyst"


class UserStatusRequest(BaseModel):
    status: Literal["active", "disabled"]


class AcceptInviteRequest(BaseModel):
    token: str = Field(..., min_length=1)
    password: str = Field(..., min_length=10, max_length=200)


class ForgotPasswordRequest(BaseModel):
    email: str = Field(..., min_length=3, max_length=320)


class ResetPasswordRequest(BaseModel):
    token: str = Field(..., min_length=1)
    password: str = Field(..., min_length=10, max_length=200)


class AuthLoginRequest(BaseModel):
    email: str = Field(..., min_length=3, max_length=320)
    password: str = Field(..., min_length=1, max_length=200)
    totp_code: str | None = Field(None, max_length=10)


class Totp2faVerifyRequest(BaseModel):
    code: str = Field(..., min_length=1, max_length=10)


class AnomalyNoteRequest(BaseModel):
    note: str = Field(..., min_length=1, max_length=4000)


class AnomalyResolveRequest(BaseModel):
    outcome: Literal["confirmed_mule", "cleared"]
    note: str = Field("", max_length=4000)


class IntakeIngestRow(BaseModel):
    src: str
    dst: str
    amount: float
    timestamp: int | None = None
    cross_border: int | None = None
    bank: str | None = None   # per-row override -- lets one Kafka batch carry transactions from several
                              # banks at once (each message says which); /intake/ingest's paste path doesn't
                              # use this, since one pasted document is always treated as one bank
    # Both optional, and both ONLY meaningful together -- when a row carries the account holder's claimed
    # name and national ID number, _auto_identity_checks() (see below) runs a real identity verification for
    # `src` automatically, the moment this row is ingested, with no separate officer step. `src` is treated
    # as the account these describe (the bank's own customer side of the transaction, by convention -- `dst`
    # is the counterparty and is never identity-checked from this field). Neither is ever persisted raw: see
    # check_identity()'s own docstring on why only a hash + last 4 digits of the ID number are ever stored.
    claimed_name: str | None = None
    id_number: str | None = None


class IntakeIngestRequest(BaseModel):
    rows: list[IntakeIngestRow] = Field(min_length=1, max_length=5000)
    bank: str | None = None   # the whole-batch bank/institution name. REQUIRED by /intake/ingest (one
                              # analyst paste = one bank, enforced there); used by /system/live-ingest only
                              # as the fallback for a row that didn't carry its own `bank` field.


class IdentityCheckRequest(BaseModel):
    account_id: str
    claimed_name: str = Field(..., min_length=1, max_length=200)
    id_number: str = Field(..., min_length=1, max_length=64)   # raw value, used in-memory only -- never persisted


class IntakeClearRequest(BaseModel):
    bank: str | None = None   # which bank's accounts to clear. IGNORED for a tenant-scoped session -- always
                              # forced to that tenant's own bank, same as /intake/ingest's `bank` handling --
                              # and REQUIRED for an admin/global session (there is no "own bank" to default to).


# ------------------------------------------------------------------ routes
@api.get("/health")
def health():
    with db() as conn:
        conn.execute("SELECT 1")
    return {"status": "ok"}


@api.post("/login")
def login(req: LoginRequest, request: Request = None):
    """The CUSTOMER/bank door -- deliberately CANNOT issue an admin session, even if someone types the real
    ADMIN_PASSKEY in here. That's not an oversight: a real security finding from a later pass was that this
    endpoint and /admin/login (below) used to be the same door, meaning the platform owner's own passkey was
    reachable from the exact same public page every bank tenant signs in on. If that admin passkey were ever
    guessed, phished, or leaked, the attacker's only way in was already sitting on the one URL every tenant
    already knows about. Splitting this into two separate endpoints (and two separate frontend pages -- see
    /admin) means a leaked admin passkey is useless unless someone also knows to try /admin in the first
    place, and a leaked/guessed ANALYST_PASSKEY or bank passkey typed in here can never escalate to admin
    under any circumstance, because this function no longer has an admin branch to fall into at all.

    Checks the submitted passkey against ANALYST_PASSKEY and every configured BANK_PASSKEYS passkey, in that
    order -- there is no per-person credential to issue or revoke here, by design. The name is NOT verified
    against anything (there's nothing to verify it against -- that's the tradeoff of a shared passkey instead
    of per-officer accounts), but it IS what gets signed into the session token and used for every action
    this person takes afterward, so the audit trail still reflects a specific person's name on every
    decision, not just 'an analyst'.

    A BANK_PASSKEYS match additionally binds that bank into the session (see issue_session_token): everything
    this person does afterward is silently scoped to that one bank, same role ('analyst') as the legacy shared
    passkey but with a real tenant boundary instead of seeing every institution's data.

    Rate-limited per caller IP (see _check_login_rate_limit): `request` defaults to None so this can still be
    called directly the way the test suite does (no real HTTP request object to hand it), in which case rate
    limiting is skipped entirely rather than crashing on a missing client -- a unit test calling this function
    in-process was never the thing this protects against."""
    ip = _client_ip(request) if request is not None else None
    if ip is not None:
        _check_login_rate_limit(ip)
    name = req.name.strip()
    if not name:
        raise HTTPException(400, "name is required")
    bank: str | None = None
    if ANALYST_PASSKEY and req.passkey == ANALYST_PASSKEY:
        role = "analyst"
    elif req.passkey in PASSKEY_TO_BANK:
        role = "analyst"
        bank = PASSKEY_TO_BANK[req.passkey]
    else:
        # Not one of the env-configured passkeys above -- check the DB-backed tenants the owner has added
        # through /owner since this process started (see the `banks` table in SCHEMA). A suspended bank's
        # passkey is deliberately treated exactly like a wrong passkey, not a distinct "account suspended"
        # error: that distinction is the owner's business on the owner page, not something to hand a locked-out
        # caller for free.
        #
        # A DB outage here must not turn a routine "wrong passkey" into an unhandled 500 -- fail closed into
        # the same 401 a real wrong passkey gets, and log it so the owner can tell "nobody could log in because
        # the DB was down" apart from "somebody mistyped a passkey".
        row = None
        try:
            with db() as conn:
                row = conn.execute(
                    "SELECT name FROM banks WHERE passkey = %s AND status = 'active'", (req.passkey,),
                ).fetchone()
        except psycopg.Error:
            log.exception("login: DB-backed passkey lookup failed, treating as invalid passkey")
        if not row:
            if ip is not None:
                _record_login_attempt(ip, False)
            raise HTTPException(401, "invalid passkey")
        role = "analyst"
        bank = row["name"]
    if ip is not None:
        _record_login_attempt(ip, True)
    token = issue_session_token(name, role, bank)
    audit("login", {"name": name, "role": role, "bank": bank})
    return {"token": token, "name": name, "role": role, "bank": bank}


@api.post("/admin/login")
def admin_login(req: LoginRequest, request: Request = None):
    """The PLATFORM OWNER's door -- the only place ADMIN_PASSKEY works at all. See login()'s own docstring
    for why this is a separate endpoint (and the frontend's /admin a separate, unlinked page) rather than one
    shared door: a bank tenant's login page should never be the same surface an attacker could try the admin
    passkey against. Rate-limited the same way login() is, on the same per-IP bucket (a brute-force attempt
    against either door counts against the same lockout, so splitting the endpoint doesn't also split an
    attacker's attempt budget in their favor)."""
    ip = _client_ip(request) if request is not None else None
    if ip is not None:
        _check_login_rate_limit(ip)
    name = req.name.strip()
    if not name:
        raise HTTPException(400, "name is required")
    if not (ADMIN_PASSKEY and req.passkey == ADMIN_PASSKEY):
        if ip is not None:
            _record_login_attempt(ip, False)
        raise HTTPException(401, "invalid passkey")
    if ip is not None:
        _record_login_attempt(ip, True)
    token = issue_session_token(name, "admin", None)
    audit("login", {"name": name, "role": "admin", "bank": None})
    return {"token": token, "name": name, "role": "admin", "bank": None}


@api.get("/whoami")
def whoami(officer: str = Depends(get_current_officer), bank: str | None = Depends(get_current_bank),
           role: str = Depends(get_current_role)):
    """Lets the frontend verify a stored session token still works (e.g. after a page reload), and display who
    is signed in (and which bank's console this is, if any), without needing to decode anything client-side --
    all three come from the signed token, the same way every other route's identity does. `role` is what the
    frontend uses to route an admin session straight to the owner backoffice (/owner) instead of the bank
    console -- see the Next.js app's root page.tsx."""
    return {"officer": officer, "bank": bank, "role": role}


@api.get("/accounts/sample")
def sample_accounts(officer: str = Depends(get_current_officer)):
    return _call("GET", f"{GNN_API_URL}/accounts/sample", params={"n": 5})


@api.get("/model/info")
def model_info(officer: str = Depends(get_current_officer)):
    return _call("GET", f"{GNN_API_URL}/model/info")


@api.get("/activity")
def activity(limit: int = 80, officer: str = Depends(get_current_officer), bank: str | None = Depends(get_current_bank)):
    """A real feedback/activity feed, so an analyst can see what's happening across both pipelines without
    opening a terminal or a database client. Built from data this service already has in Postgres -- it does
    NOT reach into the immudb ledger that audit() writes to (that's a separate, deliberately write-oriented
    tamper-evident store, not meant for casual browsing); this is the operational "what just happened" view,
    the ledger is the forensic "prove what happened" one. Three real sources, merged and sorted by time:
      - Model 1 alerts being created (a scan flagged something)
      - Model 1 alerts being decided (an officer confirmed/dismissed one)
      - Model 2 investigation_log entries (claimed / noted / resolved), flattened out of each anomaly alert's
        own JSONB history so they interleave with everything else on one timeline instead of being buried
        inside each alert's own record.

    A tenant-scoped session only ever sees its own bank's events across all three sources -- Model 1's two
    queries filter on alerts.source_bank directly; Model 2's anomaly_alerts table has no bank column of its
    own (it's keyed by account_id, not by which document/feed the account arrived through), so that query
    joins against account_banks instead. An account with no account_banks row (predates bank attribution)
    simply never matches a tenant-scoped join, which is correct -- it isn't that bank's own activity.
    """
    limit = min(max(limit, 1), 300)
    events: list[dict] = []
    with db() as conn:
        for row in conn.execute(
            "SELECT id, account_id, risk_score, created_at FROM alerts "
            + ("WHERE source_bank = %s " if bank else "")
            + "ORDER BY created_at DESC LIMIT %s",
            ((bank, limit) if bank else (limit,)),
        ).fetchall():
            events.append({
                "ts": row["created_at"].timestamp(), "pipeline": "model1", "kind": "alert_created",
                "account_id": row["account_id"], "by": None,
                "detail": f"flagged (risk {row['risk_score']:.2f})", "ref_id": row["id"],
            })
        for row in conn.execute(
            "SELECT id, account_id, status, reviewed_by, review_note, reviewed_at FROM alerts "
            "WHERE reviewed_at IS NOT NULL " + ("AND source_bank = %s " if bank else "")
            + "ORDER BY reviewed_at DESC LIMIT %s",
            ((bank, limit) if bank else (limit,)),
        ).fetchall():
            events.append({
                "ts": row["reviewed_at"].timestamp(), "pipeline": "model1", "kind": "decision",
                "account_id": row["account_id"], "by": row["reviewed_by"],
                "detail": row["status"] + (f" -- {row['review_note']}" if row["review_note"] else ""),
                "ref_id": row["id"],
            })
        if bank:
            anomaly_q = (
                "SELECT a.id, a.account_id, a.investigation_log FROM anomaly_alerts a "
                "JOIN account_banks ab ON ab.account_id = a.account_id AND ab.bank = %s "
                "WHERE jsonb_array_length(a.investigation_log) > 0 "
                "ORDER BY a.updated_at DESC NULLS LAST LIMIT %s"
            )
            anomaly_args = (bank, limit)
        else:
            anomaly_q = (
                "SELECT id, account_id, investigation_log FROM anomaly_alerts "
                "WHERE jsonb_array_length(investigation_log) > 0 ORDER BY updated_at DESC NULLS LAST LIMIT %s"
            )
            anomaly_args = (limit,)
        for row in conn.execute(anomaly_q, anomaly_args).fetchall():
            for entry in row["investigation_log"]:
                events.append({
                    "ts": entry["ts"], "pipeline": "model2", "kind": entry["status"],
                    "account_id": row["account_id"], "by": entry.get("by"),
                    "detail": entry.get("note") or entry["status"], "ref_id": row["id"],
                })
    events.sort(key=lambda e: e["ts"], reverse=True)
    return events[:limit]


@api.get("/system/status")
def system_status(officer: str = Depends(get_current_officer)):
    """Live reachability of everything this backend depends on, plus the Kafka topic a real bank feed would
    publish onto -- the "how does live data actually get into this system" page. Two honest, different kinds
    of information: the health checks below are REAL requests made right now; the Kafka fields are
    configuration this deployment already has (docker-compose.yml / .env), shown so an analyst doesn't need to
    go find them in a file -- this endpoint does not itself verify anything is currently publishing to that
    topic."""
    def check(name: str, url: str) -> dict:
        t0 = time.monotonic()
        try:
            httpx.get(url, timeout=4).raise_for_status()
            return {"name": name, "ok": True, "latency_ms": round((time.monotonic() - t0) * 1000, 1)}
        except httpx.HTTPError as exc:
            return {"name": name, "ok": False, "latency_ms": None, "error": exc.__class__.__name__}

    db_ok = True
    try:
        with db() as conn:
            conn.execute("SELECT 1")
    except psycopg.OperationalError:
        db_ok = False

    services = [
        {"name": "database", "ok": db_ok, "latency_ms": None},
        check("gnn-detection-api (Model 1 + Model 2 scoring)", f"{GNN_API_URL}/health"),
        check("xai-narrative-api (SAR narratives)", f"{XAI_API_URL}/health"),
    ]
    return {
        "services": services,
        "all_ok": all(s["ok"] for s in services),
        "kafka": {
            "external_listener": "localhost:29092 (SASL/PLAIN -- KAFKA_EXTERNAL_USER/PASSWORD in .env)",
            "transactions_topic": os.getenv("KAFKA_TOPIC_TRANSACTIONS", "aml.transactions"),
            "alerts_topic": os.getenv("KAFKA_TOPIC_ALERTS", "aml.alerts"),
            "note": "A bank's own feed publishes transaction records onto the transactions topic above. The "
                    "live-feed-consumer service (gnn_aml_core.kafka_ingest) actually consumes this topic and "
                    "adds every transaction's accounts to the live graph through the same trained, frozen "
                    "model the Intake tab uses -- this is a real, wired pipeline now, not just a display of "
                    "config. aml_synth.exporters is this project's own reference producer for that same topic, "
                    "used to generate the synthetic data this console was demoed against; pointing a real "
                    "bank's own producer at this topic with the external-listener credentials above is the "
                    "only step left to go from synthetic to real traffic -- this page cannot do that external "
                    "wiring itself, only show whether this deployment's own services are currently reachable.",
        },
    }


def _check_account_access(account_id: str, bank: str | None) -> None:
    """Raises if a tenant-scoped session (bank is not None) tries to touch an account that belongs to a
    DIFFERENT bank. An account with no attribution at all ("unknown" -- predates bank tagging, e.g. a SynthAML
    benchmark/demo account) stays visible to every tenant, since it isn't any one bank's real data to begin
    with. 404, not 403: this must not even confirm that the account exists under another bank's ownership."""
    if bank is None:
        return
    owner = lookup_bank(account_id)
    if owner not in (bank, "unknown"):
        raise HTTPException(404, "account not found")


@api.get("/accounts/{account_id}/explain")
def explain_account(account_id: str, officer: str = Depends(get_current_officer),
                    bank: str | None = Depends(get_current_bank)):
    """GNNExplainer subgraph + edge importances, drawn as the network view in the UI."""
    _check_account_access(account_id, bank)
    return _call("POST", f"{GNN_API_URL}/explain", timeout=120, json={"account_id": account_id})


@api.post("/alerts/scan")
def scan(req: ScanRequest, officer: str = Depends(get_current_officer), bank: str | None = Depends(get_current_bank)):
    _check_account_access(req.account_id, bank)
    pred = _call("POST", f"{GNN_API_URL}/predict", json={"account_ids": [req.account_id]})
    res = pred["results"][0]
    alert = None
    if res["flagged"] or req.force_alert:
        with db() as conn:
            alert = conn.execute(
                "INSERT INTO alerts (account_id, risk_score, source_bank) VALUES (%s, %s, %s) RETURNING *",
                (req.account_id, res["score"], lookup_bank(req.account_id)),
            ).fetchone()
        audit("alert_created", {"alert_id": alert["id"], "account_id": req.account_id, "score": res["score"],
                                "scanned_by": officer})
        notify_email(_bank_contact_emails(alert["source_bank"]), f"New alert #{alert['id']} -- {req.account_id}",
                     f"Account {req.account_id} was flagged (risk score {res['score']:.3f}) and scanned by "
                     f"{officer}. Open the console to review it.")
    # "score"/"threshold" (raw) are what the flagged decision and the stored alert are based on, and stay
    # exactly as before; "calibrated_score"/"calibrated_threshold" are the same pair mapped through isotonic
    # calibration, meaningful as an actual probability -- meant for display (a risk gauge showing "99.9%" for
    # every flagged account is not informative), never used for the flagged decision itself. .get() with a
    # fallback keeps this working against an older gnn-detection-api that predates calibration.
    if alert:
        alert = _attach_identity_status([alert])[0]
    return {"score": res["score"], "calibrated_score": res.get("calibrated_score", res["score"]),
            "flagged": res["flagged"], "threshold": pred["threshold"],
            "calibrated_threshold": pred.get("calibrated_threshold", pred["threshold"]),
            "latency_ms": pred["latency_ms"], "alert": alert}


@api.post("/intake/ingest")
def intake_ingest(req: IntakeIngestRequest, officer: str = Depends(get_current_officer),
                  tenant_bank: str | None = Depends(get_current_bank)):
    """Option 1's real ingestion path: a document of real transactions pasted/uploaded on the Intake tab lands
    here, gets forwarded to gnn-detection-api's POST /admin/ingest-live (which appends them to the live,
    non-synthetic account population, rebuilds graph.pt standardized with the already-trained model's own
    mean/std, and hot-reloads -- see gnn_aml_core.livegraph's module docstring), and the resulting account/
    transaction counts come straight back to the frontend. No model weights are retrained by this call.

    A tenant-scoped session (signed in with a BANK_PASSKEYS passkey) always ingests as its OWN bank -- whatever
    req.bank says is ignored, so one tenant can never tag accounts as belonging to another institution. For an
    admin/global session, req.bank is required (one pasted document = one bank, enforced below); every account
    it touches is recorded in account_banks (tag_accounts_bank()) so every alert raised on it later, by any
    path, can say which bank it came from."""
    bank = tenant_bank or (req.bank or "").strip()
    if not bank:
        raise HTTPException(422, "bank is required -- name the institution this document's accounts belong to")
    gnn_rows = [{k: v for k, v in r.model_dump().items() if k != "bank"} for r in req.rows]
    headers = {"Authorization": f"Bearer {GNN_ADMIN_TOKEN}"} if GNN_ADMIN_TOKEN else {}
    result = _call("POST", f"{GNN_API_URL}/admin/ingest-live", timeout=60, json={"rows": gnn_rows}, headers=headers)
    account_ids = sorted({r.src for r in req.rows} | {r.dst for r in req.rows})
    tag_accounts_bank(account_ids, bank)
    _auto_identity_checks(req.rows, bank, officer)
    audit("intake_ingest", {"ingested_rows": result.get("ingested_rows"), "n_accounts": result.get("n_accounts"),
                            "n_transactions": result.get("n_transactions"), "bank": bank, "by": officer})
    return {**result, "bank": bank}


@api.post("/intake/clear")
def intake_clear(req: IntakeClearRequest, officer: str = Depends(get_current_officer),
                 tenant_bank: str | None = Depends(get_current_bank)):
    """The other half of Option 1 the user asked for: a walk-in/temporary customer's accounts, pasted once for
    a quick scan on the Intake tab, shouldn't have to live in the shared graph forever. This clears every
    account tagged to one bank OUT of the live graph -- forwarded to gnn-detection-api's POST /admin/remove-live
    (gnn_aml_core.livegraph.remove_accounts + a rebuild, same pattern as /admin/ingest-live's own rebuild+reload)
    -- so those accounts become unscannable and vanish from the network view.

    Deliberately preserves the compliance-sensitive half of this bank's history: `alerts`, `anomaly_alerts`,
    and the immudb audit trail are never touched by this call (see untag_accounts_bank()'s own docstring).
    Anything already raised and decided about this bank's accounts stays exactly as it was -- this clears the
    live graph's working data, not the record of what that data already produced.

    A tenant-scoped session (BANK_PASSKEYS) always clears its OWN bank only -- whatever req.bank says is
    ignored, the same guardrail /intake/ingest already applies, so one tenant can never clear another
    institution's accounts. An admin/global session must say which bank (there's no "own bank" to default to
    for it)."""
    bank = tenant_bank or (req.bank or "").strip()
    if not bank:
        raise HTTPException(422, "bank is required -- name the institution whose live-graph accounts to clear")
    account_ids = accounts_for_bank(bank)
    if not account_ids:
        return {"status": "nothing to clear", "bank": bank, "removed_accounts": 0}
    headers = {"Authorization": f"Bearer {GNN_ADMIN_TOKEN}"} if GNN_ADMIN_TOKEN else {}
    result = _call("POST", f"{GNN_API_URL}/admin/remove-live", timeout=60,
                   json={"account_ids": account_ids}, headers=headers)
    untag_accounts_bank(account_ids)
    audit("intake_clear", {"bank": bank, "removed_accounts": len(account_ids), "by": officer})
    return {"status": "cleared", "bank": bank, "removed_accounts": len(account_ids),
            "remaining_live_accounts": result.get("n_accounts"), "remaining_live_transactions": result.get("n_transactions")}


@api.post("/system/live-ingest")
def live_ingest(req: IntakeIngestRequest, authorization: str | None = Header(None)):
    """Option 2's automated half of the same ingestion path /intake/ingest (Option 1) uses -- the
    live-feed-consumer service (gnn_aml_core.kafka_ingest) calls this for every batch it reads off the Kafka
    transactions topic, so a live feed does the exact same thing a pasted document does: it never stops,
    delays, or otherwise touches the transaction itself, it only (a) adds the accounts to the live graph via
    gnn-detection-api's POST /admin/ingest-live, standardized with the already-trained model's own frozen
    mean/std -- no retraining -- and (b) scores every account the batch touched and raises an alert, exactly
    as if an analyst had scanned it from the Scan tab, for the ones that come back flagged. That alert is the
    ENTIRE effect this has on anything: no transaction is blocked, delayed, or altered, here or anywhere else
    in this call chain.

    No officer is signed in to a background consumer, so this route authenticates with the GNN_ADMIN_TOKEN
    shared secret instead of a session token -- the same secret gnn-detection-api already requires on its own
    admin routes, not a new credential."""
    if GNN_ADMIN_TOKEN:
        token = (authorization or "").removeprefix("Bearer ").strip()
        if token != GNN_ADMIN_TOKEN:
            raise HTTPException(401, "missing or invalid Authorization header (expected 'Bearer <GNN_ADMIN_TOKEN>')")
    else:
        log.warning("POST /system/live-ingest called with GNN_ADMIN_TOKEN unset -- this endpoint is "
                    "unauthenticated. Fine for local dev; set GNN_ADMIN_TOKEN before exposing this beyond one machine.")

    headers = {"Authorization": f"Bearer {GNN_ADMIN_TOKEN}"} if GNN_ADMIN_TOKEN else {}
    gnn_rows = [{k: v for k, v in r.model_dump().items() if k != "bank"} for r in req.rows]
    ingest_result = _call("POST", f"{GNN_API_URL}/admin/ingest-live", timeout=60, json={"rows": gnn_rows}, headers=headers)

    # Each row says which bank it came from (a Kafka message can carry its own "bank"/"institution" field);
    # a row with none falls back to req.bank, the per-deployment default a single-topic consumer is
    # configured with (LIVE_FEED_DEFAULT_BANK -- see gnn_aml_core/kafka_ingest.py), and finally "unknown" if
    # neither is set, same as an account that arrives with no attribution at all.
    bank_by_account: dict[str, str] = {}
    for r in req.rows:
        b = (r.bank or req.bank or "unknown").strip() or "unknown"
        bank_by_account.setdefault(r.src, b)
        bank_by_account.setdefault(r.dst, b)
    for b in set(bank_by_account.values()):
        tag_accounts_bank([a for a, ab in bank_by_account.items() if ab == b], b)
        _auto_identity_checks([r for r in req.rows if bank_by_account.get(r.src) == b], b, "system:live-feed")

    account_ids = sorted(bank_by_account)
    alerts_created: list[int] = []
    if account_ids:
        pred = _call("POST", f"{GNN_API_URL}/predict", timeout=60, json={"account_ids": account_ids})
        for account_id, res in zip(account_ids, pred["results"]):
            if not res["flagged"]:
                continue
            with db() as conn:
                alert = conn.execute(
                    "INSERT INTO alerts (account_id, risk_score, source_bank) VALUES (%s, %s, %s) RETURNING *",
                    (account_id, res["score"], lookup_bank(account_id)),
                ).fetchone()
            alerts_created.append(alert["id"])
            audit("alert_created", {"alert_id": alert["id"], "account_id": account_id, "score": res["score"],
                                    "scanned_by": "system:live-feed"})
            notify_email(_bank_contact_emails(alert["source_bank"]), f"New alert #{alert['id']} -- {account_id}",
                         f"Account {account_id} was flagged (risk score {res['score']:.3f}) by the live feed. "
                         f"Open the console to review it.")

    return {**ingest_result, "accounts_scanned": len(account_ids), "alerts_created": alerts_created}


@api.get("/alerts")
def list_alerts(status: str | None = None, bank: str | None = None, limit: int = 100,
                officer: str = Depends(get_current_officer), tenant_bank: str | None = Depends(get_current_bank)):
    # A tenant-scoped session always gets its own bank's alerts, full stop -- the `bank` query param (the
    # Alerts tab's filter dropdown) is only honoured for an admin/global session that can see every bank and
    # is actually choosing to narrow the view. `or` short-circuits correctly here: when tenant_bank is set it
    # always wins, so nothing the caller sends in the query string can widen a tenant session's own view.
    effective_bank = tenant_bank or bank
    q, args, where = "SELECT * FROM alerts", [], []
    if status:
        where.append("status = %s")
        args.append(status)
    if effective_bank:
        where.append("source_bank = %s")
        args.append(effective_bank)
    if where:
        q += " WHERE " + " AND ".join(where)
    q += " ORDER BY created_at DESC LIMIT %s"
    args.append(min(limit, 500))
    with db() as conn:
        rows = conn.execute(q, args).fetchall()
    return _attach_identity_status(rows)


@api.get("/banks")
def list_banks(officer: str = Depends(get_current_officer), tenant_bank: str | None = Depends(get_current_bank)):
    """Every distinct bank name seen so far across alerts, for the Alerts tab's filter dropdown -- not
    account_banks directly, since what an analyst wants to filter by is "banks that actually have alerts",
    not every bank that's ever sent a transaction with nothing flagged in it. A tenant-scoped session only
    ever has one answer here (its own bank) -- no query needed, and no other bank's name is ever revealed to it."""
    if tenant_bank:
        return [tenant_bank]
    with db() as conn:
        rows = conn.execute("SELECT DISTINCT source_bank FROM alerts ORDER BY source_bank").fetchall()
    return [r["source_bank"] for r in rows]


@api.get("/admin/tenants")
def admin_tenants(role: str = Depends(get_current_role), officer: str = Depends(get_current_officer)):
    """Owner-only usage summary, one row per bank tenant -- what the frontend's separate /owner backoffice
    shows instead of the bank-facing console (see frontend/src/app/owner/page.tsx). This is deliberately NOT
    reachable from a bank-tenant or legacy shared-analyst session: 403s unless the token's role is 'admin',
    the same role ADMIN_PASSKEY issues (see issue_session_token / BANK_PASSKEYS above) -- a bank must never be
    able to see another bank's volumes, let alone the full tenant list, through any route.

    A bank counts as a tenant here if it has a row in `banks` (DB-onboarded, including one with zero activity
    yet -- the whole point of letting the owner add a tenant before it's done anything), OR an alert
    attributed to it (alerts.source_bank), OR at least one account tagged to it via Intake/Connect
    (account_banks) -- an env-configured BANK_PASSKEYS tenant has no `banks` row but still shows up once it
    has done either of the other two, same as before this table existed."""
    if role != "admin":
        raise HTTPException(403, "admin passkey required")
    with db() as conn:
        rows = conn.execute("""
            SELECT
                b.bank,
                COALESCE(bk.status, 'active')   AS status,
                bk.registration_number,
                bk.country,
                bk.region,
                bk.website,
                COALESCE(al.total_alerts, 0)    AS total_alerts,
                COALESCE(al.open_alerts, 0)     AS open_alerts,
                COALESCE(an.total_anomalies, 0) AS total_anomalies,
                COALESCE(an.open_anomalies, 0)  AS open_anomalies,
                COALESCE(acc.accounts, 0)       AS accounts,
                GREATEST(al.last_alert, an.last_anomaly) AS last_activity
            FROM (
                SELECT name AS bank FROM banks
                UNION
                SELECT bank FROM account_banks
                UNION
                SELECT source_bank AS bank FROM alerts WHERE source_bank <> 'unknown'
            ) b
            LEFT JOIN banks bk ON bk.name = b.bank
            LEFT JOIN (
                SELECT source_bank AS bank, COUNT(*) AS total_alerts,
                       COUNT(*) FILTER (WHERE status = 'open') AS open_alerts,
                       MAX(created_at) AS last_alert
                FROM alerts GROUP BY source_bank
            ) al ON al.bank = b.bank
            LEFT JOIN (
                SELECT ab.bank, COUNT(*) AS total_anomalies,
                       COUNT(*) FILTER (WHERE aa.status IN ('new', 'under_investigation')) AS open_anomalies,
                       MAX(aa.created_at) AS last_anomaly
                FROM anomaly_alerts aa JOIN account_banks ab ON ab.account_id = aa.account_id
                GROUP BY ab.bank
            ) an ON an.bank = b.bank
            LEFT JOIN (
                SELECT bank, COUNT(*) AS accounts FROM account_banks GROUP BY bank
            ) acc ON acc.bank = b.bank
            ORDER BY b.bank
        """).fetchall()
    return rows


@api.post("/admin/banks")
def admin_create_bank(req: CreateBankRequest, role: str = Depends(get_current_role),
                       officer: str = Depends(get_current_officer)):
    """Owner-only: onboards a new bank tenant without touching .env or restarting anything -- the gap flagged
    as the top priority for a real pilot (BANK_PASSKEYS alone means every new tenant needs a manual .env edit
    and a full backend restart, which briefly drops every OTHER bank's session too). Generates the passkey
    server-side (never accepted from the caller) and returns it exactly once in this response -- same
    "shown once, store it" pattern as every real API-key-issuing SaaS, since this isn't stored in a
    retrievable form here either (plaintext today, see the note below, but still never re-displayed)."""
    if role != "admin":
        raise HTTPException(403, "admin passkey required")
    name = req.name.strip()
    if not name:
        raise HTTPException(400, "bank name is required")
    passkey = secrets.token_urlsafe(18)
    with db() as conn:
        existing = conn.execute("SELECT 1 FROM banks WHERE name = %s", (name,)).fetchone()
        if existing or name in BANK_NAME_TO_PASSKEY:
            raise HTTPException(409, f"a bank named {name!r} already exists")
        conn.execute(
            "INSERT INTO banks (name, passkey, contact_email, registration_number, country, region, address, "
            "website) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
            (name, passkey, req.contact_email, req.registration_number, req.country, req.region, req.address,
             req.website),
        )
    audit("admin_create_bank", {"bank": name, "by": officer})
    return {"name": name, "passkey": passkey, "status": "active"}


@api.patch("/admin/banks/{bank_name}")
def admin_set_bank_status(bank_name: str, req: BankStatusRequest, role: str = Depends(get_current_role),
                          officer: str = Depends(get_current_officer)):
    """Owner-only: suspend (passkey stops working immediately, data untouched) or reactivate a DB-backed
    tenant. Only ever touches the `banks` table -- an env-configured BANK_PASSKEYS tenant has no row here and
    can't be suspended this way, by design (see login()'s comment on why env always wins)."""
    if role != "admin":
        raise HTTPException(403, "admin passkey required")
    with db() as conn:
        row = conn.execute(
            "UPDATE banks SET status = %s WHERE name = %s RETURNING name, status", (req.status, bank_name),
        ).fetchone()
    if not row:
        raise HTTPException(404, f"no DB-backed bank named {bank_name!r} (an env-configured BANK_PASSKEYS "
                                  f"tenant can't be suspended here)")
    audit("admin_set_bank_status", {"bank": bank_name, "status": req.status, "by": officer})
    return row


@api.post("/admin/banks/{bank_name}/regenerate-passkey")
def admin_regenerate_bank_passkey(bank_name: str, role: str = Depends(get_current_role),
                                  officer: str = Depends(get_current_officer)):
    """Owner-only: the recovery lever for the exact mistake of losing a passkey that was only ever shown once
    (admin_create_bank / admin_approve_signup_request's "shown once, store it" reveal) -- rather than having
    to delete the whole tenant and re-add it (which would also hand out a brand new bank name's worth of
    passkey, losing nothing of the bank's own data, but is a clumsy way to just get a new secret). The OLD
    passkey stops working the instant this runs -- same "rotate, don't duplicate" security posture as
    everything else here. Only works for a DB-backed tenant (one with a row in `banks`); an env-configured
    BANK_PASSKEYS tenant has no row to update here, same restriction admin_set_bank_status/admin_delete_bank
    already have, for the same reason (see login()'s comment on why env always wins)."""
    if role != "admin":
        raise HTTPException(403, "admin passkey required")
    new_passkey = secrets.token_urlsafe(18)
    with db() as conn:
        row = conn.execute(
            "UPDATE banks SET passkey = %s WHERE name = %s RETURNING name, status",
            (new_passkey, bank_name),
        ).fetchone()
    if not row:
        raise HTTPException(404, f"no DB-backed bank named {bank_name!r} (an env-configured BANK_PASSKEYS "
                                  f"tenant's passkey can't be regenerated here -- change it in .env instead)")
    audit("admin_regenerate_bank_passkey", {"bank": bank_name, "by": officer})
    return {"name": row["name"], "passkey": new_passkey, "status": row["status"]}


@api.delete("/admin/banks/{bank_name}")
def admin_delete_bank(bank_name: str, role: str = Depends(get_current_role),
                      officer: str = Depends(get_current_officer)):
    """Owner-only: removes a DB-backed tenant's passkey entirely (it stops being a valid login, period).
    Deliberately does NOT touch alerts/anomaly_alerts/account_banks -- exactly like /intake/clear, removing a
    tenant must never silently erase its compliance history."""
    if role != "admin":
        raise HTTPException(403, "admin passkey required")
    with db() as conn:
        row = conn.execute("DELETE FROM banks WHERE name = %s RETURNING name", (bank_name,)).fetchone()
    if not row:
        raise HTTPException(404, f"no DB-backed bank named {bank_name!r}")
    audit("admin_delete_bank", {"bank": bank_name, "by": officer})
    return {"removed": bank_name}


@api.post("/signup-requests", status_code=201)
def create_signup_request(req: SignupRequestBody):
    """Public, unauthenticated -- the front door for a bank that wants a pilot and doesn't have a passkey yet.
    Never creates a tenant by itself; it only queues a request the owner reviews on /owner and explicitly
    approves or declines. No email is sent from here (this deployment has no email service wired up yet --
    see notify_email()'s own docstring); the owner sees new requests by checking the backoffice, same as
    everything else there today."""
    with db() as conn:
        row = conn.execute(
            "INSERT INTO signup_requests (bank_name, contact_name, contact_email, note, registration_number, "
            "country, region, address, website) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING id, status",
            (req.bank_name.strip(), req.contact_name.strip(), req.contact_email.strip(), req.note.strip(),
             (req.registration_number or "").strip() or None, (req.country or "").strip() or None,
             (req.region or "").strip() or None, (req.address or "").strip() or None,
             (req.website or "").strip() or None),
        ).fetchone()
    return row


@api.get("/admin/signup-requests")
def admin_list_signup_requests(status: str | None = "pending", role: str = Depends(get_current_role)):
    if role != "admin":
        raise HTTPException(403, "admin passkey required")
    q, args = "SELECT * FROM signup_requests", []
    if status:
        q += " WHERE status = %s"
        args.append(status)
    q += " ORDER BY created_at DESC"
    with db() as conn:
        return conn.execute(q, args).fetchall()


@api.post("/admin/signup-requests/{request_id}/approve")
def admin_approve_signup_request(request_id: int, role: str = Depends(get_current_role),
                                 officer: str = Depends(get_current_officer)):
    """Turns a pending request into a real tenant in one action -- generates the passkey (same as
    admin_create_bank) and marks the request approved, so the owner never has to separately create the bank
    and then remember to close out the request."""
    if role != "admin":
        raise HTTPException(403, "admin passkey required")
    with db() as conn:
        req_row = conn.execute(
            "SELECT * FROM signup_requests WHERE id = %s AND status = 'pending'", (request_id,),
        ).fetchone()
        if not req_row:
            raise HTTPException(404, "no pending signup request with that id")
        name = req_row["bank_name"]
        if conn.execute("SELECT 1 FROM banks WHERE name = %s", (name,)).fetchone() or name in BANK_NAME_TO_PASSKEY:
            raise HTTPException(409, f"a bank named {name!r} already exists")
        passkey = secrets.token_urlsafe(18)
        conn.execute(
            "INSERT INTO banks (name, passkey, contact_email, registration_number, country, region, address, "
            "website) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
            (name, passkey, req_row["contact_email"], req_row.get("registration_number"), req_row.get("country"),
             req_row.get("region"), req_row.get("address"), req_row.get("website")),
        )
        conn.execute(
            "UPDATE signup_requests SET status = 'approved', decided_at = now() WHERE id = %s", (request_id,),
        )
    audit("admin_approve_signup_request", {"request_id": request_id, "bank": name, "by": officer})
    return {"name": name, "passkey": passkey, "status": "active"}


@api.post("/admin/signup-requests/{request_id}/decline")
def admin_decline_signup_request(request_id: int, role: str = Depends(get_current_role),
                                 officer: str = Depends(get_current_officer)):
    if role != "admin":
        raise HTTPException(403, "admin passkey required")
    with db() as conn:
        row = conn.execute(
            "UPDATE signup_requests SET status = 'declined', decided_at = now() "
            "WHERE id = %s AND status = 'pending' RETURNING id", (request_id,),
        ).fetchone()
    if not row:
        raise HTTPException(404, "no pending signup request with that id")
    audit("admin_decline_signup_request", {"request_id": request_id, "by": officer})
    return {"declined": request_id}


# ------------------------------------------------------------------ per-person accounts (invite / password / 2FA)
# Layered on top of the shared passkeys above, not a replacement for them -- a passkey still works. This
# exists so one person can be given (and have revoked) their OWN login, instead of everyone at a bank sharing
# one secret that has to be rotated for the whole bank the moment any one person leaves.
#
# PUBLIC_BASE_URL (e.g. "https://yourdomain.com") is used only to build a clickable link in the invite/reset
# emails this sends -- if unset, the email still includes the raw token and says which page to open it on, so
# nothing here is broken by leaving it unset, the email is just less convenient.
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "").rstrip("/")
INVITE_EXPIRY_HOURS = int(os.getenv("INVITE_EXPIRY_HOURS", "168"))     # 7 days
RESET_EXPIRY_HOURS = int(os.getenv("RESET_EXPIRY_HOURS", "2"))


def _user_public(row: dict) -> dict:
    """Strips password_hash/password_salt/totp_secret/invite_token/reset_token off a `users` row before it
    ever leaves this process -- used by every endpoint that returns a user, so there's exactly one place that
    decides what's safe to show, rather than every route remembering to exclude secrets by hand."""
    return {k: row[k] for k in ("id", "name", "email", "bank", "role", "status", "totp_enabled", "created_at") if k in row}


@api.post("/admin/users")
def admin_invite_user(req: InviteUserRequest, role: str = Depends(get_current_role),
                       officer: str = Depends(get_current_officer)):
    """Owner-only: creates an invited (not yet active) per-person account and emails them a one-time link to
    set their own password. Returns the invite token once, same "shown once" pattern as a new bank's passkey
    (admin_create_bank) -- in case SMTP isn't configured (see notify_email's own docstring), the owner can
    still hand the link to that person directly."""
    if role != "admin":
        raise HTTPException(403, "admin passkey required")
    if req.role == "admin" and req.bank:
        raise HTTPException(400, "a platform admin account can't also be scoped to one bank -- invite them "
                                  "without a bank, or as an analyst for that bank")
    name = req.name.strip()
    email = req.email.strip().lower()
    invite_token = secrets.token_urlsafe(32)
    with db() as conn:
        if conn.execute("SELECT 1 FROM users WHERE lower(email) = %s", (email,)).fetchone():
            raise HTTPException(409, f"a user with email {email!r} already exists")
        row = conn.execute(
            "INSERT INTO users (bank, name, email, role, invite_token, invite_expires) "
            "VALUES (%s, %s, %s, %s, %s, now() + interval '%s hours') "
            "RETURNING id, name, email, bank, role, status, totp_enabled, created_at",
            (req.bank, name, email, req.role, invite_token, INVITE_EXPIRY_HOURS),
        ).fetchone()
    link = f"{PUBLIC_BASE_URL}/accept-invite?token={invite_token}" if PUBLIC_BASE_URL else None
    notify_email(
        [email],
        "You've been invited to XAI-AMLBench",
        (f"{officer} invited you to sign in individually.\n\n"
         + (f"Set your password here: {link}\n\n" if link
            else f"Open the \"accept invite\" page and enter this code: {invite_token}\n\n")
         + f"This link expires in {INVITE_EXPIRY_HOURS // 24} days."),
    )
    audit("admin_invite_user", {"email": email, "bank": req.bank, "role": req.role, "by": officer})
    return {**_user_public(row), "invite_token": invite_token, "invite_link": link}


@api.get("/admin/users")
def admin_list_users(bank: str | None = None, status: str | None = None, role: str = Depends(get_current_role)):
    """`status=pending` is what the owner's "People" section uses to surface self-registered analysts (POST
    /auth/signup) waiting on a decision -- see users' own 'pending' status comment in SCHEMA. Omitted, this
    returns every account regardless of status, same as before this filter existed."""
    if role != "admin":
        raise HTTPException(403, "admin passkey required")
    q = "SELECT id, name, email, bank, role, status, totp_enabled, created_at FROM users"
    where, args = [], []
    if bank:
        where.append("bank = %s")
        args.append(bank)
    if status:
        where.append("status = %s")
        args.append(status)
    if where:
        q += " WHERE " + " AND ".join(where)
    q += " ORDER BY created_at DESC"
    with db() as conn:
        return conn.execute(q, args).fetchall()


@api.patch("/admin/users/{user_id}")
def admin_set_user_status(user_id: int, req: UserStatusRequest, role: str = Depends(get_current_role),
                           officer: str = Depends(get_current_officer)):
    """Owner-only: the actual "revoke one person" lever. Disabling blocks every FUTURE login attempt for this
    person immediately -- but because session tokens are stateless (see issue_session_token's docstring), a
    token this person already holds keeps working until it naturally expires (SESSION_MAX_AGE_S, 12 hours by
    default). For a pilot that's an acceptable tradeoff; a deployment that needs instant mid-session
    revocation would need to start checking user status on every request, not just at login."""
    if role != "admin":
        raise HTTPException(403, "admin passkey required")
    with db() as conn:
        row = conn.execute(
            "UPDATE users SET status = %s WHERE id = %s "
            "RETURNING id, name, email, bank, role, status, totp_enabled, created_at",
            (req.status, user_id),
        ).fetchone()
    if not row:
        raise HTTPException(404, "no user with that id")
    audit("admin_set_user_status", {"user_id": user_id, "status": req.status, "by": officer})
    return row


@api.delete("/admin/users/{user_id}")
def admin_delete_user(user_id: int, role: str = Depends(get_current_role),
                       officer: str = Depends(get_current_officer)):
    if role != "admin":
        raise HTTPException(403, "admin passkey required")
    with db() as conn:
        row = conn.execute("DELETE FROM users WHERE id = %s RETURNING email", (user_id,)).fetchone()
    if not row:
        raise HTTPException(404, "no user with that id")
    audit("admin_delete_user", {"user_id": user_id, "by": officer})
    return {"removed": row["email"]}


def _active_bank_names() -> set[str]:
    """Every bank an analyst is actually allowed to self-register against: the env-configured BANK_PASSKEYS
    set (always active, no `banks` row to suspend -- see login()'s own comment on why env always wins) plus
    every DB-backed tenant currently status='active'. A bank that's only `pending` in signup_requests (not yet
    approved) or `suspended` is deliberately NOT in this set -- self-registration must never let someone claim
    membership in a bank the owner hasn't actually onboarded, or has turned off."""
    with db() as conn:
        rows = conn.execute("SELECT name FROM banks WHERE status = 'active'").fetchall()
    return set(BANK_NAME_TO_PASSKEY) | {r["name"] for r in rows}


@api.get("/public/banks")
def public_banks():
    """Unauthenticated -- the list /analyst-signup's bank dropdown populates itself from, so a self-registering
    analyst picks from banks that genuinely exist and are active, never types a name freehand (which would let
    a typo, or a deliberate lie, attach their account to a bank they don't belong to -- see
    AnalystSignupRequest's own docstring). Names only: nothing else about a tenant is exposed publicly."""
    return sorted(_active_bank_names())


@api.post("/auth/signup", status_code=201)
def auth_signup(req: AnalystSignupRequest):
    """Public, unauthenticated -- the self-service door AnalystSignupRequest's docstring describes. Creates a
    status='pending' account that cannot sign in (see /auth/login's status check) until the owner approves it
    on /owner. Rejects outright, before anything is written, if `bank` isn't one of the currently active
    tenants (_active_bank_names()) -- there is no "register for a bank that doesn't exist yet" path; that bank
    has to go through /signup-requests and be approved first. Same generic-enough-not-to-leak posture as
    admin_invite_user: a duplicate email is refused with 409, same as there."""
    bank = req.bank.strip()
    if bank not in _active_bank_names():
        raise HTTPException(
            422,
            f"{bank!r} isn't a bank we have on file yet. If your bank hasn't signed up with us, request access "
            f"for it first -- an individual can only register once their own bank is already onboarded.",
        )
    name = req.name.strip()
    email = req.email.strip().lower()
    password_hash, password_salt = _hash_password(req.password)
    with db() as conn:
        if conn.execute("SELECT 1 FROM users WHERE lower(email) = %s", (email,)).fetchone():
            raise HTTPException(409, f"a user with email {email!r} already exists")
        conn.execute(
            "INSERT INTO users (bank, name, email, role, status, password_hash, password_salt) "
            "VALUES (%s, %s, %s, 'analyst', 'pending', %s, %s)",
            (bank, name, email, password_hash, password_salt),
        )
    audit("analyst_signup", {"email": email, "bank": bank})
    return {"email": email, "bank": bank, "status": "pending"}


@api.post("/auth/accept-invite")
def accept_invite(req: AcceptInviteRequest):
    """Public, unauthenticated -- the link an invited person clicks. Sets their password and activates the
    account in one step; the invite token is single-use (cleared immediately after)."""
    password_hash, password_salt = _hash_password(req.password)
    with db() as conn:
        row = conn.execute(
            "UPDATE users SET password_hash = %s, password_salt = %s, status = 'active', "
            "invite_token = NULL, invite_expires = NULL "
            "WHERE invite_token = %s AND invite_expires > now() AND status = 'invited' "
            "RETURNING id, email", (password_hash, password_salt, req.token),
        ).fetchone()
    if not row:
        raise HTTPException(400, "that invite link is invalid or has expired -- ask whoever invited you to "
                                  "send a new one")
    audit("accept_invite", {"user_id": row["id"], "email": row["email"]})
    return {"email": row["email"], "status": "active"}


@api.post("/auth/forgot-password")
def forgot_password(req: ForgotPasswordRequest):
    """Public, unauthenticated. ALWAYS returns the same message whether or not that email has an account --
    the one thing this must never do is let a caller use the response to find out which emails are
    registered."""
    email = req.email.strip().lower()
    reset_token = secrets.token_urlsafe(32)
    with db() as conn:
        row = conn.execute(
            "UPDATE users SET reset_token = %s, reset_expires = now() + interval '%s hours' "
            "WHERE lower(email) = %s AND status = 'active' RETURNING email",
            (reset_token, RESET_EXPIRY_HOURS, email),
        ).fetchone()
    if row:
        link = f"{PUBLIC_BASE_URL}/reset-password?token={reset_token}" if PUBLIC_BASE_URL else None
        notify_email(
            [email],
            "Reset your XAI-AMLBench password",
            (f"Reset your password here: {link}\n\n" if link
             else f"Open the \"reset password\" page and enter this code: {reset_token}\n\n")
            + f"This link expires in {RESET_EXPIRY_HOURS} hours. If you didn't ask for this, ignore this email.",
        )
    return {"message": "If that email has an account, we've sent a password reset link."}


@api.post("/auth/reset-password")
def reset_password(req: ResetPasswordRequest):
    password_hash, password_salt = _hash_password(req.password)
    with db() as conn:
        row = conn.execute(
            "UPDATE users SET password_hash = %s, password_salt = %s, reset_token = NULL, reset_expires = NULL "
            "WHERE reset_token = %s AND reset_expires > now() RETURNING id, email",
            (password_hash, password_salt, req.token),
        ).fetchone()
    if not row:
        raise HTTPException(400, "that reset link is invalid or has expired -- request a new one")
    audit("reset_password", {"user_id": row["id"], "email": row["email"]})
    return {"email": row["email"], "status": "ok"}


@api.post("/auth/login")
def auth_login(req: AuthLoginRequest, request: Request = None):
    """Per-person login -- the email+password counterpart to the shared-passkey /login above. Deliberately
    returns the same generic "invalid email or password" for a wrong email, a wrong password, an invited-but-
    not-yet-accepted account, and a disabled account alike: telling a caller WHICH of those is true would leak
    whether an email is registered at all. The one exception is a correct password on a 2FA-enabled account
    with no code yet -- that gets its own distinct response (`detail: "totp_required"`) so the frontend knows
    to ask for the code next, which is safe to reveal only because the password has already been proven
    correct at that point."""
    email = req.email.strip().lower()
    ip = _client_ip(request) if request is not None else None
    _check_login_rate_limit(email)
    if ip is not None:
        _check_login_rate_limit(ip)
    with db() as conn:
        user = conn.execute(
            "SELECT id, name, email, bank, role, status, password_hash, password_salt, "
            "totp_enabled, totp_secret FROM users WHERE lower(email) = %s", (email,),
        ).fetchone()
    invalid = HTTPException(401, "invalid email or password")
    if (not user or user["status"] != "active" or not user["password_hash"]
            or not _verify_password(req.password, user["password_hash"], user["password_salt"])):
        _record_login_attempt(email, False)
        if ip is not None:
            _record_login_attempt(ip, False)
        raise invalid
    if user["totp_enabled"]:
        if not req.totp_code:
            raise HTTPException(401, "totp_required")
        if not pyotp.TOTP(user["totp_secret"]).verify(req.totp_code, valid_window=1):
            _record_login_attempt(email, False)
            if ip is not None:
                _record_login_attempt(ip, False)
            raise HTTPException(401, "invalid authentication code")
    _record_login_attempt(email, True)
    if ip is not None:
        _record_login_attempt(ip, True)
    token = issue_session_token(user["name"], user["role"], user["bank"], uid=user["id"])
    audit("auth_login", {"email": email, "role": user["role"], "bank": user["bank"]})
    return {"token": token, "name": user["name"], "role": user["role"], "bank": user["bank"]}


@api.post("/auth/2fa/setup")
def setup_2fa(uid: int = Depends(get_current_user_id)):
    """Generates a new TOTP secret for the signed-in person and stores it (NOT yet enabled -- see
    /auth/2fa/enable). Calling this again before enabling just replaces the pending secret, so there's never
    more than one "in progress" setup per account. Returns the secret as plain text (type it into any
    authenticator app's "enter code manually" option) and as an otpauth:// URI (tapping it on a phone with an
    authenticator app installed adds the account directly, no QR scanner needed)."""
    with db() as conn:
        user = conn.execute("SELECT email FROM users WHERE id = %s", (uid,)).fetchone()
        if not user:
            raise HTTPException(404, "no such user")
        secret = pyotp.random_base32()
        conn.execute("UPDATE users SET totp_secret = %s, totp_enabled = false WHERE id = %s", (secret, uid))
    uri = pyotp.TOTP(secret).provisioning_uri(name=user["email"], issuer_name="XAI-AMLBench")
    return {"secret": secret, "provisioning_uri": uri}


@api.post("/auth/2fa/enable")
def enable_2fa(req: Totp2faVerifyRequest, uid: int = Depends(get_current_user_id)):
    with db() as conn:
        user = conn.execute("SELECT totp_secret FROM users WHERE id = %s", (uid,)).fetchone()
        if not user or not user["totp_secret"]:
            raise HTTPException(400, "call /auth/2fa/setup first")
        if not pyotp.TOTP(user["totp_secret"]).verify(req.code, valid_window=1):
            raise HTTPException(401, "invalid code")
        conn.execute("UPDATE users SET totp_enabled = true WHERE id = %s", (uid,))
    audit("enable_2fa", {"user_id": uid})
    return {"totp_enabled": True}


@api.post("/auth/2fa/disable")
def disable_2fa(req: Totp2faVerifyRequest, uid: int = Depends(get_current_user_id)):
    """Requires a currently-valid code to turn 2FA back off, even from an already-signed-in session -- so
    someone who merely finds a device unlocked and signed in still can't strip 2FA off the account without the
    authenticator itself."""
    with db() as conn:
        user = conn.execute(
            "SELECT totp_secret, totp_enabled FROM users WHERE id = %s", (uid,),
        ).fetchone()
        if not user or not user["totp_enabled"]:
            raise HTTPException(400, "2FA is not enabled on this account")
        if not pyotp.TOTP(user["totp_secret"]).verify(req.code, valid_window=1):
            raise HTTPException(401, "invalid code")
        conn.execute("UPDATE users SET totp_enabled = false, totp_secret = NULL WHERE id = %s", (uid,))
    audit("disable_2fa", {"user_id": uid})
    return {"totp_enabled": False}


@api.get("/public/status")
def public_status():
    """Unauthenticated, deliberately a MUCH smaller version of /system/status for a public status page testers
    can check without logging in. No Kafka config, no exception class names, no latency numbers -- just
    whether the platform as a whole is up, so this can never leak infrastructure topology to the open
    internet."""
    def ok(url: str) -> bool:
        try:
            httpx.get(url, timeout=4).raise_for_status()
            return True
        except httpx.HTTPError:
            return False
    try:
        with db() as conn:
            conn.execute("SELECT 1")
        db_ok = True
    except psycopg.OperationalError:
        db_ok = False
    all_ok = db_ok and ok(f"{GNN_API_URL}/health") and ok(f"{XAI_API_URL}/health")
    return {"all_ok": all_ok, "checked_at": time.time()}


@api.get("/public/banks/{bank_name}/branding")
def public_bank_branding(bank_name: str):
    """Unauthenticated: just enough for the login screen to show a bank's own name/logo before anyone has
    signed in (reached via /?bank=<name> -- see frontend's LoginScreen). Returns defaults rather than 404 for
    a name with no row (an env-configured BANK_PASSKEYS tenant, or a typo) so the login screen never breaks,
    it just shows the generic brand instead."""
    with db() as conn:
        row = conn.execute(
            "SELECT name, logo_url, accent_color FROM banks WHERE name = %s", (bank_name,),
        ).fetchone()
    if not row:
        return {"name": bank_name, "logo_url": None, "accent_color": None}
    return row


@api.post("/alerts/{alert_id}/narrative")
def generate_narrative(alert_id: int, officer: str = Depends(get_current_officer),
                       bank: str | None = Depends(get_current_bank)):
    with db() as conn:
        alert = conn.execute("SELECT * FROM alerts WHERE id = %s", (alert_id,)).fetchone()
    if not alert or (bank and alert["source_bank"] != bank):
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
def decide(alert_id: int, req: DecisionRequest, officer: str = Depends(get_current_officer),
          bank: str | None = Depends(get_current_bank)):
    with db() as conn:
        alert = conn.execute("SELECT * FROM alerts WHERE id = %s", (alert_id,)).fetchone()
        if not alert or (bank and alert["source_bank"] != bank):
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


# ------------------------------------------------------------------ Model 2 (anomaly) investigation workflow
# These alerts are written directly by gnn_aml_core.anomaly (its own container, running continuously -- see
# that module's docstring) straight into Postgres, not through this API; this API only handles what happens
# to an alert AFTER it exists: new -> under_investigation -> confirmed_mule | cleared. A resolve to
# confirmed_mule is read back out by gnn_aml_core.anomaly.sync_confirmed_to_model1 on its own schedule (not
# triggered synchronously by this request) and is the only thing that ever teaches Model 1 a new pattern.
@api.get("/anomalies")
def list_anomalies(status: str | None = None, limit: int = 100, officer: str = Depends(get_current_officer),
                   bank: str | None = Depends(get_current_bank)):
    # anomaly_alerts has no bank column of its own (see activity()'s docstring on why) -- a tenant-scoped
    # session joins against account_banks instead, which also naturally excludes any account with no bank
    # attribution at all, since that account isn't this tenant's own data.
    if bank:
        q = "SELECT a.* FROM anomaly_alerts a JOIN account_banks ab ON ab.account_id = a.account_id AND ab.bank = %s"
        args: list = [bank]
        if status:
            q += " AND a.status = %s"
            args.append(status)
    else:
        q, args = "SELECT * FROM anomaly_alerts", []
        if status:
            q += " WHERE status = %s"
            args.append(status)
    q += " ORDER BY created_at DESC LIMIT %s"
    args.append(min(limit, 500))
    with db() as conn:
        return conn.execute(q, args).fetchall()


def _get_anomaly_scoped(conn, anomaly_id: int, bank: str | None) -> dict:
    """Fetch + bank-ownership check in one place, used by every /anomalies/{id}/* route below. 404, not 403,
    for the same reason as _check_account_access: a tenant session must not learn that an anomaly alert
    exists for another bank's account."""
    row = conn.execute("SELECT * FROM anomaly_alerts WHERE id = %s", (anomaly_id,)).fetchone()
    if not row:
        raise HTTPException(404, "anomaly alert not found")
    if bank and lookup_bank(row["account_id"]) != bank:
        raise HTTPException(404, "anomaly alert not found")
    return row


@api.get("/anomalies/{anomaly_id}")
def get_anomaly(anomaly_id: int, officer: str = Depends(get_current_officer),
                bank: str | None = Depends(get_current_bank)):
    with db() as conn:
        return _get_anomaly_scoped(conn, anomaly_id, bank)


def _append_investigation_log(conn, anomaly_id: int, entry: dict) -> dict:
    row = conn.execute("SELECT investigation_log FROM anomaly_alerts WHERE id = %s", (anomaly_id,)).fetchone()
    if not row:
        raise HTTPException(404, "anomaly alert not found")
    updated_log = [*row["investigation_log"], entry]
    return conn.execute(
        "UPDATE anomaly_alerts SET investigation_log = %s, updated_at = now() WHERE id = %s RETURNING *",
        (Jsonb(updated_log), anomaly_id),
    ).fetchone()


@api.post("/anomalies/{anomaly_id}/claim")
def claim_anomaly(anomaly_id: int, officer: str = Depends(get_current_officer),
                  bank: str | None = Depends(get_current_bank)):
    """Moves an alert from 'new' into 'under_investigation' and assigns it to the person claiming it -- this is
    a real state transition, not a label: it's what turns a raw statistical flag into something a specific
    person is now responsible for looking into (account history, related accounts, and -- outside this
    system's reach -- things like calling the account holder or looping in investigators, as the real workflow
    this is meant to support)."""
    with db() as conn:
        alert = _get_anomaly_scoped(conn, anomaly_id, bank)
        if alert["status"] != "new":
            raise HTTPException(409, f"alert already {alert['status']}, cannot claim")
        alert = conn.execute(
            "UPDATE anomaly_alerts SET status='under_investigation', assigned_to=%s, updated_at=now() "
            "WHERE id=%s RETURNING *",
            (officer, anomaly_id),
        ).fetchone()
        alert = _append_investigation_log(conn, anomaly_id,
                                          {"ts": time.time(), "by": officer, "status": "under_investigation",
                                           "note": "claimed for investigation"})
    audit("anomaly_claimed", {"anomaly_id": anomaly_id, "account_id": alert["account_id"], "by": officer})
    return alert


@api.post("/anomalies/{anomaly_id}/notes")
def add_anomaly_note(anomaly_id: int, req: AnomalyNoteRequest, officer: str = Depends(get_current_officer),
                     bank: str | None = Depends(get_current_bank)):
    """Documents a step of the actual investigation -- e.g. 'called the account holder, no answer', 'cross-
    checked against the related shell account from last quarter', 'escalated to [investigator]' -- without
    changing the alert's status. This is the running case file; resolve() below is the final verdict."""
    with db() as conn:
        alert = _get_anomaly_scoped(conn, anomaly_id, bank)
        if alert["status"] not in ("under_investigation",):
            raise HTTPException(409, f"alert is '{alert['status']}' -- claim it before adding investigation notes")
        alert = _append_investigation_log(conn, anomaly_id,
                                          {"ts": time.time(), "by": officer, "status": alert["status"], "note": req.note})
    return alert


@api.post("/anomalies/{anomaly_id}/resolve")
def resolve_anomaly(anomaly_id: int, req: AnomalyResolveRequest, officer: str = Depends(get_current_officer),
                    bank: str | None = Depends(get_current_bank)):
    """The only way an anomaly alert becomes a verdict. 'confirmed_mule' means the investigation (account
    history, related accounts, direct contact, escalation -- whatever that took) actually established this is
    a real mule account; this is what gnn_aml_core.anomaly.sync_confirmed_to_model1 picks up and feeds into
    Model 1's fine-tune, teaching it this specific newly-discovered pattern. 'cleared' means the investigation
    found nothing -- logged as evidence, and deliberately NEVER fed into training (an unconfirmed or cleared
    alert must never quietly become a training example). Note this still feeds the one shared Model 1 for
    every tenant -- a bank-scoped resolve still only touches ITS OWN account's alert, but the resulting model
    improvement benefits every bank on this deployment, same as Model 1's detection already does for everyone;
    there is no per-tenant model to keep separate here, by design (see this module's own docstring)."""
    with db() as conn:
        alert = _get_anomaly_scoped(conn, anomaly_id, bank)
        if alert["status"] != "under_investigation":
            raise HTTPException(409, f"alert is '{alert['status']}' -- claim it and investigate before resolving")
        alert = conn.execute(
            "UPDATE anomaly_alerts SET status=%s, updated_at=now() WHERE id=%s RETURNING *",
            (req.outcome, anomaly_id),
        ).fetchone()
        alert = _append_investigation_log(conn, anomaly_id,
                                          {"ts": time.time(), "by": officer, "status": req.outcome, "note": req.note})
    record = {"anomaly_id": anomaly_id, "account_id": alert["account_id"], "outcome": req.outcome,
              "officer": officer, "note": req.note}
    record["record_sha256"] = hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()
    audit("anomaly_resolved", record)
    return alert


@api.post("/identity/check")
def identity_check(req: IdentityCheckRequest, officer: str = Depends(get_current_officer),
                   bank: str | None = Depends(get_current_bank)):
    """Runs check_identity() against whichever provider is configured and records the result. See
    check_identity()'s own docstring for exactly what each status means and what it does NOT mean -- in
    particular, 'mismatch' and 'deceased' are both reasons to follow up with a human, not final verdicts this
    system reaches on its own. The raw ID number never reaches storage, only this request's own validation and
    the provider call above see it; only its hash and last 4 characters are kept, in identity_checks below."""
    result = check_identity(req.claimed_name, req.id_number)
    row_bank = bank or lookup_bank(req.account_id)
    _store_identity_check(req.account_id, row_bank, req.claimed_name, req.id_number, result, officer)
    audit("identity_check", {"account_id": req.account_id, "status": result["status"], "by": officer,
                             "provider": IDENTITY_PROVIDER or "unconfigured"})
    with db() as conn:
        row = conn.execute(
            "SELECT * FROM identity_checks WHERE account_id = %s ORDER BY created_at DESC LIMIT 1",
            (req.account_id,),
        ).fetchone()
    return row


@api.get("/identity/checks")
def identity_checks(account_id: str | None = None, limit: int = 100,
                    bank: str | None = Depends(get_current_bank), officer: str = Depends(get_current_officer)):
    """History for the Identity Check page. A bank-tenant session only ever sees its own bank's checks, same
    scoping as every other bank-attributable route (see get_current_bank's own docstring); an admin/legacy
    session sees everything."""
    clauses, params = [], []
    if bank is not None:
        clauses.append("bank = %s")
        params.append(bank)
    if account_id:
        clauses.append("account_id = %s")
        params.append(account_id)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    with db() as conn:
        rows = conn.execute(
            f"SELECT * FROM identity_checks {where} ORDER BY created_at DESC LIMIT %s",
            (*params, max(1, min(limit, 500))),
        ).fetchall()
    return rows


app.include_router(api)
