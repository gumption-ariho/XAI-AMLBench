"""backend.manage_keys: create, list, and revoke officer API keys.

Deliberately a CLI script with direct database access, not an HTTP endpoint: an endpoint that creates new
credentials would itself need to be protected by something, which is exactly the chicken-and-egg problem this
avoids. Run this from inside the backend container (or anywhere DATABASE_URL points at the real database):

    docker compose exec backend python manage_keys.py create "Jane Smith"
    docker compose exec backend python manage_keys.py list
    docker compose exec backend python manage_keys.py revoke 3

The plaintext key is shown exactly once, at creation, and is never stored anywhere -- only its SHA-256 hash is
kept (see backend/main.py's hash_api_key for why a straight hash, not bcrypt/scrypt, is the correct choice for
a high-entropy generated key rather than a user-chosen password).
"""
from __future__ import annotations

import argparse
import os
import secrets
import sys

import psycopg
from psycopg.rows import dict_row

sys.path.insert(0, os.path.dirname(__file__))
from main import SCHEMA, hash_api_key  # noqa: E402


def _db():
    return psycopg.connect(os.environ["DATABASE_URL"], row_factory=dict_row)


def _ensure_schema() -> None:
    """Idempotently ensures the schema exists, tolerating a real, known PostgreSQL race: this really happened
    on a real run -- this script's own CREATE TABLE IF NOT EXISTS raced against the backend container's own
    startup schema creation (main.py's lifespan hook runs the identical statement), and PostgreSQL's IF NOT
    EXISTS does not fully protect against two concurrent attempts to create the same new table -- the LOSING
    side raises UniqueViolation on the underlying pg_type catalog entry, even though the table ends up existing
    either way. Runs with autocommit so a caught DDL failure does not leave a half-finished transaction behind
    (the next statement in the same transaction would otherwise fail too, since Postgres requires an explicit
    rollback after an aborted statement before anything else can run)."""
    with psycopg.connect(os.environ["DATABASE_URL"], autocommit=True) as conn:
        try:
            conn.execute(SCHEMA)
        except psycopg.errors.UniqueViolation:
            pass  # another process (most likely the backend container's own startup) created it first -- fine,
                  # the table exists either way, which is all this call was ever trying to guarantee


def create(officer_name: str) -> None:
    _ensure_schema()
    key = "officer_" + secrets.token_urlsafe(32)
    with _db() as conn:
        row = conn.execute(
            "INSERT INTO api_keys (officer_name, key_hash) VALUES (%s, %s) RETURNING id",
            (officer_name, hash_api_key(key)),
        ).fetchone()
    print(f"Created key id {row['id']} for '{officer_name}'.")
    print(f"\n  {key}\n")
    print("This is shown once and is not recoverable -- store it somewhere safe (a password manager, not a "
         "chat log or a ticket). Use it as: Authorization: Bearer <key>")


def list_keys() -> None:
    _ensure_schema()
    with _db() as conn:
        rows = conn.execute(
            "SELECT id, officer_name, created_at, revoked_at FROM api_keys ORDER BY created_at DESC"
        ).fetchall()
    if not rows:
        print("No API keys exist yet. Create one with: python manage_keys.py create \"Officer Name\"")
        return
    for r in rows:
        status = f"revoked {r['revoked_at']}" if r["revoked_at"] else "active"
        print(f"  [{r['id']}] {r['officer_name']:<30} created {r['created_at']}  ({status})")


def revoke(key_id: int) -> None:
    _ensure_schema()
    with _db() as conn:
        row = conn.execute(
            "UPDATE api_keys SET revoked_at = now() WHERE id = %s AND revoked_at IS NULL RETURNING officer_name",
            (key_id,),
        ).fetchone()
    if row:
        print(f"Revoked key id {key_id} ('{row['officer_name']}'). It will no longer authenticate.")
    else:
        print(f"No active key with id {key_id} found (already revoked, or does not exist).")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    p_create = sub.add_parser("create", help="issue a new key for an officer")
    p_create.add_argument("officer_name")

    sub.add_parser("list", help="list every key (active and revoked), never showing the plaintext key")

    p_revoke = sub.add_parser("revoke", help="revoke a key by its id (from 'list')")
    p_revoke.add_argument("key_id", type=int)

    a = ap.parse_args()
    if a.command == "create":
        create(a.officer_name)
    elif a.command == "list":
        list_keys()
    elif a.command == "revoke":
        revoke(a.key_id)


if __name__ == "__main__":
    main()
