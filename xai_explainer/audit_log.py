"""xai_explainer.audit_log: an append-only, tamper-evident record of every alert, model decision and human
review (brief 3.3). Every entry is chained to the one before it by a SHA-256 hash, the way a lightweight
blockchain is: changing, reordering or deleting a past entry breaks the chain from that point on, and
`AuditLog.verify()` detects it.

Storage: a JSON-Lines file (the "append-only log file" the brief calls for; simple, dependency-free, easy to
back up or `git log` a diff of) plus, optionally, a mirrored row in PostgreSQL for institutional storage
alongside the application's own database -- pass `database_url` to enable it. A failed Postgres write is logged
and never raises, so the file (the tamper-evident source of truth) is never blocked by a database outage.

    from xai_explainer.audit_log import AuditLog
    log = AuditLog("audit.jsonl")
    log.append({"event": "alert_flagged", "account_id": "ACC0000123", "score": 0.94})
    log.verify()          # True, unless the file has been tampered with
    list(log.read_all())  # every entry, in order

Retention: each entry carries `retain_until` (now + `retention_years`, default 5, per the brief's "5 years
minimum"). This is a labelled reminder, not an enforcement mechanism: a hash-chained log cannot have entries
deleted without invalidating every entry after them, so this module never deletes anything. Reaching a
retention decision (e.g. archiving whole sealed segments) is a policy choice for the deployment to make, not
something a benchmark-scale tool should decide for you.

Run as a script to verify a log from the command line:  python -m xai_explainer.audit_log verify audit.jsonl
"""
from __future__ import annotations

import hashlib
import json
import logging
import sys
import threading
import time
from pathlib import Path
from typing import Iterator

log = logging.getLogger("xai_explainer.audit_log")

GENESIS_HASH = "0" * 64
SECONDS_PER_YEAR = 365.25 * 86400


def _canonical_bytes(seq: int, ts: float, record: dict, prev_hash: str) -> bytes:
    """Deterministic byte encoding of everything an entry's hash covers. Sorted keys and no extra whitespace,
    so the same logical content always hashes to the same value regardless of dict insertion order."""
    return json.dumps({"seq": seq, "ts": ts, "record": record, "prev_hash": prev_hash},
                      sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")


def _entry_hash(seq: int, ts: float, record: dict, prev_hash: str) -> str:
    return hashlib.sha256(_canonical_bytes(seq, ts, record, prev_hash)).hexdigest()


class AuditLog:
    """A hash-chained, append-only log backed by a JSON-Lines file, with an optional PostgreSQL mirror."""

    def __init__(self, path: str, database_url: str | None = None, retention_years: float = 5.0):
        self.path = Path(path)
        self.retention_years = retention_years
        self._lock = threading.Lock()
        self._database_url = database_url
        self._pg_ready = False
        seq, prev_hash = self._tail_state()
        self._next_seq = seq
        self._prev_hash = prev_hash
        if database_url:
            self._pg_ensure_table()

    # ------------------------------------------------------------------------------------------------- writing
    def append(self, record: dict) -> dict:
        """Append one event. `record` can be any JSON-serialisable dict describing what happened (an alert
        raised, a model decision, an officer's review). Returns the full stored entry, including its hash."""
        with self._lock:
            seq, ts, prev_hash = self._next_seq, time.time(), self._prev_hash
            h = _entry_hash(seq, ts, record, prev_hash)
            entry = {"seq": seq, "ts": ts, "record": record, "prev_hash": prev_hash, "hash": h,
                     "retain_until": ts + self.retention_years * SECONDS_PER_YEAR}
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, sort_keys=True) + "\n")
            self._next_seq, self._prev_hash = seq + 1, h
        if self._database_url:
            self._pg_insert(entry)
        return entry

    # ------------------------------------------------------------------------------------------------- reading
    def read_all(self) -> Iterator[dict]:
        """Every entry, in order. Raises nothing on a missing file (an empty log has no entries)."""
        if not self.path.exists():
            return
        with open(self.path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)

    def __len__(self) -> int:
        return sum(1 for _ in self.read_all())

    # ------------------------------------------------------------------------------------------------ verifying
    def verify(self) -> bool:
        """Recompute the hash chain from scratch and compare it against what is stored. Returns False on the
        first inconsistency it finds: a hash that doesn't match its entry's content, a broken sequence number,
        a prev_hash that doesn't match the previous entry, or a line that isn't valid JSON at all (which is what
        a naive text edit, rather than a full chain forgery, produces). Never raises for a tampered file --
        that would defeat the purpose of a verification method a caller can safely call on untrusted input."""
        if not self.path.exists():
            return True
        expected_seq, prev_hash = 0, GENESIS_HASH
        try:
            with open(self.path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    entry = json.loads(line)
                    if entry["seq"] != expected_seq or entry["prev_hash"] != prev_hash:
                        return False
                    if _entry_hash(entry["seq"], entry["ts"], entry["record"], entry["prev_hash"]) != entry["hash"]:
                        return False
                    prev_hash = entry["hash"]
                    expected_seq += 1
        except (json.JSONDecodeError, KeyError, TypeError):
            return False
        return True

    # -------------------------------------------------------------------------------------- private: state / pg
    def _tail_state(self) -> tuple[int, str]:
        """Resume the chain correctly when re-opening an existing log (so a second `AuditLog(path)` on the same
        file continues the same chain rather than starting a new genesis block).

        Tolerates a corrupted or tampered tail rather than raising: opening a log must never crash just because
        the file has been damaged -- that would make it impossible to even construct an AuditLog in order to
        call `verify()` on it and find out. `read_all()` itself still raises on bad JSON for a caller who wants
        every entry reliably; this bootstrapping step deliberately does not."""
        seq, prev_hash = 0, GENESIS_HASH
        try:
            for entry in self.read_all():
                seq, prev_hash = entry["seq"] + 1, entry["hash"]
        except (json.JSONDecodeError, KeyError, TypeError):
            log.warning("audit log %s: could not fully resume the chain (corrupted or tampered tail); "
                       "appending will continue from the last readable entry, but verify() will report this file as invalid", self.path)
        return seq, prev_hash

    def _pg_ensure_table(self) -> None:
        try:
            import psycopg
            with psycopg.connect(self._database_url, connect_timeout=5) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS audit_log (
                        seq BIGINT PRIMARY KEY, ts DOUBLE PRECISION NOT NULL, record JSONB NOT NULL,
                        prev_hash CHAR(64) NOT NULL, hash CHAR(64) NOT NULL, retain_until DOUBLE PRECISION NOT NULL
                    )
                """)
        except Exception as exc:  # noqa: BLE001
            log.warning("audit log: could not prepare the Postgres mirror (%s); continuing file-only", exc.__class__.__name__)
            self._database_url = None

    def _pg_insert(self, entry: dict) -> None:
        try:
            import psycopg
            with psycopg.connect(self._database_url, connect_timeout=5) as conn:
                conn.execute(
                    "INSERT INTO audit_log (seq, ts, record, prev_hash, hash, retain_until) "
                    "VALUES (%s, %s, %s, %s, %s, %s) ON CONFLICT (seq) DO NOTHING",
                    (entry["seq"], entry["ts"], json.dumps(entry["record"]), entry["prev_hash"], entry["hash"], entry["retain_until"]),
                )
        except Exception as exc:  # noqa: BLE001
            log.warning("audit log: Postgres mirror write failed (%s); the file entry is unaffected", exc.__class__.__name__)


def main() -> None:
    if len(sys.argv) != 3 or sys.argv[1] != "verify":
        print("usage: python -m xai_explainer.audit_log verify <path>")
        raise SystemExit(2)
    l = AuditLog(sys.argv[2])
    ok = l.verify()
    print(f"{sys.argv[2]}: {len(l)} entries, chain {'VALID' if ok else 'BROKEN (tampering or corruption detected)'}")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
