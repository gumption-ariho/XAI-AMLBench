"""gnn_aml_core.kafka_ingest: Option 2, the Connect tab's live transaction feed -- a bank's own system (or, for
now, aml-synth-worker's `--to kafka` export) publishes transactions onto KAFKA_TOPIC_TRANSACTIONS, and this
module is the consumer that turns that stream into real, live accounts in the served graph AND gets them
scored and flagged automatically, exactly the way a pasted document on the Intake tab does -- this is the
second, previously-missing half of the same architecture, not a different one.

This process never touches the transaction itself and never calls anything that could: it only reads a copy
of the feed, batches it, and calls backend/main.py's POST /system/live-ingest -- the one place that (a) adds
the batch's accounts to the live graph via gnn-detection-api's POST /admin/ingest-live, standardized against
the already-trained model's own frozen mean/std (no retraining, no restart), and (b) scores every account the
batch touched and raises an alert for the ones that come back flagged, exactly as if an analyst had pressed
Scan on each one. An alert is the entire effect: nothing here can delay, block or alter a transaction, here or
downstream -- there is no code path back to whatever published the message. gnn_model.pt's trained weights are
never touched by any of this.

A Kafka message's value is whatever aml_synth.exporters.export_kafka published: the full synthetic-transaction
record (tx_id, src, dst, amount, currency, payment_format, timestamp, cross_border, is_laundering, typology,
pattern_id, category, memo, disputed). A real bank feed would carry its own shape; either way this module only
reads src/dst/amount (required) and timestamp/cross_border (optional, same defaults as livegraph.ingest()) and
ignores the rest -- it never reads is_laundering/typology/pattern_id, which a real feed would not have anyway
and which would be answering the exact question this system exists to answer on its own.

Messages are batched -- by count (--batch-size, default 25) or by time (--max-wait-seconds, default 5.0),
whichever comes first -- rather than ingested one at a time, because every batch triggers one full graph
rebuild + hot-reload on gnn-detection-api; a message-at-a-time consumer would thrash that far harder than the
small added latency of a short batching window costs.

    python -m gnn_aml_core.kafka_ingest                      # run the consumer loop (what the
                                                               # live-feed-consumer container's CMD runs)
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import time
import urllib.error
import urllib.request

log = logging.getLogger("gnn_aml_core.kafka_ingest")

KAFKA_BOOTSTRAP = os.environ.get("KAFKA_BOOTSTRAP", "apache-kafka:9092")
KAFKA_TOPIC_TRANSACTIONS = os.environ.get("KAFKA_TOPIC_TRANSACTIONS", "aml.transactions")
BACKEND_API_URL = os.environ.get("BACKEND_API_URL", "http://backend:8000/api")
GNN_ADMIN_TOKEN = os.environ.get("GNN_ADMIN_TOKEN", "")
# A single deployment of this consumer watches one topic, but that topic can still carry more than one
# bank's traffic if each message says which (a "bank" or "institution" field -- see _row_from_message()). A
# message with neither falls back to this per-deployment default, so a single-bank deployment (the common
# case) can just set this once instead of stamping every message. Left unset, backend's own live_ingest()
# falls further back to "unknown" -- never guessed.
LIVE_FEED_DEFAULT_BANK = os.environ.get("LIVE_FEED_DEFAULT_BANK", "")


def _row_from_message(value: dict) -> dict | None:
    """Pulls just the fields gnn_aml_core.livegraph.ingest() needs out of one Kafka message's value, plus
    whichever bank/institution field it carries (if any). Returns None for a message missing src/dst/amount
    -- skipped, same as a malformed pasted row, rather than taking down the whole consumer loop over one bad
    message."""
    src, dst = str(value.get("src", "")).strip(), str(value.get("dst", "")).strip()
    if not src or not dst:
        return None
    try:
        amount = float(value["amount"])
    except (KeyError, TypeError, ValueError):
        return None
    row: dict = {"src": src, "dst": dst, "amount": amount}
    bank = str(value.get("bank") or value.get("institution") or LIVE_FEED_DEFAULT_BANK or "").strip()
    if bank:
        row["bank"] = bank
    if value.get("timestamp") not in (None, ""):
        try:
            row["timestamp"] = int(value["timestamp"])
        except (TypeError, ValueError):
            pass
    if value.get("cross_border") not in (None, ""):
        try:
            row["cross_border"] = int(value["cross_border"])
        except (TypeError, ValueError):
            pass
    return row


def _ingest_batch(rows: list[dict]) -> dict:
    """POSTs one batch to backend's POST /system/live-ingest -- the same ingest-then-score-then-alert path the
    Intake tab's paste-a-document flow triggers manually, run here automatically for every batch off the feed.
    Authenticated with the GNN_ADMIN_TOKEN shared secret (no officer session exists for a background consumer
    to use)."""
    body = json.dumps({"rows": rows}).encode()
    req = urllib.request.Request(f"{BACKEND_API_URL.rstrip('/')}/system/live-ingest", data=body, method="POST")
    req.add_header("Content-Type", "application/json")
    if GNN_ADMIN_TOKEN:
        req.add_header("Authorization", f"Bearer {GNN_ADMIN_TOKEN}")
    with urllib.request.urlopen(req, timeout=90) as resp:
        return json.loads(resp.read())


def _make_consumer():
    """Connects to Kafka with the same retry-on-not-ready-yet pattern aml_synth.exporters.export_kafka already
    uses (the broker's healthcheck gating depends_on only guarantees the broker process is up, not that this
    consumer group's metadata fetch will succeed on the very first try)."""
    from kafka import KafkaConsumer
    from kafka import errors as kafka_errors

    NoBrokersAvailable = (getattr(kafka_errors, "NoBrokersAvailable", None)
                          or getattr(kafka_errors, "NoBrokersAvailableError", None)
                          or getattr(kafka_errors, "KafkaError"))
    for attempt in range(1, 16):
        try:
            return KafkaConsumer(
                KAFKA_TOPIC_TRANSACTIONS, bootstrap_servers=KAFKA_BOOTSTRAP,
                group_id="live-feed-consumer", auto_offset_reset="latest", enable_auto_commit=True,
                value_deserializer=lambda v: json.loads(v.decode()),
                consumer_timeout_ms=1000,   # lets the poll loop below check the batching deadline even while idle
            )
        except NoBrokersAvailable:
            log.warning("kafka not reachable at %s (attempt %d/15)", KAFKA_BOOTSTRAP, attempt)
            time.sleep(4)
    raise RuntimeError(f"Kafka not reachable at {KAFKA_BOOTSTRAP}")


def run(batch_size: int = 25, max_wait_seconds: float = 5.0) -> None:
    consumer = _make_consumer()
    log.info("live-feed-consumer: subscribed to '%s' on %s, batching up to %d message(s) or %.1fs",
             KAFKA_TOPIC_TRANSACTIONS, KAFKA_BOOTSTRAP, batch_size, max_wait_seconds)
    batch: list[dict] = []
    skipped = 0
    deadline = time.monotonic() + max_wait_seconds

    def flush() -> None:
        nonlocal batch, skipped, deadline
        if not batch:
            deadline = time.monotonic() + max_wait_seconds
            return
        try:
            result = _ingest_batch(batch)
            n_alerts = len(result.get("alerts_created", []))
            log.info("ingested %d transaction(s) (%d skipped as malformed) -- live graph now has %s account(s), "
                     "%s transaction(s); %d scored this batch, %d new alert(s) raised (nothing blocked)",
                     len(batch), skipped, result.get("n_accounts"), result.get("n_transactions"),
                     result.get("accounts_scanned", 0), n_alerts)
        except urllib.error.URLError as exc:
            log.warning("ingest-live call failed (%s) -- dropping this batch of %d message(s); Kafka offsets "
                       "were already committed, so a persistently failing gnn-detection-api will lose live "
                       "transactions rather than block the feed indefinitely -- check gnn-detection-api's health "
                       "if this keeps happening", exc, len(batch))
        batch = []
        skipped = 0
        deadline = time.monotonic() + max_wait_seconds

    try:
        while True:
            for message in consumer:
                row = _row_from_message(message.value if isinstance(message.value, dict) else {})
                if row is None:
                    skipped += 1
                else:
                    batch.append(row)
                if len(batch) >= batch_size:
                    flush()
            if time.monotonic() >= deadline:
                flush()
    finally:
        consumer.close()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--batch-size", type=int, default=int(os.environ.get("LIVE_FEED_BATCH_SIZE", 25)))
    ap.add_argument("--max-wait-seconds", type=float, default=float(os.environ.get("LIVE_FEED_MAX_WAIT_SECONDS", 5.0)))
    a = ap.parse_args()
    run(batch_size=a.batch_size, max_wait_seconds=a.max_wait_seconds)


if __name__ == "__main__":
    main()
