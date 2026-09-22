"""Export a SyntheticGraph to CSV, Parquet, JSON, Neo4j Cypher, a Kafka stream, or straight into Neo4j.

    csv      accounts.csv + transactions.csv            (spreadsheet friendly)
    parquet  accounts.parquet + transactions.parquet    (primary format for distributed processing; needs pyarrow)
    json     graph_summary.json                         (metadata, schema, summary statistics)
    cypher   import.cypher                              (run with: cypher-shell -f import.cypher)
    kafka    transaction stream, in time order
    neo4j    direct load into a running Neo4j
"""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

log = logging.getLogger("aml_synth.exporters")

TX_PROPS = ["tx_id", "amount", "currency", "payment_format", "timestamp",
            "cross_border", "is_laundering", "typology", "pattern_id"]

SCHEMA = {
    "accounts": {
        "account_id": "string, unique id (ACC0000001)", "account_type": "individual | business | shell",
        "country": "ISO-like country code (offshore set: VG KY PA SC BZ)", "opened_ts": "account opening time, unix seconds",
        "risk_score": "0..1 KYC risk rating (a weak prior, not the label)", "is_suspicious": "1 if the account takes part in a laundering pattern",
        "typology": "typology of the first pattern the account joined, or 'none'"},
    "transactions": {
        "tx_id": "string, unique", "src": "sending account_id", "dst": "receiving account_id", "amount": "USD",
        "currency": "always USD", "payment_format": "wire | ach | card | cash | crypto_ramp", "timestamp": "unix seconds (UTC)",
        "cross_border": "1 if sender and receiver countries differ", "is_laundering": "1 if part of an injected laundering pattern",
        "typology": "smurfing | scatter_gather | cyclic_loop | shell_company | cross_border_velocity | none",
        "pattern_id": "id of the injected pattern, -1 for ordinary traffic"},
}


def _out(out_dir) -> Path:
    p = Path(out_dir)
    p.mkdir(parents=True, exist_ok=True)
    return p


# --------------------------------------------------------------------- CSV
def export_csv(graph, out_dir: str = "data") -> None:
    out = _out(out_dir)
    graph.accounts.to_csv(out / "accounts.csv", index=False)
    graph.transactions.to_csv(out / "transactions.csv", index=False)
    log.info("wrote %s/accounts.csv and %s/transactions.csv", out, out)


# ----------------------------------------------------------------- Parquet
def export_parquet(graph, out_dir: str = "data") -> None:
    try:
        import pyarrow  # noqa: F401
    except ImportError as exc:
        raise ImportError("Parquet export needs pyarrow:  pip install pyarrow") from exc
    out = _out(out_dir)
    graph.accounts.to_parquet(out / "accounts.parquet", index=False)
    graph.transactions.to_parquet(out / "transactions.parquet", index=False)
    log.info("wrote %s/accounts.parquet and %s/transactions.parquet", out, out)


# -------------------------------------------------------------------- JSON
def export_json(graph, out_dir: str = "data") -> None:
    """Metadata + graph summary (not the full graph: use CSV/Parquet for that)."""
    out = _out(out_dir)
    doc = {
        "name": "XAI-AMLBench synthetic transaction graph",
        "generator_version": graph.config.get("generator_version"),
        "license": "CC-BY-4.0 (data), MIT (code)",
        "synthetic": True,
        "contains_personal_data": False,
        "config": {k: v for k, v in graph.config.items() if k not in ("generator_version",)},
        "summary": graph.summary(),
        "schema": SCHEMA,
    }
    (out / "graph_summary.json").write_text(json.dumps(doc, indent=2, default=str), encoding="utf-8")
    log.info("wrote %s/graph_summary.json", out)


# ------------------------------------------------------------------ Cypher
def _lit(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int,)) or type(v).__name__.startswith(("int", "uint")):
        return str(int(v))
    if isinstance(v, float) or type(v).__name__.startswith("float"):
        return repr(float(v))
    return json.dumps(str(v), ensure_ascii=False)          # Cypher accepts JSON-style double-quoted strings


def _map(row: dict, keys) -> str:
    return "{" + ", ".join(f"{k}: {_lit(row[k])}" for k in keys) + "}"


def export_cypher(graph, out_dir: str = "data", batch: int = 1000) -> None:
    """Writes import.cypher: constraints + batched UNWIND statements. Run with `cypher-shell -f import.cypher`."""
    out = _out(out_dir)
    acc_keys = ["account_id", "account_type", "country", "opened_ts", "risk_score", "is_suspicious", "typology"]
    tx_keys = ["src", "dst"] + TX_PROPS
    accounts = graph.accounts[acc_keys].to_dict("records")
    txs = graph.transactions[tx_keys].to_dict("records")
    with open(out / "import.cypher", "w", encoding="utf-8") as f:
        f.write("// XAI-AMLBench synthetic graph. Load with:  cypher-shell -u neo4j -p <password> -f import.cypher\n")
        f.write("CREATE CONSTRAINT account_id IF NOT EXISTS FOR (a:Account) REQUIRE a.account_id IS UNIQUE;\n")
        for i in range(0, len(accounts), batch):
            rows = ",\n  ".join(_map(r, acc_keys) for r in accounts[i:i + batch])
            f.write(f"UNWIND [\n  {rows}\n] AS row\nMERGE (a:Account {{account_id: row.account_id}}) SET a += row;\n")
        for i in range(0, len(txs), batch):
            rows = ",\n  ".join(_map(r, tx_keys) for r in txs[i:i + batch])
            f.write("UNWIND [\n  " + rows + "\n] AS row\nMATCH (s:Account {account_id: row.src}), (d:Account {account_id: row.dst})\n"
                    "CREATE (s)-[:TRANSFERRED {tx_id: row.tx_id, amount: row.amount, currency: row.currency, payment_format: row.payment_format, "
                    "timestamp: row.timestamp, cross_border: row.cross_border, is_laundering: row.is_laundering, typology: row.typology, "
                    "pattern_id: row.pattern_id}]->(d);\n")
    log.info("wrote %s/import.cypher (%d accounts, %d transactions)", out, len(accounts), len(txs))


# ------------------------------------------------------------------- Kafka
def export_kafka(graph, bootstrap: str | None = None, topic: str | None = None,
                 rate_per_sec: float | None = None) -> None:
    """Stream transactions in time order. rate_per_sec=0 sends as fast as possible."""
    from kafka import KafkaProducer
    from kafka import errors as kafka_errors
    # kafka-python has renamed/restructured this exception across major versions; try the classic name first,
    # then a couple of plausible renames, and fall back to the library's own base exception class so the retry
    # loop below still catches connection failures correctly even if none of the specific names match.
    NoBrokersAvailable = (getattr(kafka_errors, "NoBrokersAvailable", None)
                          or getattr(kafka_errors, "NoBrokersAvailableError", None)
                          or getattr(kafka_errors, "KafkaError"))

    bootstrap = bootstrap or os.getenv("KAFKA_BOOTSTRAP", "apache-kafka:9092")
    topic = topic or os.getenv("KAFKA_TOPIC_TRANSACTIONS", "aml.transactions")
    rate = float(os.getenv("KAFKA_RATE_PER_SEC", 0)) if rate_per_sec is None else rate_per_sec

    producer = None
    for attempt in range(1, 16):
        try:
            producer = KafkaProducer(
                bootstrap_servers=bootstrap, acks="all", linger_ms=20,
                value_serializer=lambda v: json.dumps(v, default=str).encode(),
                key_serializer=lambda k: k.encode(),
            )
            break
        except NoBrokersAvailable:
            log.warning("kafka not reachable at %s (attempt %d/15)", bootstrap, attempt)
            time.sleep(4)
    if producer is None:
        raise RuntimeError(f"Kafka not reachable at {bootstrap}")

    n = 0
    for row in graph.transactions.to_dict("records"):
        producer.send(topic, key=row["src"], value=row)
        n += 1
        if rate:
            time.sleep(1.0 / rate)
    producer.flush()
    log.info("published %d transactions to topic '%s'", n, topic)


# ------------------------------------------------------------------- Neo4j
def export_neo4j(graph, uri: str | None = None, user: str | None = None,
                 password: str | None = None, batch: int = 5000, wipe: bool = True) -> None:
    from neo4j import GraphDatabase

    uri = uri or os.getenv("NEO4J_URI", "bolt://neo4j-compliance-db:7687")
    user = user or os.getenv("NEO4J_USER", "neo4j")
    password = password or os.getenv("NEO4J_PASSWORD", "neo4j")

    driver = GraphDatabase.driver(uri, auth=(user, password))
    for attempt in range(1, 16):
        try:
            driver.verify_connectivity()
            break
        except Exception as exc:  # noqa: BLE001
            log.warning("neo4j not ready (%s) attempt %d/15", exc.__class__.__name__, attempt)
            time.sleep(4)
    else:
        raise RuntimeError(f"Neo4j not reachable at {uri}")

    accounts = graph.accounts.to_dict("records")
    txs = graph.transactions.to_dict("records")

    with driver, driver.session() as s:
        if wipe:
            s.run("MATCH (n:Account) DETACH DELETE n")
        s.run("CREATE CONSTRAINT account_id IF NOT EXISTS FOR (a:Account) REQUIRE a.account_id IS UNIQUE")
        for i in range(0, len(accounts), batch):
            s.run("UNWIND $rows AS row MERGE (a:Account {account_id: row.account_id}) SET a += row",
                  rows=accounts[i:i + batch])
        for i in range(0, len(txs), batch):
            s.run(
                """
                UNWIND $rows AS row
                MATCH (s:Account {account_id: row.src}), (d:Account {account_id: row.dst})
                MERGE (s)-[t:TRANSFERRED {tx_id: row.tx_id}]->(d)
                SET t += row{.amount, .currency, .payment_format, .timestamp,
                             .cross_border, .is_laundering, .typology, .pattern_id}
                """,
                rows=txs[i:i + batch],
            )
    log.info("loaded %d accounts and %d transactions into Neo4j", len(accounts), len(txs))
