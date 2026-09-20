"""Export a SyntheticGraph to CSV, Kafka (transaction stream) and Neo4j (investigation graph)."""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

log = logging.getLogger("aml_synth.exporters")

TX_PROPS = ["tx_id", "amount", "currency", "payment_format", "timestamp",
            "cross_border", "is_laundering", "typology", "pattern_id"]


# --------------------------------------------------------------------- CSV
def export_csv(graph, out_dir: str = "data") -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    graph.accounts.to_csv(out / "accounts.csv", index=False)
    graph.transactions.to_csv(out / "transactions.csv", index=False)
    log.info("wrote %s/accounts.csv and %s/transactions.csv", out, out)


# ------------------------------------------------------------------- Kafka
def export_kafka(graph, bootstrap: str | None = None, topic: str | None = None,
                 rate_per_sec: float | None = None) -> None:
    """Stream transactions in time order. rate_per_sec=0 sends as fast as possible."""
    from kafka import KafkaProducer
    from kafka.errors import NoBrokersAvailable

    bootstrap = bootstrap or os.getenv("KAFKA_BOOTSTRAP", "apache-kafka:9092")
    topic = topic or os.getenv("KAFKA_TOPIC_TRANSACTIONS", "aml.transactions")
    rate = float(os.getenv("KAFKA_RATE_PER_SEC", 0)) if rate_per_sec is None else rate_per_sec

    producer = None
    for attempt in range(1, 16):
        try:
            producer = KafkaProducer(
                bootstrap_servers=bootstrap,
                acks="all",
                linger_ms=20,
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
