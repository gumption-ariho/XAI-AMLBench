"""node_agent -- background scheduler + health monitor (stdlib only, no dependencies).

* Every HEALTH_INTERVAL seconds it probes HTTP health endpoints and TCP ports of the other containers.
* Logs one JSON line per sweep and serves the latest result at http://node_agent:9000/status.
* Add your own periodic jobs with `scheduler.every(seconds, fn)`.
"""
from __future__ import annotations

import json
import logging
import os
import socket
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("node_agent")

HEALTH_INTERVAL = int(os.getenv("HEALTH_INTERVAL", 30))
HTTP_TARGETS = {
    "backend": os.getenv("BACKEND_HEALTH", "http://backend:8000/api/health"),
    "gnn-detection-api": os.getenv("GNN_HEALTH", "http://gnn-detection-api:8000/health"),
    "xai-narrative-api": os.getenv("XAI_HEALTH", "http://xai-narrative-api:8000/health"),
    "neo4j-browser": "http://neo4j-compliance-db:7474",
}
TCP_TARGETS = {
    "postgres": ("database", 5432),
    "redis": ("redis-feature-cache", 6379),
    "kafka": ("apache-kafka", 9092),
    "neo4j-bolt": ("neo4j-compliance-db", 7687),
    "immudb": ("immudb-audit-ledger", 3322),
}
LATEST: dict = {"checked_at": None, "services": {}}


def check_http(url: str) -> dict:
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(url, timeout=3) as r:
            return {"up": 200 <= r.status < 400, "ms": round((time.perf_counter() - t0) * 1000, 1)}
    except Exception as exc:  # noqa: BLE001
        return {"up": False, "error": exc.__class__.__name__}


def check_tcp(host: str, port: int) -> dict:
    t0 = time.perf_counter()
    try:
        with socket.create_connection((host, port), timeout=3):
            return {"up": True, "ms": round((time.perf_counter() - t0) * 1000, 1)}
    except OSError as exc:
        return {"up": False, "error": exc.__class__.__name__}


def health_sweep() -> None:
    services = {n: check_http(u) for n, u in HTTP_TARGETS.items()}
    services.update({n: check_tcp(h, p) for n, (h, p) in TCP_TARGETS.items()})
    LATEST.update(checked_at=time.time(), services=services)
    down = sorted(n for n, s in services.items() if not s["up"])
    log.info(json.dumps({"event": "health_sweep", "down": down, "up_count": len(services) - len(down)}))


class Scheduler:
    def __init__(self):
        self.jobs: list[tuple[int, callable]] = []

    def every(self, seconds: int, fn) -> None:
        self.jobs.append((seconds, fn))

    def run(self) -> None:
        for seconds, fn in self.jobs:
            threading.Thread(target=self._loop, args=(seconds, fn), daemon=True).start()
        while True:
            time.sleep(3600)

    @staticmethod
    def _loop(seconds: int, fn) -> None:
        while True:
            try:
                fn()
            except Exception:  # noqa: BLE001
                log.exception("job %s failed", fn.__name__)
            time.sleep(seconds)


class Status(BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        body = json.dumps(LATEST).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_):
        pass


if __name__ == "__main__":
    threading.Thread(target=HTTPServer(("0.0.0.0", 9000), Status).serve_forever, daemon=True).start()
    scheduler = Scheduler()
    scheduler.every(HEALTH_INTERVAL, health_sweep)
    # scheduler.every(3600, rescan_high_risk_accounts)   # <- example hook for future periodic jobs
    log.info("node_agent started (interval=%ss)", HEALTH_INTERVAL)
    scheduler.run()
