#!/usr/bin/env python3
"""
check_latency_v1.py  -  a real cold/warm latency benchmark for the running GNN service, against the brief's
100ms target (section 2.1: "Inference Latency: <100ms per transaction").

Why this exists: a single anecdotal reading (one scan took 3033ms, a later one took 24ms) proves nothing on its
own -- it could be a fluke, a cold cache, or a genuinely slow account. This script measures COLD calls (a fresh
account, never scored before) and WARM calls (the same account, second time, should hit the Redis cache)
separately, across accounts stratified by connectivity (a smurfing collection account with thousands of
transactions is a fundamentally different case from an ordinary account with a handful), and reports p50/p95/p99
-- the tail, not just the average, is what a latency target is really about.

    python3 check_latency_v1.py                                  # http://localhost, models/graph.pt
    python3 check_latency_v1.py --base http://localhost --gnn http://127.0.0.1:8001   # services run by run_local.py
    python3 check_latency_v1.py --n-per-bucket 30 --json latency_report.json

Needs the stack (or run_local.py) already running, and a trained model (so the graph file used to stratify
accounts by connectivity exists). Only reads data: scores accounts via /predict, never creates alerts.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

LATENCY_TARGET_MS = 100.0


def call(base, method, path, body=None, timeout=30):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(base.rstrip("/") + path, data=data, method=method, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def percentile(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    k = (len(s) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def summarize(label: str, times_ms: list[float]) -> dict:
    if not times_ms:
        return {"label": label, "n": 0}
    return {"label": label, "n": len(times_ms), "mean": statistics.mean(times_ms), "p50": percentile(times_ms, 0.50),
            "p95": percentile(times_ms, 0.95), "p99": percentile(times_ms, 0.99), "max": max(times_ms),
            "meets_target_p95": percentile(times_ms, 0.95) <= LATENCY_TARGET_MS}


def print_summary(s: dict) -> None:
    if s["n"] == 0:
        print(f"  {s['label']:<28} no data")
        return
    flag = "OK  " if s["meets_target_p95"] else "OVER"
    print(f"  {s['label']:<28} n={s['n']:<4} mean {s['mean']:>7.1f} ms   p50 {s['p50']:>7.1f}   "
          f"p95 {s['p95']:>7.1f}   p99 {s['p99']:>7.1f}   max {s['max']:>8.1f}   [{flag} vs {LATENCY_TARGET_MS:.0f}ms target on p95]")


def stratify_by_degree(graph_path: Path, n_per_bucket: int) -> dict[str, list[str]] | None:
    """Bucket accounts into low/medium/high connectivity using the trained graph, so the benchmark deliberately
    includes hub accounts rather than relying on a random sample to stumble onto one. Returns None (caller falls
    back to a plain random sample) if PyTorch or the graph file is not available -- this script still runs
    without them, just with less diagnostic power."""
    try:
        import torch
    except ImportError:
        print("  note: PyTorch not available here, so accounts cannot be stratified by connectivity; "
              "falling back to whatever /accounts/sample returns (no note on which are hubs).")
        return None
    if not graph_path.exists():
        print(f"  note: {graph_path} not found, so accounts cannot be stratified by connectivity; "
              "falling back to whatever /accounts/sample returns.")
        return None
    g = torch.load(graph_path, map_location="cpu", weights_only=False)
    n_tx = int(g["n_tx"])
    ei = g["edge_index"][:, :n_tx]  # only the original transactions, not the reversed copies, for a true degree count
    deg = torch.zeros(g["x"].shape[0], dtype=torch.long)
    deg.scatter_add_(0, ei[0], torch.ones(n_tx, dtype=torch.long))
    deg.scatter_add_(0, ei[1], torch.ones(n_tx, dtype=torch.long))
    ids = g["account_ids"]
    order = torch.argsort(deg)
    n = len(ids)
    buckets = {
        "low (median connectivity)": [ids[i] for i in order[n // 2:n // 2 + n_per_bucket].tolist()],
        "high (top connectivity, likely hubs)": [ids[i] for i in order[-n_per_bucket:].tolist()],
    }
    print(f"  stratified {n:,} accounts by connectivity: low bucket median degree {int(deg[order[n // 2]])}, "
          f"high bucket max degree {int(deg[order[-1]])}")
    return buckets


def benchmark_bucket(base: str, label: str, account_ids: list[str]) -> tuple[dict, dict]:
    """Two calls per account, bucketed by what the API itself reports as cached rather than by call order --
    if an account was already scored by a previous run of this script (or by someone using the app), Redis may
    still hold it, and the "first" call here would silently be warm rather than cold. Trusting the API's own
    "cached" field, instead of assuming call order, means this stays correct regardless of prior state."""
    cold, warm = [], []
    for a in account_ids:
        for _ in range(2):
            t0 = time.perf_counter()
            try:
                r = call(base, "POST", "/predict", {"account_ids": [a]})
            except (urllib.error.URLError, urllib.error.HTTPError) as exc:
                print(f"    {a}: request failed ({exc.__class__.__name__}), skipping")
                break
            res = r["results"][0]
            lat = res.get("latency_ms", (time.perf_counter() - t0) * 1000)
            (warm if res.get("cached") else cold).append(lat)
    return summarize(f"{label} (cold)", cold), summarize(f"{label} (warm, cached)", warm)


def warm_up(base: str, exclude: set[str], n: int = 5) -> None:
    """Fire a handful of throwaway /predict calls before any real measurement begins. Without this, whichever
    bucket happens to be measured first unfairly absorbs one-time process warm-up costs (thread pool setup,
    memory paging, lazy kernel initialisation on the first few calls to a freshly-started service) that have
    nothing to do with account connectivity -- this is what produced a counterintuitive first result where the
    "ordinary accounts" bucket looked slower than the "hub accounts" bucket, simply because it ran first against
    a cold process. Uses accounts NOT in `exclude` so it never corrupts a bucket's own cold/warm measurement."""
    try:
        sample = call(base, "GET", "/accounts/sample?n=10")
    except Exception:  # noqa: BLE001
        return
    candidates = [a for a in (sample.get("suspicious", []) + sample.get("benign", [])) if a not in exclude][:n]
    for a in candidates:
        try:
            call(base, "POST", "/predict", {"account_ids": [a]})
        except Exception:  # noqa: BLE001
            pass


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost", help="base URL the GNN service answers at (through the gateway, or directly if --gnn is not given)")
    ap.add_argument("--gnn", default="", help="direct base URL of the model service, e.g. http://127.0.0.1:8001 (services run by run_local.py); default: --base + /gnn")
    ap.add_argument("--graph", default="models/graph.pt", help="path to the trained graph file, used to find hub accounts to stratify by")
    ap.add_argument("--n-per-bucket", type=int, default=20)
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    gnn_base = a.gnn or (a.base.rstrip("/") + "/gnn")

    print(f"XAI-AMLBench: latency benchmark  ({gnn_base})\n" + "=" * 78)
    try:
        health = call(gnn_base, "GET", "/health")
    except Exception as exc:  # noqa: BLE001
        print(f"cannot reach the model service at {gnn_base}: {exc}\n"
              "-> is the stack running? docker compose ps , or python3 run_local.py")
        return 1
    if not health.get("model_loaded"):
        print("model service is up but no model is loaded -> python -m gnn_aml_core.train --data data --out models, then restart it")
        return 1

    buckets = stratify_by_degree(Path(a.graph), a.n_per_bucket)
    if buckets is None:
        try:
            sample = call(gnn_base, "GET", f"/accounts/sample?n={a.n_per_bucket}")
        except Exception as exc:  # noqa: BLE001
            print(f"could not fetch a sample of accounts: {exc}")
            return 1
        buckets = {"sample (not stratified)": (sample["suspicious"] + sample["benign"])[:a.n_per_bucket]}

    all_bucketed_ids = {a_id for ids in buckets.values() for a_id in ids}
    print("  warming up the service with a few throwaway calls (excluded accounts stay untouched, so their "
          "cold measurement below is unaffected) ...")
    warm_up(gnn_base, exclude=all_bucketed_ids)

    print()
    results = {}
    for label, ids in buckets.items():
        if not ids:
            continue
        print(f"{label}: {len(ids)} accounts")
        cold, warm = benchmark_bucket(gnn_base, label, ids)
        print_summary(cold)
        print_summary(warm)
        if cold["n"] == 0 and warm["n"] > 0:
            print("    (every account in this bucket was already cached before this run started -- restart "
                  "Redis, or wait out PRED_CACHE_TTL, then re-run for a genuine cold measurement)")
        print()
        results[label] = {"cold": cold, "warm": warm}

    # A pooled p95 across a mix of ordinary accounts and deliberately-included hub accounts would not be a
    # meaningful single SLA number, so the honest reading is per bucket (printed above), not a single blended
    # figure. The overall verdict below is "every bucket's cold p95 is within target", which is a stricter and
    # more informative bar than a pooled average could ever be.
    print("=" * 78)
    with_cold_data = [v for v in results.values() if v["cold"]["n"] > 0]
    no_cold_data = [k for k, v in results.items() if v["cold"]["n"] == 0]
    if not with_cold_data:
        print(f"  Target: p95 <= {LATENCY_TARGET_MS:.0f} ms (brief section 2.1). No cold data was collected in "
              "any bucket (every account was already cached) -- restart Redis or wait out PRED_CACHE_TTL, then re-run.")
        overall_ok = False
    else:
        overall_ok = all(v["cold"]["meets_target_p95"] for v in with_cold_data)
        verdict = "All buckets with cold data meet it." if overall_ok else "At least one bucket exceeds it on COLD calls -- see above."
        note = f" ({len(no_cold_data)} bucket(s) had no cold data -- see notes above.)" if no_cold_data else ""
        print(f"  Target: p95 <= {LATENCY_TARGET_MS:.0f} ms (brief section 2.1). {verdict}{note}")
    print("  A bucket failing only on cold (not warm) calls means the Redis cache is doing its job; the first")
    print("  scan of a given account is the honest number to report, since a real officer scans new accounts.")

    if a.json:
        Path(a.json).write_text(json.dumps({"target_ms": LATENCY_TARGET_MS, "buckets": results}, indent=2, default=str))
        print(f"  saved {a.json}")
    return 0 if overall_ok else 1


if __name__ == "__main__":
    sys.exit(main())
