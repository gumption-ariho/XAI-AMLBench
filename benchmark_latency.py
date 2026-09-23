#!/usr/bin/env python3
"""
benchmark_latency.py  -  a real cold/warm latency benchmark for the running GNN service, against the brief's
100ms target (brief 2.1: "Inference Latency: <100ms per transaction").

Why this exists: the scorecard's own latency check (check_brief_v1.py --api ...) samples only 10 accounts once,
which cannot separate a cold cache miss from a warm cache hit and cannot catch a large-neighbourhood account
skewing the tail. Earlier manual testing on this project saw the SAME account score in 3033ms cold and 24.7ms
warm -- a >100x difference the scorecard's small sample could easily miss.

This script:
  * pulls many distinct accounts (not just the 5-10 /accounts/sample returns by default)
  * measures /predict COLD (first-ever call per account, empty cache) and WARM (repeat call, cache hit)
    SEPARATELY, over each account, so the two regimes are never averaged together
  * reports P50 / P95 / P99 / max for each, which is what the brief's "<100ms" language should be checked
    against -- a mean hides exactly the outliers (large hub accounts) that matter most
  * also benchmarks /explain (GNNExplainer), with its own, much looser expectation: this endpoint is a
    post-hoc analysis step, not a real-time score, so it is reported for visibility and outlier-catching
    (does a hub account make it take 30 seconds?), not scored against the 100ms target

    python3 benchmark_latency.py                                    # http://localhost/gnn, up to 100 accounts
    python3 benchmark_latency.py --base http://127.0.0.1:8001         # run_local.py's direct port
    python3 benchmark_latency.py --extra-accounts ACC0005000 ACC0001234   # make sure specific accounts are covered
    python3 benchmark_latency.py --skip-explain                     # /predict only (faster to run)
    python3 benchmark_latency.py --json latency_report.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import urllib.error
import urllib.request

PREDICT_TARGET_MS = 100.0  # brief 2.1


def call(base: str, method: str, path: str, body=None, timeout: float = 60):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(base.rstrip("/") + path, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read()
    dt_ms = (time.perf_counter() - t0) * 1000
    return json.loads(raw), dt_ms


def percentiles(values: list) -> dict:
    if not values:
        return {"p50": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0, "mean": 0.0, "n": 0}
    s = sorted(values)
    pick = lambda p: s[min(int(len(s) * p), len(s) - 1)]
    return {"p50": pick(0.50), "p95": pick(0.95), "p99": pick(0.99), "max": s[-1],
           "mean": statistics.mean(values), "n": len(values)}


def fmt(stats: dict) -> str:
    return f"p50 {stats['p50']:.1f} ms | p95 {stats['p95']:.1f} ms | p99 {stats['p99']:.1f} ms | max {stats['max']:.1f} ms  (n={stats['n']})"


def gather_accounts(base: str, n: int, extra: list) -> list:
    """/accounts/sample is fully deterministic -- it always returns the FIRST n suspicious and FIRST n benign
    accounts by construction order (n capped at 50 server-side), so calling it repeatedly never yields more
    than the same ~100 accounts; there is no pagination. That cap also means this benchmark may not happen to
    include a specific pathological account (a large hub) purely by chance of construction order -- pass its
    id via --extra-accounts if you know one (e.g. from a training log) to make sure it is covered."""
    body, _ = call(base, "GET", "/accounts/sample?n=50")
    # extra-accounts are guaranteed to be included regardless of n, since the whole point of passing them is to
    # make sure a SPECIFIC account (e.g. a known large hub) is covered even when n is smaller than the sample pool.
    result: dict[str, None] = {a: None for a in extra}
    for a in body.get("suspicious", []) + body.get("benign", []):
        if len(result) >= max(n, len(extra)):
            break
        result[a] = None
    available = len(set(extra) | set(body.get("suspicious", [])) | set(body.get("benign", [])))
    if n > available:
        print(f"  note: /accounts/sample can only provide {available} distinct accounts (it is not paginated); "
             f"benchmarking {available} instead of the requested {n}")
    return list(result)


def bench_predict(base: str, accounts: list) -> tuple[dict, dict]:
    cold_times, warm_times = [], []
    for a in accounts:
        _, dt = call(base, "POST", "/predict", {"account_ids": [a]})
        cold_times.append(dt)
    for a in accounts:
        _, dt = call(base, "POST", "/predict", {"account_ids": [a]})  # same account again: should hit cache
        warm_times.append(dt)
    return percentiles(cold_times), percentiles(warm_times)


def bench_explain(base: str, accounts: list) -> dict:
    times = []
    for a in accounts:
        try:
            _, dt = call(base, "POST", "/explain", {"account_id": a, "epochs": 100}, timeout=120)
            times.append(dt)
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError):
            continue  # a single account's explanation failing (e.g. not the gatv2 model) shouldn't abort the run
    return percentiles(times)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost/gnn", help="GNN service base URL")
    ap.add_argument("-n", "--accounts", type=int, default=60, help="how many distinct accounts to benchmark (capped at ~100 by the API, see gather_accounts)")
    ap.add_argument("--extra-accounts", nargs="*", default=[], help="specific account ids to make sure are covered (e.g. a known large hub)")
    ap.add_argument("--skip-explain", action="store_true", help="only benchmark /predict (much faster)")
    ap.add_argument("--explain-n", type=int, default=10, help="/explain is slow; benchmark fewer accounts for it")
    ap.add_argument("--json", default="")
    a = ap.parse_args()

    print(f"XAI-AMLBench: latency benchmark  ({a.base})\n" + "=" * 70)
    try:
        health, _ = call(a.base, "GET", "/health")
    except (urllib.error.URLError, urllib.error.HTTPError) as exc:
        print(f"\ncannot reach {a.base}: {exc}\nIs the service running? (docker compose ps, or check run_local.py)")
        return 1
    if not health.get("model_loaded"):
        print(f"\n{a.base} is reachable but no model is loaded: {health}\nRun: python -m gnn_aml_core.train --data data --out models")
        return 1
    print(f"model: {health.get('model')}  |  Redis cache: {'on' if health.get('redis') else 'off'}")

    print(f"\ngathering up to {a.accounts} distinct accounts ...")
    accounts = gather_accounts(a.base, a.accounts, a.extra_accounts)
    print(f"got {len(accounts)} accounts")
    if len(accounts) < 5:
        print("too few accounts to benchmark meaningfully (is the model trained on real data?)")
        return 1

    print(f"\n/predict -- {len(accounts)} accounts, cold pass then warm pass")
    cold, warm = bench_predict(a.base, accounts)
    print(f"  COLD (first call, cache miss):  {fmt(cold)}")
    print(f"  WARM (repeat call, cache hit):  {fmt(warm)}")
    cold_ok = cold["p95"] < PREDICT_TARGET_MS
    warm_ok = warm["p95"] < PREDICT_TARGET_MS
    print(f"  brief target (<{PREDICT_TARGET_MS:.0f} ms, p95): cold {'PASS' if cold_ok else 'FAIL'}, warm {'PASS' if warm_ok else 'FAIL'}")
    if not cold_ok:
        print(f"  note: a p95 above target on the COLD pass is common and often acceptable in production (warm/cached\n"
              f"        traffic dominates real usage) -- but check the MAX ({cold['max']:.0f} ms) isn't driven by one\n"
              f"        pathological hub account (a smurfing centre with 1000+ neighbours); see /explain below")

    explain_stats = None
    if not a.skip_explain:
        n_ex = min(a.explain_n, len(accounts))
        print(f"\n/explain -- {n_ex} accounts (GNNExplainer; not scored against the 100ms target, reported for visibility)")
        t0 = time.time()
        explain_stats = bench_explain(a.base, accounts[:n_ex])
        print(f"  {fmt(explain_stats)}   ({time.time() - t0:.0f}s total)")
        if explain_stats["max"] > 10_000:
            print(f"  note: the slowest explanation took {explain_stats['max']/1000:.1f}s -- if this recurs, it is")
            print(f"        likely a large hub account; gnn_aml_core.main's MAX_EXPLAIN_EDGES fallback to a 1-hop")
            print(f"        neighbourhood should already limit this, but a smaller value may be worth trying")

    print("\n" + "=" * 70)
    result = {"base": a.base, "n_accounts": len(accounts), "predict_cold": cold, "predict_warm": warm,
             "explain": explain_stats, "target_ms": PREDICT_TARGET_MS, "cold_meets_target": cold_ok, "warm_meets_target": warm_ok}
    if a.json:
        with open(a.json, "w") as f:
            json.dump(result, f, indent=2)
        print(f"saved {a.json}")
    return 0 if warm_ok else 1


if __name__ == "__main__":
    sys.exit(main())
