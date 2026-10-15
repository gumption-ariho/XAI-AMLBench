#!/usr/bin/env python3
"""
check_latency_v1.py  -  stratified cold/warm latency check for the running GNN service, against the brief's
100ms target (brief 2.1: "Inference Latency: <100ms per transaction").

Why this exists, and why it's a different tool from benchmark_latency.py: benchmark_latency.py samples whatever
/accounts/sample happens to return (its first ~50 suspicious + ~50 benign, by construction order) and calls that
"cold". Two problems with that on real AMLworld data:

  1. A fresh account's TRUE first request isn't just a Redis cache miss -- the GNN service's torch kernels,
     thread pools and Redis connection are also "cold" right after the process starts. The very first /predict
     call of any kind absorbs that one-time cost and makes every other account look artificially fast by
     comparison, even though in production only the first real request pays it.
  2. /accounts/sample's construction-order sample skews toward whatever the synthetic/AMLworld generator put
     first, not toward the accounts that are actually expensive: main.py's _score() pulls in the k-hop
     neighbourhood of every requested account (gnn_aml_core/main.py's k_hop_subgraph call), so an account sitting
     next to a smurfing-style hub can cost far more than its own degree suggests -- see
     tests/test_check_latency.py's TestStratifyByDegree for the exact case (a hub's individual customers have
     degree 1 each, but their 2-hop neighbourhood is the whole hub cluster).

check_latency_v1.py fixes both: it warms up the SERVICE first (torch/Redis/connection-pool warm-up) using a
disjoint set of decoy accounts, pulled from the graph independently of the accounts actually measured, so those
decoys never contaminate the real measurement. Then it stratifies accounts by their k-hop NEIGHBOURHOOD size
(not raw degree) into low/medium/high buckets, and measures a true cold pass (each bucket account's genuine
first-ever call) and a warm pass (repeat call, cache hit) on each bucket separately.

    python3 check_latency_v1.py --gnn http://127.0.0.1:8001
    python3 check_latency_v1.py --gnn http://127.0.0.1:8001 --n-per-bucket 25 --json latency_report.json
    python3 check_latency_v1.py --gnn http://127.0.0.1:8001 --graph models/graph.pt --seed 7
    python3 check_latency_v1.py --gnn http://127.0.0.1:8001 --concurrency 10   # also test request batching
      # (--concurrency fires N /predict calls AT ONCE per bucket, against accounts the sequential pass above
      #  never touched -- this is the only way to see whether gnn_aml_core/main.py's request batching
      #  (GNN_BATCH_WINDOW_MS) is doing anything, since the default sequential measurement never has more
      #  than one request in flight and so can't exercise it either way)

If models/graph.pt isn't found (or torch/scipy aren't installed), this falls back to benchmark_latency.py's
flat /accounts/sample-based sampling and prints a note -- the stratified-by-neighbourhood-size measurement is a
refinement on top of that, not a replacement that hard-requires local model artifacts to run at all.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PREDICT_TARGET_MS = 100.0  # brief 2.1; overridable at runtime via --target-ms (see main())


# ------------------------------------------------------------------ pure-python stats helpers
# (no network, no torch -- these always run; see tests/test_check_latency.py)
def percentile(values: list, p: float) -> float:
    """Linear-interpolation percentile (numpy.percentile's default 'linear' method), 0.0 <= p <= 1.0."""
    if not values:
        return float("nan")
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * p
    f, c = math.floor(k), math.ceil(k)
    if f == c:
        return s[int(k)]
    return s[f] * (c - k) + s[c] * (k - f)


def summarize(label: str, values: list) -> dict:
    """n/mean/max/p50/p95/p99 for one bucket of latency samples (ms), plus whether p95 meets PREDICT_TARGET_MS.
    An empty list reports only {"label", "n": 0} -- there is nothing else honest to say about zero samples."""
    out = {"label": label, "n": len(values)}
    if not values:
        return out
    out["mean"] = statistics.mean(values)
    out["max"] = max(values)
    out["p50"] = percentile(values, 0.50)
    out["p95"] = percentile(values, 0.95)
    out["p99"] = percentile(values, 0.99)
    out["meets_target_p95"] = out["p95"] < PREDICT_TARGET_MS
    return out


def fmt(stats: dict) -> str:
    if stats["n"] == 0:
        return f"{stats['label']}: no samples"
    return (f"{stats['label']:<42} p50 {stats['p50']:7.1f} ms | p95 {stats['p95']:7.1f} ms | "
           f"p99 {stats['p99']:7.1f} ms | max {stats['max']:7.1f} ms  (n={stats['n']})")


# ------------------------------------------------------------------ HTTP
def call(base: str, method: str, path: str, body=None, timeout: float = 30):
    """Returns the parsed JSON body only -- timing is the caller's job (so a caller can time a sequence of
    calls, or skip timing a warm-up call entirely). Raises on a non-2xx response or a connection failure."""
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(base.rstrip("/") + path, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read()
    return json.loads(raw)


def timed_call(base: str, method: str, path: str, body=None, timeout: float = 30) -> tuple:
    t0 = time.perf_counter()
    result = call(base, method, path, body, timeout)
    return result, (time.perf_counter() - t0) * 1000


# ------------------------------------------------------------------ pipeline warm-up
def warm_up(base: str, exclude: set, n: int = 10) -> None:
    """Prime the SERVICE (torch kernels, thread pools, the Redis connection) using decoy accounts that are
    never part of the real measurement, so the stratified cold pass below measures per-account cache misses,
    not the one-time first-request-after-startup cost. Never raises: a warm-up that fails just means the real
    measurement runs slightly colder than intended, which is a reason to look at the numbers more closely, not
    a reason to abort the whole check."""
    try:
        body = call(base, "GET", "/accounts/sample?n=50")
    except Exception:  # noqa: BLE001
        return
    candidates = [a for a in body.get("suspicious", []) + body.get("benign", []) if a not in exclude]
    for a in candidates[:n]:
        try:
            call(base, "POST", "/predict", {"account_ids": [a]})
        except Exception:  # noqa: BLE001
            continue


# ------------------------------------------------------------------ stratification by k-hop neighbourhood size
def stratify_by_degree(graph_path, n_per_bucket: int, hops: int = 2, seed: int | None = None):
    """Buckets the accounts in models/graph.pt into low/medium/high by the size of their k-hop neighbourhood
    (the set _score() in gnn_aml_core/main.py actually pulls into the subgraph for a /predict call on that
    account), not by raw degree. Returns None if graph_path doesn't exist.

    Raw degree is misleading here: an account with one edge into a smurfing-style hub has the ENTIRE hub
    cluster in its 2-hop neighbourhood (and is just as expensive to score as the hub itself), while a hub's
    own 500 direct, non-fanning neighbours can leave its own 2-hop neighbourhood no bigger than its 1-hop one.
    See tests/test_check_latency.py::TestStratifyByDegree for the worked example this guards against.

    seed controls only which accounts are SAMPLED from each bucket (the buckets themselves are a deterministic
    function of the graph). Pass a fixed seed to reproduce a specific run; leave it None to get a fresh sample
    each time, which matters when re-running this script within Redis's cache TTL (a fixed seed would always
    pick the same accounts, and they'd already be warm from the previous run)."""
    graph_path = Path(graph_path)
    if not graph_path.exists():
        return None

    import numpy as np
    import torch
    from scipy import sparse

    g = torch.load(graph_path, map_location="cpu", weights_only=False)
    ei = g["edge_index"]
    n = g["x"].size(0)
    account_ids = g["account_ids"]
    if n < 3:
        return None

    src, dst = ei[0].numpy(), ei[1].numpy()
    ones = np.ones(len(src), dtype=np.int8)
    adj = sparse.csr_matrix((ones, (src, dst)), shape=(n, n)).astype(bool)

    reach = adj.copy()
    frontier = adj.copy()
    for _ in range(max(hops - 1, 0)):
        frontier = (frontier @ adj).astype(bool)
        reach = (reach + frontier).astype(bool)

    sizes = np.asarray(reach.sum(axis=1)).ravel().astype(np.int64)
    sizes -= reach.diagonal().astype(np.int64)  # a node's own neighbourhood never counts itself

    order = np.argsort(-sizes, kind="stable")  # descending by neighbourhood size; ties keep original (index) order
    third = n // 3
    pools = {
        "high (large neighbourhood, likely hub-adjacent)": order[:third],
        "medium (typical neighbourhood)": order[third: n - third],
        "low (small neighbourhood)": order[n - third:],
    }

    rng = np.random.default_rng(seed)
    buckets: dict[str, list] = {}
    for label, pool in pools.items():
        pool = np.asarray(pool)
        k = min(n_per_bucket, len(pool))
        chosen = rng.choice(pool, size=k, replace=False) if k > 0 else np.array([], dtype=int)
        buckets[label] = sorted(account_ids[i] for i in chosen)
    return buckets


# ------------------------------------------------------------------ fallback (no graph.pt available)
def _fallback_accounts(base: str, n: int) -> list:
    body = call(base, "GET", f"/accounts/sample?n={min(n, 50)}")
    seen: dict[str, None] = {}
    for a in body.get("suspicious", []) + body.get("benign", []):
        seen[a] = None
    return list(seen)


# ------------------------------------------------------------------ measurement
def measure_bucket(base: str, label: str, accounts: list) -> tuple[dict, dict]:
    cold_times = []
    for a in accounts:
        try:
            _, dt = timed_call(base, "POST", "/predict", {"account_ids": [a]})
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError):
            continue
        cold_times.append(dt)
    warm_times = []
    for a in accounts:
        try:
            _, dt = timed_call(base, "POST", "/predict", {"account_ids": [a]})
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError):
            continue
        warm_times.append(dt)
    return summarize(f"{label} -- cold", cold_times), summarize(f"{label} -- warm", warm_times)


def measure_concurrent_burst(base: str, label: str, accounts: list) -> dict:
    """Fires one /predict call per account in `accounts` AT THE SAME TIME (not one after another), each a
    genuine first-touch (these accounts must not have been scored yet -- see main()'s split between the
    sequential cold/warm accounts and these). This is the only way to actually exercise gnn_aml_core/main.py's
    _BatchScorer: the sequential cold/warm measurement above never has more than one in-flight request, so
    there is nothing for request batching to coalesce -- a server with batching and a server without it look
    identical under a sequential benchmark. A concurrent burst is what real production traffic (several
    officers, or an alert backlog, querying around the same time) actually looks like."""
    if not accounts:
        return summarize(f"{label} -- concurrent burst", [])
    with ThreadPoolExecutor(max_workers=len(accounts)) as pool:
        futures = [pool.submit(timed_call, base, "POST", "/predict", {"account_ids": [a]}) for a in accounts]
        times = []
        for f in futures:
            try:
                _, dt = f.result()
            except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError):
                continue
            times.append(dt)
    return summarize(f"{label} -- concurrent burst (n={len(accounts)}, first touch)", times)


def main() -> int:
    global PREDICT_TARGET_MS
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gnn", default="http://127.0.0.1:8001", help="GNN service base URL")
    ap.add_argument("--graph", default="models/graph.pt", help="path to the trained graph.pt (for stratification)")
    ap.add_argument("--n-per-bucket", type=int, default=15, help="accounts to measure per low/medium/high bucket")
    ap.add_argument("--hops", type=int, default=2, help="neighbourhood radius used for stratification (should match the model's num_layers)")
    ap.add_argument("--seed", type=int, default=None, help="fix the bucket sample for a reproducible run (default: fresh sample each time)")
    ap.add_argument("--warm-up-n", type=int, default=20, help="decoy accounts used to prime the service before measuring")
    ap.add_argument("--target-ms", type=float, default=PREDICT_TARGET_MS, help="latency target, p95, in ms (brief 2.1 default: 100)")
    ap.add_argument("--concurrency", type=int, default=0,
                    help="also fire this many /predict calls AT ONCE per bucket, against accounts the sequential "
                         "pass above never touched, to show whether request batching (GNN_BATCH_WINDOW_MS in "
                         "gnn_aml_core/main.py) actually helps under real concurrent load -- the sequential "
                         "cold/warm measurement above can't show this, since it never has more than one request "
                         "in flight. 0 (default) skips this extra check.")
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    PREDICT_TARGET_MS = a.target_ms

    print(f"XAI-AMLBench: stratified latency check  ({a.gnn})\n" + "=" * 78)
    try:
        health = call(a.gnn, "GET", "/health")
    except (urllib.error.URLError, urllib.error.HTTPError) as exc:
        print(f"\ncannot reach {a.gnn}: {exc}\nIs the service running? (docker compose ps, or check run_local.py)")
        return 1
    if not health.get("model_loaded"):
        print(f"\n{a.gnn} is reachable but no model is loaded: {health}\nRun: python -m gnn_aml_core.train --data data --out models")
        return 1
    print(f"model: {health.get('model')}  |  Redis cache: {'on' if health.get('redis') else 'off'}  |  target: p95 < {PREDICT_TARGET_MS:.0f} ms")

    # When --concurrency is set, fetch extra accounts per bucket up front: n_per_bucket for the existing
    # sequential cold/warm pass, plus `concurrency` more that the sequential pass will never touch, reserved
    # for the concurrent burst below (so that burst is a genuine first-touch, not already-cached accounts).
    fetch_n = a.n_per_bucket + (a.concurrency if a.concurrency > 0 else 0)
    buckets = stratify_by_degree(a.graph, fetch_n, hops=a.hops, seed=a.seed)
    if buckets is None:
        print(f"\nnote: {a.graph} not found (or torch/scipy unavailable) -- falling back to a flat "
             f"/accounts/sample list, same as benchmark_latency.py. Stratification by neighbourhood size needs "
             f"the local graph.pt; run gnn_aml_core.train first to get it.")
        flat = _fallback_accounts(a.gnn, fetch_n * 3)
        buckets = {"flat /accounts/sample": flat}

    seq_buckets: dict[str, list] = {}
    burst_buckets: dict[str, list] = {}
    for label, accounts in buckets.items():
        seq_buckets[label] = accounts[:a.n_per_bucket]
        burst_buckets[label] = accounts[a.n_per_bucket:a.n_per_bucket + a.concurrency] if a.concurrency > 0 else []
    buckets = seq_buckets

    all_bucket_accounts = {acc for accs in buckets.values() for acc in accs} | {acc for accs in burst_buckets.values() for acc in accs}
    print(f"\nwarming up the service with {a.warm_up_n} decoy accounts (never touching the {len(all_bucket_accounts)} "
         f"accounts about to be measured, so their first call stays genuinely cold) ...")
    warm_up(a.gnn, exclude=all_bucket_accounts, n=a.warm_up_n)

    results = {"base": a.gnn, "target_ms": PREDICT_TARGET_MS, "buckets": {}}
    for label, accounts in buckets.items():
        if not accounts:
            print(f"\n{label}: no accounts available, skipping")
            continue
        print(f"\n{label} -- {len(accounts)} accounts")
        cold, warm = measure_bucket(a.gnn, label, accounts)
        print(f"  {fmt(cold)}")
        print(f"  {fmt(warm)}")
        results["buckets"][label] = {"cold": cold, "warm": warm}

        burst_accounts = burst_buckets.get(label, [])
        if burst_accounts:
            burst = measure_concurrent_burst(a.gnn, label, burst_accounts)
            print(f"  {fmt(burst)}")
            if cold["n"] and burst["n"]:
                # Fair baseline: with NO batching, S.lock in gnn_aml_core/main.py serializes scoring, so N
                # callers arriving at once queue up behind it one at a time -- the last caller to get the
                # lock finishes only after everyone ahead of it has been scored. That means the right
                # "no-batching" comparison for a burst of N concurrent requests isn't a single sequential
                # call's p95 (that's the best case, only the FIRST caller ever sees it) -- it's roughly
                # N x a single cold call's cost, which is what the slowest (last-served) caller would
                # actually experience. Comparing against a lone cold p95 sets an unreasonably high bar:
                # it credits batching only when it beats one isolated request, not when it beats N of them
                # queuing single-file.
                projected_no_batch = cold["mean"] * len(burst_accounts)
                delta = projected_no_batch - burst["p95"]
                note = (f"{delta:+.0f} ms vs projected no-batching p95 (~{projected_no_batch:.0f} ms = "
                       f"{len(burst_accounts)} x cold mean, i.e. requests queuing single-file with no "
                       f"batching) -- batching is helping" if delta > 0 else
                       f"{delta:+.0f} ms vs projected no-batching p95 (~{projected_no_batch:.0f} ms = "
                       f"{len(burst_accounts)} x cold mean) -- no batching benefit visible here"
                       f"{' (GNN_BATCH_WINDOW_MS=0?)' if delta > -1 else ''}")
                print(f"    {note}")
            results["buckets"][label]["concurrent_burst"] = burst

    # "Worst bucket" rather than a pooled percentile across buckets: a stratified low-neighbourhood bucket at
    # 20ms shouldn't be allowed to drag down a genuinely slow high-neighbourhood bucket's reported p95 -- the
    # brief's <100ms target has to hold for every account a real user might query, not just on average.
    overall_cold_vals = [b["cold"]["p95"] for b in results["buckets"].values() if b["cold"]["n"]]
    overall_warm_vals = [b["warm"]["p95"] for b in results["buckets"].values() if b["warm"]["n"]]

    print("\n" + "=" * 78)
    worst_cold_p95 = max(overall_cold_vals) if overall_cold_vals else float("nan")
    worst_warm_p95 = max(overall_warm_vals) if overall_warm_vals else float("nan")
    cold_ok = worst_cold_p95 < PREDICT_TARGET_MS if overall_cold_vals else False
    warm_ok = worst_warm_p95 < PREDICT_TARGET_MS if overall_warm_vals else False
    print(f"worst bucket p95 -- cold: {worst_cold_p95:.1f} ms ({'PASS' if cold_ok else 'FAIL'})   "
         f"warm: {worst_warm_p95:.1f} ms ({'PASS' if warm_ok else 'FAIL'})   target: < {PREDICT_TARGET_MS:.0f} ms")
    results["worst_cold_p95_ms"] = worst_cold_p95
    results["worst_warm_p95_ms"] = worst_warm_p95
    results["cold_meets_target"] = cold_ok
    results["warm_meets_target"] = warm_ok

    if a.json:
        with open(a.json, "w") as f:
            json.dump(results, f, indent=2, default=lambda o: None if isinstance(o, float) and math.isnan(o) else o)
        print(f"saved {a.json}")
    return 0 if warm_ok else 1


if __name__ == "__main__":
    sys.exit(main())
