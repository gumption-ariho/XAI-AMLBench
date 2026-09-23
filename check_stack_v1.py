#!/usr/bin/env python3
"""
check_stack_v1.py  -  is everything connected?  Tests the whole chain through the gateway, exactly like the browser does.

    python3 check_stack_v1.py                          # http://localhost
    python3 check_stack_v1.py --base http://localhost --json stack_report.json
    python3 check_stack_v1.py --base http://localhost:3000 --gnn http://127.0.0.1:8001 --xai http://127.0.0.1:8002   # services run by run_local.py

The chain:   browser -> Traefik(:80) -> frontend (/)  and  backend (/api) -> GNN service (/gnn) + narrative service (/xai) -> Postgres

What it does to your data: it scans ONE suspicious sample account (that creates one alert), generates its narrative and marks the
alert "dismissed" by a reviewer called "check_stack". Nothing else is written. Standard library only.
Start the stack first:   docker compose --profile app --profile ml up -d --build
"""
import argparse
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

RESULTS = []          # (status, name, detail, hint)


def call(base, method, path, body=None, timeout=30):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(base.rstrip("/") + path, data=data, method=method, headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read()
            code = r.status
    except urllib.error.HTTPError as e:
        raw, code = e.read(), e.code
    dt = (time.perf_counter() - t0) * 1000
    text = raw.decode("utf-8", "replace")
    try:
        return code, json.loads(text), dt
    except ValueError:
        return code, text, dt


def step(name, fn, hint=""):
    """Run one check; fn returns (ok, detail). Any exception becomes a FAIL with the reason."""
    try:
        ok, detail = fn()
    except urllib.error.URLError as e:
        ok, detail = False, f"cannot connect: {getattr(e, 'reason', e)}"
    except Exception as e:  # noqa: BLE001
        ok, detail = False, f"{e.__class__.__name__}: {e}"
    RESULTS.append(("PASS" if ok else "FAIL", name, str(detail), "" if ok else hint))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"   [{detail}]" if ok and detail else ""))
    if not ok:
        print(f"          {detail}")
        if hint:
            print(f"          -> {hint}")
    return ok


def upstream_hint(code, body):
    if code in (502, 503, 504):
        return "the gateway or backend cannot reach the next service (is it running?  docker compose ps)"
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost")
    ap.add_argument("--json", default="")
    ap.add_argument("--traefik", default="", help="gateway dashboard/API URL (default: same host, port 8081)")
    ap.add_argument("--gnn", default="", help="base URL of the model service when it runs outside Docker, e.g. http://127.0.0.1:8001")
    ap.add_argument("--xai", default="", help="base URL of the narrative service when it runs outside Docker, e.g. http://127.0.0.1:8002")
    a = ap.parse_args()
    B = a.base
    state = {}
    print(f"XAI-AMLBench: stack check  ({B})\n" + "=" * 70)

    u = urllib.parse.urlparse(B)
    if u.port in (None, 80) or a.traefik:
        print("\n0  Gateway routes (does Traefik see your containers?)")
        tbase = a.traefik or f"{u.scheme}://{u.hostname}:8081"

        def routes():
            code, body, _ = call(tbase, "GET", "/api/http/routers")
            if code != 200 or not isinstance(body, list):
                return False, f"HTTP {code} from {tbase}/api/http/routers"
            names = [r.get("name", "").split("@")[0] for r in body if r.get("name", "").endswith("@docker")]
            want = {"frontend": "website", "backend": "backend API", "gnn": "model service"}
            missing = [f"{k} ({v})" for k, v in want.items() if k not in names]
            if not any(n.startswith("xai") for n in names):
                missing.append("xai (narrative service)")
            return not missing, (f"routes: {', '.join(sorted(names))}" if not missing else f"the gateway knows {len(names)} application route(s); missing: {', '.join(missing)}")
        step("gateway has a route for every service", routes,
             "the containers run but Traefik does not see them. With Docker Engine 29+ an old Traefik cannot talk to Docker: docker compose logs traefik-edge-router --tail 20 "
             "(look for 'too old'). Fix: use a newer Traefik image (image: traefik:v3 in docker-compose.yml), docker compose pull traefik-edge-router, docker compose up -d traefik-edge-router. Also check that PUBLIC_HOST in .env matches the address you open.")

    print("\n1  Gateway and website")
    def front():
        code, body, dt = call(B, "GET", "/")
        return code == 200 and "XAI" in str(body), f"HTTP {code}, {dt:.0f} ms" if code == 200 else f"HTTP {code}: {str(body)[:120]}"
    step("frontend answers at /", front, "docker compose --profile app up -d --build   (the frontend builds with `next build`, which takes a few minutes)")

    print("\n2  Backend (ProjectXY API -> Postgres)")
    def health():
        code, body, dt = call(B, "GET", "/api/health")
        return code == 200 and isinstance(body, dict) and body.get("status") == "ok", f"{body}" if code != 200 else f"database reachable, {dt:.0f} ms"
    step("GET /api/health", health, "backend or Postgres not up: docker compose ps ; docker compose logs backend --tail 30")

    print("\n3  GNN service (your trained model)")
    def gnn_health():
        code, body, _ = call(a.gnn, "GET", "/health") if a.gnn else call(B, "GET", "/gnn/health")
        state["gnn_health"] = body
        return code == 200 and isinstance(body, dict) and bool(body.get("model_loaded")), body if code != 200 or not (isinstance(body, dict) and body.get("model_loaded")) else f"model {body.get('model')}, redis cache {'on' if body.get('redis') else 'off'}"
    step("GET /gnn/health: model loaded", gnn_health, "no model in ./models? run: python -m gnn_aml_core.train --data data --out models ; then docker compose restart gnn-detection-api")

    def model_info():
        code, body, dt = call(B, "GET", "/api/model/info")
        ok = code == 200 and isinstance(body, dict) and "metrics" in body
        return ok, (f"{body.get('model')} AUC {body['metrics']['test_auc_roc']:.3f}, threshold {body['threshold']:.2f}" if ok else f"HTTP {code}: {str(body)[:160]}")
    step("backend -> GNN: GET /api/model/info", model_info, upstream_hint(0, "") or "the backend cannot reach gnn-detection-api (profile ml): docker compose --profile ml up -d")

    def sample():
        code, body, _ = call(B, "GET", "/api/accounts/sample")
        ok = code == 200 and isinstance(body, dict) and body.get("suspicious")
        if ok:
            state["acct"] = body["suspicious"][0]
        return ok, f"e.g. {state.get('acct')}" if ok else f"HTTP {code}: {str(body)[:160]}"
    step("backend -> GNN: sample accounts", sample, "same cause as above")

    print("\n4  Scoring, explanation and narrative (the real workflow)")
    def scan():
        code, body, dt = call(B, "POST", "/api/alerts/scan", {"account_id": state["acct"]})
        ok = code == 200 and isinstance(body, dict) and body.get("flagged") and body.get("alert")
        if ok:
            state["alert"] = body["alert"]["id"]
        return ok, (f"score {body['score']:.3f} (calibrated {body.get('calibrated_score', body['score']):.3f}), threshold {body['threshold']:.2f}, "
                    f"model {body['latency_ms']} ms, round trip {dt:.0f} ms, alert #{body['alert']['id']}" if ok
                    else f"HTTP {code}: {str(body)[:200]}")
    if "acct" in state:
        step("POST /api/alerts/scan flags a suspicious account and creates an alert", scan, "check docker compose logs gnn-detection-api --tail 30")

    def explain():
        code, body, dt = call(B, "GET", f"/api/accounts/{state['acct']}/explain", timeout=180)
        ok = code == 200 and isinstance(body, dict) and body.get("edges") and body.get("nodes")
        return ok, (f"{len(body['nodes'])} accounts, {len(body['edges'])} transactions, top feature {body['top_features'][0]['feature'] if body.get('top_features') else 'n/a'}, {dt/1000:.1f} s" if ok
                    else f"HTTP {code}: {str(body)[:200]}")
    if "acct" in state:
        step("GNNExplainer returns a network (Network tab)", explain, "explanations need a GATv2 model; see docker compose logs gnn-detection-api --tail 40")

    def xai_health():
        code, body, _ = call(a.xai, "GET", "/health") if a.xai else call(B, "GET", "/xai/health")
        return code == 200 and isinstance(body, dict) and body.get("status") == "ok", f"backend '{body.get('backend')}'" if isinstance(body, dict) and code == 200 else f"HTTP {code}: {str(body)[:120]}"
    step("GET /xai/health: narrative service up", xai_health, "no narrative service. CPU-only machine: docker compose --profile app up -d --build (includes xai-narrative-lite); GPU machine: --profile llm")

    def narrative():
        code, body, dt = call(B, "POST", f"/api/alerts/{state['alert']}/narrative", timeout=240)
        ok = code == 200 and isinstance(body, dict) and body.get("narrative")
        if ok:
            state["narr"] = body["narrative"]
        return ok, (f"{len(body['narrative'].split())} words, source '{body['narrative_source']}', typology '{body['typology']}', {dt/1000:.1f} s" if ok else f"HTTP {code}: {str(body)[:200]}")
    if "alert" in state:
        step("POST /api/alerts/{id}/narrative: SAR narrative generated and stored", narrative, "backend -> narrative service failed: docker compose logs backend --tail 30 ; docker compose logs xai-narrative-lite --tail 30")

    def listing():
        code, body, _ = call(B, "GET", "/api/alerts")
        mine = [x for x in body if x.get("id") == state["alert"]] if isinstance(body, list) else []
        if not mine:
            return False, f"HTTP {code}: alert #{state['alert']} not found in the list"
        if not mine[0].get("narrative"):
            return False, f"alert #{state['alert']} exists but has no narrative saved"
        return True, f"{len(body)} alerts, ours has its narrative"
    if "alert" in state:
        step("GET /api/alerts returns the alert with its narrative (Alerts tab)", listing, "alert or narrative not saved in Postgres")

    def decide():
        code, body, dt = call(B, "POST", f"/api/alerts/{state['alert']}/decision", {"officer": "check_stack", "decision": "dismissed", "note": "automated stack check"})
        return code == 200 and isinstance(body, dict) and body.get("status") == "dismissed", f"status {body.get('status')}, written to the audit ledger" if code == 200 else f"HTTP {code}: {str(body)[:160]}"
    if "alert" in state:
        step("POST /api/alerts/{id}/decision: officer decision recorded", decide, "docker compose logs backend --tail 30 (immudb audit failures are logged there)")

    fails = sum(1 for r in RESULTS if r[0] == "FAIL")
    print("\n" + "=" * 70)
    print(f"  {len(RESULTS) - fails}/{len(RESULTS)} checks passed" + ("   -> everything is connected" if not fails else f"   -> {fails} broken link(s), see the hints above"))
    if a.json:
        with open(a.json, "w") as f:
            json.dump([dict(status=s, name=n, detail=d, hint=h) for s, n, d, h in RESULTS], f, indent=2)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
