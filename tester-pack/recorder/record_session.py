#!/usr/bin/env python3
"""XAI-AMLBench test recorder.

You use the system normally in the browser this opens. While you do, the recorder measures it silently against
published standards (performance, reliability, accessibility, security, Kenyan data-protection law) and records
what you did. You never mark anything. At the end you press "Finish test" and write your overall opinion and
suggestions. One zip file is produced for you to email.
"""
import argparse, datetime, html, json, os, platform, re, subprocess, sys, time, urllib.request, zipfile
from pathlib import Path
from urllib.parse import urlparse

try:
    from playwright.sync_api import sync_playwright
except ImportError:
    sys.exit("Playwright is not installed. Run:  pip install playwright  then  playwright install chromium")

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from thresholds import RULES, CAPS, FIX, PROTECTED_PROBES

now = time.time
iso = lambda t: datetime.datetime.fromtimestamp(t).strftime("%Y-%m-%d %H:%M:%S")
AXE_URLS = ["https://cdn.jsdelivr.net/npm/axe-core@4.10.2/axe.min.js", "https://unpkg.com/axe-core@4.10.2/axe.min.js"]


def run(cmd, cwd=None):
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=cwd, timeout=60)
        return (r.stdout + r.stderr).strip()
    except Exception as e:
        return f"(could not run: {e})"


def pct(vals, q):
    v = sorted(vals)
    if not v: return None
    return v[min(len(v) - 1, max(0, int(round(q * len(v) + 0.4999)) - 1))]


def norm(path):
    p = re.sub(r"/accounts/[^/]+", "/accounts/:name", path)
    p = re.sub(r"/admin/banks/[^/]+", "/admin/banks/:name", p)
    p = re.sub(r"/public/banks/[^/]+", "/public/banks/:name", p)
    return re.sub(r"/\d+", "/:id", p)


def ensure_axe():
    f = HERE / "axe.min.js"
    if f.exists() and f.stat().st_size > 100000: return f
    for u in AXE_URLS:
        try:
            data = urllib.request.urlopen(u, timeout=30).read()
            if len(data) > 100000: f.write_bytes(data); return f
        except Exception:
            continue
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tester", required=True, help="your name, no spaces")
    ap.add_argument("--url", default="https://xai.localhost")
    ap.add_argument("--project", default=".", help="the project folder, used to record docker status")
    ap.add_argument("--out", default="test-results")
    ap.add_argument("--selftest", action="store_true", help=argparse.SUPPRESS)
    a = ap.parse_args()
    base = a.url.rstrip("/")
    out = Path(a.out) / a.tester
    for d in ("screenshots", "video"): (out / d).mkdir(parents=True, exist_ok=True)
    statef = out / "state.json"
    S = {"tester": a.tester, "url": base, "started": now(), "events": [], "api": [], "docs": [], "vitals": {}, "axe": {},
         "sittings": [], "opinion": {}, "shots": [], "pages": [], "cookies": []}
    if statef.exists():
        try: S.update(json.loads(statef.read_text()))
        except Exception: pass
    S["sittings"].append({"start": now()})
    TOK = {}               # session tokens, memory only, never written to disk
    queue, last_shot, seen_paths, axe_done = [], [0.0], set(), set(S.get("axe", {}).keys())
    fin = []

    def save(): statef.write_text(json.dumps(S))
    def ev(**k): k["t"] = now(); S["events"].append(k)

    axe_src = None
    if os.environ.get("XAI_AXE_STUB"): axe_src = "window.axe={run:()=>Promise.resolve({violations:[{id:'label',impact:'serious',help:'Form elements must have labels',nodes:[1,2]}]})}"
    else:
        f = ensure_axe()
        axe_src = f.read_text(encoding="utf8") if f else None
        if not f: print("Note: the accessibility checker could not be downloaded, so accessibility will be reported as not measured.")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=a.selftest)
        ctx = browser.new_context(viewport={"width": 1366, "height": 800}, record_video_dir=str(out / "video"),
                                  record_video_size={"width": 1280, "height": 720}, ignore_https_errors=True)
        def f_api(o):
            o["t"] = now(); o["p"] = norm(o["p"])
            tok = o.pop("token", None)
            if tok: TOK[f"{o.get('bank') or o.get('role') or 'user'}|{len(TOK)}"] = {"token": tok, "bank": o.get("bank"), "role": o.get("role")}
            S["api"].append(o)
            if o["s"] >= 400 or o["s"] == 0: queue.append(("shot", f"api-{o['s']}"))
        ctx.expose_function("xai_api", f_api)
        ctx.expose_function("xai_event", lambda o: ev(**o))
        def f_fin(o): S["opinion"] = o; save(); fin.append(1)
        ctx.expose_function("xai_finish", f_fin)
        for js in ("vitals.js", "events.js", "badge.js"): ctx.add_init_script(path=str(HERE / js))

        def hook(page):
            def nav(fr):
                if fr == page.main_frame:
                    ev(type="page", url=fr.url); queue.append(("page", fr.url))
            page.on("framenavigated", nav)
            def on_resp(r):
                try:
                    if r.request.resource_type == "document" and r.request.frame.parent_frame is None:
                        t = r.request.timing
                        S["docs"].append({"t": now(), "url": r.url.split("?")[0], "status": r.status,
                                          "headers": {k.lower(): v for k, v in r.headers.items() if k.lower() in
                                          ("x-content-type-options", "content-security-policy", "x-frame-options", "referrer-policy", "strict-transport-security")},
                                          "ttfb": round(t.get("responseStart", -1)) if t else None})
                except Exception: pass
            page.on("response", on_resp)
            page.on("console", lambda m: ev(type="console", text=m.text[:200]) if m.type == "error" else None)
            page.on("pageerror", lambda e: ev(type="pageerror", text=str(e)[:200]))
        ctx.on("page", hook)
        page = ctx.new_page()
        try: page.goto(base, timeout=45000)
        except Exception as e:
            page.set_content(f"<body style='font-family:Times New Roman;padding:2em'><h2>Could not open {html.escape(base)}</h2><p>Start the project first, wait until it opens in a normal browser, then run the recorder again.</p>")
        print("\nThe test browser is open. Use the system normally. The recorder measures it silently.")
        print("Press 'Finish test' (bottom left of the page) when you are done, or press Ctrl+C here.\n")

        shot_n = [len(list((out / "screenshots").glob("*.png")))]
        def shoot(label):
            if now() - last_shot[0] < 1.2: return
            try:
                pg = ctx.pages[-1]
                pg.evaluate("document.getElementById('xai-root')&&(document.getElementById('xai-root').style.visibility='hidden')")
                shot_n[0] += 1; name = f"{shot_n[0]:03d}-{label}.png"
                pg.screenshot(path=str(out / "screenshots" / name))
                pg.evaluate("document.getElementById('xai-root')&&(document.getElementById('xai-root').style.visibility='visible')")
                last_shot[0] = now(); S["shots"].append({"file": name, "url": pg.url, "label": label})
            except Exception: pass

        def poll_vitals(pg):
            try:
                v = pg.evaluate("window.__xaiV")
                if v: S["vitals"][urlparse(pg.url).path or "/"] = v
            except Exception: pass

        def run_axe(pg, path):
            if axe_src is None or path in axe_done or len(axe_done) >= 30: return
            axe_done.add(path)
            try:
                pg.evaluate(axe_src)
                res = pg.evaluate("axe.run({exclude:[['#xai-root']]},{runOnly:{type:'tag',values:['wcag2a','wcag2aa','wcag21a','wcag21aa','wcag22aa']}}).then(r=>r.violations.map(v=>({id:v.id,impact:v.impact,help:v.help,nodes:v.nodes.length})))")
                S["axe"][path] = res
            except Exception as e:
                S["axe"][path] = [{"id": "checker-error", "impact": "unknown", "help": str(e)[:100], "nodes": 0}]

        script = None
        if a.selftest:
            script = os.environ.get("XAI_SELFTEST_STEPS", "").split(",")
        quiet = now(); tick = 0
        try:
            while not fin:
                try: ctx.pages[-1].wait_for_timeout(250)
                except Exception: break
                if not ctx.pages: break
                tick += 1
                while queue:
                    kind, val = queue.pop(0)
                    if kind == "shot": time.sleep(0.4); shoot(val)
                    elif kind == "page":
                        path = urlparse(val).path or "/"
                        if path not in seen_paths:
                            seen_paths.add(path); S["pages"].append(path)
                            time.sleep(1.5); shoot("page"); run_axe(ctx.pages[-1], path)
                if tick % 8 == 0: poll_vitals(ctx.pages[-1])
                if tick % 40 == 0: save()
                if script and now() - quiet > 1.0:
                    quiet = now(); step = script.pop(0) if script else None
                    if step:
                        pg = ctx.pages[-1]
                        if step.startswith("go:"): pg.goto(base + step[3:])
                        elif step.startswith("click:"): pg.get_by_role("button", name=step[6:]).click()
                        elif step == "finish":
                            pg.get_by_role("button", name="Finish test").click()
                            pg.locator("#op").fill("self test opinion"); pg.locator("#sg").fill("self test suggestion")
                            pg.locator("#score").select_option("4"); pg.locator("#help").select_option("Yes")
                            pg.get_by_role("button", name="Save and finish").click()
        except KeyboardInterrupt:
            print("\nStopped with Ctrl+C. Building your report...")
        poll_vitals(ctx.pages[-1]) if ctx.pages else None
        try: S["cookies"] = ctx.cookies()
        except Exception: pass
        S["sittings"][-1]["end"] = now(); save()
        # active probes with the tester's own sessions and with none
        probes = run_probes(p, base, TOK, S)
        try: ctx.close(); browser.close()
        except Exception: pass
    package(S, out, a, probes)


def run_probes(p, base, TOK, S):
    pr = {"unauth": {}, "badtoken": {}, "cors": None, "iso": None, "audit": None, "error": None}
    try:
        rc = p.request.new_context(ignore_https_errors=True, base_url=base)
        for path in PROTECTED_PROBES:
            pr["unauth"][path] = rc.get(path, timeout=15000).status
            pr["badtoken"][path] = rc.get(path, headers={"Authorization": "Bearer invalid.token.value"}, timeout=15000).status
        r = rc.get("/api/public/status", headers={"Origin": "https://evil.example"}, timeout=15000)
        h = {k.lower(): v for k, v in r.headers.items()}
        pr["cors"] = {"acao": h.get("access-control-allow-origin"), "creds": h.get("access-control-allow-credentials")}
        banks = {}
        for v in TOK.values():
            if v["bank"] and v["role"] != "admin": banks[v["bank"]] = v["token"]
        if len(banks) >= 2:
            names = list(banks)[:2]; data = {}
            for n in names:
                hd = {"Authorization": f"Bearer {banks[n]}"}
                al = rc.get("/api/alerts?limit=500", headers=hd, timeout=20000)
                an = rc.get("/api/anomalies?limit=500", headers=hd, timeout=20000)
                data[n] = {"alerts": al.json() if al.ok else [], "anoms": an.json() if an.ok else []}
            a_, b_ = names; leak = []
            ids = lambda n, k: {str(x.get("id")) for x in data[n][k] if isinstance(x, dict)}
            for k in ("alerts", "anoms"):
                if ids(a_, k) & ids(b_, k): leak.append(f"same {k} ids visible to both banks")
            for x, y in ((a_, b_), (b_, a_)):
                for al in [t for t in data[x]["alerts"] if isinstance(t, dict) and t.get("account_id")][:3]:
                    rr = rc.get(f"/api/accounts/{al['account_id']}/explain", headers={"Authorization": f"Bearer {banks[y]}"}, timeout=30000)
                    if rr.status == 200: leak.append(f"{y} could open {x}'s account")
            pr["iso"] = {"banks": names, "leaks": leak}
        for v in TOK.values():
            if v["role"] != "admin":
                rr = rc.get("/api/activity?limit=50", headers={"Authorization": f"Bearer {v['token']}"}, timeout=20000)
                if rr.ok:
                    j = rr.json(); j = j if isinstance(j, list) else (j.get("items") if isinstance(j, dict) else [])
                    pr["audit"] = {"entries": len(j or [])}; break
    except Exception as e:
        pr["error"] = str(e)[:200]
    return pr


def verdicts(S, pr):
    R = {}
    def put(i, status, measured, detail=""): R[i] = {"status": status, "measured": measured, "detail": detail}
    V = S["vitals"]; col = lambda k: [v[k] for v in V.values() if v.get(k) is not None]
    for i, k, lim, unit in (("P1", "ttfb", 800, "ms"), ("P2", "lcp", 2500, "ms"), ("P3", "inp", 200, "ms"), ("P4", "cls", 0.1, "")):
        vals = col(k)
        if not vals: put(i, "Not measured", "no page data", "No pages produced this measurement.")
        else:
            x = pct(vals, .75); put(i, "Pass" if x <= lim else "Fail", f"{round(x,3)} {unit} (75th percentile, {len(vals)} pages)")
    api = S["api"]; ms = [e["ms"] for e in api if e["s"] > 0]
    if ms:
        x = pct(ms, .95); put("P5", "Pass" if x <= 1000 else "Fail", f"{x} ms (95th percentile, {len(ms)} calls)",
            "; ".join(sorted({f"{e['m']} {e['p']} {e['ms']} ms" for e in api if e['ms'] > 1000})[:6]))
    else: put("P5", "Not measured", "no server calls")
    lat = [e["latency"] for e in api if e.get("latency") is not None]
    if lat: x = pct(lat, .95); put("P6", "Pass" if x <= 100 else "Fail", f"{round(x,1)} ms (95th percentile, {len(lat)} scans)")
    else: put("P6", "Not exercised", "no account was scanned")
    e5 = [e for e in api if e["s"] >= 500]
    if api: put("P7", "Pass" if not e5 else "Fail", f"{len(e5)} server errors in {len(api)} calls", "; ".join(sorted({f"{e['m']} {e['p']} -> {e['s']}" for e in e5})[:8]))
    else: put("P7", "Not measured", "no server calls")
    if not S["axe"]: put("A1", "Not measured", "checker not available")
    else:
        bad = {pth: [v for v in vs if v["impact"] in ("critical", "serious")] for pth, vs in S["axe"].items()}
        bad = {k: v for k, v in bad.items() if v}
        put("A1", "Fail" if bad else "Pass", f"{len(bad)} of {len(S['axe'])} pages have critical or serious issues",
            "; ".join(f"{k}: " + ", ".join(f"{v['id']} ({v['nodes']})" for v in vs) for k, vs in list(bad.items())[:6]))
    for i, key in (("S1", "unauth"), ("S2", "badtoken")):
        d = pr[key]
        if not d: put(i, "Not measured", "probe could not run", pr.get("error") or "")
        else:
            bad = [f"{k} -> {v}" for k, v in d.items() if v not in (401, 403, 404)]
            put(i, "Fail" if bad else "Pass", f"{len(d)-len(bad)} of {len(d)} endpoints refused", "; ".join(bad))
    docs = S["docs"]
    if not docs: put("S3", "Not measured", "no page responses")
    else:
        miss = set()
        for d in docs:
            h = d["headers"]
            if h.get("x-content-type-options", "").lower() != "nosniff": miss.add("X-Content-Type-Options")
            if "content-security-policy" not in h: miss.add("Content-Security-Policy")
            if "x-frame-options" not in h and "frame-ancestors" not in h.get("content-security-policy", ""): miss.add("framing protection")
            if "referrer-policy" not in h: miss.add("Referrer-Policy")
        put("S3", "Fail" if miss else "Pass", f"{len(miss)} missing" if miss else "all present", ", ".join(sorted(miss)))
    host = urlparse(S["url"]).hostname or ""; local = host.endswith("localhost") or host in ("127.0.0.1", "::1")
    https = S["url"].startswith("https")
    if not https: put("S4", "N/A" if local else "Fail", "plain HTTP", "Local test over HTTP; required on the public domain." if local else "Public site is not HTTPS.")
    else:
        hs = [d["headers"].get("strict-transport-security", "") for d in docs]
        ok = bool(hs) and all(re.search(r"max-age=(\d+)", h) and int(re.search(r"max-age=(\d+)", h).group(1)) >= 31536000 for h in hs)
        if local and not ok: put("S4", "N/A", "local test address", "HSTS is checked on the public domain in the pilot.")
        else: put("S4", "Pass" if ok else "Fail", "HSTS present" if ok else "HSTS missing or under one year")
    c = pr["cors"]
    if c is None: put("S5", "Not measured", "probe could not run")
    else:
        bad = c["acao"] == "https://evil.example" and str(c["creds"]).lower() == "true"
        put("S5", "Fail" if bad else "Pass", f"Allow-Origin: {c['acao'] or 'none'}")
    tr = [e for e in api if e.get("trace")]
    put("S6", "Fail" if tr else ("Pass" if api else "Not measured"), f"{len(tr)} responses with internal details" if api else "no server calls", "; ".join(sorted({f"{e['m']} {e['p']}" for e in tr})[:6]))
    ck = S["cookies"]
    if not ck: put("S7", "N/A", "no cookies were set")
    else:
        bad = [c["name"] for c in ck if not c.get("httpOnly") or c.get("sameSite") in (None, "None") or (https and not c.get("secure"))]
        put("S7", "Fail" if bad else "Pass", f"{len(ck)-len(bad)} of {len(ck)} cookies protected", ", ".join(bad))
    iso = pr["iso"]
    if not iso: put("S8", "Not exercised", "needs two different banks signed in during the test")
    else: put("S8", "Fail" if iso["leaks"] else "Pass", f"{iso['banks'][0]} and {iso['banks'][1]} checked", "; ".join(iso["leaks"]))
    fails = [e for e in api if e["p"] in ("/api/login", "/api/auth/login", "/api/admin/login") and e["s"] in (400, 401, 403, 429)]
    if sum(1 for e in fails if e["s"] != 429) >= 5: put("S9", "Pass" if any(e["s"] == 429 for e in fails) else "Fail", f"{len(fails)} failed sign-ins")
    else: put("S9", "Not exercised", "fewer than five failed sign-ins were made")
    leaks = sorted({f"{e['m']} {e['p']}: {k}" for e in api for k in e["found"] if (k in ("password", "password_hash", "hashed_password", "totp_secret", "secret") or (k in ("passkey", "invite_token") and not e["p"].startswith("/api/admin/")))})
    put("S10", "Fail" if leaks else ("Pass" if api else "Not measured"), f"{len(leaks)} found" if api else "no server calls", "; ".join(leaks[:6]))
    ids = sorted({f"{e['m']} {e['p']}" for e in api if "id_number" in e["found"]})
    put("K1", "Fail" if ids else ("Pass" if api else "Not measured"), f"{len(ids)} responses with a raw ID field" if api else "no server calls", "; ".join(ids[:6]))
    if https: put("K2", "Pass", "HTTPS in use")
    else: put("K2", "N/A" if local else "Fail", "plain HTTP", "Local test over HTTP; HTTPS is required on the public domain.")
    wr = [e for e in api if e["m"] in ("POST", "PATCH", "DELETE") and 200 <= e["s"] < 300 and not e["p"].startswith("/api/auth") and "login" not in e["p"]]
    au = pr["audit"]
    if not wr: put("K3", "Not exercised", "no write action was made")
    elif au is None: put("K3", "Not measured", "no tenant session to read the log")
    else: put("K3", "Pass" if au["entries"] > 0 else "Fail", f"{au['entries']} log entries after {len(wr)} write actions")
    return R


def capabilities(S):
    out = []; api = S["api"]
    banks = {e.get("bank") for e in api if e["p"] in ("/api/login", "/api/auth/login") and 200 <= e["s"] < 300 and e.get("bank")}
    for cid, title, specs in CAPS:
        if specs[0][0] == "BANKS":
            out.append({"id": cid, "title": title, "status": "Pass" if len(banks) >= 2 else "Not exercised", "calls": len(banks), "detail": ""}); continue
        m = specs[0][0]; mode = specs[0][2] if len(specs[0]) > 2 else ""
        pat = specs[0][1]
        calls = [e for e in api if e["m"] == m and re.match(pat, e["p"])]
        st = sorted({e["s"] for e in calls})
        if mode == "reject": ok = any(e["s"] in (400, 422) for e in calls)
        elif mode == "x3": ok = sum(1 for e in calls if 200 <= e["s"] < 300) >= 3
        else: ok = any(200 <= e["s"] < 300 for e in calls)
        if not calls: s = "Not exercised"
        elif any(e["s"] >= 500 for e in calls): s = "Fail"
        elif ok: s = "Pass"
        else: s = "Needs review"
        out.append({"id": cid, "title": title, "status": s, "calls": len(calls), "detail": "statuses " + ", ".join(map(str, st)) if calls else ""})
    return out


def package(S, out, a, pr):
    R = verdicts(S, pr); caps = capabilities(S)
    sysinfo = [f"Tester: {S['tester']}", f"Address tested: {S['url']}", f"First started: {iso(S['started'])}", f"Computer: {platform.platform()}",
               f"Python: {sys.version.split()[0]}", "Docker: " + run("docker --version"), "Docker Compose: " + run("docker compose version"),
               "Project version: " + run("git describe --tags --always", cwd=a.project), "Project commit: " + run("git rev-parse --short HEAD", cwd=a.project),
               "", "docker compose ps:", run("docker compose ps", cwd=a.project)]
    (out / "system.txt").write_text("\n".join(sysinfo))
    active = sum(max(0, s.get("end", now()) - s["start"]) for s in S["sittings"])
    rules = []
    for rid, cat, title, thr, src in RULES:
        r = R[rid]; rules.append({"id": rid, "category": cat, "title": title, "threshold": thr, "source": src, **r,
                                  "suggestion": FIX.get(rid, "") if r["status"] == "Fail" else ""})
    cnt = lambda L, k: sum(1 for x in L if x["status"] == k)
    res = {"tester": S["tester"], "url": S["url"], "started": iso(S["started"]), "active_minutes": round(active / 60, 1),
           "opinion": S["opinion"], "rules": rules, "capabilities": caps, "pages": S["pages"],
           "counts": {k: cnt(rules, k) for k in ("Pass", "Fail", "Not measured", "Not exercised", "N/A")},
           "cap_counts": {k: cnt(caps, k) for k in ("Pass", "Fail", "Needs review", "Not exercised")},
           "measurements": {"api_ms": [e["ms"] for e in S["api"] if e["s"] > 0][:3000], "scan_ms": [e["latency"] for e in S["api"] if e.get("latency") is not None][:1000],
                            "n_api": len(S["api"]), "n_5xx": sum(1 for e in S["api"] if e["s"] >= 500), "vitals": S["vitals"]},
           "system": "\n".join(sysinfo)}
    (out / "results.json").write_text(json.dumps(res, indent=1))
    c = res["counts"]; o = S["opinion"] or {}
    md = [f"# XAI-AMLBench Test Report: {S['tester']}", "", f"Address tested: {S['url']}  ", f"Started: {iso(S['started'])}  ", f"Active testing time: {res['active_minutes']} minutes  ",
          f"Pages visited: {len(S['pages'])}  ", f"Server calls measured: {len(S['api'])}", "", "## Summary of results",
          f"Technical checks: {c['Pass']} passed, {c['Fail']} failed, {c['Not exercised']} not exercised, {c['Not measured']} not measured, {c['N/A']} not applicable on a local test.  ",
          f"Features used: {res['cap_counts']['Pass']} worked, {res['cap_counts']['Fail']} failed, {res['cap_counts']['Needs review']} need review, {res['cap_counts']['Not exercised']} not used.", "",
          "## Technical results (measured automatically against published standards)", "", "| ID | Check | Standard and source | Result | Measured | Detail |", "|---|---|---|---|---|---|"]
    for r in rules: md.append(f"| {r['id']} | {r['title']} | {r['threshold']}. {r['source']} | {r['status']} | {r['measured']} | {r['detail'].replace('|','/')} |")
    md += ["", "## Features used and their outcome", "", "| ID | Feature | Result | Calls | Detail |", "|---|---|---|---|---|"]
    for x in caps: md.append(f"| {x['id']} | {x['title']} | {x['status']} | {x['calls']} | {x['detail']} |")
    md += ["", "## Suggestions for improvement from the measurements"]
    sug = [f"- {r['id']} {r['title']}: {r['suggestion']}" for r in rules if r["suggestion"]]
    md += sug or ["- No measured check failed."]
    md += ["", "## Tester's overall overview (opinion, kept separate from the measurements)", f"Setup without help: {o.get('help') or 'not answered'}  ",
           f"Overall impression (1 to 5): {o.get('score') or 'not given'}  ", "", f"Opinion: {o.get('opinion') or '(none written)'}", "", f"Suggestions: {o.get('suggestions') or '(none written)'}"]
    (out / "REPORT.md").write_text("\n".join(md))
    h = ["<html><head><meta charset='utf-8'><style>body{font-family:'Times New Roman',serif;margin:2em;color:#000}table{border-collapse:collapse;width:100%;margin-bottom:1.5em}td,th{border:1px solid #000;padding:4px;font-size:13px;vertical-align:top}</style></head><body>",
         f"<h1>XAI-AMLBench test report: {html.escape(S['tester'])}</h1><p>Started {iso(S['started'])}. Active time {res['active_minutes']} minutes. {len(S['api'])} server calls measured.</p>",
         f"<h2>Summary</h2><p>Technical checks: {c['Pass']} passed, {c['Fail']} failed, {c['Not exercised']} not exercised, {c['Not measured']} not measured, {c['N/A']} not applicable.</p>",
         "<h2>Technical results</h2><table><tr><th>ID</th><th>Check</th><th>Standard and source</th><th>Result</th><th>Measured</th><th>Detail</th></tr>"]
    for r in rules: h.append(f"<tr><td>{r['id']}</td><td>{html.escape(r['title'])}</td><td>{html.escape(r['threshold'])}.<br><i>{html.escape(r['source'])}</i></td><td><b>{r['status']}</b></td><td>{html.escape(str(r['measured']))}</td><td>{html.escape(r['detail'])}</td></tr>")
    h.append("</table><h2>Features used</h2><table><tr><th>ID</th><th>Feature</th><th>Result</th><th>Calls</th><th>Detail</th></tr>")
    for x in caps: h.append(f"<tr><td>{x['id']}</td><td>{html.escape(x['title'])}</td><td>{x['status']}</td><td>{x['calls']}</td><td>{html.escape(x['detail'])}</td></tr>")
    h.append("</table><h2>Suggestions for improvement</h2><ul>" + "".join(f"<li>{html.escape(s[2:])}</li>" for s in sug or ["- No measured check failed."]) + "</ul>")
    h.append(f"<h2>Tester's overall overview (opinion)</h2><p>Setup without help: {html.escape(o.get('help') or 'not answered')}. Overall impression: {html.escape(str(o.get('score') or 'not given'))}.</p><p>{html.escape(o.get('opinion') or '')}</p><p><b>Suggestions:</b> {html.escape(o.get('suggestions') or '')}</p></body></html>")
    (out / "report.html").write_text("\n".join(h))
    zname = Path(a.out) / f"xai-test-{S['tester']}.zip"
    with zipfile.ZipFile(zname, "w", zipfile.ZIP_DEFLATED) as z:
        for f in out.rglob("*"):
            if f.is_file() and f.name != "state.json": z.write(f, f.relative_to(out.parent))
    print(f"\nDone. Technical checks: {c['Pass']} passed, {c['Fail']} failed, {c['Not exercised']} not exercised.")
    print(f"Email this file, with your report, to the project owner:\n  {zname.resolve()}\n  (your readable report is {(out/'REPORT.md').resolve()})\n")


if __name__ == "__main__":
    main()
