"use client";

import { useCallback, useEffect, useState } from "react";

async function api(path, options) {
  const res = await fetch(`/api${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  const text = await res.text();
  let data;
  try { data = JSON.parse(text); } catch { data = { detail: text }; }
  if (!res.ok) throw new Error(typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail));
  return data;
}

export default function Home() {
  const [accountId, setAccountId] = useState("");
  const [samples, setSamples] = useState(null);
  const [scan, setScan] = useState(null);
  const [alerts, setAlerts] = useState([]);
  const [officer, setOfficer] = useState("");
  const [busy, setBusy] = useState("");
  const [error, setError] = useState("");

  const load = useCallback(async () => {
    try { setAlerts(await api("/alerts")); } catch (e) { setError(e.message); }
  }, []);

  useEffect(() => {
    load();
    api("/accounts/sample").then(setSamples).catch(() => {});
  }, [load]);

  const run = async (key, fn) => {
    setBusy(key); setError("");
    try { await fn(); await load(); } catch (e) { setError(e.message); } finally { setBusy(""); }
  };

  const doScan = () => run("scan", async () => {
    setScan(await api("/alerts/scan", { method: "POST", body: JSON.stringify({ account_id: accountId.trim() }) }));
  });
  const doNarrative = (id) => run(`n${id}`, () => api(`/alerts/${id}/narrative`, { method: "POST" }));
  const doDecision = (id, decision) => {
    if (!officer.trim()) { setError("Enter your name as the reviewing officer first."); return; }
    run(`d${id}`, () => api(`/alerts/${id}/decision`, {
      method: "POST", body: JSON.stringify({ officer: officer.trim(), decision, note: "" }),
    }));
  };

  return (
    <main>
      <h1>XAI-AMLBench Compliance Console</h1>
      <div className="sub">Scan an account, review the explainable alert, and record an auditable decision.</div>

      <section className="card">
        <h2>Scan account</h2>
        <div className="row">
          <input placeholder="Account ID, e.g. ACC0000123" value={accountId} onChange={(e) => setAccountId(e.target.value)} />
          <button disabled={!accountId.trim() || busy === "scan"} onClick={doScan}>{busy === "scan" ? "Scanning…" : "Scan"}</button>
        </div>
        {samples && (
          <div className="row" style={{ marginTop: 10 }}>
            <span className="muted">Try:</span>
            {samples.suspicious.slice(0, 3).map((a) => <span key={a} className="chip" onClick={() => setAccountId(a)}>{a} (suspicious)</span>)}
            {samples.benign.slice(0, 2).map((a) => <span key={a} className="chip" onClick={() => setAccountId(a)}>{a} (benign)</span>)}
          </div>
        )}
        {scan && (
          <p className="muted" style={{ marginTop: 12 }}>
            Score <b>{(scan.score * 100).toFixed(1)}%</b> (alert threshold {(scan.threshold * 100).toFixed(1)}%) ·{" "}
            {scan.flagged ? "flagged, alert created" : "below threshold, no alert"} · {scan.latency_ms} ms
          </p>
        )}
        {error && <div className="err">{error}</div>}
      </section>

      <section className="card">
        <div className="row" style={{ justifyContent: "space-between" }}>
          <h2>Alerts ({alerts.length})</h2>
          <div className="row">
            <input placeholder="Reviewing officer name" value={officer} onChange={(e) => setOfficer(e.target.value)} />
            <button className="ghost" onClick={load}>Refresh</button>
          </div>
        </div>
        {alerts.length === 0 && <p className="muted">No alerts yet. Scan a suspicious account above.</p>}
        {alerts.map((a) => (
          <div className="alert" key={a.id}>
            <div className="row" style={{ justifyContent: "space-between" }}>
              <div className="row">
                <b>#{a.id}</b> <span>{a.account_id}</span>
                <span className={`pill ${a.status}`}>{a.status}</span>
                {a.typology && <span className="muted">{a.typology.replaceAll("_", " ")}</span>}
              </div>
              <div className="row">
                <div className="bar"><div style={{ width: `${Math.round(a.risk_score * 100)}%` }} /></div>
                <span className="muted">{(a.risk_score * 100).toFixed(0)}%</span>
              </div>
            </div>

            {a.narrative && (
              <div className="narr">
                {a.narrative}
                <div className="muted" style={{ marginTop: 6 }}>source: {a.narrative_source} · sha256 {a.narrative_sha256?.slice(0, 12)}…</div>
              </div>
            )}

            <div className="row" style={{ marginTop: 8 }}>
              <button className="ghost" disabled={busy === `n${a.id}`} onClick={() => doNarrative(a.id)}>
                {busy === `n${a.id}` ? "Generating…" : a.narrative ? "Regenerate narrative" : "Generate SAR narrative"}
              </button>
              {a.status === "open" && a.narrative && (
                <>
                  <button className="red" disabled={busy === `d${a.id}`} onClick={() => doDecision(a.id, "confirmed")}>Confirm suspicious</button>
                  <button className="green" disabled={busy === `d${a.id}`} onClick={() => doDecision(a.id, "dismissed")}>Dismiss</button>
                </>
              )}
              {a.reviewed_by && <span className="muted">reviewed by {a.reviewed_by}</span>}
            </div>
          </div>
        ))}
      </section>
    </main>
  );
}