"use client";

import { Fragment, useCallback, useEffect, useMemo, useState } from "react";
import type { ReactElement } from "react";
import Background from "@/components/Background";
import FloatingDock from "@/components/FloatingDock";
import type { DockTab } from "@/components/FloatingDock";
import Glass from "@/components/Glass";
import { IconBell, IconGraph, IconHome, IconRadar, IconShield } from "@/components/Icons";
import LoginScreen from "@/components/LoginScreen";
import NetworkGraph from "@/components/NetworkGraph";
import RiskGauge from "@/components/RiskGauge";
import { useCountUp, useTypewriter } from "@/components/hooks";
import { api, clearApiKey, errorMessage, hasStoredApiKey, pct, sleep } from "@/lib/api";
import { isDemo } from "@/lib/demo";
import type {
  Alert,
  AlertStatus,
  Explanation,
  ModelInfo,
  SampleAccounts,
  ScanResponse,
  ScanResult,
  TabId,
  Toast,
  ToastKind,
  WhoamiResponse,
} from "@/types";

const niceType = (t: string | null): string => (t ? t.replaceAll("_", " ") : "");

type Tone = "blue" | "amber" | "red" | "cyan";
type Filter = "all" | AlertStatus;
const FILTERS: Filter[] = ["all", "open", "confirmed", "dismissed"];

/* ----------------------------------------------------------------- small pieces */
interface StatProps {
  label: string;
  value: number;
  tone: Tone;
  hint: string;
  suffix?: string;
}

function Stat({ label, value, tone, hint, suffix = "" }: StatProps) {
  const v = useCountUp(value);
  return (
    <Glass as="div" className={`stat tone-${tone}`}>
      <div className="stat-label">{label}</div>
      <div className="stat-value">
        {v.toFixed(0)}
        {suffix}
      </div>
      <div className="stat-hint">{hint}</div>
      <span className="stat-spark" />
    </Glass>
  );
}

const PIPE: string[] = ["Transactions", "Kafka stream", "GNN detection", "Explainability", "SAR narrative", "immudb ledger"];

function Pipeline() {
  return (
    <div className="pipe">
      {PIPE.map((s, i) => (
        <Fragment key={s}>
          <div className="pipe-item">
            <div className="pipe-node" style={{ animationDelay: `${i * 0.4}s` }}>
              <b>{i + 1}</b>
            </div>
            <span>{s}</span>
          </div>
          {i < PIPE.length - 1 && <i className="pipe-link" style={{ animationDelay: `${i * 0.25}s` }} />}
        </Fragment>
      ))}
    </div>
  );
}

interface NarrativeProps {
  text: string;
  source: string | null;
  sha256: string | null;
  animate: boolean;
  onDone: () => void;
}

function Narrative({ text, source, sha256, animate, onDone }: NarrativeProps) {
  const shown = useTypewriter(text, animate, onDone);
  return (
    <div className="narr">
      <span>{shown}</span>
      {animate && shown.length < text.length && <i className="caret" />}
      <div className="narr-meta">
        source: {source ?? "n/a"} · sha256 {sha256 ? `${sha256.slice(0, 12)}…` : "n/a"}
      </div>
    </div>
  );
}

function ModelCard({ info }: { info: ModelInfo | null }) {
  const auc = info?.metrics.test_auc_roc ?? 0;
  const target = info?.targets.auc_roc ?? 0.87;
  const aucAnim = useCountUp(auc * 100, 1100);
  return (
    <Glass className="model-card">
      <h3>Detection model</h3>
      {info === null && <p className="muted">Model not loaded yet. Train one (see README), then refresh.</p>}
      {info !== null && (
        <>
          <div className="auc-row">
            <span>AUC-ROC</span>
            <b>{(aucAnim / 100).toFixed(3)}</b>
            <span className={`pill ${info.metrics.meets_auc_target ? "ok" : "warn"}`}>
              {info.metrics.meets_auc_target ? `meets ≥ ${target}` : `below ${target}`}
            </span>
          </div>
          <div className="auc-bar">
            <div className="auc-fill" style={{ width: `${Math.min(100, auc * 100)}%` }} />
            <i className="auc-target" style={{ left: `${target * 100}%` }} />
          </div>
          <div className="mini-grid">
            <div><span>Model</span><b>{info.model.toUpperCase()}</b></div>
            <div><span>Precision</span><b>{pct(info.metrics.test_precision, 0)}</b></div>
            <div><span>Recall</span><b>{pct(info.metrics.test_recall, 0)}</b></div>
            <div><span>Threshold</span><b>{pct(info.threshold, 0)}</b></div>
          </div>
        </>
      )}
    </Glass>
  );
}

/* ----------------------------------------------------------------------- page */
export default function Home(): ReactElement {
  const demoMode = isDemo();
  // In demo mode we must install the mock backend BEFORE the first request, so wait for it.
  const [ready, setReady] = useState<boolean>(!demoMode);
  const [tab, setTab] = useState<TabId>("overview");
  const [alerts, setAlerts] = useState<Alert[]>([]);
  const [online, setOnline] = useState<boolean | null>(null);
  const [samples, setSamples] = useState<SampleAccounts | null>(null);
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null);
  const [accountId, setAccountId] = useState<string>("");
  const [scan, setScan] = useState<ScanResult | null>(null);
  const [authedOfficer, setAuthedOfficer] = useState<string | null>(null);
  const [authChecked, setAuthChecked] = useState<boolean>(demoMode);   // demo mode skips auth entirely
  const [filter, setFilter] = useState<Filter>("all");
  const [busy, setBusy] = useState<string>("");
  const [fresh, setFresh] = useState<Set<number>>(() => new Set<number>());
  const [netId, setNetId] = useState<string>("");
  const [graph, setGraph] = useState<Explanation | null>(null);
  const [toasts, setToasts] = useState<Toast[]>([]);

  const notify = useCallback((kind: ToastKind, msg: string): void => {
    const id = Date.now() + Math.random();
    setToasts((t) => [...t, { id, kind, msg }]);
    setTimeout(() => setToasts((t) => t.filter((x) => x.id !== id)), 5200);
  }, []);

  const load = useCallback(async (): Promise<void> => {
    try {
      setAlerts(await api<Alert[]>("/alerts"));
      setOnline(true);
    } catch {
      setOnline(false);
    }
  }, []);

  useEffect(() => {
    if (!demoMode) return;
    void import("../../demo/mockApi").then((m) => {
      m.installMockApi();
      setReady(true);
    });
  }, [demoMode]);

  useEffect(() => {
    if (demoMode) return;   // demo mode has no real backend to authenticate against at all
    if (!hasStoredApiKey()) {
      setAuthChecked(true);
      return;
    }
    // a key is already stored (e.g. this tab's sessionStorage survived a reload) -- verify it still works
    // rather than trusting it blindly, since it may have been revoked since the last page load
    api<WhoamiResponse>("/whoami")
      .then((who) => setAuthedOfficer(who.officer))
      .catch(() => clearApiKey())
      .finally(() => setAuthChecked(true));
  }, [demoMode]);

  const handleLoginSuccess = useCallback((officerName: string): void => {
    setAuthedOfficer(officerName);
  }, []);

  const handleLogout = useCallback((): void => {
    clearApiKey();
    setAuthedOfficer(null);
  }, []);

  useEffect(() => {
    if (!ready) return undefined;
    void load();
    api<SampleAccounts>("/accounts/sample").then(setSamples).catch(() => undefined);
    api<ModelInfo>("/model/info").then(setModelInfo).catch(() => undefined);
    const id = setInterval(() => void load(), 15000);
    return () => clearInterval(id);
  }, [load, ready]);

  const stats = useMemo(() => {
    const open = alerts.filter((a) => a.status === "open").length;
    const confirmed = alerts.filter((a) => a.status === "confirmed").length;
    const avg = alerts.length ? alerts.reduce((s, a) => s + a.risk_score, 0) / alerts.length : 0;
    return { total: alerts.length, open, confirmed, avg };
  }, [alerts]);

  /* ---- actions ---- */
  const doScan = async (): Promise<void> => {
    const id = accountId.trim();
    if (!id) return;
    setBusy("scan");
    setScan(null);
    try {
      const [res] = await Promise.all([
        api<ScanResponse>("/alerts/scan", { method: "POST", body: JSON.stringify({ account_id: id }) }),
        sleep(900),
      ]);
      setScan({ ...res, account_id: id });
      if (res.alert) {
        notify("warn", `Alert #${res.alert.id} created for ${id}`);
        void load();
      }
    } catch (e) {
      notify("err", errorMessage(e));
    } finally {
      setBusy("");
    }
  };

  const doNarrative = async (id: number): Promise<void> => {
    setBusy(`n${id}`);
    try {
      const updated = await api<Alert>(`/alerts/${id}/narrative`, { method: "POST" });
      setFresh((s) => new Set(s).add(id));
      setAlerts((list) => list.map((x) => (x.id === id ? updated : x)));
      notify("ok", `SAR narrative ready (${updated.narrative_source ?? "unknown"})`);
    } catch (e) {
      notify("err", errorMessage(e));
    } finally {
      setBusy("");
    }
  };

  const doDecision = async (id: number, decision: "confirmed" | "dismissed"): Promise<void> => {
    setBusy(`d${id}`);
    try {
      const updated = await api<Alert>(`/alerts/${id}/decision`, {
        method: "POST",
        body: JSON.stringify({ decision, note: "" }),
      });
      setAlerts((list) => list.map((x) => (x.id === id ? updated : x)));
      notify("ok", `Alert #${id} ${decision} and written to the audit ledger`);
    } catch (e) {
      notify("err", errorMessage(e));
    } finally {
      setBusy("");
    }
  };

  const doExplain = async (id?: string): Promise<void> => {
    const acct = (id ?? netId).trim();
    if (!acct) return;
    setNetId(acct);
    setTab("network");
    setBusy("explain");
    setGraph(null);
    try {
      const [data] = await Promise.all([api<Explanation>(`/accounts/${encodeURIComponent(acct)}/explain`), sleep(600)]);
      setGraph(data);
    } catch (e) {
      notify("err", errorMessage(e));
    } finally {
      setBusy("");
    }
  };

  const clearFresh = (id: number): void =>
    setFresh((s) => {
      const next = new Set(s);
      next.delete(id);
      return next;
    });

  const tabs: DockTab[] = [
    { id: "overview", label: "Overview", icon: <IconHome /> },
    { id: "scan", label: "Scan", icon: <IconRadar /> },
    { id: "alerts", label: "Alerts", icon: <IconBell />, badge: stats.open },
    { id: "network", label: "Network", icon: <IconGraph /> },
  ];

  const shown = alerts.filter((a) => filter === "all" || a.status === filter);
  const latencyTarget = modelInfo?.targets.latency_ms ?? 100;

  /* ---- views ---- */
  const overview = (
    <>
      <section className="hero">
        <span className="eyebrow">Explainable AML · FinCEN-ready audit trail</span>
        <h1>
          <span className="grad-text">Follow the money.</span>
          <br />
          Explain every flag.
        </h1>
        <p className="lead">
          Graph neural networks spot laundering typologies across multi-hop transaction networks, and every alert
          comes with a human-readable SAR narrative and a tamper-proof audit record.
        </p>
        <div className="row">
          <button className="btn" onClick={() => setTab("scan")}>Scan an account</button>
          <button className="btn ghost" onClick={() => setTab("alerts")}>Review alerts</button>
        </div>
      </section>

      <div className="stats">
        <Stat label="Total alerts" value={stats.total} tone="blue" hint="all time" />
        <Stat label="Awaiting review" value={stats.open} tone="amber" hint="open cases" />
        <Stat label="Confirmed" value={stats.confirmed} tone="red" hint="marked suspicious" />
        <Stat label="Average risk" value={stats.avg * 100} suffix="%" tone="cyan" hint="across alerts" />
      </div>

      <Glass>
        <h3>Detection pipeline</h3>
        <Pipeline />
      </Glass>

      <div className="two">
        <ModelCard info={modelInfo} />
        <Glass>
          <div className="row between">
            <h3>Latest alerts</h3>
            <button className="link" onClick={() => setTab("alerts")}>View all →</button>
          </div>
          {alerts.length === 0 && <p className="muted">Nothing yet. Scan a suspicious account to create the first alert.</p>}
          {alerts.slice(0, 4).map((a, i) => (
            <div className="mini-alert rise" key={a.id} style={{ animationDelay: `${i * 70}ms` }} onClick={() => setTab("alerts")}>
              <span className={`dot ${a.status}`} />
              <b>{a.account_id}</b>
              <span className="muted">{niceType(a.typology) || "no narrative yet"}</span>
              <span className="grow" />
              <span className="risk">{pct(a.risk_score, 0)}</span>
            </div>
          ))}
        </Glass>
      </div>
    </>
  );

  const scanView = (
    <Glass className="scan-card">
      <h2>Scan an account</h2>
      <p className="muted">The model scores the account from its 2-hop transaction neighbourhood.</p>
      <div className="row">
        <input
          className="input grow"
          placeholder="Account ID, e.g. ACC0000123"
          value={accountId}
          onChange={(e) => setAccountId(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter") void doScan();
          }}
        />
        <button className="btn shine" disabled={!accountId.trim() || busy === "scan"} onClick={() => void doScan()}>
          {busy === "scan" ? "Scanning…" : "Scan"}
        </button>
      </div>
      {samples !== null && (
        <div className="row chips">
          <span className="muted">Try:</span>
          {samples.suspicious.slice(0, 3).map((a) => (
            <button key={a} className="chip bad" onClick={() => setAccountId(a)}>{a}</button>
          ))}
          {samples.benign.slice(0, 2).map((a) => (
            <button key={a} className="chip" onClick={() => setAccountId(a)}>{a}</button>
          ))}
        </div>
      )}

      {busy === "scan" && (
        <div className="scan-wait">
          <div className="radar" />
          <p className="muted">Running graph inference…</p>
        </div>
      )}
      {scan !== null && busy !== "scan" && (
        <div className="scan-result rise">
          <RiskGauge key={`${scan.account_id}-${scan.score}`} score={scan.calibrated_score ?? scan.score} threshold={scan.calibrated_threshold ?? scan.threshold} />
          <div className="scan-meta">
            <div className={`verdict ${scan.flagged ? "bad" : "good"}`}>
              {scan.flagged ? "Flagged: suspicious pattern" : "Below alert threshold"}
            </div>
            <p className="muted">
              {scan.account_id} · threshold {pct(scan.calibrated_threshold ?? scan.threshold)} · inference {scan.latency_ms} ms
              {scan.latency_ms <= latencyTarget && <span className="ok-text"> ✓ within {latencyTarget} ms target</span>}
            </p>
            <div className="row">
              <button className="btn ghost" onClick={() => void doExplain(scan.account_id)}>Explain network</button>
              {scan.alert !== null && (
                <button className="btn" onClick={() => setTab("alerts")}>Open alert #{scan.alert.id}</button>
              )}
            </div>
          </div>
        </div>
      )}
    </Glass>
  );

  const alertsView = (
    <>
      <Glass>
        <div className="row between">
          <div className="seg">
            {FILTERS.map((f) => (
              <button key={f} className={filter === f ? "on" : ""} onClick={() => setFilter(f)}>{f}</button>
            ))}
          </div>
          <div className="row">
            <button className="btn ghost" onClick={() => void load()}>Refresh</button>
          </div>
        </div>
      </Glass>

      {shown.length === 0 && (
        <Glass>
          <p className="muted">No alerts in this view.</p>
        </Glass>
      )}
      {shown.map((a, i) => (
        <Glass className="alert-card rise" key={a.id} style={{ animationDelay: `${Math.min(i, 8) * 60}ms` }}>
          <div className="row between">
            <div className="row">
              <span className="alert-id">#{a.id}</span>
              <b>{a.account_id}</b>
              <span className={`pill s-${a.status}`}>{a.status}</span>
              {a.typology && <span className="pill neutral">{niceType(a.typology)}</span>}
            </div>
            <div className="row">
              <div className="bar"><div style={{ width: `${Math.round(a.risk_score * 100)}%` }} /></div>
              <b className="risk">{pct(a.risk_score, 0)}</b>
            </div>
          </div>

          {a.narrative && (
            <Narrative
              text={a.narrative}
              source={a.narrative_source}
              sha256={a.narrative_sha256}
              animate={fresh.has(a.id)}
              onDone={() => clearFresh(a.id)}
            />
          )}

          <div className="row actions">
            <button className="btn ghost" disabled={busy === `n${a.id}`} onClick={() => void doNarrative(a.id)}>
              {busy === `n${a.id}` ? "Generating…" : a.narrative ? "Regenerate narrative" : "Generate SAR narrative"}
            </button>
            <button className="btn ghost" onClick={() => void doExplain(a.account_id)}>View network</button>
            {a.status === "open" && a.narrative && (
              <>
                <button className="btn red" disabled={busy === `d${a.id}`} onClick={() => void doDecision(a.id, "confirmed")}>
                  Confirm suspicious
                </button>
                <button className="btn green" disabled={busy === `d${a.id}`} onClick={() => void doDecision(a.id, "dismissed")}>
                  Dismiss
                </button>
              </>
            )}
            {a.reviewed_by && <span className="muted">reviewed by {a.reviewed_by}</span>}
          </div>
        </Glass>
      ))}
    </>
  );

  const networkView = (
    <>
      <Glass>
        <div className="row">
          <input
            className="input grow"
            placeholder="Account ID to explain, e.g. ACC0000123"
            value={netId}
            onChange={(e) => setNetId(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") void doExplain();
            }}
          />
          <button className="btn shine" disabled={!netId.trim() || busy === "explain"} onClick={() => void doExplain()}>
            {busy === "explain" ? "Explaining…" : "Explain"}
          </button>
        </div>
        {samples !== null && (
          <div className="row chips">
            <span className="muted">Try:</span>
            {samples.suspicious.slice(0, 3).map((a) => (
              <button key={a} className="chip bad" onClick={() => setNetId(a)}>{a}</button>
            ))}
          </div>
        )}
      </Glass>
      {busy === "explain" && (
        <Glass>
          <div className="scan-wait">
            <div className="radar" />
            <p className="muted">Running GNNExplainer…</p>
          </div>
        </Glass>
      )}
      {graph !== null && busy !== "explain" && (
        <div className="rise">
          <NetworkGraph key={graph.account_id} data={graph} />
        </div>
      )}
      {graph === null && busy !== "explain" && (
        <Glass>
          <p className="muted">Explain an account to see the transaction network the model reasoned over.</p>
        </Glass>
      )}
    </>
  );

  if (!authChecked) {
    return <Background />;   // brief, near-instant check of an already-stored key; nothing worth a spinner for
  }
  if (!demoMode && !authedOfficer) {
    return (
      <>
        <Background />
        <LoginScreen onSuccess={handleLoginSuccess} />
      </>
    );
  }

  return (
    <>
      <Background />
      {demoMode && <div className="demo-badge">Demo mode &middot; sample data &middot; no live backend</div>}
      <div className="toasts">
        {toasts.map((t) => (
          <div key={t.id} className={`toast ${t.kind}`}>{t.msg}</div>
        ))}
      </div>

      <div className={`app${demoMode ? " demo" : ""}`}>
        <header className="topbar">
          <div className="brand">
            <div className="logo">
              <span className="logo-ring" />
              <IconShield />
            </div>
            <div>
              <div className="brand-name">XAI<span>·</span>AMLBench</div>
              <div className="brand-sub">Compliance Console</div>
            </div>
          </div>
          <div className={`status ${online === false ? "off" : online ? "on" : ""}`}>
            <i />
            {online === null ? "Connecting…" : online ? "Backend online" : "Backend offline"}
          </div>
          {authedOfficer && (
            <div className="row officer-badge">
              <span className="muted">Signed in as {authedOfficer}</span>
              <button className="btn ghost" onClick={handleLogout}>Sign out</button>
            </div>
          )}
        </header>

        <main key={tab} className="view">
          {tab === "overview" && overview}
          {tab === "scan" && scanView}
          {tab === "alerts" && alertsView}
          {tab === "network" && networkView}
        </main>
      </div>

      <FloatingDock tabs={tabs} active={tab} onChange={setTab} />
    </>
  );
}
