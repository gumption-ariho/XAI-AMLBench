"use client";

import { Fragment, useCallback, useEffect, useMemo, useState } from "react";
import type { ReactElement } from "react";
import { useRouter } from "next/navigation";
import Background from "@/components/Background";
import Sidebar from "@/components/Sidebar";
import type { DockTab } from "@/components/Sidebar";
import Glass from "@/components/Glass";
import {
  IconBell, IconFlow, IconGraph, IconHome, IconList, IconPlug, IconRadar, IconSearch, IconShield, IconUpload,
} from "@/components/Icons";
import LoginScreen from "@/components/LoginScreen";
import NetworkGraph from "@/components/NetworkGraph";
import RiskGauge from "@/components/RiskGauge";
import { useCountUp, useTypewriter } from "@/components/hooks";
import { api, clearApiKey, errorMessage, hasStoredApiKey, pct, sleep } from "@/lib/api";
import { isDemo } from "@/lib/demo";
import type {
  ActivityEntry,
  Alert,
  AlertStatus,
  AnomalyAlert,
  AnomalyStatus,
  Explanation,
  IntakeIngestResponse,
  IntakeClearResponse,
  ModelInfo,
  SampleAccounts,
  ScanResponse,
  ScanResult,
  SystemStatus,
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
  icon: ReactElement;
}

function Stat({ label, value, tone, hint, suffix = "", icon }: StatProps) {
  const v = useCountUp(value);
  return (
    <Glass as="div" className={`stat tone-${tone}`}>
      <span className="stat-icon">{icon}</span>
      <div className="stat-label">{label}</div>
      <div className="stat-value">
        {v.toFixed(0)}
        {suffix}
      </div>
      <div className="stat-hint">{hint}</div>
    </Glass>
  );
}

const PIPE: string[] = ["Transactions", "Kafka stream", "GNN detection", "Explainability", "SAR narrative", "immudb ledger"];

/** The real-world investigation toolkit, organised the way an actual AML investigation runs. Two kinds of
 * item: "action" jumps to something this console can genuinely do right now (the scan/network tools above);
 * "link" opens a real external lookup in a new tab; everything else is a checklist item this console has no
 * live data source for (core banking, KYC file, credit bureau, SWIFT tracing, law enforcement databases --
 * all outside what a self-contained benchmark project can reach), tracked here so the step isn't forgotten
 * and the finding gets written into the case file either way. */
type ToolkitAction =
  | "scan" | "risk-compare" | "alerts-history" | "anomaly-history" | "investigation-log" | "activity-history"
  | "network" | "explain-summary" | "counterparties" | "explain-features";
interface ToolkitItem { id: string; label: string; kind: "action" | "link" | "checklist"; action?: ToolkitAction; url?: (q: string) => string }
interface ToolkitSection { title: string; items: ToolkitItem[] }
const TOOLKIT_SECTIONS: ToolkitSection[] = [
  {
    title: "Internal bank data",
    items: [
      { id: "txn_monitoring", label: "Transaction monitoring / Model 1 cross-check (runs here)", kind: "action", action: "scan" },
      { id: "risk_compare", label: "Risk score vs. alert threshold (runs here)", kind: "action", action: "risk-compare" },
      { id: "alerts_history", label: "Model 1 alert history for this account (runs here)", kind: "action", action: "alerts-history" },
      { id: "anomaly_history", label: "Model 2 anomaly history for this account (runs here)", kind: "action", action: "anomaly-history" },
      { id: "investigation_log_tool", label: "Investigation log on file for this account (runs here)", kind: "action", action: "investigation-log" },
      { id: "activity_history", label: "Recent system activity for this account (runs here)", kind: "action", action: "activity-history" },
      { id: "core_banking", label: "Core banking record (account opening info, signatures, balance history)", kind: "checklist" },
      { id: "kyc_file", label: "KYC/CDD file (ID docs, proof of address, stated income, source of funds)", kind: "checklist" },
      { id: "login_data", label: "Login/session data (IP, device fingerprint, geolocation)", kind: "checklist" },
      { id: "call_notes", label: "Call center / branch interaction notes", kind: "checklist" },
    ],
  },
  {
    title: "Identity & background checks",
    items: [
      { id: "sanctions", label: "OFAC sanctions list search ↗", kind: "link", url: () => "https://sanctionssearch.ofac.treas.gov/" },
      { id: "un_list", label: "UN Security Council consolidated list ↗", kind: "link", url: () => "https://www.un.org/securitycouncil/content/un-sc-consolidated-list" },
      { id: "corp_registry", label: "Corporate registry / beneficial ownership (OpenCorporates) ↗", kind: "link", url: (q) => `https://opencorporates.com/companies?q=${encodeURIComponent(q)}` },
      { id: "osint", label: "OSINT / web search on the account holder ↗", kind: "link", url: (q) => `https://www.google.com/search?q=${encodeURIComponent(q)}` },
      { id: "gov_id", label: "Government ID verification", kind: "checklist" },
      { id: "credit_bureau", label: "Credit bureau report", kind: "checklist" },
    ],
  },
  {
    title: "Network / link analysis",
    items: [
      { id: "graph_view", label: "Open full network graph (this console's Network tab)", kind: "action", action: "network" },
      { id: "flow_summary", label: "Flow summary -- accounts, volume, cross-border % (runs here)", kind: "action", action: "explain-summary" },
      { id: "counterparties", label: "List counterparties seen in the graph (runs here)", kind: "action", action: "counterparties" },
      { id: "top_features", label: "Top risk features (XAI) for this account (runs here)", kind: "action", action: "explain-features" },
      { id: "shared_attrs", label: "Shared address / phone / device / IP across \"unrelated\" accounts", kind: "checklist" },
    ],
  },
  {
    title: "Cross-institution / external data sharing",
    items: [
      { id: "swift_trace", label: "SWIFT message tracing", kind: "checklist" },
      { id: "correspondent", label: "Correspondent bank inquiry", kind: "checklist" },
      { id: "info_sharing", label: "314(b) / interbank information-sharing request", kind: "checklist" },
      { id: "law_enforcement", label: "Law enforcement database check (where legally accessible)", kind: "checklist" },
    ],
  },
  {
    title: "Direct outreach",
    items: [
      { id: "phone_call", label: "Phone call to the account holder", kind: "checklist" },
      { id: "letter", label: "Letter / request for source-of-funds documentation", kind: "checklist" },
      { id: "branch_visit", label: "In-person branch verification", kind: "checklist" },
    ],
  },
  {
    title: "Reporting",
    items: [
      { id: "sar_filing", label: "SAR/STR filed with the relevant FIU", kind: "checklist" },
      { id: "case_doc", label: "Case fully documented in the audit trail", kind: "checklist" },
    ],
  },
];

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
  const router = useRouter();
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
  const [authedBank, setAuthedBank] = useState<string | null>(null);
  const [authChecked, setAuthChecked] = useState<boolean>(demoMode);   // demo mode skips auth entirely
  const [filter, setFilter] = useState<Filter>("all");
  const [bankFilter, setBankFilter] = useState<string>("all");
  const [banks, setBanks] = useState<string[]>([]);
  const [busy, setBusy] = useState<string>("");
  const [fresh, setFresh] = useState<Set<number>>(() => new Set<number>());
  const [netId, setNetId] = useState<string>("");
  const [graph, setGraph] = useState<Explanation | null>(null);
  const [toasts, setToasts] = useState<Toast[]>([]);
  const [anomalies, setAnomalies] = useState<AnomalyAlert[]>([]);
  const [anomalyFilter, setAnomalyFilter] = useState<AnomalyStatus | "all">("all");
  const [noteDraft, setNoteDraft] = useState<Record<number, string>>({});
  const [crossCheck, setCrossCheck] = useState<Record<number, ScanResult | "pending" | "error">>({});
  const [lookupId, setLookupId] = useState<string>("");
  const [lookupResult, setLookupResult] = useState<ScanResult | "pending" | "error" | null>(null);
  const [toolkitOpen, setToolkitOpen] = useState<boolean>(false);
  const [toolkitChecks, setToolkitChecks] = useState<Record<string, boolean>>({});
  const [toolkitQuery, setToolkitQuery] = useState<string>("");
  const [toolkitFindings, setToolkitFindings] = useState<string>("");
  const [toolkitRunningId, setToolkitRunningId] = useState<string>("");
  const [toolkitResult, setToolkitResult] = useState<{ tool: string; lines: string[] } | null>(null);
  const [toolkitExplainCache, setToolkitExplainCache] = useState<{ account_id: string; data: Explanation } | null>(null);
  const [activity, setActivity] = useState<ActivityEntry[]>([]);
  const [sysStatus, setSysStatus] = useState<SystemStatus | null>(null);
  const [intakeText, setIntakeText] = useState<string>("");
  const [intakeResults, setIntakeResults] = useState<ScanResult[]>([]);
  const [intakeRunning, setIntakeRunning] = useState<boolean>(false);
  const [intakeProgress, setIntakeProgress] = useState<{ done: number; total: number }>({ done: 0, total: 0 });
  const [ingestBank, setIngestBank] = useState<string>("");
  const [ingestText, setIngestText] = useState<string>("");
  const [ingestRunning, setIngestRunning] = useState<boolean>(false);
  const [ingestResult, setIngestResult] = useState<IntakeIngestResponse | null>(null);
  const [clearRunning, setClearRunning] = useState<boolean>(false);
  const [clearResult, setClearResult] = useState<IntakeClearResponse | null>(null);

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

  const loadAnomalies = useCallback(async (): Promise<void> => {
    try {
      setAnomalies(await api<AnomalyAlert[]>("/anomalies"));
    } catch {
      // the overview badge/stat just won't update this round; the Investigate tab's own refresh button covers it
    }
  }, []);

  const loadActivity = useCallback(async (): Promise<void> => {
    try {
      setActivity(await api<ActivityEntry[]>("/activity?limit=80"));
    } catch {
      // Activity tab shows its own "couldn't load" state; nothing else depends on this succeeding
    }
  }, []);

  const loadStatus = useCallback(async (): Promise<void> => {
    try {
      setSysStatus(await api<SystemStatus>("/system/status"));
    } catch {
      setSysStatus(null);
    }
  }, []);

  const loadBanks = useCallback(async (): Promise<void> => {
    try {
      setBanks(await api<string[]>("/banks"));
    } catch {
      // the Alerts tab's bank filter just won't offer a dropdown this round; "all" still works
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
    if (demoMode) return;   // demo mode (sample-data mode) has no real backend to authenticate against at all
    // Always attempt /whoami, even with no key stored at all: if the backend's own DEMO_LOGIN_BYPASS is on
    // (a reviewer-demo convenience, off by default -- see backend/main.py), an unauthenticated call to this
    // same endpoint succeeds and returns "Reviewer (DEMO MODE)", letting this skip the login screen
    // automatically with no frontend-side flag needed at all. With bypass off (the normal case), an
    // unauthenticated call here correctly fails with 401 and the login screen still shows, same as before --
    // this only ever makes login EASIER when the backend explicitly allows it, never less secure on its own.
    if (!hasStoredApiKey()) {
      api<WhoamiResponse>("/whoami")
        .then((who) => {
          // An admin passkey has no bank console of its own -- this page is the customer/bank console, so
          // send an admin session straight to the owner backoffice instead of rendering it here. See
          // handleLoginSuccess below for the same redirect right after login.
          if (who.role === "admin") { router.replace("/owner"); return; }
          setAuthedOfficer(who.officer); setAuthedBank(who.bank);
        })
        .catch(() => {})   // no stored key and bypass is off: expected, fall through to the login screen
        .finally(() => setAuthChecked(true));
      return;
    }
    // a key is already stored (e.g. this tab's sessionStorage survived a reload) -- verify it still works
    // rather than trusting it blindly, since it may have been revoked since the last page load
    api<WhoamiResponse>("/whoami")
      .then((who) => {
        if (who.role === "admin") { router.replace("/owner"); return; }
        setAuthedOfficer(who.officer); setAuthedBank(who.bank);
      })
      .catch(() => clearApiKey())
      .finally(() => setAuthChecked(true));
  }, [demoMode, router]);

  const handleLoginSuccess = useCallback((officerName: string, bank: string | null, role: "admin" | "analyst"): void => {
    // Same split as the whoami check above: the owner passkey has no business on the bank console.
    if (role === "admin") { router.replace("/owner"); return; }
    setAuthedOfficer(officerName);
    setAuthedBank(bank);
  }, [router]);

  const handleLogout = useCallback((): void => {
    clearApiKey();
    setAuthedOfficer(null);
    setAuthedBank(null);
  }, []);

  useEffect(() => {
    if (!ready) return undefined;
    void load();
    void loadAnomalies();
    void loadActivity();
    void loadStatus();
    void loadBanks();
    api<SampleAccounts>("/accounts/sample").then(setSamples).catch(() => undefined);
    api<ModelInfo>("/model/info").then(setModelInfo).catch(() => undefined);
    const id = setInterval(() => {
      void load();
      void loadAnomalies();
      void loadActivity();
      void loadStatus();
      void loadBanks();
    }, 15000);
    return () => clearInterval(id);
  }, [load, loadAnomalies, loadActivity, loadStatus, loadBanks, ready]);

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

  /** Parses whatever was pasted or uploaded into real transaction rows for POST /intake/ingest -- this is the
   * actual "add real accounts to the live graph" path. Accepts a CSV with a header row containing at least
   * src,dst,amount (timestamp and cross_border are optional and default sensibly server-side -- see
   * gnn_aml_core/livegraph.py's ingest()), or the same columns with no header in that fixed order. A malformed
   * row (missing src/dst/amount, or a non-numeric amount) is skipped rather than failing the whole paste, same
   * as the backend's own ingest() does. */
  const parseIntakeTx = (raw: string): { src: string; dst: string; amount: number; timestamp?: number; cross_border?: number }[] => {
    const lines = raw.split(/\r?\n/).map((l) => l.trim()).filter(Boolean);
    const rows: { src: string; dst: string; amount: number; timestamp?: number; cross_border?: number }[] = [];
    let cols = ["src", "dst", "amount", "timestamp", "cross_border"];
    let start = 0;
    if (lines.length > 0) {
      const headerCells = lines[0].split(",").map((c) => c.trim().toLowerCase());
      if (headerCells.includes("src") && headerCells.includes("dst") && headerCells.includes("amount")) {
        cols = headerCells;
        start = 1;
      }
    }
    for (let i = start; i < lines.length; i++) {
      const cells = lines[i].split(",").map((c) => c.trim());
      const rec: Record<string, string> = {};
      cols.forEach((c, idx) => { if (cells[idx] !== undefined) rec[c] = cells[idx]; });
      const src = rec.src, dst = rec.dst;
      const amount = Number(rec.amount);
      if (!src || !dst || !Number.isFinite(amount)) continue;
      const row: { src: string; dst: string; amount: number; timestamp?: number; cross_border?: number } = { src, dst, amount };
      if (rec.timestamp && Number.isFinite(Number(rec.timestamp))) row.timestamp = Number(rec.timestamp);
      if (rec.cross_border && Number.isFinite(Number(rec.cross_border))) row.cross_border = Number(rec.cross_border);
      rows.push(row);
    }
    return rows;
  };

  const doIngest = async (): Promise<void> => {
    // A tenant session (authedBank set) always ingests as its own bank -- the backend enforces this
    // server-side regardless of what's sent here, so the field is pre-filled and locked to match (see the
    // Bank/institution input below) rather than letting someone type something that'll just be overridden.
    const bank = authedBank || ingestBank.trim();
    if (!bank) {
      notify("warn", "Name the bank/institution this document's accounts belong to first.");
      return;
    }
    const rows = parseIntakeTx(ingestText);
    if (rows.length === 0) {
      notify("warn", "No valid transaction rows found (expected src,dst,amount per line, with an optional header).");
      return;
    }
    setIngestRunning(true);
    setIngestResult(null);
    try {
      const res = await api<IntakeIngestResponse>("/intake/ingest", { method: "POST", body: JSON.stringify({ rows, bank }) });
      setIngestResult(res);
      notify("ok", `Ingested ${res.ingested_rows} transaction(s) for ${bank} -- live graph now has ${res.n_accounts} account(s)`);
      void load();
      void loadBanks();
    } catch (e) {
      notify("err", errorMessage(e));
    } finally {
      setIngestRunning(false);
    }
  };

  /** POST /intake/clear -- for a walk-in/temporary customer who pasted a document for a one-off scan and
   * shouldn't have their accounts live in the shared graph forever. Removes every account tagged to this
   * bank from the live graph (unscannable, gone from the network view afterward); any alert already raised
   * on those accounts, and the decision/audit history behind it, is deliberately left untouched -- this
   * clears working data, not the compliance record. A tenant session always clears its own bank; an
   * admin/global session isn't offered this control at all (see the Intake tab JSX below), since there's no
   * single "own bank" for it to default to and a blanket clear-everything button is too easy to misclick. */
  const doClear = async (): Promise<void> => {
    const bank = authedBank;
    if (!bank) return;
    if (!window.confirm(
      `Clear ${bank}'s accounts from the live graph? They'll stop showing up in the network view and can't ` +
      `be scanned afterward. Any alerts already raised for ${bank}, and their decision history, are kept.`
    )) {
      return;
    }
    setClearRunning(true);
    setClearResult(null);
    try {
      const res = await api<IntakeClearResponse>("/intake/clear", { method: "POST", body: JSON.stringify({}) });
      setClearResult(res);
      setIngestResult(null);
      notify("ok", res.removed_accounts > 0
        ? `Cleared ${res.removed_accounts} account(s) for ${bank} from the live graph. Alerts and decision history were kept.`
        : `Nothing to clear -- ${bank} has no accounts in the live graph right now.`);
      void load();
      void loadBanks();
    } catch (e) {
      notify("err", errorMessage(e));
    } finally {
      setClearRunning(false);
    }
  };

  const onIngestFile = (file: File): void => {
    const reader = new FileReader();
    reader.onload = () => setIngestText(String(reader.result ?? ""));
    reader.readAsText(file);
  };

  /** Parses whatever was pasted or uploaded into a flat list of account IDs. Accepts one ID per line, a
   * comma/whitespace-separated list, or a CSV with an "account_id" (or first) column -- this is the "paste a
   * document of accounts through the frontend" intake path. It does NOT ingest raw transaction ledgers into
   * the graph itself (that would mean rebuilding the account graph gnn_aml_core.main was trained against --
   * a separate, much bigger pipeline than this console can trigger); what it does do is run every account ID
   * found through the real, already-trained Model 1 scan (the same /alerts/scan this page's Scan tab uses),
   * in bulk, so a whole list from a bank extract can be screened in one sitting instead of one at a time.
   *
   * If what's pasted is actually a transaction CSV (the same src/dst/amount/timestamp/cross_border shape the
   * ingestion card above takes), this pulls ONLY the src/dst columns as account IDs -- otherwise the amount
   * and timestamp columns get swept up as "account IDs" too (every cell matches the same loose
   * alphanumeric pattern an account ID does), producing a pile of bogus not-found IDs. Anything else falls
   * back to the original permissive per-cell parse, so a plain list of IDs (one per line, comma-separated,
   * or under an "account_id" CSV header) still works exactly as before. */
  const parseIntakeIds = (raw: string): string[] => {
    const lines = raw.split(/\r?\n/).map((l) => l.trim()).filter(Boolean);
    if (lines.length > 0) {
      const headerCells = lines[0].split(",").map((c) => c.trim().toLowerCase());
      if (headerCells.includes("src") && headerCells.includes("dst") && headerCells.includes("amount")) {
        const srcIdx = headerCells.indexOf("src");
        const dstIdx = headerCells.indexOf("dst");
        const ids: string[] = [];
        for (const line of lines.slice(1)) {
          const cells = line.split(",").map((c) => c.trim());
          if (cells[srcIdx]) ids.push(cells[srcIdx]);
          if (cells[dstIdx]) ids.push(cells[dstIdx]);
        }
        return Array.from(new Set(ids));
      }
    }
    const ids: string[] = [];
    for (const line of lines) {
      const cells = line.split(/[,\t]/).map((c) => c.trim()).filter(Boolean);
      for (const cell of cells) {
        if (/^account[_ ]?id$/i.test(cell)) continue;   // skip an obvious CSV header cell
        if (/^[A-Za-z0-9_-]{3,40}$/.test(cell)) ids.push(cell);
      }
    }
    return Array.from(new Set(ids));   // de-duplicate -- a pasted extract often repeats an account over many rows
  };

  const doBatchScan = async (): Promise<void> => {
    const ids = parseIntakeIds(intakeText);
    if (ids.length === 0) {
      notify("warn", "No account IDs found in that text (expected one per line, or a CSV column).");
      return;
    }
    setIntakeRunning(true);
    setIntakeResults([]);
    setIntakeProgress({ done: 0, total: ids.length });
    let created = 0;
    for (let i = 0; i < ids.length; i++) {
      const id = ids[i];
      try {
        const res = await api<ScanResponse>("/alerts/scan", { method: "POST", body: JSON.stringify({ account_id: id }) });
        setIntakeResults((r) => [...r, { ...res, account_id: id }]);
        if (res.alert) created++;
      } catch (e) {
        setIntakeResults((r) => [...r, {
          account_id: id, score: 0, flagged: false, threshold: 0, latency_ms: 0, alert: null,
        }]);
        notify("err", `${id}: ${errorMessage(e)}`);
      }
      setIntakeProgress({ done: i + 1, total: ids.length });
    }
    setIntakeRunning(false);
    void load();
    notify(created > 0 ? "warn" : "ok", `Scanned ${ids.length} account${ids.length === 1 ? "" : "s"} -- ${created} new alert${created === 1 ? "" : "s"}`);
  };

  const onIntakeFile = (file: File): void => {
    const reader = new FileReader();
    reader.onload = () => setIntakeText(String(reader.result ?? ""));
    reader.readAsText(file);
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

  const doClaim = async (id: number): Promise<void> => {
    setBusy(`claim${id}`);
    try {
      const updated = await api<AnomalyAlert>(`/anomalies/${id}/claim`, { method: "POST" });
      setAnomalies((list) => list.map((x) => (x.id === id ? updated : x)));
      notify("ok", `Claimed anomaly #${id} for investigation`);
    } catch (e) {
      notify("err", errorMessage(e));
    } finally {
      setBusy("");
    }
  };

  const doAddNote = async (id: number): Promise<void> => {
    const note = (noteDraft[id] ?? "").trim();
    if (!note) return;
    setBusy(`note${id}`);
    try {
      const updated = await api<AnomalyAlert>(`/anomalies/${id}/notes`, {
        method: "POST",
        body: JSON.stringify({ note }),
      });
      setAnomalies((list) => list.map((x) => (x.id === id ? updated : x)));
      setNoteDraft((d) => ({ ...d, [id]: "" }));
    } catch (e) {
      notify("err", errorMessage(e));
    } finally {
      setBusy("");
    }
  };

  const doResolveAnomaly = async (id: number, outcome: "confirmed_mule" | "cleared"): Promise<void> => {
    setBusy(`resolve${id}`);
    try {
      const updated = await api<AnomalyAlert>(`/anomalies/${id}/resolve`, {
        method: "POST",
        body: JSON.stringify({ outcome, note: noteDraft[id] ?? "" }),
      });
      setAnomalies((list) => list.map((x) => (x.id === id ? updated : x)));
      setNoteDraft((d) => ({ ...d, [id]: "" }));
      notify(outcome === "confirmed_mule" ? "warn" : "ok",
        outcome === "confirmed_mule"
          ? `#${id} confirmed as a mule account -- this pattern will be taught to the detection model`
          : `#${id} cleared -- logged, not used for training`);
    } catch (e) {
      notify("err", errorMessage(e));
    } finally {
      setBusy("");
    }
  };

  /** Runs the account through Model 1's own scan (the same endpoint the Scan tab uses) so an analyst
   * investigating a Model-2-only anomaly can see, inline, whether the baseline classifier agrees there's
   * anything unusual here at all -- a real cross-check against a second, independent signal, not a second
   * opinion from the same model. */
  const doCrossCheck = async (anomalyId: number, accountId: string): Promise<void> => {
    setCrossCheck((c) => ({ ...c, [anomalyId]: "pending" }));
    try {
      const res = await api<ScanResponse>("/alerts/scan", { method: "POST", body: JSON.stringify({ account_id: accountId }) });
      setCrossCheck((c) => ({ ...c, [anomalyId]: { ...res, account_id: accountId } }));
    } catch {
      setCrossCheck((c) => ({ ...c, [anomalyId]: "error" }));
    }
  };

  const NOTE_TEMPLATES: string[] = [
    "Called the account holder -- no answer, left message",
    "Reached the account holder -- explanation given, verifying",
    "Cross-checked related/linked accounts -- see network view",
    "Verified KYC documents on file -- no discrepancy found",
    "Verified KYC documents on file -- discrepancy found",
    "Escalated to compliance / law enforcement",
  ];

  /** Not a resolution -- the actual calling-the-owner-in / involving-investigators-or-police step happens
   * outside this system, as the Model 2 docstring and this tab's own banner both already say. This just logs
   * that it happened, distinctly tagged, so it shows up in the case file and the Activity feed without
   * forcing a premature confirmed/cleared verdict before that outside step is done. */
  const doEscalate = async (id: number): Promise<void> => {
    const extra = (noteDraft[id] ?? "").trim();
    const note = `[ESCALATED] ${extra || "Escalated to compliance / law enforcement for further action."}`;
    setBusy(`escalate${id}`);
    try {
      const updated = await api<AnomalyAlert>(`/anomalies/${id}/notes`, { method: "POST", body: JSON.stringify({ note }) });
      setAnomalies((list) => list.map((x) => (x.id === id ? updated : x)));
      setNoteDraft((d) => ({ ...d, [id]: "" }));
      notify("warn", `#${id} marked escalated`);
    } catch (e) {
      notify("err", errorMessage(e));
    } finally {
      setBusy("");
    }
  };

  const copyAccountId = (accountId: string): void => {
    navigator.clipboard?.writeText(accountId)
      .then(() => notify("ok", `Copied ${accountId}`))
      .catch(() => notify("err", "Couldn't copy to clipboard"));
  };

  /** The Investigate tab's own toolbar, for any account -- not only ones Model 2 has already flagged. An
   * analyst following up on a tip, a call, or something outside this system entirely still wants the same
   * tools (Model 1 cross-check, network view, copy ID) without needing a formal anomaly_alerts row to exist
   * first. Runs the real Model 1 scan, same as the Scan tab and the per-alert cross-check below. */
  const doLookup = async (): Promise<void> => {
    const acct = lookupId.trim();
    if (!acct) return;
    setLookupResult("pending");
    try {
      const res = await api<ScanResponse>("/alerts/scan", { method: "POST", body: JSON.stringify({ account_id: acct }) });
      setLookupResult({ ...res, account_id: acct });
    } catch {
      setLookupResult("error");
    }
  };

  /** Compiles whatever's checked in the toolkit plus the free-text findings into one note. If the account
   * being looked at has a claimed (under_investigation) Model 2 case open, it's written straight into that
   * case's own append-only log via the real /notes endpoint -- same record a resolve() later reads back.
   * Without a claimed case to attach to (an ad-hoc lookup with nothing flagged), there's nowhere in this
   * system to persist it yet, so it's copied to the clipboard instead with an honest explanation why. */
  const doSaveChecklist = async (accountId: string): Promise<void> => {
    const checked = TOOLKIT_SECTIONS.flatMap((s) => s.items.filter((it) => toolkitChecks[it.id])
      .map((it) => `  [x] ${it.label.replace(" ↗", "")}`));
    const lines = [
      `Investigation checklist for ${accountId}:`,
      ...(checked.length ? checked : ["  (no checklist items marked)"]),
      ...(toolkitFindings.trim() ? ["", `Findings: ${toolkitFindings.trim()}`] : []),
    ];
    const text = lines.join("\n");
    const match = anomalies.find((a) => a.account_id === accountId && a.status === "under_investigation");
    if (match) {
      setBusy(`checklist${match.id}`);
      try {
        const updated = await api<AnomalyAlert>(`/anomalies/${match.id}/notes`, { method: "POST", body: JSON.stringify({ note: text }) });
        setAnomalies((list) => list.map((x) => (x.id === match.id ? updated : x)));
        notify("ok", `Checklist saved to case #${match.id}`);
      } catch (e) {
        notify("err", errorMessage(e));
      } finally {
        setBusy("");
      }
      return;
    }
    navigator.clipboard?.writeText(text)
      .then(() => notify("warn", "No claimed Model 2 case for this account yet -- checklist copied to clipboard instead."))
      .catch(() => notify("err", "Couldn't copy the checklist"));
  };

  /** Fetches the real /accounts/{id}/explain payload for the toolkit's network-derived tools, caching it per
   * account so "flow summary", "counterparties" and "top features" don't each re-fetch -- they share one real
   * GNNExplainer call. Deliberately a separate cache from the Network tab's own `graph` state, so running these
   * toolkit tools never clobbers whatever the analyst has open on the Network tab. */
  const getToolkitExplain = async (acct: string): Promise<Explanation> => {
    if (toolkitExplainCache && toolkitExplainCache.account_id === acct) return toolkitExplainCache.data;
    const data = await api<Explanation>(`/accounts/${encodeURIComponent(acct)}/explain`);
    setToolkitExplainCache({ account_id: acct, data });
    return data;
  };

  /** Runs one toolkit "action" tool against lookupId and renders its result inline, right in this toolkit panel
   * -- no tab switch, except for the one tool whose whole point IS the Network tab's graphic (action:"network").
   * Every other tool here is backed by a real endpoint this console already has (Model 1's /alerts/scan, the
   * GNNExplainer /explain payload) or by data already loaded client-side (alerts, anomalies, activity) -- none
   * of it is invented or mocked. */
  const runToolkitTool = async (item: ToolkitItem): Promise<void> => {
    const acct = lookupId.trim();
    if (!acct) { notify("warn", "Enter an account ID above first."); return; }
    if (item.action === "network") { setTab("network"); void doExplain(acct); return; }
    setToolkitRunningId(item.id);
    setToolkitResult(null);
    try {
      switch (item.action) {
        case "scan": {
          const res = await api<ScanResponse>("/alerts/scan", { method: "POST", body: JSON.stringify({ account_id: acct }) });
          setToolkitResult({ tool: item.label, lines: [
            `Model 1 says: ${res.flagged ? "unusual" : "looks normal"}`,
            `Risk score ${pct(res.score, 1)} (threshold ${pct(res.threshold, 1)})`,
            res.alert ? `Alert #${res.alert.id} on file -- status ${res.alert.status}` : "No Model 1 alert on file for this account",
          ] });
          break;
        }
        case "risk-compare": {
          const res = await api<ScanResponse>("/alerts/scan", { method: "POST", body: JSON.stringify({ account_id: acct }) });
          const gap = res.score - res.threshold;
          setToolkitResult({ tool: item.label, lines: [
            `Score ${pct(res.score, 1)} vs. threshold ${pct(res.threshold, 1)}`,
            gap >= 0 ? `${pct(gap, 1)} ABOVE the alerting threshold` : `${pct(-gap, 1)} below the alerting threshold`,
            ...(res.calibrated_score !== undefined ? [`Calibrated score: ${pct(res.calibrated_score, 1)}`] : []),
          ] });
          break;
        }
        case "alerts-history": {
          const rows = alerts.filter((a) => a.account_id === acct);
          setToolkitResult({ tool: item.label, lines: rows.length
            ? rows.map((a) => `#${a.id} -- ${a.status} -- risk ${pct(a.risk_score, 1)} -- ${new Date(a.created_at).toLocaleString()}`)
            : ["No Model 1 alert history for this account in this session's data."] });
          break;
        }
        case "anomaly-history": {
          const rows = anomalies.filter((a) => a.account_id === acct);
          setToolkitResult({ tool: item.label, lines: rows.length
            ? rows.map((a) => `#${a.id} -- ${a.status} -- anomaly score ${a.anomaly_score.toFixed(4)} -- ${new Date(a.created_at).toLocaleString()}`)
            : ["No Model 2 (anomaly) history for this account."] });
          break;
        }
        case "investigation-log": {
          const match = anomalies.find((a) => a.account_id === acct && a.investigation_log.length > 0);
          setToolkitResult({ tool: item.label, lines: match
            ? match.investigation_log.map((l) => `${new Date(l.ts * 1000).toLocaleString()} -- ${l.by} -- ${l.status}${l.note ? `: ${l.note}` : ""}`)
            : ["No investigation log entries on file for this account."] });
          break;
        }
        case "activity-history": {
          const rows = activity.filter((a) => a.account_id === acct);
          setToolkitResult({ tool: item.label, lines: rows.length
            ? rows.slice(0, 20).map((a) => `${new Date(a.ts * 1000).toLocaleString()} -- [${a.pipeline}] ${a.kind}: ${a.detail}`)
            : ["No recent activity-feed entries for this account (activity feed only covers this session's loaded window)."] });
          break;
        }
        case "explain-summary": {
          const data = await getToolkitExplain(acct);
          const txVolume = data.edges.reduce((s, e) => s + e.amount, 0);
          const crossBorder = data.edges.filter((e) => e.cross_border).length;
          setToolkitResult({ tool: item.label, lines: [
            `${data.nodes.length} account(s), ${data.edges.length} transaction(s) in this account's graph neighborhood`,
            `Total transaction volume shown: ${txVolume.toLocaleString(undefined, { maximumFractionDigits: 0 })}`,
            `Cross-border: ${crossBorder} of ${data.edges.length} transaction(s)`,
          ] });
          break;
        }
        case "counterparties": {
          const data = await getToolkitExplain(acct);
          const others = new Set<string>();
          data.edges.forEach((e) => { if (e.src !== acct) others.add(e.src); if (e.dst !== acct) others.add(e.dst); });
          setToolkitResult({ tool: item.label, lines: others.size
            ? [`${others.size} distinct counterparties visible in the graph:`, ...Array.from(others).slice(0, 25)]
            : ["No counterparties found in the graph for this account."] });
          break;
        }
        case "explain-features": {
          const data = await getToolkitExplain(acct);
          setToolkitResult({ tool: item.label, lines: data.top_features.length
            ? data.top_features.map((f) => `${f.feature} -- weight ${f.weight.toFixed(3)}`)
            : ["No top features returned by the explainer for this account."] });
          break;
        }
        default:
          break;
      }
    } catch (e) {
      setToolkitResult({ tool: item.label, lines: [`Couldn't run this tool: ${errorMessage(e)}`] });
    } finally {
      setToolkitRunningId("");
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

  const anomalyOpenCount = anomalies.filter((a) => a.status === "new" || a.status === "under_investigation").length;

  const tabs: DockTab[] = [
    { id: "overview", label: "Overview", icon: <IconHome />, group: "Monitoring", desc: "Live counts & system pulse" },
    { id: "alerts", label: "Alerts", icon: <IconBell />, badge: stats.open, group: "Monitoring", desc: "Flagged accounts awaiting review" },
    { id: "activity", label: "Activity", icon: <IconList />, group: "Monitoring", desc: "Full pipeline event log" },
    { id: "scan", label: "Scan", icon: <IconRadar />, group: "Detection", desc: "Score a single account on demand" },
    { id: "intake", label: "Intake", icon: <IconUpload />, group: "Detection", desc: "Bulk-screen pasted or uploaded accounts" },
    { id: "pipeline", label: "Pipeline", icon: <IconFlow />, group: "Detection", desc: "How Models 1 & 2 process a transaction" },
    { id: "investigate", label: "Investigate", icon: <IconSearch />, badge: anomalyOpenCount, group: "Investigation", desc: "Explainability & SAR toolkit" },
    { id: "network", label: "Network", icon: <IconGraph />, group: "Investigation", desc: "Transaction graph & feature weights" },
    { id: "connect", label: "Connect", icon: <IconPlug />, badge: sysStatus && !sysStatus.all_ok ? 1 : undefined, group: "System", desc: "Live feed status & bank enrollment" },
  ];

  const shown = alerts.filter((a) => (filter === "all" || a.status === filter)
                                    && (bankFilter === "all" || a.source_bank === bankFilter));
  const shownAnomalies = anomalies.filter((a) => anomalyFilter === "all" || a.status === anomalyFilter);
  const ANOMALY_FILTERS: (AnomalyStatus | "all")[] = ["all", "new", "under_investigation", "confirmed_mule", "cleared"];
  const latencyTarget = modelInfo?.targets.latency_ms ?? 100;

  /* ---- views ---- */
  const overview = (
    <>
      <div className="hero-grid">
        <section className="hero">
          <span className="eyebrow">Explainable AML · FinCEN-ready audit trail</span>
          <h1>
            <span className="grad-text">Follow the money.</span>
            <br />
            Explain every flag.
          </h1>
          <p className="lead">
            Graph neural networks spot laundering typologies across multi-hop transaction networks, and every
            alert comes with a human-readable SAR narrative and a tamper-proof audit record.
          </p>
          <div className="row">
            <button className="btn" onClick={() => setTab("scan")}>Scan an account</button>
            <button className="btn ghost" onClick={() => setTab("alerts")}>Review alerts</button>
          </div>
        </section>

        <div className="stats-side">
          <Stat label="Total alerts" value={stats.total} tone="blue" hint="all time" icon={<IconBell />} />
          <Stat label="Awaiting review" value={stats.open} tone="amber" hint="open cases" icon={<IconList />} />
          <Stat label="Confirmed" value={stats.confirmed} tone="red" hint="marked suspicious" icon={<IconShield />} />
          <Stat label="Average risk" value={stats.avg * 100} suffix="%" tone="cyan" hint="across alerts" icon={<IconRadar />} />
          <Stat label="Anomalies (Model 2)" value={anomalyOpenCount} tone="amber" hint="awaiting investigation" icon={<IconGraph />} />
        </div>
      </div>

      <div className="two reverse">
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
        <ModelCard info={modelInfo} />
      </div>

      <Glass>
        <h3>Detection pipeline</h3>
        <Pipeline />
      </Glass>
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
            {/* A tenant session (authedBank set) only ever sees its own bank's alerts no matter what this
               dropdown says -- server-scoped, not just filtered client-side -- so it adds nothing and is
               hidden rather than shown as a no-op control. */}
            {!authedBank && banks.length > 0 && (
              <select className="input" value={bankFilter} onChange={(e) => setBankFilter(e.target.value)}
                     style={{ width: "auto" }} aria-label="Filter by bank">
                <option value="all">All banks</option>
                {banks.map((b) => <option key={b} value={b}>{b}</option>)}
              </select>
            )}
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
              <span className="pill neutral" title="Which bank this account's data came from">{a.source_bank}</span>
              {a.typology && <span className="pill neutral">{niceType(a.typology)}</span>}
              {a.identity_status && a.identity_status !== "unconfigured" && (
                <span
                  className={`pill ${a.identity_status === "match" ? "ok" : "s-confirmed"}`}
                  title={a.identity_detail ?? ""}
                >
                  ID: {a.identity_status.replace("_", " ")}
                </span>
              )}
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

  const investigateView = (
    <>
      <Glass>
        <h3>Investigate any account</h3>
        <p className="muted" style={{ marginBottom: 12 }}>
          These tools work on any account, whether or not Model 2 has flagged it -- a tip, a call-in, or
          something outside this system entirely still deserves the same toolkit.
        </p>
        <div className="row">
          <input
            className="input grow"
            placeholder="Account ID, e.g. ACC0000123"
            value={lookupId}
            onChange={(e) => setLookupId(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") void doLookup();
            }}
          />
          <button className="btn shine" disabled={!lookupId.trim() || lookupResult === "pending"} onClick={() => void doLookup()}>
            {lookupResult === "pending" ? "Checking…" : "Cross-check with Model 1"}
          </button>
          <button className="btn ghost" disabled={!lookupId.trim()} onClick={() => void doExplain(lookupId.trim())}>
            View network →
          </button>
          <button className="btn ghost" disabled={!lookupId.trim()} onClick={() => copyAccountId(lookupId.trim())}>
            Copy ID
          </button>
        </div>
        {lookupResult && lookupResult !== "pending" && (
          lookupResult === "error" ? (
            <p className="muted small" style={{ marginTop: 10 }}>Couldn&apos;t reach Model 1 for that account.</p>
          ) : (
            <p className="muted small" style={{ marginTop: 10 }}>
              Model 1 says: {lookupResult.flagged ? "unusual" : "looks normal"} · risk score {pct(lookupResult.score, 1)}
              {lookupResult.alert ? ` · alert #${lookupResult.alert.id} created/exists` : " · no alert"}
            </p>
          )
        )}
        <button className="link" style={{ marginTop: 10 }} onClick={() => setToolkitOpen((o) => !o)}>
          {toolkitOpen ? "Hide full investigation toolkit ▲" : "Open full investigation toolkit ▼"}
        </button>
      </Glass>

      {toolkitOpen && (
        <Glass>
          <h3>Investigation toolkit</h3>
          <p className="muted" style={{ marginBottom: 10 }}>
            The checklist a real mule-account investigation runs through. The ▶ tools run for real, right here,
            against this account -- backed by Model 1&apos;s live scan, the GNNExplainer graph, and this
            session&apos;s own alert/anomaly/activity data, nothing mocked; the OSINT/sanctions ↗ links open real
            public lookups in a new tab; everything else needs a system this benchmark project has no live
            connection to (core banking, KYC storage, credit bureaus, SWIFT, law enforcement databases) -- check
            it off once you&apos;ve done it elsewhere, and it still gets written into the case file below.
          </p>
          <div className="row">
            <input
              className="input grow"
              placeholder="Account holder name or search term, for the OSINT/registry links (optional)"
              value={toolkitQuery}
              onChange={(e) => setToolkitQuery(e.target.value)}
            />
          </div>
          <div className="toolkit-grid">
            {TOOLKIT_SECTIONS.map((section) => (
              <div className="toolkit-section" key={section.title}>
                <h4>{section.title}</h4>
                {section.items.map((item) => (
                  <label key={item.id} className="toolkit-item">
                    {item.kind === "checklist" && (
                      <input
                        type="checkbox"
                        checked={!!toolkitChecks[item.id]}
                        onChange={(e) => setToolkitChecks((c) => ({ ...c, [item.id]: e.target.checked }))}
                      />
                    )}
                    {item.kind === "action" && (
                      <button
                        type="button"
                        className="toolkit-action"
                        disabled={toolkitRunningId === item.id}
                        onClick={() => void runToolkitTool(item)}
                      >
                        {toolkitRunningId === item.id ? "…" : "▶"}
                      </button>
                    )}
                    {item.kind === "link" && (
                      <button
                        type="button"
                        className="toolkit-action"
                        onClick={() => window.open(item.url!(toolkitQuery || lookupId), "_blank", "noopener,noreferrer")}
                      >
                        ↗
                      </button>
                    )}
                    <span>{item.label}</span>
                  </label>
                ))}
              </div>
            ))}
          </div>
          {toolkitResult && (
            <div className="toolkit-result">
              <b>{toolkitResult.tool}</b>
              {toolkitResult.lines.length === 0 ? (
                <p className="muted small">No result.</p>
              ) : (
                <ul>
                  {toolkitResult.lines.map((line, i) => <li key={i}>{line}</li>)}
                </ul>
              )}
            </div>
          )}
          <textarea
            className="input"
            style={{ width: "100%", marginTop: 12 }}
            rows={3}
            placeholder="Findings from the checked items above…"
            value={toolkitFindings}
            onChange={(e) => setToolkitFindings(e.target.value)}
          />
          <div className="row" style={{ marginTop: 10 }}>
            <button
              className="btn shine"
              disabled={!lookupId.trim()}
              onClick={() => void doSaveChecklist(lookupId.trim())}
            >
              Save checklist to case file
            </button>
          </div>
        </Glass>
      )}

      <Glass>
        <p className="muted">
          Model 2&apos;s alerts below -- accounts Model 1 currently calls normal, but that look statistically
          unlike the rest of that same population. Nothing here is auto-decided: claim one to start
          investigating, document what you find, then resolve it. Only a &quot;confirmed mule&quot; resolution
          teaches the detection model this pattern; a cleared one is kept only as evidence.
        </p>
      </Glass>
      <Glass>
        <div className="row between">
          <div className="seg">
            {ANOMALY_FILTERS.map((f) => (
              <button key={f} className={anomalyFilter === f ? "on" : ""} onClick={() => setAnomalyFilter(f)}>
                {f.replaceAll("_", " ")}
              </button>
            ))}
          </div>
          <button className="btn ghost" onClick={() => void loadAnomalies()}>Refresh</button>
        </div>
      </Glass>

      {shownAnomalies.length === 0 && (
        <Glass>
          <p className="muted">No anomaly alerts in this view yet -- Model 2&apos;s next scan pass may still be running. The lookup tools above work regardless.</p>
        </Glass>
      )}
      {shownAnomalies.map((a, i) => (
        <Glass className="alert-card rise" key={a.id} style={{ animationDelay: `${Math.min(i, 8) * 60}ms` }}>
          <div className="row between">
            <div className="row">
              <span className="alert-id">#{a.id}</span>
              <b>{a.account_id}</b>
              <span className={`pill s-${a.status === "confirmed_mule" ? "confirmed" : a.status === "cleared" ? "dismissed" : a.status === "under_investigation" ? "open" : "neutral"}`}>
                {a.status.replaceAll("_", " ")}
              </span>
              {a.assigned_to && <span className="pill neutral">assigned: {a.assigned_to}</span>}
            </div>
            <div className="muted small">
              anomaly score {a.anomaly_score.toFixed(4)} · threshold {a.threshold.toFixed(4)}
            </div>
          </div>

          <div className="row actions investigator-tools">
            <button className="chip" onClick={() => void doExplain(a.account_id)}>View network →</button>
            <button className="chip" disabled={crossCheck[a.id] === "pending"} onClick={() => void doCrossCheck(a.id, a.account_id)}>
              {crossCheck[a.id] === "pending" ? "Checking Model 1…" : "Cross-check with Model 1"}
            </button>
            <button className="chip" onClick={() => copyAccountId(a.account_id)}>Copy account ID</button>
          </div>
          {crossCheck[a.id] && crossCheck[a.id] !== "pending" && (
            crossCheck[a.id] === "error" ? (
              <p className="muted small">Model 1 cross-check failed -- try again.</p>
            ) : (
              <p className="muted small">
                Model 1 says: {(crossCheck[a.id] as ScanResult).flagged ? "also unusual" : "looks normal to it"} ·
                {" "}risk score {pct((crossCheck[a.id] as ScanResult).score, 1)}
                {(crossCheck[a.id] as ScanResult).alert ? ` · alert #${(crossCheck[a.id] as ScanResult).alert!.id} already exists there` : " · no Model 1 alert exists"}
              </p>
            )
          )}

          {a.investigation_log.length > 0 && (
            <div className="narr" style={{ display: "flex", flexDirection: "column", gap: 8 }}>
              {a.investigation_log.map((entry, idx) => (
                <div key={idx} className="small">
                  <b>{entry.by}</b> <span className="muted">· {new Date(entry.ts * 1000).toLocaleString()} · {entry.status.replaceAll("_", " ")}</span>
                  <div>{entry.note}</div>
                </div>
              ))}
            </div>
          )}

          <div className="row actions">
            {a.status === "new" && (
              <button className="btn" disabled={busy === `claim${a.id}`} onClick={() => void doClaim(a.id)}>
                {busy === `claim${a.id}` ? "Claiming…" : "Claim for investigation"}
              </button>
            )}
            {a.status === "under_investigation" && (
              <div style={{ display: "flex", flexDirection: "column", gap: 10, width: "100%" }}>
                <div className="row chips">
                  <span className="muted small">Quick notes:</span>
                  {NOTE_TEMPLATES.map((t) => (
                    <button key={t} className="chip" onClick={() => setNoteDraft((d) => ({ ...d, [a.id]: t }))}>
                      {t.length > 34 ? `${t.slice(0, 34)}…` : t}
                    </button>
                  ))}
                </div>
                <div className="row">
                  <input
                    className="input grow"
                    placeholder="Investigation note (e.g. called the account holder, cross-checked related accounts...)"
                    value={noteDraft[a.id] ?? ""}
                    onChange={(e) => setNoteDraft((d) => ({ ...d, [a.id]: e.target.value }))}
                  />
                  <button className="btn ghost" disabled={busy === `note${a.id}` || !(noteDraft[a.id] ?? "").trim()} onClick={() => void doAddNote(a.id)}>
                    Add note
                  </button>
                </div>
                <div className="row">
                  <button className="btn red" disabled={busy === `resolve${a.id}`} onClick={() => void doResolveAnomaly(a.id, "confirmed_mule")}>
                    Confirm mule
                  </button>
                  <button className="btn green" disabled={busy === `resolve${a.id}`} onClick={() => void doResolveAnomaly(a.id, "cleared")}>
                    Clear
                  </button>
                  <span className="grow" />
                  <button className="btn ghost" disabled={busy === `escalate${a.id}`} onClick={() => void doEscalate(a.id)}>
                    {busy === `escalate${a.id}` ? "Escalating…" : "Escalate to compliance / police"}
                  </button>
                </div>
              </div>
            )}
            {a.status === "confirmed_mule" && (
              <span className="muted small">
                {a.trained_at ? "Taught to the detection model." : "Queued to teach the detection model (next sync cycle)."}
              </span>
            )}
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

  const pipelineView = (
    <>
      <Glass>
        <h3>Model 1 · known-typology classifier (baseline)</h3>
        <p className="muted">
          Every scan runs through here first. Trained on labelled laundering typologies, it flags the patterns
          it already knows -- the obvious cases. Its own alerts never retrain it; only a confirmed Model 2
          discovery does (see below), which is a deliberate choice: everyday confirm/dismiss clicks on already-
          obvious cases would make the model drift with reviewer habits, not with genuinely new information.
        </p>
        <Pipeline />
      </Glass>

      <div className="two">
        <Glass>
          <h3>Model 2 · deep anomaly detector (blind-spot watcher)</h3>
          <p className="muted">
            A separate graph autoencoder, trained only on accounts Model 1 currently calls normal -- it learns
            what "normal" looks like in depth, then flags anything that stops fitting that pattern, however
            small. It scores only Model 1's blind spot, so a pattern Model 1 already catches never reaches it.
          </p>
          <div className="mini-grid">
            <div><span>Open anomalies</span><b>{anomalyOpenCount}</b></div>
            <div><span>Confirmed mules found</span><b>{anomalies.filter((a) => a.status === "confirmed_mule").length}</b></div>
            <div><span>Cleared (false alarms)</span><b>{anomalies.filter((a) => a.status === "cleared").length}</b></div>
            <div><span>Taught back to Model 1</span><b>{anomalies.filter((a) => a.trained_at !== null).length}</b></div>
          </div>
        </Glass>
        <Glass>
          <h3>The feedback loop</h3>
          <ol className="loop-steps">
            <li>Model 2 flags an account Model 1 called normal.</li>
            <li>An analyst claims it on the Investigate tab and actually investigates (account history, related
              accounts, contacting the owner, escalation -- outside this system's reach).</li>
            <li>Resolved <b>confirmed mule</b> -- the pattern is fed into a short, warm-started fine-tune of
              Model 1 (never a retrain from scratch), gated so it only ever goes live if it measurably beats
              the model currently in production.</li>
            <li>Resolved <b>cleared</b> -- logged as evidence only, never used for training.</li>
          </ol>
          <p className="muted" style={{ marginTop: 10 }}>
            This is the only path that changes Model 1. Routine officer decisions on Model 1&apos;s own alerts
            (the Alerts tab) are audit evidence only and never retrain anything.
          </p>
        </Glass>
      </div>
    </>
  );

  const intakeView = (
    <>
      <Glass>
        <h3>Add real accounts (ingest transactions)</h3>
        <p className="muted">
          Paste or upload a document of real transactions -- a CSV with <code>src,dst,amount</code> columns
          (<code>timestamp</code> and <code>cross_border</code> are optional), or the same four/five columns
          with no header, one transaction per line. These accounts do not need to already exist: this adds
          them to the live graph for the first time, standardized with the already-trained model's own
          normalization stats and scored by the exact same frozen weights as every other account -- nothing
          is retrained. The trained model (<code>gnn_model.pt</code>) is never modified by this.
        </p>
        <label className="muted" style={{ display: "block", marginBottom: 6 }}>Bank / institution</label>
        <input
          className="input"
          placeholder="e.g. Equity Bank Kenya"
          value={authedBank ?? ingestBank}
          onChange={(e) => setIngestBank(e.target.value)}
          disabled={ingestRunning || !!authedBank}
          title={authedBank ? "Your session is signed in as this bank -- every account you add here is attributed to it, no matter what this field says." : undefined}
          style={{ marginBottom: 10 }}
        />
        <textarea
          className="input intake-textarea"
          placeholder={"src,dst,amount,timestamp,cross_border\nACC1001,ACC2002,4800,,0\nACC2002,ACC3003,15200,,1"}
          value={ingestText}
          onChange={(e) => setIngestText(e.target.value)}
          disabled={ingestRunning}
          rows={8}
        />
        <div className="row" style={{ marginTop: 10 }}>
          <label className="btn ghost" style={{ cursor: "pointer" }}>
            Upload file
            <input
              type="file"
              accept=".csv,.txt"
              style={{ display: "none" }}
              onChange={(e) => {
                const f = e.target.files?.[0];
                if (f) onIngestFile(f);
                e.target.value = "";
              }}
            />
          </label>
          <button className="btn shine" disabled={!ingestText.trim() || !(authedBank || ingestBank.trim()) || ingestRunning} onClick={() => void doIngest()}>
            {ingestRunning ? "Ingesting…" : "Add to live graph"}
          </button>
          {ingestText.trim() && !ingestRunning && (
            <span className="muted">{parseIntakeTx(ingestText).length} transaction row(s) detected</span>
          )}
        </div>
      </Glass>

      {ingestResult && (
        <Glass>
          <h3>Live graph updated{ingestResult.bank ? ` -- ${ingestResult.bank}` : ""}</h3>
          <div className="row" style={{ gap: 18, flexWrap: "wrap" }}>
            <div><b>{ingestResult.ingested_rows}</b><div className="muted">transactions ingested this batch</div></div>
            <div><b>{ingestResult.n_accounts}</b><div className="muted">accounts now live (total)</div></div>
            <div><b>{ingestResult.n_transactions}</b><div className="muted">transactions now live (total)</div></div>
          </div>
          <p className="muted" style={{ marginTop: 10 }}>
            These accounts can now be scanned from the Scan tab, investigated, or watched for anomalies --
            same as any other account the graph knows about.
          </p>
        </Glass>
      )}

      {authedBank && (
        <Glass>
          <h3>Clear temporary data</h3>
          <p className="muted">
            For a walk-in customer who pasted a document for a one-off scan -- not someone you want living in
            the shared graph indefinitely. This removes every account tagged to <b>{authedBank}</b> from the
            live graph: they stop showing up in the network view and can&apos;t be scanned afterward. Any
            alert already raised for {authedBank}, and its decision history, is kept exactly as it is --
            this only clears working data, never the audit trail.
          </p>
          <div className="row" style={{ marginTop: 10 }}>
            <button className="btn ghost" disabled={clearRunning} onClick={() => void doClear()}>
              {clearRunning ? "Clearing…" : `Clear ${authedBank}'s live data`}
            </button>
          </div>
          {clearResult && (
            <div className="row" style={{ gap: 18, flexWrap: "wrap", marginTop: 14 }}>
              <div><b>{clearResult.removed_accounts}</b><div className="muted">account(s) removed</div></div>
              {clearResult.remaining_live_accounts !== undefined && (
                <div><b>{clearResult.remaining_live_accounts}</b><div className="muted">accounts still live (total)</div></div>
              )}
            </div>
          )}
        </Glass>
      )}

      <Glass>
        <h3>Screen a batch of accounts</h3>
        <p className="muted">
          Paste a list of account IDs (one per line, or a CSV with an <code>account_id</code> column) from a
          bank extract, or upload a .csv/.txt file below -- pasting the same <code>src,dst,amount,…</code>
          transaction CSV the ingestion card above takes also works here, and only its account columns are
          used. Each account runs through the real Model 1 scan -- the same check the Scan tab does one at a
          time -- so a whole extract can be screened in one sitting instead of typed in by hand. This screens
          accounts already in the live graph; use the ingestion card above first for accounts that aren't
          live yet.
        </p>
        <textarea
          className="input intake-textarea"
          placeholder={"ACC0000123\nACC0000456\nACC0000789\n…or paste a CSV with an account_id column"}
          value={intakeText}
          onChange={(e) => setIntakeText(e.target.value)}
          disabled={intakeRunning}
          rows={8}
        />
        <div className="row" style={{ marginTop: 10 }}>
          <label className="btn ghost" style={{ cursor: "pointer" }}>
            Upload file
            <input
              type="file"
              accept=".csv,.txt"
              style={{ display: "none" }}
              onChange={(e) => {
                const f = e.target.files?.[0];
                if (f) onIntakeFile(f);
                e.target.value = "";
              }}
            />
          </label>
          <button className="btn shine" disabled={!intakeText.trim() || intakeRunning} onClick={() => void doBatchScan()}>
            {intakeRunning ? `Scanning ${intakeProgress.done}/${intakeProgress.total}…` : "Scan all"}
          </button>
          {intakeText.trim() && !intakeRunning && (
            <span className="muted">{parseIntakeIds(intakeText).length} account ID(s) detected</span>
          )}
        </div>
      </Glass>

      {intakeResults.length > 0 && (
        <Glass>
          <div className="row between">
            <h3>Results ({intakeResults.length})</h3>
            <span className="muted">{intakeResults.filter((r) => r.alert).length} flagged</span>
          </div>
          <div className="intake-results">
            {intakeResults.map((r, i) => (
              <div className={`mini-alert rise${r.alert ? " bad" : ""}`} key={`${r.account_id}-${i}`}>
                <span className={`dot ${r.alert ? "open" : "dismissed"}`} />
                <b>{r.account_id}</b>
                <span className="muted">{r.alert ? `alert #${r.alert.id} created` : "below threshold"}</span>
                <span className="grow" />
                <span className="risk">{pct(r.score, 0)}</span>
              </div>
            ))}
          </div>
        </Glass>
      )}
    </>
  );

  const activityView = (
    <Glass>
      <div className="row between">
        <h3>Activity feed</h3>
        <button className="link" onClick={() => void loadActivity()}>Refresh</button>
      </div>
      <p className="muted" style={{ marginBottom: 12 }}>
        Every scan, decision, claim, note and resolution across both pipelines, newest first -- so an analyst
        can see what happened without opening a terminal or a database client.
      </p>
      {activity.length === 0 && <p className="muted">Nothing yet.</p>}
      <div className="activity-feed">
        {activity.map((e, i) => (
          <div className="activity-item rise" key={`${e.ref_id}-${e.ts}-${i}`} style={{ animationDelay: `${Math.min(i, 12) * 30}ms` }}>
            <span className={`pipeline-tag ${e.pipeline}`}>{e.pipeline === "model1" ? "M1" : "M2"}</span>
            <div className="activity-body">
              <div className="row">
                <b>{e.account_id}</b>
                <span className="muted">{e.kind.replaceAll("_", " ")}</span>
                {e.by && <span className="muted">· {e.by}</span>}
              </div>
              <div className="muted activity-detail">{e.detail}</div>
            </div>
            <span className="activity-ts">{new Date(e.ts * 1000).toLocaleString()}</span>
          </div>
        ))}
      </div>
    </Glass>
  );

  const connectView = (
    <>
      <Glass>
        <div className="row between">
          <h3>Live system status</h3>
          <button className="link" onClick={() => void loadStatus()}>Refresh</button>
        </div>
        {sysStatus === null && <p className="muted">Couldn&apos;t reach the backend to check status.</p>}
        {sysStatus !== null && (
          <div className="status-grid">
            {sysStatus.services.map((s) => (
              <div key={s.name} className={`status-pill ${s.ok ? "ok" : "bad"}`}>
                <i />
                <div>
                  <b>{s.name}</b>
                  <span className="muted">
                    {s.ok ? (s.latency_ms !== null ? `reachable · ${s.latency_ms}ms` : "reachable") : `unreachable${s.error ? ` (${s.error})` : ""}`}
                  </span>
                </div>
              </div>
            ))}
          </div>
        )}
      </Glass>

      <div className="two">
        <Glass>
          <h3>Option 1 · paste data through the frontend</h3>
          <p className="muted">
            The Intake tab lets an analyst paste or upload a list of account IDs (a CSV extract from a bank,
            say) and screen all of them through Model 1 in one go -- no terminal, no direct database access.
            See the Intake tab to use it.
          </p>
          <button className="btn ghost" onClick={() => setTab("intake")}>Open Intake →</button>
        </Glass>
        <Glass>
          <h3>Option 2 · connect a live transaction feed</h3>
          <p className="muted">
            Wired and running: a background service (<code>live-feed-consumer</code>) consumes the topic
            below, adds every transaction&apos;s accounts to the live graph through the exact same trained,
            frozen model the Intake tab uses, and scores them automatically -- a flagged account raises an
            alert on the Alerts tab, same as a manual scan would. That alert is the only effect this has on
            anything: nothing here can delay, block, or alter a transaction, now or ever -- there is no path
            back to whatever published it, only a one-way read. A bank&apos;s own system publishes onto this
            topic to go live; <code>aml_synth</code>&apos;s own synthetic producer (<code>--to kafka</code>)
            can be pointed at it too, to see the feed work end to end before a real one is connected. Every
            account this brings in is attributed to a bank -- either a <code>bank</code>/<code>institution</code>
            field on the message itself, or <code>LIVE_FEED_DEFAULT_BANK</code> if this deployment is set up
            for one bank's traffic -- so alerts raised from this feed show which bank they came from on the
            Alerts tab, same as a pasted document on the Intake tab does.
          </p>
          {sysStatus !== null && (
            <>
              <div className="mini-grid">
                <div><span>Transactions topic</span><b>{sysStatus.kafka.transactions_topic}</b></div>
                <div><span>Alerts topic</span><b>{sysStatus.kafka.alerts_topic}</b></div>
                <div><span>External listener</span><b>{sysStatus.kafka.external_listener}</b></div>
              </div>
              <p className="muted" style={{ marginTop: 10 }}>{sysStatus.kafka.note}</p>
            </>
          )}
        </Glass>
      </div>
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

      <div className={`shell${demoMode ? " demo" : ""}`}>
        <aside className="sidebar-col">
          <div className="brand">
            <div className="logo"><IconShield /></div>
            <div>
              <div className="brand-name">XAI·AMLBench</div>
              <div className="brand-sub">Compliance Console</div>
            </div>
          </div>
          <Sidebar tabs={tabs} active={tab} onChange={setTab} />
        </aside>

        <div className="app">
          <header className="topbar">
            <div className="module-head">
              <div className="eyebrow-label">Active section</div>
              <h1 className="module-title">{tabs.find((t) => t.id === tab)?.label ?? ""}</h1>
            </div>
            <div className="row" style={{ gap: 14 }}>
              <div className={`status ${online === false ? "off" : online ? "on" : ""}`}>
                <i />
                {online === null ? "Connecting…" : online ? "Backend online" : "Backend offline"}
              </div>
              {authedOfficer && (
                <div className="row officer-badge">
                  {authedBank && <span className="tenant-badge">{authedBank}</span>}
                  <span className="muted">Signed in as {authedOfficer}</span>
                  <button className="btn ghost" onClick={() => router.push("/identity")}>Identity check</button>
                  <button className="btn ghost" onClick={handleLogout}>Sign out</button>
                </div>
              )}
            </div>
          </header>

          <main key={tab} className="view">
            {tab === "overview" && overview}
            {tab === "pipeline" && pipelineView}
            {tab === "scan" && scanView}
            {tab === "intake" && intakeView}
            {tab === "alerts" && alertsView}
            {tab === "investigate" && investigateView}
            {tab === "network" && networkView}
            {tab === "activity" && activityView}
            {tab === "connect" && connectView}
          </main>
        </div>
      </div>
    </>
  );
}
