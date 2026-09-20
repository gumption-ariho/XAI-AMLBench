/**
 * Offline demo backend. `installMockApi()` replaces window.fetch for /api/* so the real UI code
 * (src/app/page.tsx and components) can run in a browser with no Docker, Postgres or GNN service.
 *
 * Shapes are checked against src/types.ts, so the mock cannot drift from what the real backend returns.
 */
import type { Alert, AlertStatus, ModelInfo, SampleAccounts, ScanResponse } from "@/types";
import { DEMO_ACCOUNTS, DEMO_EXPLANATIONS, DEMO_NARRATIVES } from "./demoData";

const THRESHOLD = 0.62;
const T0 = Date.now();
const iso = (minutesAgo: number): string => new Date(T0 - minutesAgo * 60_000).toISOString();

const suspicious = DEMO_ACCOUNTS.filter((a) => a.label !== "benign");
const benign = DEMO_ACCOUNTS.filter((a) => a.label === "benign");

const byLabel = (label: string) => DEMO_ACCOUNTS.find((a) => a.label === label);

function makeAlert(
  id: number,
  accountId: string,
  status: AlertStatus,
  minutesAgo: number,
  withNarrative: boolean,
  reviewer: string | null = null,
): Alert {
  const account = DEMO_ACCOUNTS.find((a) => a.id === accountId);
  const narrative = withNarrative ? DEMO_NARRATIVES[accountId] : undefined;
  return {
    id,
    account_id: accountId,
    risk_score: account?.score ?? 0.9,
    status,
    typology: narrative?.typology ?? null,
    narrative: narrative?.text ?? null,
    narrative_source: narrative ? "template" : null,
    narrative_sha256: narrative?.sha256 ?? null,
    reviewed_by: reviewer,
    review_note: reviewer ? "" : null,
    created_at: iso(minutesAgo),
    reviewed_at: reviewer ? iso(Math.max(minutesAgo - 20, 1)) : null,
  };
}

const alerts: Alert[] = [];
const seed = (label: string, status: AlertStatus, mins: number, narr: boolean, who: string | null = null): void => {
  const a = byLabel(label);
  if (a) alerts.push(makeAlert(alerts.length + 1, a.id, status, mins, narr, who));
};
seed("smurfing", "open", 42, true);
seed("shell_company", "open", 95, false);
seed("cyclic_loop", "confirmed", 610, true, "M. Wanjiru");
seed("cross_border_velocity", "dismissed", 1300, true, "M. Wanjiru");
let nextId = alerts.length + 1;

const MODEL_INFO: ModelInfo = {
  model: "gatv2",
  hparams: { hidden: 64, num_layers: 2 },
  threshold: THRESHOLD,
  metrics: {
    val_auc_roc: 0.981, test_auc_roc: 0.983, test_pr_auc: 0.94, test_precision: 0.91, test_recall: 0.94,
    threshold: THRESHOLD, auc_target: 0.87, meets_auc_target: true,
  },
  targets: { auc_roc: 0.87, latency_ms: 100 },
  n_nodes: 5488,
  n_features: 26,
};

interface Reply {
  status: number;
  body: unknown;
  delay: number;
}
const ok = (body: unknown, delay = 220): Reply => ({ status: 200, body, delay });
const fail = (status: number, detail: string, delay = 220): Reply => ({ status, body: { detail }, delay });

function route(method: string, path: string, data: Record<string, unknown>): Reply {
  if (method === "GET" && path === "/alerts") {
    return ok([...alerts].sort((a, b) => b.created_at.localeCompare(a.created_at)));
  }
  if (method === "GET" && path === "/accounts/sample") {
    const body: SampleAccounts = { suspicious: suspicious.map((a) => a.id), benign: benign.map((a) => a.id) };
    return ok(body, 120);
  }
  if (method === "GET" && path === "/model/info") return ok(MODEL_INFO, 150);

  if (method === "POST" && path === "/alerts/scan") {
    const id = String(data.account_id ?? "").trim();
    const account = DEMO_ACCOUNTS.find((a) => a.id === id);
    if (!account) {
      return fail(404, `unknown account_id '${id}'. In demo mode try one of the "Try:" accounts.`);
    }
    const flagged = account.score >= THRESHOLD;
    let alert: Alert | null = null;
    if (flagged) {
      alert = makeAlert(nextId++, account.id, "open", 0, false);
      alerts.push(alert);
    }
    const body: ScanResponse = {
      score: account.score,
      flagged,
      threshold: THRESHOLD,
      latency_ms: Math.round((6 + Math.random() * 12) * 10) / 10,
      alert,
    };
    return ok(body, 260);
  }

  let m = path.match(/^\/accounts\/([^/]+)\/explain$/);
  if (method === "GET" && m) {
    const id = decodeURIComponent(m[1]);
    const explanation = DEMO_EXPLANATIONS[id];
    return explanation ? ok(explanation, 500) : fail(404, `unknown account_id '${id}'`);
  }

  m = path.match(/^\/alerts\/(\d+)\/narrative$/);
  if (method === "POST" && m) {
    const alert = alerts.find((a) => a.id === Number(m?.[1]));
    if (!alert) return fail(404, "alert not found");
    const narrative = DEMO_NARRATIVES[alert.account_id];
    if (!narrative) return fail(502, "no narrative available for this account in demo mode");
    alert.narrative = narrative.text;
    alert.narrative_source = "template"; // the demo has no GPU/LLM, so this is the validated template path
    alert.narrative_sha256 = narrative.sha256;
    alert.typology = narrative.typology;
    return ok(alert, 1300); // pretend the LLM takes a moment
  }

  m = path.match(/^\/alerts\/(\d+)\/decision$/);
  if (method === "POST" && m) {
    const alert = alerts.find((a) => a.id === Number(m?.[1]));
    if (!alert) return fail(404, "alert not found");
    if (alert.status !== "open") return fail(409, `alert already ${alert.status}`);
    const officer = String(data.officer ?? "").trim();
    const decision = data.decision;
    if (officer.length < 2) return fail(422, "officer name is required");
    if (decision !== "confirmed" && decision !== "dismissed") return fail(422, "invalid decision");
    alert.status = decision;
    alert.reviewed_by = officer;
    alert.review_note = String(data.note ?? "");
    alert.reviewed_at = new Date().toISOString();
    return ok(alert, 450);
  }

  return fail(404, `Not found: ${method} ${path}`, 80);
}

export function installMockApi(): void {
  const realFetch = window.fetch.bind(window);
  window.fetch = async (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
    const url = typeof input === "string" ? input : input instanceof URL ? input.pathname : input.url;
    if (!url.startsWith("/api/")) return realFetch(input, init);

    let data: Record<string, unknown> = {};
    if (typeof init?.body === "string") {
      try {
        data = JSON.parse(init.body) as Record<string, unknown>;
      } catch {
        data = {};
      }
    }
    const reply = route((init?.method ?? "GET").toUpperCase(), url.slice(4), data);
    await new Promise<void>((resolve) => setTimeout(resolve, reply.delay));
    return new Response(JSON.stringify(reply.body), {
      status: reply.status,
      headers: { "Content-Type": "application/json" },
    });
  };
}
