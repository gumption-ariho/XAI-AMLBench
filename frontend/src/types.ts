/** Types shared between the UI and the backend JSON (see backend/main.py and gnn_aml_core/main.py). */

export type AlertStatus = "open" | "confirmed" | "dismissed";
export type TabId = "overview" | "scan" | "alerts" | "network";
export type ToastKind = "ok" | "err" | "warn";

export interface Alert {
  id: number;
  account_id: string;
  risk_score: number;
  status: AlertStatus;
  typology: string | null;
  narrative: string | null;
  narrative_source: string | null;
  narrative_sha256: string | null;
  reviewed_by: string | null;
  review_note: string | null;
  created_at: string;
  reviewed_at: string | null;
}

export interface ScanResponse {
  score: number;
  calibrated_score?: number;
  flagged: boolean;
  threshold: number;
  calibrated_threshold?: number;
  latency_ms: number;
  alert: Alert | null;
}

export interface ScanResult extends ScanResponse {
  account_id: string;
}

export interface SampleAccounts {
  suspicious: string[];
  benign: string[];
}

export interface ModelMetrics {
  val_auc_roc: number;
  test_auc_roc: number;
  test_pr_auc: number;
  test_precision: number;
  test_recall: number;
  threshold: number;
  auc_target: number;
  meets_auc_target: boolean;
}

export interface ModelInfo {
  model: string;
  hparams: Record<string, number>;
  threshold: number;
  metrics: ModelMetrics;
  targets: { auc_roc: number; latency_ms: number };
  n_nodes: number;
  n_features: number;
}

/* ---- GNNExplainer payload returned by POST /explain ---- */
export interface ExplainNode {
  account_id: string;
  account_type: string; // individual | business | shell
  country: string;
}

export interface ExplainEdge {
  tx_id: string;
  src: string;
  dst: string;
  amount: number;
  timestamp: number;
  cross_border: number; // 0 | 1
  importance: number;
}

export interface TopFeature {
  feature: string;
  weight: number;
}

export interface Explanation {
  account_id: string;
  risk_score: number;
  raw_risk_score?: number;
  threshold: number;
  reporting_threshold: number;
  model: string;
  method: string;
  nodes: ExplainNode[];
  edges: ExplainEdge[];
  top_features: TopFeature[];
}

export interface Toast {
  id: number;
  kind: ToastKind;
  msg: string;
}
