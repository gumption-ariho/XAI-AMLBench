/** Types shared between the UI and the backend JSON (see backend/main.py and gnn_aml_core/main.py). */

export type AlertStatus = "open" | "confirmed" | "dismissed";
export type TabId = "overview" | "pipeline" | "scan" | "intake" | "alerts" | "investigate" | "network" | "activity" | "connect";
export type ToastKind = "ok" | "err" | "warn";

/* ---- GET /api/activity -- a merged, real feedback-log feed (see backend/main.py's activity() docstring) ---- */
export interface ActivityEntry {
  ts: number;
  pipeline: "model1" | "model2";
  kind: string;
  account_id: string;
  by: string | null;
  detail: string;
  ref_id: number;
}

/* ---- GET /api/system/status -- live reachability + Kafka wiring info (see backend/main.py) ---- */
export interface ServiceStatus {
  name: string;
  ok: boolean;
  latency_ms: number | null;
  error?: string;
}

export interface SystemStatus {
  services: ServiceStatus[];
  all_ok: boolean;
  kafka: {
    external_listener: string;
    transactions_topic: string;
    alerts_topic: string;
    note: string;
  };
}

export interface WhoamiResponse {
  officer: string;
  /** Set for a bank-tenant session (signed in with that bank's own BANK_PASSKEYS passkey) -- every alert,
   * anomaly, and account this session can see or touch is scoped to this one bank. null for an admin or
   * legacy shared-analyst session, which still sees every bank's data, exactly as before this field existed. */
  bank: string | null;
  /** "admin" routes straight to the owner backoffice (/owner) instead of the bank console -- see the root
   * page's mount-time whoami check. */
  role: "admin" | "analyst";
}

export interface LoginResponse {
  token: string;
  name: string;
  role: "admin" | "analyst";
  bank: string | null;
}

/* ---- Owner backoffice (GET /admin/tenants) -- admin-role sessions only, see backend/main.py ---- */
export interface TenantSummary {
  bank: string;
  status: "active" | "suspended";
  registration_number?: string | null;
  country?: string | null;
  region?: string | null;
  website?: string | null;
  total_alerts: number;
  open_alerts: number;
  total_anomalies: number;
  open_anomalies: number;
  accounts: number;
  last_activity: string | null;
}

export interface NewBankResponse {
  name: string;
  passkey: string;
  status: "active" | "suspended";
}

export interface SignupRequest {
  id: number;
  bank_name: string;
  contact_name: string;
  contact_email: string;
  note: string;
  status: "pending" | "approved" | "declined";
  created_at: string;
  decided_at: string | null;
  // Full registration detail -- see backend/main.py's banks/signup_requests ALTER TABLE comments. All
  // optional/nullable: an older request, or one filled in quickly, may not have every field.
  registration_number?: string | null;
  country?: string | null;
  region?: string | null;
  address?: string | null;
  website?: string | null;
}

export interface PublicStatus {
  all_ok: boolean;
  checked_at: number;
}

export interface BankBranding {
  name: string;
  logo_url: string | null;
  accent_color: string | null;
}

/* ---- Per-person accounts (on top of the shared admin/analyst/bank passkeys) -- see backend/main.py's
   admin_invite_user / auth_login / setup_2fa and friends. ---- */
export interface PersonUser {
  id: number;
  name: string;
  email: string;
  bank: string | null;
  role: "admin" | "analyst";
  // "pending" is a self-registered analyst (POST /auth/signup) awaiting the owner's approval -- see
  // backend/main.py's users table comment. They cannot sign in until an admin flips this to "active".
  status: "invited" | "pending" | "active" | "disabled";
  totp_enabled: boolean;
  created_at: string;
}

export interface InviteUserResponse extends PersonUser {
  invite_token: string;
  invite_link: string | null;
}

/* ---- POST /auth/signup -- the public, self-service analyst registration form (frontend/src/app/
   analyst-signup/page.tsx), the counterpart to the owner-driven invite flow above. ---- */
export interface AnalystSignupResponse {
  email: string;
  bank: string;
  status: "pending";
}

/** The second POST /auth/login call throws this exact string as the error detail when the password was
 * correct but the account has 2FA enabled and no code was sent yet -- the signal to show the code field and
 * resubmit, see frontend/src/app/signin/page.tsx. */
export const TOTP_REQUIRED = "totp_required";

export interface Setup2faResponse {
  secret: string;
  provisioning_uri: string;
}

/* ---- Model 2 (anomaly detector) -- see gnn_aml_core/anomaly.py and backend/main.py's /api/anomalies/* ---- */
export type AnomalyStatus = "new" | "under_investigation" | "confirmed_mule" | "cleared";

export interface InvestigationLogEntry {
  ts: number;
  by: string;
  status: string;
  note: string;
}

export interface AnomalyAlert {
  id: number;
  account_id: string;
  anomaly_score: number;
  threshold: number;
  status: AnomalyStatus;
  assigned_to: string | null;
  investigation_log: InvestigationLogEntry[];
  created_at: string;
  updated_at: string | null;
  trained_at: string | null;
}

export interface Alert {
  id: number;
  account_id: string;
  risk_score: number;
  status: AlertStatus;
  source_bank: string;   // which bank this account's data came from ("unknown" if it predates attribution
                         // or was never tagged -- see backend/main.py's account_banks table)
  typology: string | null;
  narrative: string | null;
  narrative_source: string | null;
  narrative_sha256: string | null;
  reviewed_by: string | null;
  review_note: string | null;
  created_at: string;
  reviewed_at: string | null;
  // Most recent identity_checks result for this account, if ANY has ever been run (manually, or
  // automatically at intake -- see backend/main.py's _auto_identity_checks). null/undefined on an alert
  // whose account has never had an identity check -- that means "nothing checked yet", not "verified clean".
  identity_status?: IdentityCheckStatus | null;
  identity_detail?: string | null;
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

/* ---- POST /intake/ingest -- the Intake tab's real-transaction ingestion path (see backend/main.py and
   gnn_aml_core/livegraph.py). Adds the given transactions to the live, non-synthetic account population and
   rebuilds the served graph standardized with the already-trained model's own mean/std -- no weights change,
   nothing is retrained. ---- */
export interface IntakeIngestResponse {
  status: string;
  ingested_rows: number;
  n_accounts: number;
  n_transactions: number;
  bank?: string;
  model_version?: string;
  [key: string]: unknown;   // _reload_model() may include extra fields (e.g. reload timing); pass through as-is
}

/* ---- POST /api/intake/clear (backend/main.py's intake_clear -- Option 1's "clear temporary data" control,
   gnn_aml_core/livegraph.py's remove_accounts). Removes a bank's accounts from the live graph only --
   alerts/anomaly_alerts and the immudb audit trail are deliberately never touched by this call. ---- */
export interface IntakeClearResponse {
  status: string;
  bank: string;
  removed_accounts: number;
  remaining_live_accounts?: number;
  remaining_live_transactions?: number;
  [key: string]: unknown;
}

/* ---- Identity Check page -- POST /api/identity/check, GET /api/identity/checks (backend/main.py's
   check_identity()). "unconfigured" means no provider is set in .env yet, not that the check failed; see
   backend/main.py's check_identity() docstring for exactly what each status does and doesn't mean -- in
   particular "mismatch"/"deceased" are reasons to follow up with a human, never a verdict this system reaches
   on its own, and NOTHING here can ever confirm whether the real ID owner is aware this account exists. ---- */
export type IdentityCheckStatus = "unconfigured" | "match" | "mismatch" | "not_found" | "deceased" | "provider_error";

export interface IdentityCheckRecord {
  id: number;
  account_id: string;
  bank: string;
  claimed_name: string;
  id_number_last4: string;
  provider: string;
  status: IdentityCheckStatus;
  detail: string;
  raw_response: unknown;
  checked_by: string;
  created_at: string;
}

export interface Toast {
  id: number;
  kind: ToastKind;
  msg: string;
}
