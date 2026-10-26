"use client";

/** The platform OWNER's backoffice -- a page every bank tenant and legacy analyst session is explicitly kept
 * out of (see this page's own mount-time whoami check, and page.tsx's redirect of any admin session away
 * from the bank console and onto this one instead). This is deliberately a separate route, not a tab inside
 * the customer console: a customer (bank tenant) must never be one click away from platform-wide tenant
 * volumes, even if every data call were correctly scoped server-side -- the UI itself should make that
 * mistake structurally impossible, not just policy-impossible.
 *
 * Everything here reads from GET /admin/tenants (403s server-side for anything but an admin-role token --
 * see backend/main.py) plus the existing platform-wide /system/status, which was already bank-agnostic. */

import { useCallback, useEffect, useState } from "react";
import type { ReactElement } from "react";
import { useRouter } from "next/navigation";
import Background from "@/components/Background";
import Glass from "@/components/Glass";
import LoginScreen from "@/components/LoginScreen";
import { api, clearApiKey, errorMessage, hasStoredApiKey } from "@/lib/api";
import type {
  InviteUserResponse, IntakeClearResponse, NewBankResponse, PersonUser, SignupRequest, SystemStatus,
  TenantSummary, WhoamiResponse,
} from "@/types";

function relativeTime(iso: string | null): string {
  if (!iso) return "no activity yet";
  const ms = Date.now() - new Date(iso).getTime();
  const mins = Math.round(ms / 60000);
  if (mins < 1) return "just now";
  if (mins < 60) return `${mins}m ago`;
  const hours = Math.round(mins / 60);
  if (hours < 24) return `${hours}h ago`;
  return `${Math.round(hours / 24)}d ago`;
}

export default function OwnerBackoffice(): ReactElement {
  const router = useRouter();
  const [authChecked, setAuthChecked] = useState<boolean>(false);
  const [authedOfficer, setAuthedOfficer] = useState<string | null>(null);
  const [deniedRole, setDeniedRole] = useState<boolean>(false);   // signed in, but as a bank/analyst -- not admin
  const [tenants, setTenants] = useState<TenantSummary[] | null>(null);
  const [status, setStatus] = useState<SystemStatus | null>(null);
  const [loadError, setLoadError] = useState<string>("");
  const [requests, setRequests] = useState<SignupRequest[] | null>(null);

  // New-bank form + the "shown once" new passkey reveal
  const [newBankName, setNewBankName] = useState<string>("");
  const [newBankEmail, setNewBankEmail] = useState<string>("");
  const [addingBank, setAddingBank] = useState<boolean>(false);
  const [addBankError, setAddBankError] = useState<string>("");
  const [revealed, setRevealed] = useState<NewBankResponse | null>(null);
  const [busyBank, setBusyBank] = useState<string>("");   // which tenant row has a suspend/remove call in flight
  const [busyRequest, setBusyRequest] = useState<number | null>(null);
  const [clearedBank, setClearedBank] = useState<IntakeClearResponse | null>(null);

  // People (per-person accounts) -- invite form + the "shown once" invite link reveal
  const [users, setUsers] = useState<PersonUser[] | null>(null);
  const [newUserName, setNewUserName] = useState<string>("");
  const [newUserEmail, setNewUserEmail] = useState<string>("");
  const [newUserBank, setNewUserBank] = useState<string>("");       // "" = platform admin, no bank
  const [newUserRole, setNewUserRole] = useState<"admin" | "analyst">("analyst");
  const [invitingUser, setInvitingUser] = useState<boolean>(false);
  const [inviteUserError, setInviteUserError] = useState<string>("");
  const [revealedInvite, setRevealedInvite] = useState<InviteUserResponse | null>(null);
  const [busyUser, setBusyUser] = useState<number | null>(null);

  const checkAuth = useCallback((): void => {
    api<WhoamiResponse>("/whoami")
      .then((who) => {
        if (who.role !== "admin") {
          // A bank or legacy-analyst passkey landed here directly (typed the URL, bookmarked it, whatever) --
          // this console has nothing for them, so bounce back to the one they actually belong on rather than
          // silently rendering an empty owner page that only hints at what it would have shown.
          clearApiKey();
          setDeniedRole(true);
          return;
        }
        setAuthedOfficer(who.officer);
      })
      .catch(() => clearApiKey())
      .finally(() => setAuthChecked(true));
  }, []);

  useEffect(() => {
    if (hasStoredApiKey()) checkAuth();
    else setAuthChecked(true);
  }, [checkAuth]);

  const load = useCallback(async (): Promise<void> => {
    try {
      const [t, s, r, u] = await Promise.all([
        api<TenantSummary[]>("/admin/tenants"),
        api<SystemStatus>("/system/status"),
        api<SignupRequest[]>("/admin/signup-requests?status=pending"),
        api<PersonUser[]>("/admin/users"),
      ]);
      setTenants(t);
      setStatus(s);
      setRequests(r);
      setUsers(u);
      setLoadError("");
    } catch (e) {
      setLoadError(errorMessage(e));
    }
  }, []);

  const handleAddBank = useCallback(async (): Promise<void> => {
    const name = newBankName.trim();
    if (!name) return;
    setAddingBank(true);
    setAddBankError("");
    try {
      const res = await api<NewBankResponse>("/admin/banks", {
        method: "POST",
        body: JSON.stringify({ name, contact_email: newBankEmail.trim() || null }),
      });
      setRevealed(res);
      setNewBankName("");
      setNewBankEmail("");
      void load();
    } catch (e) {
      setAddBankError(errorMessage(e));
    } finally {
      setAddingBank(false);
    }
  }, [newBankName, newBankEmail, load]);

  const handleSetBankStatus = useCallback(async (bank: string, status: "active" | "suspended"): Promise<void> => {
    setBusyBank(bank);
    try {
      await api(`/admin/banks/${encodeURIComponent(bank)}`, { method: "PATCH", body: JSON.stringify({ status }) });
      void load();
    } catch (e) {
      setLoadError(errorMessage(e));
    } finally {
      setBusyBank("");
    }
  }, [load]);

  const handleRegeneratePasskey = useCallback(async (bank: string): Promise<void> => {
    if (!window.confirm(
      `Generate a new passkey for ${bank}? Their OLD passkey stops working the instant this runs -- anyone ` +
      `still using it (including you, if you lost it) will need the new one shown next.`
    )) return;
    setBusyBank(bank);
    try {
      const res = await api<NewBankResponse>(`/admin/banks/${encodeURIComponent(bank)}/regenerate-passkey`, {
        method: "POST",
      });
      setRevealed(res);
    } catch (e) {
      setLoadError(errorMessage(e));
    } finally {
      setBusyBank("");
    }
  }, []);

  const handleRemoveBank = useCallback(async (bank: string): Promise<void> => {
    if (!window.confirm(
      `Remove ${bank}'s passkey entirely? This bank can no longer sign in. Its alerts and decision history ` +
      `are kept -- this doesn't touch any of that, same as suspending, just permanent for the login itself.`
    )) return;
    setBusyBank(bank);
    try {
      await api(`/admin/banks/${encodeURIComponent(bank)}`, { method: "DELETE" });
      void load();
    } catch (e) {
      setLoadError(errorMessage(e));
    } finally {
      setBusyBank("");
    }
  }, [load]);

  const handleInviteUser = useCallback(async (): Promise<void> => {
    const name = newUserName.trim();
    const email = newUserEmail.trim();
    if (!name || !email) return;
    setInvitingUser(true);
    setInviteUserError("");
    try {
      const res = await api<InviteUserResponse>("/admin/users", {
        method: "POST",
        body: JSON.stringify({
          name, email, role: newUserRole, bank: newUserRole === "admin" ? null : (newUserBank.trim() || null),
        }),
      });
      setRevealedInvite(res);
      setNewUserName("");
      setNewUserEmail("");
      setNewUserBank("");
      setNewUserRole("analyst");
      void load();
    } catch (e) {
      setInviteUserError(errorMessage(e));
    } finally {
      setInvitingUser(false);
    }
  }, [newUserName, newUserEmail, newUserBank, newUserRole, load]);

  const handleSetUserStatus = useCallback(async (id: number, status: "active" | "disabled"): Promise<void> => {
    setBusyUser(id);
    try {
      await api(`/admin/users/${id}`, { method: "PATCH", body: JSON.stringify({ status }) });
      void load();
    } catch (e) {
      setLoadError(errorMessage(e));
    } finally {
      setBusyUser(null);
    }
  }, [load]);

  const handleRemoveUser = useCallback(async (id: number, email: string): Promise<void> => {
    if (!window.confirm(`Remove ${email}'s account? They'll no longer be able to sign in individually.`)) return;
    setBusyUser(id);
    try {
      await api(`/admin/users/${id}`, { method: "DELETE" });
      void load();
    } catch (e) {
      setLoadError(errorMessage(e));
    } finally {
      setBusyUser(null);
    }
  }, [load]);

  const handleClearBank = useCallback(async (bank: string): Promise<void> => {
    if (!window.confirm(
      `Clear ${bank}'s live-graph data? This removes every account tagged to ${bank} from the live graph -- ` +
      `they stop showing up in the network view and can't be scanned afterward. Any alert already raised for ` +
      `${bank}, and its decision history, is kept exactly as it is -- this only clears working data, never ` +
      `the audit trail.`
    )) return;
    setBusyBank(bank);
    setClearedBank(null);
    try {
      const res = await api<IntakeClearResponse>("/intake/clear", {
        method: "POST", body: JSON.stringify({ bank }),
      });
      setClearedBank(res);
      void load();
    } catch (e) {
      setLoadError(errorMessage(e));
    } finally {
      setBusyBank("");
    }
  }, [load]);

  const handleApproveRequest = useCallback(async (id: number): Promise<void> => {
    setBusyRequest(id);
    try {
      const res = await api<NewBankResponse>(`/admin/signup-requests/${id}/approve`, { method: "POST" });
      setRevealed(res);
      void load();
    } catch (e) {
      setLoadError(errorMessage(e));
    } finally {
      setBusyRequest(null);
    }
  }, [load]);

  const handleDeclineRequest = useCallback(async (id: number): Promise<void> => {
    setBusyRequest(id);
    try {
      await api(`/admin/signup-requests/${id}/decline`, { method: "POST" });
      void load();
    } catch (e) {
      setLoadError(errorMessage(e));
    } finally {
      setBusyRequest(null);
    }
  }, [load]);

  useEffect(() => {
    if (!authedOfficer) return undefined;
    void load();
    const id = setInterval(() => void load(), 20000);
    return () => clearInterval(id);
  }, [authedOfficer, load]);

  const handleLoginSuccess = useCallback((officerName: string, _bank: string | null, role: "admin" | "analyst"): void => {
    if (role !== "admin") {
      clearApiKey();
      setDeniedRole(true);
      return;
    }
    setDeniedRole(false);
    setAuthedOfficer(officerName);
  }, []);

  const handleLogout = useCallback((): void => {
    clearApiKey();
    setAuthedOfficer(null);
  }, []);

  if (!authChecked) return <Background />;

  if (!authedOfficer) {
    return (
      <>
        <Background />
        <LoginScreen
          onSuccess={handleLoginSuccess}
          endpoint="/admin/login"
          variant="admin"
          subtitle="Owner backoffice"
          hint={
            deniedRole
              ? "A bank/analyst passkey doesn't work here at all -- this door only ever accepts the admin passkey. Wrong page? Your bank console is at the site's root address."
              : "This is the platform owner's view, kept on a separate sign-in from every bank tenant's own door. The admin passkey only ever works here, never on the main site."
          }
        />
      </>
    );
  }

  const totals = (tenants ?? []).reduce(
    (acc, t) => ({
      banks: acc.banks + 1,
      alerts: acc.alerts + t.total_alerts,
      openAlerts: acc.openAlerts + t.open_alerts,
      anomalies: acc.anomalies + t.total_anomalies,
      openAnomalies: acc.openAnomalies + t.open_anomalies,
      accounts: acc.accounts + t.accounts,
    }),
    { banks: 0, alerts: 0, openAlerts: 0, anomalies: 0, openAnomalies: 0, accounts: 0 },
  );

  return (
    <>
      <Background />
      <div className="owner-page">
        <div className="row between" style={{ marginBottom: 18 }}>
          <div>
            <div className="login-title" style={{ fontSize: 22 }}>Owner backoffice</div>
            <div className="muted small">Signed in as {authedOfficer} &middot; platform-wide view, every tenant</div>
          </div>
          <div className="row">
            <button className="btn ghost" onClick={() => router.push("/")}>Open a bank console</button>
            <button className="btn ghost" onClick={() => router.push("/identity")}>Identity check</button>
            <button className="btn ghost" onClick={handleLogout}>Sign out</button>
          </div>
        </div>

        {loadError && (
          <Glass className="alert-card" style={{ marginBottom: 16 }}>
            <span className="pill s-confirmed">Couldn&apos;t load</span> {loadError}
          </Glass>
        )}

        <div className="row" style={{ gap: 14, marginBottom: 20, flexWrap: "wrap" }}>
          <Glass className="alert-card" style={{ minWidth: 140 }}>
            <div className="muted small">Banks onboarded</div>
            <div style={{ fontSize: 26, fontWeight: 700 }}>{totals.banks}</div>
          </Glass>
          <Glass className="alert-card" style={{ minWidth: 140 }}>
            <div className="muted small">Open alerts</div>
            <div style={{ fontSize: 26, fontWeight: 700 }}>{totals.openAlerts}</div>
            <div className="muted small">of {totals.alerts} total</div>
          </Glass>
          <Glass className="alert-card" style={{ minWidth: 140 }}>
            <div className="muted small">Open anomalies</div>
            <div style={{ fontSize: 26, fontWeight: 700 }}>{totals.openAnomalies}</div>
            <div className="muted small">of {totals.anomalies} total</div>
          </Glass>
          <Glass className="alert-card" style={{ minWidth: 140 }}>
            <div className="muted small">Accounts, all banks</div>
            <div style={{ fontSize: 26, fontWeight: 700 }}>{totals.accounts}</div>
          </Glass>
          <Glass className="alert-card" style={{ minWidth: 140 }}>
            <div className="muted small">Platform status</div>
            <div style={{ fontSize: 18, fontWeight: 700 }}>
              {status ? (
                <span className={`pill ${status.all_ok ? "ok" : "s-confirmed"}`}>
                  {status.all_ok ? "All services up" : "Degraded"}
                </span>
              ) : (
                <span className="muted small">checking…</span>
              )}
            </div>
          </Glass>
        </div>

        {revealed && (
          <Glass className="alert-card" style={{ marginBottom: 16, borderColor: "var(--green)" }}>
            <div className="row between">
              <div>
                <span className="pill ok">New passkey for {revealed.name}</span>
                <div style={{ marginTop: 8, fontFamily: "monospace", fontSize: 15, wordBreak: "break-all" }}>
                  {revealed.passkey}
                </div>
                <div className="muted small" style={{ marginTop: 6 }}>
                  Shown once -- copy it now and send it to {revealed.name}. It isn&apos;t stored anywhere you
                  can come back and re-read it from.
                </div>
              </div>
              <button className="btn ghost" onClick={() => setRevealed(null)}>Dismiss</button>
            </div>
          </Glass>
        )}

        <div className="muted small" style={{ marginBottom: 8 }}>Add a bank</div>
        <Glass className="alert-card" style={{ marginBottom: 20 }}>
          <div className="row" style={{ gap: 10, flexWrap: "wrap" }}>
            <input
              className="input"
              placeholder="Bank name"
              value={newBankName}
              onChange={(e) => setNewBankName(e.target.value)}
              disabled={addingBank}
              style={{ flex: "1 1 200px" }}
            />
            <input
              className="input"
              placeholder="Contact email (optional -- for alert notifications)"
              value={newBankEmail}
              onChange={(e) => setNewBankEmail(e.target.value)}
              disabled={addingBank}
              style={{ flex: "1 1 260px" }}
            />
            <button className="btn shine" disabled={addingBank || !newBankName.trim()} onClick={() => void handleAddBank()}>
              {addingBank ? "Adding…" : "Add bank"}
            </button>
          </div>
          {addBankError && <p className="login-error" style={{ marginTop: 8 }}>{addBankError}</p>}
        </Glass>

        {requests !== null && requests.length > 0 && (
          <>
            <div className="muted small" style={{ marginBottom: 8 }}>
              Pending access requests ({requests.length})
            </div>
            {requests.map((r) => (
              <Glass className="alert-card" key={r.id} style={{ marginBottom: 10 }}>
                <div className="row between">
                  <div>
                    <span className="tenant-badge">{r.bank_name}</span>{" "}
                    <span className="muted small">{r.contact_name} &lt;{r.contact_email}&gt;</span>
                    {(r.registration_number || r.country || r.region || r.address || r.website) && (
                      <div className="muted small" style={{ marginTop: 4 }}>
                        {[r.registration_number && `reg# ${r.registration_number}`, r.region, r.country, r.website]
                          .filter(Boolean).join(" · ")}
                        {r.address && <div>{r.address}</div>}
                      </div>
                    )}
                    {r.note && <div className="muted small" style={{ marginTop: 4 }}>&ldquo;{r.note}&rdquo;</div>}
                  </div>
                  <div className="row" style={{ gap: 8 }}>
                    <button className="btn ghost" disabled={busyRequest === r.id}
                      onClick={() => void handleDeclineRequest(r.id)}>
                      Decline
                    </button>
                    <button className="btn shine" disabled={busyRequest === r.id}
                      onClick={() => void handleApproveRequest(r.id)}>
                      {busyRequest === r.id ? "Approving…" : "Approve"}
                    </button>
                  </div>
                </div>
              </Glass>
            ))}
          </>
        )}

        {clearedBank && (
          <Glass className="alert-card" style={{ marginBottom: 16, borderColor: "var(--green)" }}>
            <div className="row between">
              <div>
                <span className="pill ok">Cleared {clearedBank.bank}</span>{" "}
                <span className="muted small">
                  {clearedBank.removed_accounts} account(s) removed from the live graph
                  {clearedBank.remaining_live_accounts !== undefined
                    ? ` -- ${clearedBank.remaining_live_accounts} accounts still live platform-wide` : ""}.
                  Its alerts and decision history are untouched.
                </span>
              </div>
              <button className="btn ghost" onClick={() => setClearedBank(null)}>Dismiss</button>
            </div>
          </Glass>
        )}

        <div className="muted small" style={{ marginBottom: 8 }}>
          Tenants &middot; &ldquo;Clear live data&rdquo; removes a bank&apos;s temporary/walk-in accounts from
          the live graph (not its alerts or history) -- the same control each bank has on its own console,
          available here for every tenant at once.
        </div>
        {tenants === null && !loadError && <Glass className="alert-card">Loading tenants…</Glass>}
        {tenants !== null && tenants.length === 0 && (
          <Glass className="alert-card">
            No banks have been onboarded yet -- add one above, or wait for an access request to approve.
          </Glass>
        )}
        {tenants !== null && tenants.map((t) => (
          <Glass className="alert-card" key={t.bank} style={{ marginBottom: 10 }}>
            <div className="row between">
              <div className="row" style={{ gap: 10 }}>
                <span className="tenant-badge">{t.bank}</span>
                {t.status === "suspended" && <span className="pill s-confirmed">suspended</span>}
                {(t.region || t.country) && (
                  <span className="pill neutral">{[t.region, t.country].filter(Boolean).join(", ")}</span>
                )}
                {t.registration_number && <span className="muted small">reg# {t.registration_number}</span>}
                <span className="muted small">last activity {relativeTime(t.last_activity)}</span>
              </div>
              <div className="row" style={{ gap: 8 }}>
                <span className="pill neutral">{t.accounts} accounts</span>
                <span className={`pill ${t.open_alerts > 0 ? "s-open" : "neutral"}`}>
                  {t.open_alerts} open / {t.total_alerts} alerts
                </span>
                <span className={`pill ${t.open_anomalies > 0 ? "s-open" : "neutral"}`}>
                  {t.open_anomalies} open / {t.total_anomalies} anomalies
                </span>
                {t.status === "active" ? (
                  <button className="btn ghost" disabled={busyBank === t.bank}
                    onClick={() => void handleSetBankStatus(t.bank, "suspended")}>
                    Suspend
                  </button>
                ) : (
                  <button className="btn ghost" disabled={busyBank === t.bank}
                    onClick={() => void handleSetBankStatus(t.bank, "active")}>
                    Reactivate
                  </button>
                )}
                <button className="btn ghost" disabled={busyBank === t.bank || t.accounts === 0}
                  title={t.accounts === 0 ? "No live-graph accounts to clear for this bank" : undefined}
                  onClick={() => void handleClearBank(t.bank)}>
                  {busyBank === t.bank ? "Clearing…" : "Clear live data"}
                </button>
                <button className="btn ghost" disabled={busyBank === t.bank}
                  title="Lost or never saved this bank's passkey? Get a new one -- the old one stops working immediately."
                  onClick={() => void handleRegeneratePasskey(t.bank)}>
                  Regenerate passkey
                </button>
                <button className="btn ghost" disabled={busyBank === t.bank}
                  onClick={() => void handleRemoveBank(t.bank)}>
                  Remove
                </button>
              </div>
            </div>
          </Glass>
        ))}

        {revealedInvite && (
          <Glass className="alert-card" style={{ marginBottom: 16, borderColor: "var(--green)" }}>
            <div className="row between">
              <div>
                <span className="pill ok">Invited {revealedInvite.email}</span>
                <div className="muted small" style={{ marginTop: 6 }}>
                  {revealedInvite.invite_link ? (
                    <>They&apos;ve been emailed a link to set their password. If that email doesn&apos;t
                      arrive, you can also send them this link directly: </>
                  ) : (
                    <>No email provider is configured, so send this link to them yourself: </>
                  )}
                </div>
                <div style={{ marginTop: 4, fontFamily: "monospace", fontSize: 13, wordBreak: "break-all" }}>
                  {revealedInvite.invite_link ?? `/accept-invite?token=${revealedInvite.invite_token}`}
                </div>
              </div>
              <button className="btn ghost" onClick={() => setRevealedInvite(null)}>Dismiss</button>
            </div>
          </Glass>
        )}

        <div className="muted small" style={{ marginBottom: 8 }}>
          Invite a person (individual sign-in, on top of shared passkeys)
        </div>
        <Glass className="alert-card" style={{ marginBottom: 20 }}>
          <div className="row" style={{ gap: 10, flexWrap: "wrap" }}>
            <input
              className="input"
              placeholder="Their name"
              value={newUserName}
              onChange={(e) => setNewUserName(e.target.value)}
              disabled={invitingUser}
              style={{ flex: "1 1 160px" }}
            />
            <input
              className="input"
              type="email"
              placeholder="Their email"
              value={newUserEmail}
              onChange={(e) => setNewUserEmail(e.target.value)}
              disabled={invitingUser}
              style={{ flex: "1 1 220px" }}
            />
            <select
              className="input"
              value={newUserRole}
              onChange={(e) => setNewUserRole(e.target.value as "admin" | "analyst")}
              disabled={invitingUser}
              style={{ flex: "0 0 160px" }}
            >
              <option value="analyst">Analyst (one bank)</option>
              <option value="admin">Owner admin</option>
            </select>
            {newUserRole === "analyst" && (
              <input
                className="input"
                placeholder="Which bank (exact name)"
                value={newUserBank}
                onChange={(e) => setNewUserBank(e.target.value)}
                disabled={invitingUser}
                style={{ flex: "1 1 200px" }}
              />
            )}
            <button
              className="btn shine"
              disabled={invitingUser || !newUserName.trim() || !newUserEmail.trim()
                || (newUserRole === "analyst" && !newUserBank.trim())}
              onClick={() => void handleInviteUser()}
            >
              {invitingUser ? "Inviting…" : "Send invite"}
            </button>
          </div>
          {inviteUserError && <p className="login-error" style={{ marginTop: 8 }}>{inviteUserError}</p>}
        </Glass>

        <div className="muted small" style={{ marginBottom: 8 }}>
          People
          {users !== null && users.some((u) => u.status === "pending") && (
            <> &middot; {users.filter((u) => u.status === "pending").length} analyst sign-up(s) waiting on your
              approval</>
          )}
        </div>
        {users !== null && users.length === 0 && (
          <Glass className="alert-card" style={{ marginBottom: 20 }}>
            Nobody has an individual account yet -- everyone's still using a shared passkey. Invite someone
            above, or point your banks' analysts at <code>/analyst-signup</code> to register themselves.
          </Glass>
        )}
        {users !== null && [...users].sort((a, b) => (a.status === "pending" ? -1 : b.status === "pending" ? 1 : 0)).map((u) => (
          <Glass className="alert-card" key={u.id} style={{ marginBottom: 10 }}>
            <div className="row between">
              <div className="row" style={{ gap: 10 }}>
                <span className="tenant-badge">{u.name}</span>
                <span className="muted small">{u.email}</span>
                <span className="pill neutral">{u.role}{u.bank ? ` · ${u.bank}` : ""}</span>
                <span className={`pill ${u.status === "disabled" ? "s-confirmed" : u.status === "pending" ? "s-open" : u.status === "invited" ? "neutral" : "ok"}`}>
                  {u.status === "pending" ? "pending approval" : u.status}
                </span>
                {u.totp_enabled && <span className="pill ok">2FA on</span>}
              </div>
              <div className="row" style={{ gap: 8 }}>
                {u.status === "pending" ? (
                  <button className="btn shine" disabled={busyUser === u.id}
                    onClick={() => void handleSetUserStatus(u.id, "active")}>
                    {busyUser === u.id ? "Approving…" : "Approve"}
                  </button>
                ) : u.status === "disabled" ? (
                  <button className="btn ghost" disabled={busyUser === u.id}
                    onClick={() => void handleSetUserStatus(u.id, "active")}>
                    Reactivate
                  </button>
                ) : u.status === "active" ? (
                  <button className="btn ghost" disabled={busyUser === u.id}
                    onClick={() => void handleSetUserStatus(u.id, "disabled")}>
                    Disable
                  </button>
                ) : null}
                <button className="btn ghost" disabled={busyUser === u.id}
                  onClick={() => void handleRemoveUser(u.id, u.email)}>
                  {u.status === "pending" ? "Decline" : "Remove"}
                </button>
              </div>
            </div>
          </Glass>
        ))}

        {status && !status.all_ok && (
          <>
            <div className="muted small" style={{ margin: "20px 0 8px" }}>Service health</div>
            {status.services.filter((s) => !s.ok).map((s) => (
              <Glass className="alert-card" key={s.name} style={{ marginBottom: 8 }}>
                <span className="pill s-confirmed">{s.name}</span> {s.error ?? "not responding"}
              </Glass>
            ))}
          </>
        )}
      </div>
    </>
  );
}
