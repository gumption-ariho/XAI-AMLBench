"use client";

/** Identity Check -- a SEPARATE page from the main console and the owner backoffice (linked from both), for
 * one question only: does a claimed name + national ID number match a real identity on file with an external
 * KYC/identity-verification provider? This is deliberately NOT the same thing as the OSINT/sanctions link on
 * the Scan tab (a manual click-through to a sanctions list or a web search) -- this page calls a real
 * provider API and records the result. See backend/main.py's check_identity() docstring for exactly what
 * each result status means.
 *
 * What this can tell you:
 *   - the ID number is real and its registered owner's name matches the account ("match")
 *   - the ID number is real but belongs to someone ELSE ("mismatch" -- the classic stolen-ID pattern)
 *   - the ID number has no record at all ("not_found")
 *   - the provider's own registry marks that ID's owner as deceased ("deceased"), IF the configured
 *     provider actually exposes that flag -- some do, some don't, this page never guesses
 * What this can NEVER tell you, no matter which provider is configured: whether the real ID owner is aware
 * this account exists. A mismatch or a deceased result both mean "an officer needs to follow up with the
 * real person on file", not "confirmed fraud" and not "confirmed clean". */

import { useCallback, useEffect, useState } from "react";
import type { FormEvent, ReactElement } from "react";
import { useRouter } from "next/navigation";
import Background from "@/components/Background";
import Glass from "@/components/Glass";
import { IconShield } from "@/components/Icons";
import { api, errorMessage, hasStoredSessionToken } from "@/lib/api";
import type { IdentityCheckRecord } from "@/types";

const STATUS_LABEL: Record<string, string> = {
  unconfigured: "Not verified -- no provider configured",
  match: "Match -- name matches the ID's registered owner",
  mismatch: "Mismatch -- real ID, different real owner",
  not_found: "Not found -- no record of this ID number",
  deceased: "Registered owner marked deceased",
  provider_error: "Provider error -- try again",
};

const STATUS_PILL: Record<string, string> = {
  unconfigured: "pill s-open",
  match: "pill ok",
  mismatch: "pill s-confirmed",
  not_found: "pill s-open",
  deceased: "pill s-confirmed",
  provider_error: "pill s-open",
};

export default function IdentityCheckPage(): ReactElement {
  const router = useRouter();
  const signedIn = hasStoredSessionToken();
  const [accountId, setAccountId] = useState("");
  const [claimedName, setClaimedName] = useState("");
  const [idNumber, setIdNumber] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState<IdentityCheckRecord | null>(null);
  const [history, setHistory] = useState<IdentityCheckRecord[]>([]);

  const loadHistory = useCallback(async (): Promise<void> => {
    try {
      setHistory(await api<IdentityCheckRecord[]>("/identity/checks"));
    } catch {
      // the page still works with no history loaded -- not worth an error banner on a background refresh
    }
  }, []);

  useEffect(() => {
    if (signedIn) void loadHistory();
  }, [signedIn, loadHistory]);

  const submit = async (e: FormEvent): Promise<void> => {
    e.preventDefault();
    if (!accountId.trim() || !claimedName.trim() || !idNumber.trim()) return;
    setBusy(true);
    setError("");
    try {
      const res = await api<IdentityCheckRecord>("/identity/check", {
        method: "POST",
        body: JSON.stringify({ account_id: accountId.trim(), claimed_name: claimedName.trim(), id_number: idNumber.trim() }),
      });
      setResult(res);
      setIdNumber("");          // never leave the raw ID number sitting in the form after a submit
      void loadHistory();
    } catch (e2) {
      setError(errorMessage(e2));
    } finally {
      setBusy(false);
    }
  };

  return (
    <>
      <Background />
      <div className="login-screen">
        <div className="login-card-wrap" style={{ maxWidth: 640 }}>
          <Glass as="div" className="login-card">
            <div className="row between" style={{ marginBottom: 6 }}>
              <div className="login-brand" style={{ marginBottom: 0 }}>
                <span className="login-shield"><IconShield /></span>
                <div>
                  <div className="login-title">Identity check</div>
                  <div className="login-subtitle">Verify an account holder&apos;s claimed ID against a real registry</div>
                </div>
              </div>
              {signedIn && <button className="btn ghost" onClick={() => router.back()}>Back</button>}
            </div>

            {!signedIn ? (
              <p className="login-hint">
                <a href="/">Sign in</a> first -- this page needs a session the same as the rest of the console.
              </p>
            ) : (
              <>
                <form onSubmit={(e) => void submit(e)} style={{ marginTop: 14 }}>
                  <div className="muted small" style={{ marginBottom: 4 }}>Account ID</div>
                  <input className="input login-input" value={accountId} onChange={(e) => setAccountId(e.target.value)}
                         placeholder="e.g. ACC-10234" disabled={busy} />
                  <div className="muted small" style={{ margin: "10px 0 4px" }}>Claimed name on the account</div>
                  <input className="input login-input" value={claimedName} onChange={(e) => setClaimedName(e.target.value)}
                         placeholder="Full name as it appears on the account" disabled={busy} />
                  <div className="muted small" style={{ margin: "10px 0 4px" }}>National ID number</div>
                  <input className="input login-input" value={idNumber} onChange={(e) => setIdNumber(e.target.value)}
                         placeholder="Used once, in memory -- never stored in full" disabled={busy} />
                  <p className="login-hint" style={{ fontSize: 13, marginTop: 8 }}>
                    Only a hash and the last 4 digits of the ID number are ever saved -- the full number is
                    used once, in memory, to call the verification provider, and is never written to the
                    database or the audit log.
                  </p>
                  {error && <p className="login-error">{error}</p>}
                  <button className="btn shine login-submit" type="submit" disabled={busy}>
                    {busy ? "Checking…" : "Run identity check"}
                  </button>
                </form>

                {result && (
                  <Glass className="alert-card" style={{ marginTop: 16, borderColor: "var(--green)" }}>
                    <span className={STATUS_PILL[result.status] ?? "pill s-open"}>
                      {STATUS_LABEL[result.status] ?? result.status}
                    </span>
                    <p style={{ margin: "8px 0 0" }}>{result.detail}</p>
                    {(result.status === "mismatch" || result.status === "deceased") && (
                      <p className="muted small" style={{ margin: "8px 0 0" }}>
                        This is not a verdict -- flag account <strong>{result.account_id}</strong> for a human
                        officer to follow up with the real person on record before deciding anything.
                      </p>
                    )}
                  </Glass>
                )}

                {history.length > 0 && (
                  <div style={{ marginTop: 24 }}>
                    <div className="muted small" style={{ marginBottom: 8 }}>Recent checks</div>
                    {history.slice(0, 20).map((h) => (
                      <Glass className="alert-card" key={h.id} style={{ marginBottom: 8 }}>
                        <div className="row between">
                          <div>
                            <strong>{h.account_id}</strong>{" "}
                            <span className="muted small">{h.claimed_name} &middot; ID ····{h.id_number_last4}</span>
                          </div>
                          <span className={STATUS_PILL[h.status] ?? "pill s-open"}>
                            {STATUS_LABEL[h.status] ?? h.status}
                          </span>
                        </div>
                        <div className="muted small" style={{ marginTop: 4 }}>checked by {h.checked_by}</div>
                      </Glass>
                    ))}
                  </div>
                )}
              </>
            )}
          </Glass>
        </div>
      </div>
    </>
  );
}
