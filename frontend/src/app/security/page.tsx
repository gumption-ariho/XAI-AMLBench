"use client";

/** Account security for a signed-in PER-PERSON account (one that signed in via /signin, not a shared
 * passkey) -- set up or turn off two-factor authentication. See backend/main.py's setup_2fa/enable_2fa/
 * disable_2fa and get_current_user_id's own docstring on why a shared-passkey session can't use this at all:
 * there's no individual account row for 2FA to attach to. */

import { useState } from "react";
import type { FormEvent, ReactElement } from "react";
import Background from "@/components/Background";
import Glass from "@/components/Glass";
import { IconShield } from "@/components/Icons";
import { api, errorMessage, hasStoredSessionToken } from "@/lib/api";
import type { Setup2faResponse } from "@/types";

export default function Security(): ReactElement {
  const [setup, setSetup] = useState<Setup2faResponse | null>(null);
  const [code, setCode] = useState<string>("");
  const [enabled, setEnabled] = useState<boolean | null>(null);
  const [busy, setBusy] = useState<boolean>(false);
  const [error, setError] = useState<string>("");
  const [message, setMessage] = useState<string>("");

  const signedIn = hasStoredSessionToken();

  const startSetup = async (): Promise<void> => {
    setBusy(true);
    setError("");
    try {
      const res = await api<Setup2faResponse>("/auth/2fa/setup", { method: "POST" });
      setSetup(res);
    } catch (e) {
      setError(errorMessage(e));
    } finally {
      setBusy(false);
    }
  };

  const confirmEnable = async (e: FormEvent): Promise<void> => {
    e.preventDefault();
    setBusy(true);
    setError("");
    try {
      await api("/auth/2fa/enable", { method: "POST", body: JSON.stringify({ code: code.trim() }) });
      setEnabled(true);
      setSetup(null);
      setCode("");
      setMessage("Two-factor authentication is on. You'll be asked for a code like this one every time you sign in.");
    } catch (e2) {
      setError(errorMessage(e2));
    } finally {
      setBusy(false);
    }
  };

  const confirmDisable = async (e: FormEvent): Promise<void> => {
    e.preventDefault();
    setBusy(true);
    setError("");
    try {
      await api("/auth/2fa/disable", { method: "POST", body: JSON.stringify({ code: code.trim() }) });
      setEnabled(false);
      setCode("");
      setMessage("Two-factor authentication is off.");
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
        <div className="login-card-wrap">
          <Glass as="div" className="login-card">
            <div className="login-brand">
              <span className="login-shield"><IconShield /></span>
              <div>
                <div className="login-title">Account security</div>
                <div className="login-subtitle">Two-factor authentication</div>
              </div>
            </div>

            {!signedIn ? (
              <p className="login-hint">
                <a href="/signin">Sign in with your own account</a> first -- this only applies to an
                individual account, not a shared passkey.
              </p>
            ) : message ? (
              <p className="login-hint">{message}</p>
            ) : setup ? (
              <form onSubmit={(e) => void confirmEnable(e)}>
                <p className="login-hint">
                  Add this account to your authenticator app (Google Authenticator, 1Password, Authy, …),
                  then enter the 6-digit code it shows to turn 2FA on.
                </p>
                <p className="login-hint" style={{ wordBreak: "break-all" }}>
                  <a href={setup.provisioning_uri}>Open in your authenticator app</a>, or enter this key
                  manually: <code>{setup.secret}</code>
                </p>
                <input
                  className="input login-input"
                  type="text"
                  inputMode="numeric"
                  placeholder="6-digit code"
                  value={code}
                  onChange={(e) => setCode(e.target.value)}
                  autoFocus
                  disabled={busy}
                />
                {error && <p className="login-error">{error}</p>}
                <button className="btn shine login-submit" type="submit" disabled={busy || !code.trim()}>
                  {busy ? "Checking…" : "Turn on 2FA"}
                </button>
              </form>
            ) : enabled === false ? null : (
              <>
                <p className="login-hint">
                  Two-factor authentication adds a code from your phone to every sign-in, on top of your
                  password -- so a stolen password alone can't be used to sign in as you.
                </p>
                {error && <p className="login-error">{error}</p>}
                <button className="btn shine login-submit" type="button" disabled={busy} onClick={() => void startSetup()}>
                  {busy ? "Starting…" : "Set up two-factor authentication"}
                </button>
              </>
            )}

            {signedIn && enabled !== false && !setup && !message && (
              <form onSubmit={(e) => void confirmDisable(e)} style={{ marginTop: 14 }}>
                <p className="login-hint">Already have 2FA on and want to turn it off? Enter a current code:</p>
                <input
                  className="input login-input"
                  type="text"
                  inputMode="numeric"
                  placeholder="6-digit code"
                  value={code}
                  onChange={(e) => setCode(e.target.value)}
                  disabled={busy}
                />
                <button className="btn login-submit" type="submit" disabled={busy || !code.trim()}>
                  Turn off 2FA
                </button>
              </form>
            )}
          </Glass>
        </div>
      </div>
    </>
  );
}
