"use client";

/** The link an invited person clicks (?token=... -- see backend/main.py's admin_invite_user/accept_invite).
 * Sets their own password and activates the account in one step. Public/unauthenticated by design -- the
 * token itself is the credential here, exactly like a password-reset link. */

import { Suspense, useState } from "react";
import type { FormEvent, ReactElement } from "react";
import { useSearchParams } from "next/navigation";
import Background from "@/components/Background";
import Glass from "@/components/Glass";
import { IconShield } from "@/components/Icons";
import { api, errorMessage } from "@/lib/api";

export default function AcceptInvite(): ReactElement {
  return (
    <Suspense fallback={null}>
      <AcceptInviteInner />
    </Suspense>
  );
}

function AcceptInviteInner(): ReactElement {
  const searchParams = useSearchParams();
  const tokenFromLink = searchParams.get("token") ?? "";
  const [token, setToken] = useState<string>(tokenFromLink);
  const [password, setPassword] = useState<string>("");
  const [confirm, setConfirm] = useState<string>("");
  const [busy, setBusy] = useState<boolean>(false);
  const [error, setError] = useState<string>("");
  const [done, setDone] = useState<boolean>(false);

  const handleSubmit = async (e: FormEvent): Promise<void> => {
    e.preventDefault();
    if (!token.trim()) {
      setError("Paste the code from your invite email.");
      return;
    }
    if (password.length < 10) {
      setError("Password must be at least 10 characters.");
      return;
    }
    if (password !== confirm) {
      setError("Passwords don't match.");
      return;
    }
    setBusy(true);
    setError("");
    try {
      await api("/auth/accept-invite", {
        method: "POST",
        body: JSON.stringify({ token: token.trim(), password }),
      });
      setDone(true);
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
        <form className="login-card-wrap" onSubmit={(e) => void handleSubmit(e)}>
          <Glass as="div" className="login-card">
            <div className="login-brand">
              <span className="login-shield"><IconShield /></span>
              <div>
                <div className="login-title">XAI·AMLBench</div>
                <div className="login-subtitle">Set your password</div>
              </div>
            </div>
            {done ? (
              <p className="login-hint">
                Your account is active. <a href="/signin">Sign in</a> with your email and new password.
              </p>
            ) : (
              <>
                <p className="login-hint">
                  Finish setting up your own sign-in. This link only works once and expires after a few days.
                </p>
                {!tokenFromLink && (
                  <input
                    className="input login-input"
                    placeholder="Invite code"
                    value={token}
                    onChange={(e) => setToken(e.target.value)}
                    disabled={busy}
                    autoFocus
                  />
                )}
                <input
                  className="input login-input"
                  type="password"
                  placeholder="New password (min. 10 characters)"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  disabled={busy}
                  autoFocus={!!tokenFromLink}
                  autoComplete="new-password"
                />
                <input
                  className="input login-input"
                  type="password"
                  placeholder="Confirm password"
                  value={confirm}
                  onChange={(e) => setConfirm(e.target.value)}
                  disabled={busy}
                  autoComplete="new-password"
                />
                {error && <p className="login-error">{error}</p>}
                <button className="btn shine login-submit" type="submit" disabled={busy}>
                  {busy ? "Setting up…" : "Activate my account"}
                </button>
              </>
            )}
          </Glass>
        </form>
      </div>
    </>
  );
}
