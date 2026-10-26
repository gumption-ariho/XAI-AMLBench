"use client";

/** The link from a forgot-password email (?token=... -- see backend/main.py's POST /auth/reset-password). */

import { Suspense, useState } from "react";
import type { FormEvent, ReactElement } from "react";
import { useSearchParams } from "next/navigation";
import Background from "@/components/Background";
import Glass from "@/components/Glass";
import { IconShield } from "@/components/Icons";
import { api, errorMessage } from "@/lib/api";

export default function ResetPassword(): ReactElement {
  return (
    <Suspense fallback={null}>
      <ResetPasswordInner />
    </Suspense>
  );
}

function ResetPasswordInner(): ReactElement {
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
      setError("Paste the code from your reset email.");
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
      await api("/auth/reset-password", {
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
                <div className="login-subtitle">Choose a new password</div>
              </div>
            </div>
            {done ? (
              <p className="login-hint">
                Your password has been changed. <a href="/signin">Sign in</a> with it now.
              </p>
            ) : (
              <>
                {!tokenFromLink && (
                  <input
                    className="input login-input"
                    placeholder="Reset code"
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
                  {busy ? "Saving…" : "Change password"}
                </button>
              </>
            )}
          </Glass>
        </form>
      </div>
    </>
  );
}
