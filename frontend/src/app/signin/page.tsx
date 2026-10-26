"use client";

/** Per-person sign-in -- the email+password counterpart to the shared admin/analyst/bank passkey on the main
 * login screen (see components/LoginScreen.tsx and backend/main.py's POST /auth/login). Reached from a link
 * on that screen ("Sign in with your own account instead"), for anyone who's accepted an invite from the
 * owner page's People section.
 *
 * Two-step form: email + password first; if the account has 2FA enabled, the backend's first response is a
 * 401 with detail exactly "totp_required" (a deliberately distinct error from "wrong password" -- see
 * auth_login's own docstring) rather than a token, so this re-shows the form with a code field and resubmits
 * the same email/password plus that code. */

import { useState } from "react";
import type { FormEvent, ReactElement } from "react";
import { useRouter } from "next/navigation";
import Background from "@/components/Background";
import Glass from "@/components/Glass";
import { IconShield } from "@/components/Icons";
import { api, errorMessage, setSessionToken } from "@/lib/api";
import { TOTP_REQUIRED, type LoginResponse } from "@/types";

export default function SignIn(): ReactElement {
  const router = useRouter();
  const [email, setEmail] = useState<string>("");
  const [password, setPassword] = useState<string>("");
  const [totpCode, setTotpCode] = useState<string>("");
  const [needsTotp, setNeedsTotp] = useState<boolean>(false);
  const [error, setError] = useState<string>("");
  const [busy, setBusy] = useState<boolean>(false);

  const handleSubmit = async (e: FormEvent): Promise<void> => {
    e.preventDefault();
    if (!email.trim() || !password) {
      setError("Enter your email and password.");
      return;
    }
    setBusy(true);
    setError("");
    try {
      const res = await api<LoginResponse>("/auth/login", {
        method: "POST",
        body: JSON.stringify({
          email: email.trim(),
          password,
          totp_code: needsTotp ? totpCode.trim() : undefined,
        }),
      });
      setSessionToken(res.token);
      router.replace(res.role === "admin" ? "/owner" : "/");
    } catch (e2) {
      const msg = errorMessage(e2);
      if (msg === TOTP_REQUIRED) {
        setNeedsTotp(true);
        setError("");
      } else {
        setError(needsTotp ? "That code wasn't accepted. Check your authenticator app and try again."
          : "That email or password wasn't accepted.");
      }
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
                <div className="login-subtitle">{needsTotp ? "Enter your authenticator code" : "Sign in"}</div>
              </div>
            </div>
            {!needsTotp && (
              <p className="login-hint">
                Sign in with your own email and password, if you&apos;ve accepted an invite. Using a shared
                passkey instead? <a href="/">Go back to the main sign-in</a>.
              </p>
            )}
            {!needsTotp ? (
              <>
                <input
                  className="input login-input"
                  type="email"
                  placeholder="Email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  autoFocus
                  disabled={busy}
                  autoComplete="username"
                />
                <input
                  className="input login-input"
                  type="password"
                  placeholder="Password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  disabled={busy}
                  autoComplete="current-password"
                />
              </>
            ) : (
              <input
                className="input login-input"
                type="text"
                inputMode="numeric"
                placeholder="6-digit code"
                value={totpCode}
                onChange={(e) => setTotpCode(e.target.value)}
                autoFocus
                disabled={busy}
                autoComplete="one-time-code"
              />
            )}
            {error && <p className="login-error">{error}</p>}
            <button
              className="btn shine login-submit"
              type="submit"
              disabled={busy || !email.trim() || !password || (needsTotp && !totpCode.trim())}
            >
              {busy ? "Checking…" : needsTotp ? "Verify" : "Sign in"}
            </button>
            {!needsTotp && (
              <p className="login-hint" style={{ marginTop: 10 }}>
                <a href="/forgot-password">Forgot your password?</a>
              </p>
            )}
          </Glass>
        </form>
      </div>
    </>
  );
}
