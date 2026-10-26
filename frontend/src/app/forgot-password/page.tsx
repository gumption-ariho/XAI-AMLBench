"use client";

/** Public, unauthenticated -- posts to backend/main.py's POST /auth/forgot-password, which always returns the
 * same generic message whether or not that email has an account (never reveals which emails are registered).
 * This page mirrors that: it shows the same "check your email" message no matter what was typed. */

import { useState } from "react";
import type { FormEvent, ReactElement } from "react";
import Background from "@/components/Background";
import Glass from "@/components/Glass";
import { IconShield } from "@/components/Icons";
import { api, errorMessage } from "@/lib/api";

export default function ForgotPassword(): ReactElement {
  const [email, setEmail] = useState<string>("");
  const [busy, setBusy] = useState<boolean>(false);
  const [error, setError] = useState<string>("");
  const [done, setDone] = useState<boolean>(false);

  const handleSubmit = async (e: FormEvent): Promise<void> => {
    e.preventDefault();
    if (!email.trim()) {
      setError("Enter the email you signed up with.");
      return;
    }
    setBusy(true);
    setError("");
    try {
      await api("/auth/forgot-password", {
        method: "POST",
        body: JSON.stringify({ email: email.trim() }),
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
                <div className="login-subtitle">Reset your password</div>
              </div>
            </div>
            {done ? (
              <p className="login-hint">
                If that email has an account, we&apos;ve sent a link to reset the password. Check your inbox
                (and spam folder) -- the link expires in a couple of hours.
              </p>
            ) : (
              <>
                <p className="login-hint">Enter the email you signed in with and we&apos;ll send a reset link.</p>
                <input
                  className="input login-input"
                  type="email"
                  placeholder="Email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  autoFocus
                  disabled={busy}
                />
                {error && <p className="login-error">{error}</p>}
                <button className="btn shine login-submit" type="submit" disabled={busy || !email.trim()}>
                  {busy ? "Sending…" : "Send reset link"}
                </button>
              </>
            )}
          </Glass>
        </form>
      </div>
    </>
  );
}
