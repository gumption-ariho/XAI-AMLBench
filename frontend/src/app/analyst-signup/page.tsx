"use client";

/** The public front door for an INDIVIDUAL analyst at a bank that's already signed up -- no admin invite
 * needed to get started, but no immediate access either: this creates a status='pending' account (see
 * backend/main.py's POST /auth/signup) that cannot sign in until the owner reviews and approves it on /owner.
 * That review step is the whole point -- self-registering proves who you SAY you are and which bank you SAY
 * you belong to, nothing more; the owner's approval is what turns that claim into an actual account. Each
 * account is tied to exactly one bank, chosen here from the list of banks that have actually been onboarded
 * (GET /public/banks) -- never typed freehand, so nobody can attach themselves to a bank that doesn't exist
 * or isn't active. */

import { useEffect, useState } from "react";
import type { FormEvent, ReactElement } from "react";
import Background from "@/components/Background";
import Glass from "@/components/Glass";
import { IconShield } from "@/components/Icons";
import { api, errorMessage } from "@/lib/api";
import type { AnalystSignupResponse } from "@/types";

export default function AnalystSignup(): ReactElement {
  const [banks, setBanks] = useState<string[] | null>(null);
  const [banksError, setBanksError] = useState<string>("");
  const [name, setName] = useState<string>("");
  const [email, setEmail] = useState<string>("");
  const [bank, setBank] = useState<string>("");
  const [password, setPassword] = useState<string>("");
  const [confirmPassword, setConfirmPassword] = useState<string>("");
  const [busy, setBusy] = useState<boolean>(false);
  const [error, setError] = useState<string>("");
  const [done, setDone] = useState<AnalystSignupResponse | null>(null);

  useEffect(() => {
    api<string[]>("/public/banks")
      .then((b) => {
        setBanks(b);
        if (b.length === 1) setBank(b[0]);
      })
      .catch((e) => setBanksError(errorMessage(e)));
  }, []);

  const handleSubmit = async (e: FormEvent): Promise<void> => {
    e.preventDefault();
    if (!name.trim() || !email.trim() || !bank) {
      setError("Your name, email, and which bank you belong to are required.");
      return;
    }
    if (password.length < 10) {
      setError("Password must be at least 10 characters.");
      return;
    }
    if (password !== confirmPassword) {
      setError("Passwords don't match.");
      return;
    }
    setBusy(true);
    setError("");
    try {
      const res = await api<AnalystSignupResponse>("/auth/signup", {
        method: "POST",
        body: JSON.stringify({ name: name.trim(), email: email.trim(), password, bank }),
      });
      setDone(res);
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
                <div className="login-subtitle">Analyst sign-up</div>
              </div>
            </div>
            {done ? (
              <p className="login-hint">
                Thanks, {done.email} -- your registration for {done.bank} is in. It&apos;s <strong>pending</strong>{" "}
                until your bank&apos;s administrator reviews and approves it; you can&apos;t sign in until then.
                Ask them to check the &ldquo;People&rdquo; section of their owner console if it&apos;s been a
                while.
              </p>
            ) : (
              <>
                <p className="login-hint">
                  For individuals at a bank that&apos;s already onboarded with us. Every account belongs to one
                  specific bank, and needs that bank&apos;s administrator to approve it before it can sign in --
                  nobody gets in on a self-registration alone.
                </p>
                {banksError && <p className="login-error">{banksError}</p>}
                {banks !== null && banks.length === 0 && !banksError && (
                  <p className="login-hint">
                    No banks are onboarded yet. If your bank hasn&apos;t signed up with us, {" "}
                    <a href="/request-access">request access for it</a> first -- then come back here.
                  </p>
                )}
                <input
                  className="input login-input"
                  placeholder="Your full name"
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  disabled={busy}
                  autoFocus
                  autoComplete="name"
                />
                <input
                  className="input login-input"
                  type="email"
                  placeholder="Your work email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  disabled={busy}
                  autoComplete="email"
                />
                <select
                  className="input login-input"
                  value={bank}
                  onChange={(e) => setBank(e.target.value)}
                  disabled={busy || !banks || banks.length === 0}
                >
                  <option value="">{banks === null ? "Loading banks…" : "Which bank do you belong to?"}</option>
                  {banks?.map((b) => (
                    <option key={b} value={b}>{b}</option>
                  ))}
                </select>
                <input
                  className="input login-input"
                  type="password"
                  placeholder="Choose a password (10+ characters)"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  disabled={busy}
                  autoComplete="new-password"
                />
                <input
                  className="input login-input"
                  type="password"
                  placeholder="Confirm password"
                  value={confirmPassword}
                  onChange={(e) => setConfirmPassword(e.target.value)}
                  disabled={busy}
                  autoComplete="new-password"
                />
                {error && <p className="login-error">{error}</p>}
                <button className="btn shine login-submit" type="submit" disabled={busy}>
                  {busy ? "Submitting…" : "Register"}
                </button>
                <p className="login-hint" style={{ marginTop: 10 }}>
                  Already approved? <a href="/signin">Sign in with your account</a>.
                </p>
              </>
            )}
          </Glass>
        </form>
      </div>
    </>
  );
}
