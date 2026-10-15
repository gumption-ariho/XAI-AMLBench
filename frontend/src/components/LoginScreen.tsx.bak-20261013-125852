"use client";

import { useState } from "react";
import type { FormEvent, ReactElement } from "react";
import Glass from "@/components/Glass";
import { IconShield } from "@/components/Icons";
import { api, setApiKey } from "@/lib/api";
import type { WhoamiResponse } from "@/types";

interface LoginScreenProps {
  /** Called once the entered key is verified against /whoami -- the real officer name comes back from the
   * backend, never from anything typed into this form, so the audit trail always reflects who actually
   * authenticated, not a self-reported name. */
  onSuccess: (officer: string) => void;
}

/** Replaces the earlier prompt()-based placeholder with a real form: the key is verified against /whoami
 * before it is treated as valid, and a wrong or revoked key gets a clear, specific error rather than silently
 * failing on the first real request. */
export default function LoginScreen({ onSuccess }: LoginScreenProps): ReactElement {
  const [key, setKey] = useState<string>("");
  const [error, setError] = useState<string>("");
  const [busy, setBusy] = useState<boolean>(false);

  const handleSubmit = async (e: FormEvent): Promise<void> => {
    e.preventDefault();
    const trimmed = key.trim();
    if (!trimmed) {
      setError("Enter your officer API key.");
      return;
    }
    setBusy(true);
    setError("");
    setApiKey(trimmed);   // must be stored before calling api(), which reads the key from storage
    try {
      const who = await api<WhoamiResponse>("/whoami");
      onSuccess(who.officer);
    } catch {
      setError("That key was not accepted. Check it for typos, or ask whoever ran manage_keys.py to confirm it hasn't been revoked.");
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="login-screen">
      <form className="login-card-wrap" onSubmit={(e) => void handleSubmit(e)}>
        <Glass as="div" className="login-card">
          <div className="login-brand">
            <span className="login-shield"><IconShield /></span>
            <div>
              <div className="login-title">XAI·AMLBench</div>
              <div className="login-subtitle">Compliance Console</div>
            </div>
          </div>
          <p className="login-hint">
            Sign in with your officer API key. Issued with <code>manage_keys.py create</code> -- ask your
            administrator for one if you don&apos;t have it.
          </p>
          <input
            className="input login-input"
            type="password"
            placeholder="officer_…"
            value={key}
            onChange={(e) => setKey(e.target.value)}
            autoFocus
            disabled={busy}
            autoComplete="off"
            spellCheck={false}
          />
          {error && <p className="login-error">{error}</p>}
          <button className="btn shine login-submit" type="submit" disabled={busy || !key.trim()}>
            {busy ? "Checking…" : "Sign in"}
          </button>
        </Glass>
      </form>
    </div>
  );
}
