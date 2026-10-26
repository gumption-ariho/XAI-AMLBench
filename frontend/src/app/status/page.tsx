"use client";

/** Public, unauthenticated status page -- so a tester can check "is it down right now" without signing in or
 * asking the owner directly. Deliberately thin: GET /public/status (backend/main.py) returns only an overall
 * up/down boolean, nothing about which internal service, no error detail, no infra names -- this page is
 * meant to be safely linkable to anyone. */

import { useEffect, useState } from "react";
import type { ReactElement } from "react";
import Background from "@/components/Background";
import Glass from "@/components/Glass";
import { IconShield } from "@/components/Icons";
import { api, errorMessage } from "@/lib/api";
import type { PublicStatus } from "@/types";

export default function StatusPage(): ReactElement {
  const [status, setStatus] = useState<PublicStatus | null>(null);
  const [error, setError] = useState<string>("");

  useEffect(() => {
    const load = (): void => {
      api<PublicStatus>("/public/status").then(setStatus).catch((e) => setError(errorMessage(e)));
    };
    load();
    const id = setInterval(load, 30000);
    return () => clearInterval(id);
  }, []);

  return (
    <>
      <Background />
      <div className="login-screen">
        <div className="login-card-wrap">
          <Glass as="div" className="login-card">
            <div className="login-brand">
              <span className="login-shield"><IconShield /></span>
              <div>
                <div className="login-title">XAI·AMLBench</div>
                <div className="login-subtitle">Platform status</div>
              </div>
            </div>
            {error && <p className="login-error">Couldn&apos;t reach the status check: {error}</p>}
            {!error && status === null && <p className="login-hint">Checking…</p>}
            {status && (
              <>
                <p className={`pill ${status.all_ok ? "ok" : "s-confirmed"}`} style={{ fontSize: 16, padding: "8px 16px" }}>
                  {status.all_ok ? "All systems normal" : "Degraded -- some services aren't responding"}
                </p>
                <p className="login-hint" style={{ marginTop: 14 }}>
                  Last checked {new Date(status.checked_at * 1000).toLocaleTimeString()}. This refreshes every
                  30 seconds.
                </p>
              </>
            )}
          </Glass>
        </div>
      </div>
    </>
  );
}
