"use client";

import { Suspense, useEffect, useState } from "react";
import type { FormEvent, ReactElement } from "react";
import { useSearchParams } from "next/navigation";
import Glass from "@/components/Glass";
import { IconShield } from "@/components/Icons";
import { api, setSessionToken } from "@/lib/api";
import type { BankBranding, LoginResponse } from "@/types";

interface LoginScreenProps {
  /** Called once POST /api/login accepts the passkey -- the signed-in name is whatever was typed below,
   * since there's no per-person account to look it up against (see backend/main.py's module docstring on why
   * that's an intentional tradeoff of a shared, predefined passkey). `bank` is set when this passkey belongs
   * to one specific institution (a BANK_PASSKEYS tenant login) rather than an admin/global session. `role`
   * is what the caller uses to route an admin passkey to the owner backoffice instead of the bank console. */
  onSuccess: (name: string, bank: string | null, role: "admin" | "analyst") => void;
  /** Overrides the default brand/hint copy -- the /owner page uses this to make clear an analyst/bank
   * passkey won't get in there, without duplicating the whole form. */
  subtitle?: string;
  hint?: string;
  /** Which backend route this form posts to. Defaults to "/login" (the customer/bank door, which cannot
   * issue an admin session no matter what passkey is typed -- see backend/main.py's login() docstring).
   * The /owner page passes "/admin/login" instead -- the ONLY place ADMIN_PASSKEY works -- so a bank
   * tenant's login page is never the same endpoint an admin-passkey guess could be tried against. */
  endpoint?: "/login" | "/admin/login";
  /** "admin" gives this form a visibly more technical treatment (monospace, darker) -- deliberate: the
   * owner/operator is expected to be comfortable with that, while a bank's own staff (often non-technical,
   * e.g. accountants) get the plain, friendly default. Cosmetic only; the real separation is `endpoint`. */
  variant?: "customer" | "admin";
}

/** Next.js requires useSearchParams() to sit inside a Suspense boundary (it can otherwise de-opt the whole
 * page out of static rendering, or fail the build) -- this one-line wrapper is that boundary, so every
 * caller of LoginScreen keeps working without needing to know this detail. The fallback is just the branding
 * lookup not having resolved yet (one render tick, invisible in practice), not a loading state worth showing. */
export default function LoginScreen(props: LoginScreenProps): ReactElement {
  return (
    <Suspense fallback={<LoginScreenInner {...props} bankSlugOverride={null} />}>
      <LoginScreenWithBranding {...props} />
    </Suspense>
  );
}

function LoginScreenWithBranding(props: LoginScreenProps): ReactElement {
  const searchParams = useSearchParams();
  return <LoginScreenInner {...props} bankSlugOverride={searchParams.get("bank")} />;
}

/** One shared passkey per role (admin/analyst), plus one passkey per bank for a multi-tenant deployment --
 * no per-person account to create or manage either way. The name typed here is what every action gets
 * attributed to afterward, so it's asked for up front, not assumed. */
function LoginScreenInner({ onSuccess, subtitle, hint, bankSlugOverride, endpoint = "/login", variant = "customer" }: LoginScreenProps & { bankSlugOverride: string | null }): ReactElement {
  const [name, setName] = useState<string>("");
  const [passkey, setPasskey] = useState<string>("");
  const [error, setError] = useState<string>("");
  const [busy, setBusy] = useState<boolean>(false);

  // A light "white-label" touch for a pilot: a link sent as /?bank=<name> shows that bank's own name (and
  // logo, if it set one) on the login screen before anyone has signed in, pulled from the unauthenticated
  // GET /public/banks/{name}/branding. `subtitle` passed in by the caller (the /owner page does this) always
  // wins -- this never overrides an explicit caller-set subtitle.
  const [branding, setBranding] = useState<BankBranding | null>(null);
  useEffect(() => {
    if (!bankSlugOverride || subtitle) return;
    api<BankBranding>(`/public/banks/${encodeURIComponent(bankSlugOverride)}/branding`).then(setBranding).catch(() => undefined);
  }, [bankSlugOverride, subtitle]);

  const handleSubmit = async (e: FormEvent): Promise<void> => {
    e.preventDefault();
    const trimmedName = name.trim();
    if (!trimmedName) {
      setError("Enter your name.");
      return;
    }
    if (!passkey) {
      setError("Enter the admin or analyst passkey.");
      return;
    }
    setBusy(true);
    setError("");
    try {
      const res = await api<LoginResponse>(endpoint, {
        method: "POST",
        body: JSON.stringify({ name: trimmedName, passkey }),
      });
      setSessionToken(res.token);
      onSuccess(res.name, res.bank, res.role);
    } catch {
      setError(endpoint === "/admin/login"
        ? "That passkey wasn't accepted. This door only takes the admin passkey."
        : "That passkey wasn't accepted. Check it for typos, or ask whoever set up this console for your passkey.");
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="login-screen">
      <form className="login-card-wrap" onSubmit={(e) => void handleSubmit(e)}>
        <Glass as="div" className={`login-card ${variant === "admin" ? "admin-variant" : ""}`}>
          <div className="login-brand">
            {branding?.logo_url ? (
              // eslint-disable-next-line @next/next/no-img-element -- a tenant-supplied URL, not a build asset
              <img src={branding.logo_url} alt="" className="login-shield" style={{ objectFit: "contain" }} />
            ) : (
              <span className="login-shield"><IconShield /></span>
            )}
            <div>
              <div className="login-title">{branding ? branding.name : variant === "admin" ? "XAI·AMLBench // OPERATOR" : "XAI·AMLBench"}</div>
              <div className="login-subtitle">{subtitle ?? "Compliance Console"}</div>
            </div>
          </div>
          <p className="login-hint">
            {hint ?? "Sign in with your name and your passkey -- the shared analyst passkey, or your bank's own "
              + "passkey if this is a multi-bank deployment. Every action you take is recorded under the name "
              + "you enter here."}
          </p>
          {!subtitle && (
            <p className="login-hint" style={{ marginTop: -6, marginBottom: 10 }}>
              Have your own individual account instead? <a href="/signin">Sign in with it</a>.
              {variant === "customer" && (
                <> No account yet? <a href="/analyst-signup">Register as an analyst</a>.</>
              )}
            </p>
          )}
          <input
            className="input login-input"
            type="text"
            placeholder="Your name"
            value={name}
            onChange={(e) => setName(e.target.value)}
            autoFocus
            disabled={busy}
            autoComplete="name"
          />
          <input
            className="input login-input"
            type="password"
            placeholder="Passkey"
            value={passkey}
            onChange={(e) => setPasskey(e.target.value)}
            disabled={busy}
            autoComplete="off"
            spellCheck={false}
          />
          {error && <p className="login-error">{error}</p>}
          <button className="btn shine login-submit" type="submit" disabled={busy || !name.trim() || !passkey}>
            {busy ? "Checking…" : "Sign in"}
          </button>
        </Glass>
      </form>
    </div>
  );
}
