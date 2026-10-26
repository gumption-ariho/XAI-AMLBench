"use client";

/** The public front door for a bank that wants a pilot and doesn't have a passkey yet -- no login required,
 * reachable by anyone with the link (see backend/main.py's POST /signup-requests, which this posts to). It
 * never creates a tenant by itself; it only queues a request the owner reviews and approves on /owner. */

import { useState } from "react";
import type { FormEvent, ReactElement } from "react";
import Background from "@/components/Background";
import Glass from "@/components/Glass";
import { IconShield } from "@/components/Icons";
import { api, errorMessage } from "@/lib/api";

export default function RequestAccess(): ReactElement {
  const [bankName, setBankName] = useState<string>("");
  const [contactName, setContactName] = useState<string>("");
  const [contactEmail, setContactEmail] = useState<string>("");
  // Full registration detail -- this is what lets the owner, governing several banks at once, actually tell
  // which bank this is and where it operates, not just a name typed in a box. All optional: a bank filling
  // this in quickly may not have every field to hand right now.
  const [registrationNumber, setRegistrationNumber] = useState<string>("");
  const [country, setCountry] = useState<string>("");
  const [region, setRegion] = useState<string>("");
  const [address, setAddress] = useState<string>("");
  const [website, setWebsite] = useState<string>("");
  const [note, setNote] = useState<string>("");
  const [busy, setBusy] = useState<boolean>(false);
  const [error, setError] = useState<string>("");
  const [done, setDone] = useState<boolean>(false);

  const handleSubmit = async (e: FormEvent): Promise<void> => {
    e.preventDefault();
    if (!bankName.trim() || !contactName.trim() || !contactEmail.trim()) {
      setError("Bank name, your name, and an email are required.");
      return;
    }
    setBusy(true);
    setError("");
    try {
      await api("/signup-requests", {
        method: "POST",
        body: JSON.stringify({
          bank_name: bankName.trim(),
          contact_name: contactName.trim(),
          contact_email: contactEmail.trim(),
          registration_number: registrationNumber.trim() || null,
          country: country.trim() || null,
          region: region.trim() || null,
          address: address.trim() || null,
          website: website.trim() || null,
          note: note.trim(),
        }),
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
                <div className="login-subtitle">Request pilot access</div>
              </div>
            </div>
            {done ? (
              <p className="login-hint">
                Request sent. The platform owner will review it and send your bank a passkey once approved --
                no separate confirmation email yet, so check back or ask them directly if you don&apos;t hear
                back.
              </p>
            ) : (
              <>
                <p className="login-hint">
                  No passkey yet? Tell us who you are and we&apos;ll set your bank up for a pilot.
                </p>
                <input
                  className="input login-input"
                  placeholder="Bank name"
                  value={bankName}
                  onChange={(e) => setBankName(e.target.value)}
                  disabled={busy}
                  autoFocus
                />
                <input
                  className="input login-input"
                  placeholder="Your name"
                  value={contactName}
                  onChange={(e) => setContactName(e.target.value)}
                  disabled={busy}
                />
                <input
                  className="input login-input"
                  type="email"
                  placeholder="Your email"
                  value={contactEmail}
                  onChange={(e) => setContactEmail(e.target.value)}
                  disabled={busy}
                />
                <p className="login-hint" style={{ marginTop: 10, marginBottom: 4 }}>
                  Registration detail -- helps us tell your bank apart from every other one we work with
                  (optional, but the more you give us the faster we can review this).
                </p>
                <input
                  className="input login-input"
                  placeholder="Bank registration / license number"
                  value={registrationNumber}
                  onChange={(e) => setRegistrationNumber(e.target.value)}
                  disabled={busy}
                />
                <div className="row" style={{ gap: 8 }}>
                  <input
                    className="input login-input"
                    placeholder="Country"
                    value={country}
                    onChange={(e) => setCountry(e.target.value)}
                    disabled={busy}
                    style={{ flex: 1 }}
                  />
                  <input
                    className="input login-input"
                    placeholder="Region / city"
                    value={region}
                    onChange={(e) => setRegion(e.target.value)}
                    disabled={busy}
                    style={{ flex: 1 }}
                  />
                </div>
                <input
                  className="input login-input"
                  placeholder="Registered address (optional)"
                  value={address}
                  onChange={(e) => setAddress(e.target.value)}
                  disabled={busy}
                />
                <input
                  className="input login-input"
                  placeholder="Website (optional)"
                  value={website}
                  onChange={(e) => setWebsite(e.target.value)}
                  disabled={busy}
                />
                <textarea
                  className="input login-input"
                  placeholder="Anything else worth knowing (optional)"
                  value={note}
                  onChange={(e) => setNote(e.target.value)}
                  disabled={busy}
                  rows={3}
                />
                {error && <p className="login-error">{error}</p>}
                <button className="btn shine login-submit" type="submit" disabled={busy}>
                  {busy ? "Sending…" : "Request access"}
                </button>
                <p className="login-hint" style={{ marginTop: 10 }}>
                  Already a signed-up bank? Your own analysts register individually at{" "}
                  <a href="/analyst-signup">analyst sign-up</a> once your bank is approved.
                </p>
              </>
            )}
          </Glass>
        </form>
      </div>
    </>
  );
}
