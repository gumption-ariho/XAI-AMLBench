/** Thin typed wrapper around fetch for the /api routes (served by Traefik or the Next.js rewrite).
 *
 * Every route but /api/health and /api/login requires "Authorization: Bearer <session-token>". The token
 * comes back from POST /api/login (a predefined admin/analyst passkey plus the signer's own name -- see
 * backend/main.py's module docstring) and is kept in sessionStorage -- cleared when the tab closes, never
 * sent to localStorage or baked into the build (either of those would defeat the point: a token baked into
 * the compiled JS is visible to anyone via the browser's dev tools, and localStorage persists indefinitely,
 * widening an XSS attack's window). This replaced an earlier per-officer API-key system; the storage key name
 * below keeps its old name for simplicity, but what it holds is now a signed session token, not a raw key. */
const SESSION_STORAGE_KEY = "xai_amlbench_api_key";

export function getStoredSessionToken(): string {
  if (typeof window === "undefined") return "";           // server-side render: no browser storage to read
  return window.sessionStorage.getItem(SESSION_STORAGE_KEY) ?? "";
}

export function hasStoredSessionToken(): boolean {
  return getStoredSessionToken() !== "";
}

export function setSessionToken(token: string): void {
  if (typeof window !== "undefined") window.sessionStorage.setItem(SESSION_STORAGE_KEY, token);
}

export function clearSessionToken(): void {
  if (typeof window !== "undefined") window.sessionStorage.removeItem(SESSION_STORAGE_KEY);
}

// Kept as aliases so any older call site (or a skipped spot in this round's refactor) still compiles.
export const getStoredApiKey = getStoredSessionToken;
export const hasStoredApiKey = hasStoredSessionToken;
export const setApiKey = setSessionToken;
export const clearApiKey = clearSessionToken;

export async function api<T>(path: string, options?: RequestInit): Promise<T> {
  const headers: Record<string, string> = { "Content-Type": "application/json" };
  if (path !== "/health" && path !== "/login") {
    const token = getStoredSessionToken();
    if (token) headers["Authorization"] = `Bearer ${token}`;
  }
  const res = await fetch(`/api${path}`, {
    headers: { ...headers, ...(options?.headers as Record<string, string> | undefined) },
    ...options,
  });
  const text = await res.text();

  let data: unknown;
  try {
    data = JSON.parse(text);
  } catch {
    data = { detail: text };
  }

  if (res.status === 401 && path !== "/login") {
    clearSessionToken();   // the stored token was rejected (wrong, expired, or the server restarted and
                           // rotated its signing secret) -- drop it so the app re-shows the login screen
                           // rather than silently retrying the same bad token forever
  }
  if (!res.ok) {
    const detail = (data as { detail?: unknown }).detail;
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  return data as T;
}

export const sleep = (ms: number): Promise<void> => new Promise((resolve) => setTimeout(resolve, ms));

export const pct = (x: number, digits = 1): string => `${(x * 100).toFixed(digits)}%`;

export const money = (n: number): string => `$${Math.round(n).toLocaleString("en-US")}`;

export const errorMessage = (e: unknown): string => (e instanceof Error ? e.message : String(e));
