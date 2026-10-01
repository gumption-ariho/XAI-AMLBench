/** Thin typed wrapper around fetch for the /api routes (served by Traefik or the Next.js rewrite).
 *
 * Every route but /api/health requires "Authorization: Bearer <key>" (a real security fix: the backend
 * previously had no authentication at all). The key is kept in sessionStorage -- cleared when the tab closes,
 * never sent to localStorage or baked into the build (either of those would defeat the point: a key baked into
 * the compiled JS is visible to anyone via the browser's dev tools, and localStorage persists indefinitely,
 * widening an XSS attack's window). A real login screen (components/LoginScreen.tsx) collects the key and
 * verifies it via /whoami before storing it -- this module no longer falls back to a prompt() if the key is
 * missing; that was a deliberately temporary placeholder, now replaced. */
const API_KEY_STORAGE_KEY = "xai_amlbench_api_key";

export function getStoredApiKey(): string {
  if (typeof window === "undefined") return "";           // server-side render: no browser storage to read
  return window.sessionStorage.getItem(API_KEY_STORAGE_KEY) ?? "";
}

export function hasStoredApiKey(): boolean {
  return getStoredApiKey() !== "";
}

export function setApiKey(key: string): void {
  if (typeof window !== "undefined") window.sessionStorage.setItem(API_KEY_STORAGE_KEY, key);
}

export function clearApiKey(): void {
  if (typeof window !== "undefined") window.sessionStorage.removeItem(API_KEY_STORAGE_KEY);
}

export async function api<T>(path: string, options?: RequestInit): Promise<T> {
  const headers: Record<string, string> = { "Content-Type": "application/json" };
  if (path !== "/health") {
    const key = getStoredApiKey();
    if (key) headers["Authorization"] = `Bearer ${key}`;
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

  if (res.status === 401) {
    clearApiKey();   // the stored key was rejected (wrong or revoked) -- drop it so the app re-shows the
                     // login screen rather than silently retrying the same bad key forever
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
