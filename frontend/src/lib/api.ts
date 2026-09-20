/** Thin typed wrapper around fetch for the /api routes (served by Traefik or the Next.js rewrite). */
export async function api<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`/api${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  const text = await res.text();

  let data: unknown;
  try {
    data = JSON.parse(text);
  } catch {
    data = { detail: text };
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
