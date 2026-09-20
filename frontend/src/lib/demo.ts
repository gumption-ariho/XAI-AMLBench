/**
 * Demo mode: run the real UI with built-in sample data and NO backend.
 *
 *   npm run dev:demo      ->  http://localhost:3000
 *
 * next.config.js inlines NEXT_PUBLIC_DEMO at build time ("1" only for the dev:demo script), so in a normal
 * production build this is a constant `false` and the mock backend code is dropped from the bundle.
 * The try/catch keeps it safe in plain-browser bundles (demo/build-demo.cjs) where `process` does not exist.
 */
export const isDemo = (): boolean => {
  try {
    return process.env.NEXT_PUBLIC_DEMO === "1";
  } catch {
    return false;
  }
};
