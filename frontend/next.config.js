const BACKEND_URL = process.env.BACKEND_URL || "http://backend:8000";

/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  // Pin the project root so a stray package-lock.json in your home folder cannot confuse Next.js.
  outputFileTracingRoot: __dirname,
  // Inlined at build time. "1" only for `npm run dev:demo`; a normal build gets "0" and drops the demo code.
  env: { NEXT_PUBLIC_DEMO: process.env.NEXT_PUBLIC_DEMO === "1" ? "1" : "0" },
  // When you open the app directly on :3000 (not through Traefik on :80), /api still reaches the backend.
  async rewrites() {
    return [{ source: "/api/:path*", destination: `${BACKEND_URL}/api/:path*` }];
  },
};

module.exports = nextConfig;
