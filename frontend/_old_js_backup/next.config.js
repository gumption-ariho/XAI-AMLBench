/** @type {import('next').NextConfig} */
const BACKEND_URL = process.env.BACKEND_URL || "http://backend:8000";

module.exports = {
  reactStrictMode: true,
  // When you open the app directly on :3000 (not through Traefik on :80), /api still reaches the backend.
  async rewrites() {
    return [{ source: "/api/:path*", destination: `${BACKEND_URL}/api/:path*` }];
  },
};