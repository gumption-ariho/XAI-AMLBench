#!/usr/bin/env python3
"""
setup_frontend_v2.py  -  installs the TypeScript frontend into your existing  frontend/  folder.

Run it from your project root (the folder that contains frontend/, backend/, docker-compose.yml):

    python3 setup_frontend_v2.py --dry-run     # show what WOULD change, touch nothing
    python3 setup_frontend_v2.py               # do it

What it does
  * writes every file of the TypeScript frontend (see the tree printed at the end)
  * moves the old JavaScript files  src/app/layout.js  and  src/app/page.js  into  frontend/_old_js_backup/
    (Next.js would fail if  page.js  and  page.tsx  sat side by side)
  * before replacing any file that differs (package.json, next.config.js, globals.css ...), saves the
    original into  frontend/_old_js_backup/  as well. Nothing is ever deleted.
  * running it twice is safe: unchanged files are left alone and the first backup is kept.

Only the frontend/ folder is touched. Every project file is stored below the PAYLOAD line as comment
lines starting with "#|", between "@@@@FILE: <path>" and "@@@@END" markers, so you can read it here too.
"""
import shutil
import sys
from pathlib import Path

VERSION = "v3  (calibrated risk scores on the gauge and network panel; demo mode: npm run dev:demo, Next 15.5.21 + React 19)"
MARK_FILE, MARK_END, PAYLOAD = "@@@@FILE: ", "@@@@END", "#==== PAYLOAD BELOW ===="
OLD_JS = ["src/app/layout.js", "src/app/page.js"]
BACKUP = "_old_js_backup"


def read_payload():
    text = Path(__file__).read_text(encoding="utf-8")
    body = text.split(PAYLOAD + "\n", 1)[1]
    files, path, buf = {}, None, []
    for raw in body.split("\n"):
        if not raw.startswith("#|"):
            continue
        line = raw[2:]
        if line.startswith(MARK_FILE):
            path, buf = line[len(MARK_FILE):].strip(), []
        elif line == MARK_END and path:
            files[path] = "\n".join(buf) + "\n"
            path = None
        elif path is not None:
            buf.append(line)
    return files


def find_target(args):
    if args:
        return Path(args[0]).expanduser()
    if Path("frontend").is_dir():
        return Path("frontend")
    if Path.cwd().name == "frontend":
        return Path(".")
    return Path("frontend")          # will be created


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    dry = "--dry-run" in sys.argv
    files = read_payload()
    if "--list" in sys.argv:
        for p in files:
            print(f"{len(files[p].splitlines()):>5} lines  {p}")
        print(f"\n{len(files)} files")
        return

    print(f"{Path(__file__).name} {VERSION}\n")
    target = find_target(args)
    backup = target / BACKUP
    print(("DRY RUN - nothing will be changed\n" if dry else "") + f"Target folder: {target.resolve()}\n")

    def stash(rel, move):
        src = target / rel
        dst = backup / rel
        if not dst.exists():                     # keep the very first original
            print(f"  backup   {rel}  ->  {BACKUP}/{rel}")
            if not dry:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
        if move:
            print(f"  remove   {rel}   (old JavaScript version, saved in {BACKUP}/)")
            if not dry:
                src.unlink()

    for rel in OLD_JS:
        if (target / rel).exists():
            stash(rel, move=True)

    created = replaced = same = 0
    for rel, content in files.items():
        dest = target / rel
        if dest.exists():
            if dest.read_text(encoding="utf-8") == content:
                same += 1
                continue
            stash(rel, move=False)
            print(f"  replace  {rel}")
            replaced += 1
        else:
            print(f"  create   {rel}")
            created += 1
        if not dry:
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(content, encoding="utf-8", newline="\n")

    print(f"\n{'Would write' if dry else 'Done'}: {created} created, {replaced} replaced, {same} already up to date.")
    if dry:
        print("Run again without --dry-run to apply.")
        return
    print("""
Next steps
  cd frontend
  npm install                 # syncs the packages (quick when node_modules already exists)
  npm run dev:demo            # the app on built-in sample data -> http://localhost:3000  (no backend needed)
  npm run typecheck           # strict TypeScript check

Run it for real
  docker compose --profile app up -d --build           # from the project root, opens on http://localhost
  BACKEND_URL=http://localhost npm run dev             # or: local dev server on :3000 talking to the Docker stack
""")


if __name__ == "__main__":
    main()

#==== PAYLOAD BELOW ====
#|@@@@FILE: .dockerignore
#|node_modules
#|.next
#|preview
#|_old_js_backup
#|*.tsbuildinfo
#|@@@@END
#|@@@@FILE: Dockerfile
#|# Build context = ./frontend
#|FROM node:20-alpine
#|
#|WORKDIR /app
#|ENV NEXT_TELEMETRY_DISABLED=1
#|# Rewrites are baked in at build time; "backend" is the compose service name.
#|ENV BACKEND_URL=http://backend:8000
#|
#|COPY package.json ./
#|RUN npm install
#|
#|COPY . .
#|RUN npm run build
#|
#|EXPOSE 3000
#|CMD ["npm", "start"]
#|@@@@END
#|@@@@FILE: next-env.d.ts
#|/// <reference types="next" />
#|/// <reference types="next/image-types/global" />
#|
#|// NOTE: This file is regenerated by Next.js. Do not edit it.
#|@@@@END
#|@@@@FILE: next.config.js
#|const BACKEND_URL = process.env.BACKEND_URL || "http://backend:8000";
#|
#|/** @type {import('next').NextConfig} */
#|const nextConfig = {
#|  reactStrictMode: true,
#|  // Pin the project root so a stray package-lock.json in your home folder cannot confuse Next.js.
#|  outputFileTracingRoot: __dirname,
#|  // Inlined at build time. "1" only for `npm run dev:demo`; a normal build gets "0" and drops the demo code.
#|  env: { NEXT_PUBLIC_DEMO: process.env.NEXT_PUBLIC_DEMO === "1" ? "1" : "0" },
#|  // When you open the app directly on :3000 (not through Traefik on :80), /api still reaches the backend.
#|  async rewrites() {
#|    return [{ source: "/api/:path*", destination: `${BACKEND_URL}/api/:path*` }];
#|  },
#|};
#|
#|module.exports = nextConfig;
#|@@@@END
#|@@@@FILE: package.json
#|{
#|  "name": "xai-amlbench-frontend",
#|  "version": "0.2.0",
#|  "private": true,
#|  "scripts": {
#|    "dev": "next dev -p 3000",
#|    "dev:demo": "NEXT_PUBLIC_DEMO=1 next dev -p 3000",
#|    "build": "next build",
#|    "start": "next start -p 3000",
#|    "typecheck": "tsc --noEmit",
#|    "demo": "node demo/build-demo.cjs"
#|  },
#|  "dependencies": {
#|    "next": "15.5.21",
#|    "react": "^19.2.3",
#|    "react-dom": "^19.2.3"
#|  },
#|  "devDependencies": {
#|    "@types/node": "^20.14.0",
#|    "@types/react": "^19.0.0",
#|    "@types/react-dom": "^19.0.0",
#|    "typescript": "^5.5.4"
#|  },
#|  "engines": {
#|    "node": ">=18.18.0"
#|  }
#|}
#|@@@@END
#|@@@@FILE: tsconfig.json
#|{
#|  "compilerOptions": {
#|    "target": "ES2017",
#|    "lib": [
#|      "dom",
#|      "dom.iterable",
#|      "esnext"
#|    ],
#|    "allowJs": false,
#|    "skipLibCheck": true,
#|    "strict": true,
#|    "noEmit": true,
#|    "noFallthroughCasesInSwitch": true,
#|    "noImplicitOverride": true,
#|    "forceConsistentCasingInFileNames": true,
#|    "esModuleInterop": true,
#|    "module": "esnext",
#|    "moduleResolution": "bundler",
#|    "resolveJsonModule": true,
#|    "isolatedModules": true,
#|    "jsx": "preserve",
#|    "incremental": true,
#|    "plugins": [
#|      {
#|        "name": "next"
#|      }
#|    ],
#|    "baseUrl": ".",
#|    "paths": {
#|      "@/*": [
#|        "./src/*"
#|      ]
#|    }
#|  },
#|  "include": [
#|    "next-env.d.ts",
#|    "**/*.ts",
#|    "**/*.tsx",
#|    ".next/types/**/*.ts"
#|  ],
#|  "exclude": [
#|    "node_modules",
#|    "preview",
#|    "_old_js_backup"
#|  ]
#|}
#|@@@@END
#|@@@@FILE: demo/build-demo.cjs
#|#!/usr/bin/env node
#|/**
#| * Compiles the real TypeScript sources (src/**, demo/**) into ONE self-contained HTML page that runs
#| * in any browser: React 18 comes from cdnjs, everything else is inlined. No Next.js, Docker or backend.
#| *
#| *   npm install            # provides `typescript`
#| *   node demo/build-demo.cjs        ->  preview/demo.html
#| */
#|const fs = require("fs");
#|const path = require("path");
#|const ts = require("typescript");
#|
#|const ROOT = path.resolve(__dirname, "..");
#|const REACT = "https://cdnjs.cloudflare.com/ajax/libs/react/18.3.1/umd/react.production.min.js";
#|const REACT_DOM = "https://cdnjs.cloudflare.com/ajax/libs/react-dom/18.3.1/umd/react-dom.production.min.js";
#|
#|const walk = (dir) =>
#|  fs.readdirSync(dir, { withFileTypes: true }).flatMap((e) =>
#|    e.isDirectory() ? walk(path.join(dir, e.name)) : [path.join(dir, e.name)]);
#|
#|/** Tiny CommonJS-style module registry, so `import x from "@/..."` keeps working inside one file. */
#|const RUNTIME = `(function () {
#|  var defs = {}, cache = {};
#|  window.__d = function (id, fn) { defs[id] = fn; };
#|  function norm(p) { var out = []; p.split("/").forEach(function (s) { if (s === "..") out.pop(); else if (s && s !== ".") out.push(s); }); return out.join("/"); }
#|  function dirname(id) { return id.split("/").slice(0, -1).join("/"); }
#|  function load(id) {
#|    if (cache[id]) return cache[id].exports;
#|    if (!defs[id]) throw new Error("Module not found: " + id);
#|    var m = (cache[id] = { exports: {} });
#|    defs[id].call(m.exports, m.exports, function (r) { return resolve(id, r); }, m);
#|    return m.exports;
#|  }
#|  function resolve(from, r) {
#|    if (r === "react") return window.React;
#|    if (r === "react-dom" || r === "react-dom/client") return window.ReactDOM;
#|    if (r.indexOf("@/") === 0) return load(norm("src/" + r.slice(2)));
#|    if (r.charAt(0) === ".") return load(norm(dirname(from) + "/" + r));
#|    throw new Error("Unbundled module: " + r);
#|  }
#|  window.__r = load;
#|})();
#|`;
#|
#|function buildBundle() {
#|  const files = [...walk(path.join(ROOT, "src")), ...walk(path.join(ROOT, "demo"))].filter(
#|    (f) => /\.(ts|tsx)$/.test(f) && !f.endsWith(".d.ts") && !f.endsWith(path.join("app", "layout.tsx")));
#|
#|  let mods = "";
#|  for (const f of files) {
#|    const id = path.relative(ROOT, f).split(path.sep).join("/").replace(/\.(ts|tsx)$/, "");
#|    const out = ts.transpileModule(fs.readFileSync(f, "utf8"), {
#|      fileName: f,
#|      reportDiagnostics: true,
#|      compilerOptions: {
#|        module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2019, jsx: ts.JsxEmit.React, esModuleInterop: true,
#|      },
#|    });
#|    if (out.diagnostics && out.diagnostics.length) {
#|      throw new Error(id + ": " + out.diagnostics.map((d) => ts.flattenDiagnosticMessageText(d.messageText, "\n")).join("; "));
#|    }
#|    mods += `__d(${JSON.stringify(id)}, function (exports, require, module) {\n${out.outputText}\n});\n`;
#|  }
#|  const js = (RUNTIME + mods + `__r("demo/entry");\n`).replace(/<\/script/gi, "<\\/script");
#|  const css = fs.readFileSync(path.join(ROOT, "src/app/globals.css"), "utf8");
#|  return { js, css, modules: files.length };
#|}
#|
#|const DEMO_CSS = `
#|:root { color-scheme: dark; box-sizing: border-box; padding-top: env(safe-area-inset-top, 0px); padding-bottom: env(safe-area-inset-bottom, 0px); }
#|html { scroll-padding-top: env(safe-area-inset-top, 0px); }
#|.dock { bottom: calc(22px + env(safe-area-inset-bottom, 0px)); }
#|.app { padding-top: 48px; }
#|`;
#|
#|function page({ js, css }, scripts) {
#|  return `<!doctype html>
#|<html lang="en">
#|<head>
#|<meta charset="utf-8">
#|<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
#|<meta name="theme-color" content="#030712">
#|<title>XAI-AMLBench | Compliance Console (demo)</title>
#|<style>${css}${DEMO_CSS}</style>
#|</head>
#|<body>
#|<div class="demo-badge">Demo mode &middot; sample data &middot; no live backend</div>
#|<div id="root"></div>
#|${scripts}
#|<script>
#|if (!window.React || !window.ReactDOM) {
#|  document.getElementById("root").innerHTML = '<p style="padding:40px;color:#e8f0ff;font-family:system-ui">Could not load React from cdnjs.cloudflare.com. Check your internet connection and reload.</p>';
#|} else {
#|${js}
#|}
#|</script>
#|</body>
#|</html>
#|`;
#|}
#|
#|module.exports = { buildBundle, page, REACT, REACT_DOM };
#|
#|if (require.main === module) {
#|  const bundle = buildBundle();
#|  const html = page(bundle, `<script src="${REACT}"></script>\n<script src="${REACT_DOM}"></script>`);
#|  const out = path.join(ROOT, "preview", "demo.html");
#|  fs.mkdirSync(path.dirname(out), { recursive: true });
#|  fs.writeFileSync(out, html);
#|  console.log(`built ${out}  (${bundle.modules} TypeScript modules, ${(html.length / 1024).toFixed(0)} KB)`);
#|}
#|@@@@END
#|@@@@FILE: demo/demoData.ts
#|/* AUTO-GENERATED by the demo data script from real aml_synth output + the real sar_generator.
#| * Used only by demo/mockApi.ts (the offline demo). The production app never imports this file. */
#|import type { Explanation } from "@/types";
#|
#|export interface DemoAccount {
#|  id: string;
#|  score: number;
#|  label: string;
#|}
#|
#|export const DEMO_ACCOUNTS: DemoAccount[] = [
#| {
#|  "id": "ACC0001449",
#|  "score": 0.97,
#|  "label": "smurfing"
#| },
#| {
#|  "id": "ACC0000528",
#|  "score": 0.96,
#|  "label": "shell_company"
#| },
#| {
#|  "id": "ACC0000283",
#|  "score": 0.91,
#|  "label": "cyclic_loop"
#| },
#| {
#|  "id": "ACC0003271",
#|  "score": 0.93,
#|  "label": "scatter_gather"
#| },
#| {
#|  "id": "ACC0005473",
#|  "score": 0.88,
#|  "label": "cross_border_velocity"
#| },
#| {
#|  "id": "ACC0002825",
#|  "score": 0.04,
#|  "label": "benign"
#| },
#| {
#|  "id": "ACC0000819",
#|  "score": 0.07,
#|  "label": "benign"
#| },
#| {
#|  "id": "ACC0003621",
#|  "score": 0.11,
#|  "label": "benign"
#| }
#|];
#|
#|/** One explained account per line. Each entry is what POST /explain returns. */
#|export const DEMO_EXPLANATIONS: Record<string, Explanation> = {
#|  "ACC0001449": {"account_id": "ACC0001449", "risk_score": 0.97, "threshold": 0.62, "reporting_threshold": 10000, "model": "gatv2", "method": "GNNExplainer", "nodes": [{"account_id": "ACC0001449", "account_type": "individual", "country": "FR"}, {"account_id": "ACC0005009", "account_type": "individual", "country": "NG"}, {"account_id": "ACC0005002", "account_type": "individual", "country": "IN"}, {"account_id": "ACC0005004", "account_type": "individual", "country": "AE"}, {"account_id": "ACC0005008", "account_type": "individual", "country": "IN"}, {"account_id": "ACC0005003", "account_type": "individual", "country": "FR"}, {"account_id": "ACC0005007", "account_type": "individual", "country": "US"}, {"account_id": "ACC0005005", "account_type": "individual", "country": "NG"}, {"account_id": "ACC0005010", "account_type": "individual", "country": "BR"}, {"account_id": "ACC0005006", "account_type": "individual", "country": "US"}, {"account_id": "ACC0005011", "account_type": "individual", "country": "GB"}, {"account_id": "ACC0005000", "account_type": "business", "country": "DE"}, {"account_id": "ACC0005001", "account_type": "individual", "country": "DE"}, {"account_id": "ACC0005012", "account_type": "individual", "country": "AE"}], "edges": [{"tx_id": "TX000001531", "src": "ACC0001449", "dst": "ACC0005009", "amount": 8748.4, "timestamp": 1767327013, "cross_border": 1, "importance": 0.73095}, {"tx_id": "TX000001747", "src": "ACC0001449", "dst": "ACC0005002", "amount": 9327.9, "timestamp": 1767341273, "cross_border": 1, "importance": 0.77391}, {"tx_id": "TX000001810", "src": "ACC0001449", "dst": "ACC0005004", "amount": 8961.11, "timestamp": 1767345429, "cross_border": 1, "importance": 0.91968}, {"tx_id": "TX000001882", "src": "ACC0001449", "dst": "ACC0005008", "amount": 9315.29, "timestamp": 1767349570, "cross_border": 1, "importance": 0.73626}, {"tx_id": "TX000001972", "src": "ACC0001449", "dst": "ACC0005003", "amount": 8456.88, "timestamp": 1767355171, "cross_border": 0, "importance": 0.75314}, {"tx_id": "TX000001995", "src": "ACC0001449", "dst": "ACC0005007", "amount": 9432.58, "timestamp": 1767356663, "cross_border": 1, "importance": 0.78495}, {"tx_id": "TX000002000", "src": "ACC0001449", "dst": "ACC0005005", "amount": 8277.01, "timestamp": 1767357071, "cross_border": 1, "importance": 0.62386}, {"tx_id": "TX000002157", "src": "ACC0001449", "dst": "ACC0005010", "amount": 8227.41, "timestamp": 1767367070, "cross_border": 1, "importance": 0.75476}, {"tx_id": "TX000002170", "src": "ACC0001449", "dst": "ACC0005006", "amount": 8972.81, "timestamp": 1767367768, "cross_border": 1, "importance": 0.80195}, {"tx_id": "TX000002186", "src": "ACC0001449", "dst": "ACC0005011", "amount": 8648.84, "timestamp": 1767368766, "cross_border": 1, "importance": 0.86719}, {"tx_id": "TX000002188", "src": "ACC0005002", "dst": "ACC0005000", "amount": 9111.48, "timestamp": 1767368864, "cross_border": 1, "importance": 0.15577}, {"tx_id": "TX000002225", "src": "ACC0005005", "dst": "ACC0005000", "amount": 7952.88, "timestamp": 1767371500, "cross_border": 1, "importance": 0.23529}, {"tx_id": "TX000002248", "src": "ACC0001449", "dst": "ACC0005001", "amount": 8466.42, "timestamp": 1767372983, "cross_border": 1, "importance": 0.58627}, {"tx_id": "TX000002314", "src": "ACC0005009", "dst": "ACC0005000", "amount": 8622.65, "timestamp": 1767377716, "cross_border": 1, "importance": 0.42766}, {"tx_id": "TX000002323", "src": "ACC0005004", "dst": "ACC0005000", "amount": 8618.88, "timestamp": 1767378465, "cross_border": 1, "importance": 0.38351}, {"tx_id": "TX000002326", "src": "ACC0005008", "dst": "ACC0005000", "amount": 8863.72, "timestamp": 1767378731, "cross_border": 1, "importance": 0.13591}, {"tx_id": "TX000002435", "src": "ACC0005011", "dst": "ACC0005000", "amount": 8556.03, "timestamp": 1767384990, "cross_border": 1, "importance": 0.49323}, {"tx_id": "TX000002562", "src": "ACC0001449", "dst": "ACC0005012", "amount": 8760.89, "timestamp": 1767393627, "cross_border": 1, "importance": 0.9359}, {"tx_id": "TX000002743", "src": "ACC0005003", "dst": "ACC0005000", "amount": 8346.14, "timestamp": 1767404671, "cross_border": 1, "importance": 0.36849}, {"tx_id": "TX000002792", "src": "ACC0005007", "dst": "ACC0005000", "amount": 9041.85, "timestamp": 1767407852, "cross_border": 1, "importance": 0.35391}, {"tx_id": "TX000002879", "src": "ACC0005006", "dst": "ACC0005000", "amount": 8727.65, "timestamp": 1767412307, "cross_border": 1, "importance": 0.17985}, {"tx_id": "TX000003084", "src": "ACC0005010", "dst": "ACC0005000", "amount": 8003.49, "timestamp": 1767425230, "cross_border": 1, "importance": 0.1257}, {"tx_id": "TX000003143", "src": "ACC0005001", "dst": "ACC0005000", "amount": 8120.17, "timestamp": 1767429258, "cross_border": 0, "importance": 0.32078}, {"tx_id": "TX000003286", "src": "ACC0005012", "dst": "ACC0005000", "amount": 8414.65, "timestamp": 1767438922, "cross_border": 1, "importance": 0.14263}], "top_features": [{"feature": "near_thr_ratio", "weight": 0.31}, {"feature": "burst_6h", "weight": 0.19}, {"feature": "flow_ratio", "weight": 0.15}, {"feature": "out_uniq", "weight": 0.12}, {"feature": "retained_frac", "weight": 0.09}]},
#|  "ACC0000528": {"account_id": "ACC0000528", "risk_score": 0.96, "threshold": 0.62, "reporting_threshold": 10000, "model": "gatv2", "method": "GNNExplainer", "nodes": [{"account_id": "ACC0000528", "account_type": "individual", "country": "CA"}, {"account_id": "ACC0005398", "account_type": "shell", "country": "BZ"}, {"account_id": "ACC0005399", "account_type": "shell", "country": "KY"}, {"account_id": "ACC0005400", "account_type": "shell", "country": "BZ"}, {"account_id": "ACC0005397", "account_type": "business", "country": "US"}], "edges": [{"tx_id": "TX000027163", "src": "ACC0000528", "dst": "ACC0005398", "amount": 217855.54, "timestamp": 1768956500, "cross_border": 1, "importance": 0.62608}, {"tx_id": "TX000028488", "src": "ACC0005398", "dst": "ACC0005399", "amount": 217086.63, "timestamp": 1769045035, "cross_border": 1, "importance": 0.21194}, {"tx_id": "TX000028796", "src": "ACC0005399", "dst": "ACC0005400", "amount": 214467.46, "timestamp": 1769065773, "cross_border": 1, "importance": 0.13143}, {"tx_id": "TX000029551", "src": "ACC0005400", "dst": "ACC0005397", "amount": 213073.76, "timestamp": 1769111464, "cross_border": 1, "importance": 0.2963}], "top_features": [{"feature": "retained_frac", "weight": 0.28}, {"feature": "xb_out_ratio", "weight": 0.21}, {"feature": "out_amt_max", "weight": 0.16}, {"feature": "flow_ratio", "weight": 0.11}, {"feature": "is_offshore", "weight": 0.08}]},
#|  "ACC0000283": {"account_id": "ACC0000283", "risk_score": 0.91, "threshold": 0.62, "reporting_threshold": 10000, "model": "gatv2", "method": "GNNExplainer", "nodes": [{"account_id": "ACC0000283", "account_type": "individual", "country": "US"}, {"account_id": "ACC0005353", "account_type": "business", "country": "IN"}, {"account_id": "ACC0005354", "account_type": "business", "country": "FR"}, {"account_id": "ACC0005355", "account_type": "business", "country": "CA"}, {"account_id": "ACC0005356", "account_type": "business", "country": "JP"}], "edges": [{"tx_id": "TX000013336", "src": "ACC0000283", "dst": "ACC0005353", "amount": 151105.12, "timestamp": 1768072206, "cross_border": 1, "importance": 0.72621}, {"tx_id": "TX000014109", "src": "ACC0005353", "dst": "ACC0005354", "amount": 149983.67, "timestamp": 1768121519, "cross_border": 1, "importance": 0.44012}, {"tx_id": "TX000014598", "src": "ACC0005354", "dst": "ACC0005355", "amount": 149218.98, "timestamp": 1768152246, "cross_border": 1, "importance": 0.31727}, {"tx_id": "TX000015124", "src": "ACC0005355", "dst": "ACC0005356", "amount": 147847.55, "timestamp": 1768186008, "cross_border": 1, "importance": 0.36331}, {"tx_id": "TX000015365", "src": "ACC0005356", "dst": "ACC0000283", "amount": 146890.18, "timestamp": 1768201732, "cross_border": 1, "importance": 0.74991}, {"tx_id": "TX000016122", "src": "ACC0000283", "dst": "ACC0005353", "amount": 144730.38, "timestamp": 1768248500, "cross_border": 1, "importance": 0.81498}, {"tx_id": "TX000016671", "src": "ACC0005353", "dst": "ACC0005354", "amount": 142846.04, "timestamp": 1768285861, "cross_border": 1, "importance": 0.29379}, {"tx_id": "TX000017109", "src": "ACC0005354", "dst": "ACC0005355", "amount": 142476.01, "timestamp": 1768313013, "cross_border": 1, "importance": 0.2257}, {"tx_id": "TX000017871", "src": "ACC0005355", "dst": "ACC0005356", "amount": 141335.35, "timestamp": 1768359826, "cross_border": 1, "importance": 0.49911}, {"tx_id": "TX000018411", "src": "ACC0005356", "dst": "ACC0000283", "amount": 139235.95, "timestamp": 1768393477, "cross_border": 1, "importance": 0.94828}, {"tx_id": "TX000018671", "src": "ACC0000283", "dst": "ACC0005353", "amount": 137530.53, "timestamp": 1768411021, "cross_border": 1, "importance": 0.88609}, {"tx_id": "TX000019432", "src": "ACC0005353", "dst": "ACC0005354", "amount": 135807.83, "timestamp": 1768460709, "cross_border": 1, "importance": 0.38897}, {"tx_id": "TX000020036", "src": "ACC0005354", "dst": "ACC0005355", "amount": 134856.89, "timestamp": 1768498137, "cross_border": 1, "importance": 0.23981}, {"tx_id": "TX000020120", "src": "ACC0005355", "dst": "ACC0005356", "amount": 133424.12, "timestamp": 1768503032, "cross_border": 1, "importance": 0.20727}, {"tx_id": "TX000020214", "src": "ACC0005356", "dst": "ACC0000283", "amount": 133023.3, "timestamp": 1768508693, "cross_border": 1, "importance": 0.66562}], "top_features": [{"feature": "flow_ratio", "weight": 0.3}, {"feature": "retained_frac", "weight": 0.22}, {"feature": "in_deg", "weight": 0.14}, {"feature": "out_deg", "weight": 0.12}, {"feature": "out_amt_sum", "weight": 0.09}]},
#|  "ACC0003271": {"account_id": "ACC0003271", "risk_score": 0.93, "threshold": 0.62, "reporting_threshold": 10000, "model": "gatv2", "method": "GNNExplainer", "nodes": [{"account_id": "ACC0003271", "account_type": "business", "country": "AE"}, {"account_id": "ACC0005218", "account_type": "business", "country": "US"}, {"account_id": "ACC0005212", "account_type": "business", "country": "US"}, {"account_id": "ACC0005219", "account_type": "individual", "country": "US"}, {"account_id": "ACC0005211", "account_type": "individual", "country": "AE"}, {"account_id": "ACC0005209", "account_type": "business", "country": "FR"}, {"account_id": "ACC0005214", "account_type": "individual", "country": "CA"}, {"account_id": "ACC0005215", "account_type": "individual", "country": "IN"}, {"account_id": "ACC0005217", "account_type": "business", "country": "US"}, {"account_id": "ACC0005213", "account_type": "business", "country": "US"}, {"account_id": "ACC0005216", "account_type": "business", "country": "NG"}, {"account_id": "ACC0005208", "account_type": "business", "country": "US"}, {"account_id": "ACC0005210", "account_type": "business", "country": "AE"}], "edges": [{"tx_id": "TX000003945", "src": "ACC0003271", "dst": "ACC0005218", "amount": 48495.35, "timestamp": 1767480177, "cross_border": 1, "importance": 0.57809}, {"tx_id": "TX000003977", "src": "ACC0003271", "dst": "ACC0005212", "amount": 54295.48, "timestamp": 1767482199, "cross_border": 1, "importance": 0.85652}, {"tx_id": "TX000004070", "src": "ACC0003271", "dst": "ACC0005219", "amount": 30337.61, "timestamp": 1767488286, "cross_border": 1, "importance": 0.71016}, {"tx_id": "TX000004099", "src": "ACC0003271", "dst": "ACC0005211", "amount": 82803.15, "timestamp": 1767490082, "cross_border": 0, "importance": 0.88863}, {"tx_id": "TX000004103", "src": "ACC0003271", "dst": "ACC0005209", "amount": 50731.07, "timestamp": 1767490542, "cross_border": 1, "importance": 0.70461}, {"tx_id": "TX000004214", "src": "ACC0003271", "dst": "ACC0005214", "amount": 1834.39, "timestamp": 1767498393, "cross_border": 1, "importance": 0.93322}, {"tx_id": "TX000004216", "src": "ACC0003271", "dst": "ACC0005215", "amount": 57615.79, "timestamp": 1767498510, "cross_border": 1, "importance": 0.88892}, {"tx_id": "TX000004221", "src": "ACC0003271", "dst": "ACC0005217", "amount": 11.26, "timestamp": 1767498693, "cross_border": 1, "importance": 0.55022}, {"tx_id": "TX000004255", "src": "ACC0003271", "dst": "ACC0005213", "amount": 2052.36, "timestamp": 1767500829, "cross_border": 1, "importance": 0.63389}, {"tx_id": "TX000004271", "src": "ACC0003271", "dst": "ACC0005216", "amount": 22207.16, "timestamp": 1767501857, "cross_border": 1, "importance": 0.91411}, {"tx_id": "TX000004312", "src": "ACC0005218", "dst": "ACC0005208", "amount": 47313.63, "timestamp": 1767504065, "cross_border": 0, "importance": 0.2986}, {"tx_id": "TX000004434", "src": "ACC0003271", "dst": "ACC0005210", "amount": 39873.45, "timestamp": 1767512052, "cross_border": 0, "importance": 0.94214}, {"tx_id": "TX000004629", "src": "ACC0005217", "dst": "ACC0005208", "amount": 11.16, "timestamp": 1767524551, "cross_border": 0, "importance": 0.27102}, {"tx_id": "TX000004962", "src": "ACC0005210", "dst": "ACC0005208", "amount": 39574.94, "timestamp": 1767545578, "cross_border": 1, "importance": 0.14775}, {"tx_id": "TX000005108", "src": "ACC0005219", "dst": "ACC0005208", "amount": 30060.4, "timestamp": 1767554651, "cross_border": 0, "importance": 0.35919}, {"tx_id": "TX000005234", "src": "ACC0005212", "dst": "ACC0005208", "amount": 53712.63, "timestamp": 1767562515, "cross_border": 0, "importance": 0.41583}, {"tx_id": "TX000005265", "src": "ACC0005213", "dst": "ACC0005208", "amount": 1993.67, "timestamp": 1767564310, "cross_border": 0, "importance": 0.22251}, {"tx_id": "TX000005279", "src": "ACC0005214", "dst": "ACC0005208", "amount": 1803.59, "timestamp": 1767565459, "cross_border": 1, "importance": 0.15311}, {"tx_id": "TX000005371", "src": "ACC0005215", "dst": "ACC0005208", "amount": 56100.39, "timestamp": 1767570369, "cross_border": 1, "importance": 0.24638}, {"tx_id": "TX000005578", "src": "ACC0005211", "dst": "ACC0005208", "amount": 81876.18, "timestamp": 1767583924, "cross_border": 1, "importance": 0.48635}, {"tx_id": "TX000005757", "src": "ACC0005216", "dst": "ACC0005208", "amount": 21851.32, "timestamp": 1767594746, "cross_border": 1, "importance": 0.40806}, {"tx_id": "TX000005768", "src": "ACC0005209", "dst": "ACC0005208", "amount": 49272.75, "timestamp": 1767595122, "cross_border": 1, "importance": 0.16484}], "top_features": [{"feature": "out_uniq", "weight": 0.29}, {"feature": "flow_ratio", "weight": 0.2}, {"feature": "retained_frac", "weight": 0.16}, {"feature": "in_uniq", "weight": 0.1}, {"feature": "out_amt_sum", "weight": 0.08}]},
#|  "ACC0005473": {"account_id": "ACC0005473", "risk_score": 0.88, "threshold": 0.62, "reporting_threshold": 10000, "model": "gatv2", "method": "GNNExplainer", "nodes": [{"account_id": "ACC0005473", "account_type": "individual", "country": "IN"}, {"account_id": "ACC0002541", "account_type": "business", "country": "DE"}, {"account_id": "ACC0001809", "account_type": "individual", "country": "CA"}, {"account_id": "ACC0003808", "account_type": "business", "country": "ZA"}, {"account_id": "ACC0004402", "account_type": "individual", "country": "JP"}, {"account_id": "ACC0003165", "account_type": "individual", "country": "US"}, {"account_id": "ACC0000997", "account_type": "individual", "country": "GB"}, {"account_id": "ACC0000684", "account_type": "individual", "country": "BR"}, {"account_id": "ACC0003376", "account_type": "business", "country": "GB"}, {"account_id": "ACC0001724", "account_type": "individual", "country": "US"}, {"account_id": "ACC0000935", "account_type": "individual", "country": "GB"}, {"account_id": "ACC0004078", "account_type": "individual", "country": "CA"}, {"account_id": "ACC0000613", "account_type": "individual", "country": "US"}, {"account_id": "ACC0001181", "account_type": "individual", "country": "GB"}, {"account_id": "ACC0001600", "account_type": "individual", "country": "US"}, {"account_id": "ACC0003326", "account_type": "business", "country": "US"}, {"account_id": "ACC0002048", "account_type": "individual", "country": "NG"}, {"account_id": "ACC0003861", "account_type": "individual", "country": "JP"}, {"account_id": "ACC0000864", "account_type": "individual", "country": "US"}, {"account_id": "ACC0002464", "account_type": "individual", "country": "ZA"}, {"account_id": "ACC0001986", "account_type": "individual", "country": "GB"}, {"account_id": "ACC0004445", "account_type": "individual", "country": "US"}, {"account_id": "ACC0001045", "account_type": "individual", "country": "BR"}, {"account_id": "ACC0001517", "account_type": "individual", "country": "US"}, {"account_id": "ACC0001679", "account_type": "individual", "country": "JP"}, {"account_id": "ACC0004858", "account_type": "individual", "country": "US"}, {"account_id": "ACC0002258", "account_type": "individual", "country": "GB"}, {"account_id": "ACC0001539", "account_type": "individual", "country": "DE"}, {"account_id": "ACC0002329", "account_type": "business", "country": "US"}, {"account_id": "ACC0000929", "account_type": "individual", "country": "CA"}, {"account_id": "ACC0001379", "account_type": "individual", "country": "ZA"}, {"account_id": "ACC0002718", "account_type": "individual", "country": "AE"}, {"account_id": "ACC0001762", "account_type": "individual", "country": "KE"}], "edges": [{"tx_id": "TX000008371", "src": "ACC0005473", "dst": "ACC0002541", "amount": 2109.37, "timestamp": 1767760385, "cross_border": 1, "importance": 0.64856}, {"tx_id": "TX000008383", "src": "ACC0005473", "dst": "ACC0001809", "amount": 4420.42, "timestamp": 1767761079, "cross_border": 1, "importance": 0.59042}, {"tx_id": "TX000008389", "src": "ACC0003808", "dst": "ACC0005473", "amount": 2716.85, "timestamp": 1767761541, "cross_border": 1, "importance": 0.57396}, {"tx_id": "TX000008408", "src": "ACC0005473", "dst": "ACC0004402", "amount": 7495.99, "timestamp": 1767762580, "cross_border": 1, "importance": 0.86881}, {"tx_id": "TX000008438", "src": "ACC0003165", "dst": "ACC0005473", "amount": 3692.16, "timestamp": 1767764052, "cross_border": 1, "importance": 0.62107}, {"tx_id": "TX000008464", "src": "ACC0005473", "dst": "ACC0000997", "amount": 5408.19, "timestamp": 1767765338, "cross_border": 1, "importance": 0.77372}, {"tx_id": "TX000008479", "src": "ACC0000684", "dst": "ACC0005473", "amount": 4058.83, "timestamp": 1767766057, "cross_border": 1, "importance": 0.72897}, {"tx_id": "TX000008489", "src": "ACC0005473", "dst": "ACC0003376", "amount": 6973.2, "timestamp": 1767766744, "cross_border": 1, "importance": 0.62627}, {"tx_id": "TX000008492", "src": "ACC0001724", "dst": "ACC0005473", "amount": 5724.02, "timestamp": 1767767013, "cross_border": 1, "importance": 0.84276}, {"tx_id": "TX000008524", "src": "ACC0000935", "dst": "ACC0005473", "amount": 5319.81, "timestamp": 1767768497, "cross_border": 1, "importance": 0.60239}, {"tx_id": "TX000008531", "src": "ACC0004078", "dst": "ACC0005473", "amount": 9343.06, "timestamp": 1767768854, "cross_border": 1, "importance": 0.80749}, {"tx_id": "TX000008551", "src": "ACC0000613", "dst": "ACC0005473", "amount": 3027.92, "timestamp": 1767770045, "cross_border": 1, "importance": 0.5966}, {"tx_id": "TX000008552", "src": "ACC0005473", "dst": "ACC0001181", "amount": 3945.88, "timestamp": 1767770057, "cross_border": 1, "importance": 0.7183}, {"tx_id": "TX000008558", "src": "ACC0001600", "dst": "ACC0005473", "amount": 3587.28, "timestamp": 1767770512, "cross_border": 1, "importance": 0.63515}, {"tx_id": "TX000008559", "src": "ACC0005473", "dst": "ACC0003326", "amount": 8430.75, "timestamp": 1767770523, "cross_border": 1, "importance": 0.65792}, {"tx_id": "TX000008564", "src": "ACC0002048", "dst": "ACC0005473", "amount": 3747.78, "timestamp": 1767770688, "cross_border": 1, "importance": 0.93837}, {"tx_id": "TX000008567", "src": "ACC0005473", "dst": "ACC0003861", "amount": 2576.04, "timestamp": 1767770884, "cross_border": 1, "importance": 0.87136}, {"tx_id": "TX000008571", "src": "ACC0005473", "dst": "ACC0000864", "amount": 3189.61, "timestamp": 1767771211, "cross_border": 1, "importance": 0.67166}, {"tx_id": "TX000008576", "src": "ACC0002464", "dst": "ACC0005473", "amount": 5301.43, "timestamp": 1767771476, "cross_border": 1, "importance": 0.90395}, {"tx_id": "TX000008577", "src": "ACC0001986", "dst": "ACC0005473", "amount": 4895.67, "timestamp": 1767771490, "cross_border": 1, "importance": 0.63428}, {"tx_id": "TX000008609", "src": "ACC0005473", "dst": "ACC0004445", "amount": 6728.51, "timestamp": 1767772901, "cross_border": 1, "importance": 0.70771}, {"tx_id": "TX000008611", "src": "ACC0001045", "dst": "ACC0005473", "amount": 3635.64, "timestamp": 1767772959, "cross_border": 1, "importance": 0.89175}, {"tx_id": "TX000008625", "src": "ACC0001517", "dst": "ACC0005473", "amount": 5391.47, "timestamp": 1767773867, "cross_border": 1, "importance": 0.80673}, {"tx_id": "TX000008650", "src": "ACC0005473", "dst": "ACC0001679", "amount": 5779.57, "timestamp": 1767775708, "cross_border": 1, "importance": 0.59013}, {"tx_id": "TX000008655", "src": "ACC0005473", "dst": "ACC0004858", "amount": 7549.63, "timestamp": 1767775992, "cross_border": 1, "importance": 0.94572}, {"tx_id": "TX000008678", "src": "ACC0002258", "dst": "ACC0005473", "amount": 1060.81, "timestamp": 1767777498, "cross_border": 1, "importance": 0.6353}, {"tx_id": "TX000008689", "src": "ACC0005473", "dst": "ACC0001539", "amount": 1301.13, "timestamp": 1767778245, "cross_border": 1, "importance": 0.65331}, {"tx_id": "TX000008692", "src": "ACC0005473", "dst": "ACC0002329", "amount": 4024.56, "timestamp": 1767778346, "cross_border": 1, "importance": 0.85908}, {"tx_id": "TX000008707", "src": "ACC0005473", "dst": "ACC0000929", "amount": 4863.41, "timestamp": 1767779553, "cross_border": 1, "importance": 0.68158}, {"tx_id": "TX000008720", "src": "ACC0001379", "dst": "ACC0005473", "amount": 6754.46, "timestamp": 1767780479, "cross_border": 1, "importance": 0.66853}, {"tx_id": "TX000008738", "src": "ACC0002718", "dst": "ACC0005473", "amount": 7812.01, "timestamp": 1767781164, "cross_border": 1, "importance": 0.57936}, {"tx_id": "TX000008740", "src": "ACC0005473", "dst": "ACC0001762", "amount": 6000.96, "timestamp": 1767781360, "cross_border": 1, "importance": 0.58605}], "top_features": [{"feature": "burst_6h", "weight": 0.3}, {"feature": "n_cp_countries", "weight": 0.24}, {"feature": "xb_in_ratio", "weight": 0.16}, {"feature": "xb_out_ratio", "weight": 0.12}, {"feature": "in_deg", "weight": 0.07}]},
#|  "ACC0002825": {"account_id": "ACC0002825", "risk_score": 0.04, "threshold": 0.62, "reporting_threshold": 10000, "model": "gatv2", "method": "GNNExplainer", "nodes": [{"account_id": "ACC0002825", "account_type": "individual", "country": "IN"}, {"account_id": "ACC0004648", "account_type": "business", "country": "IN"}, {"account_id": "ACC0001936", "account_type": "business", "country": "IN"}, {"account_id": "ACC0003589", "account_type": "individual", "country": "IN"}, {"account_id": "ACC0000200", "account_type": "individual", "country": "IN"}, {"account_id": "ACC0002494", "account_type": "individual", "country": "IN"}, {"account_id": "ACC0003900", "account_type": "individual", "country": "IN"}, {"account_id": "ACC0003609", "account_type": "individual", "country": "KE"}, {"account_id": "ACC0003059", "account_type": "individual", "country": "IN"}, {"account_id": "ACC0003525", "account_type": "individual", "country": "IN"}], "edges": [{"tx_id": "TX000000026", "src": "ACC0002825", "dst": "ACC0004648", "amount": 117.26, "timestamp": 1767227880, "cross_border": 0, "importance": 0.16655}, {"tx_id": "TX000004197", "src": "ACC0001936", "dst": "ACC0002825", "amount": 20.17, "timestamp": 1767497395, "cross_border": 0, "importance": 0.0986}, {"tx_id": "TX000010238", "src": "ACC0003589", "dst": "ACC0002825", "amount": 2032.54, "timestamp": 1767876696, "cross_border": 0, "importance": 0.17026}, {"tx_id": "TX000024404", "src": "ACC0002825", "dst": "ACC0000200", "amount": 154.27, "timestamp": 1768781859, "cross_border": 0, "importance": 0.12434}, {"tx_id": "TX000024434", "src": "ACC0002494", "dst": "ACC0002825", "amount": 257.12, "timestamp": 1768783062, "cross_border": 0, "importance": 0.14064}, {"tx_id": "TX000028586", "src": "ACC0002825", "dst": "ACC0003900", "amount": 188.96, "timestamp": 1769050881, "cross_border": 0, "importance": 0.24183}, {"tx_id": "TX000028904", "src": "ACC0003609", "dst": "ACC0002825", "amount": 102.28, "timestamp": 1769073156, "cross_border": 1, "importance": 0.14674}, {"tx_id": "TX000037327", "src": "ACC0002825", "dst": "ACC0003059", "amount": 436.02, "timestamp": 1769603960, "cross_border": 0, "importance": 0.16491}, {"tx_id": "TX000040372", "src": "ACC0003525", "dst": "ACC0002825", "amount": 108.03, "timestamp": 1769802950, "cross_border": 0, "importance": 0.22331}], "top_features": [{"feature": "in_deg", "weight": 0.04}, {"feature": "out_amt_mean", "weight": 0.03}, {"feature": "age_days", "weight": 0.03}]},
#|  "ACC0000819": {"account_id": "ACC0000819", "risk_score": 0.07, "threshold": 0.62, "reporting_threshold": 10000, "model": "gatv2", "method": "GNNExplainer", "nodes": [{"account_id": "ACC0000819", "account_type": "business", "country": "US"}, {"account_id": "ACC0001988", "account_type": "individual", "country": "US"}, {"account_id": "ACC0004466", "account_type": "individual", "country": "US"}, {"account_id": "ACC0000831", "account_type": "individual", "country": "JP"}, {"account_id": "ACC0001879", "account_type": "individual", "country": "US"}, {"account_id": "ACC0001202", "account_type": "business", "country": "US"}, {"account_id": "ACC0002204", "account_type": "individual", "country": "US"}, {"account_id": "ACC0002194", "account_type": "individual", "country": "US"}, {"account_id": "ACC0003658", "account_type": "individual", "country": "US"}, {"account_id": "ACC0000129", "account_type": "individual", "country": "US"}], "edges": [{"tx_id": "TX000000036", "src": "ACC0000819", "dst": "ACC0001988", "amount": 469.02, "timestamp": 1767228443, "cross_border": 0, "importance": 0.08657}, {"tx_id": "TX000001079", "src": "ACC0004466", "dst": "ACC0000819", "amount": 4021.45, "timestamp": 1767298455, "cross_border": 0, "importance": 0.08083}, {"tx_id": "TX000002564", "src": "ACC0000831", "dst": "ACC0000819", "amount": 503.17, "timestamp": 1767393810, "cross_border": 1, "importance": 0.23168}, {"tx_id": "TX000011129", "src": "ACC0000819", "dst": "ACC0001879", "amount": 300.19, "timestamp": 1767931660, "cross_border": 0, "importance": 0.21356}, {"tx_id": "TX000023073", "src": "ACC0001202", "dst": "ACC0000819", "amount": 526.39, "timestamp": 1768692869, "cross_border": 0, "importance": 0.0999}, {"tx_id": "TX000026226", "src": "ACC0000819", "dst": "ACC0002204", "amount": 573.68, "timestamp": 1768896614, "cross_border": 0, "importance": 0.08796}, {"tx_id": "TX000027159", "src": "ACC0000819", "dst": "ACC0002194", "amount": 158.27, "timestamp": 1768956137, "cross_border": 0, "importance": 0.19788}, {"tx_id": "TX000028441", "src": "ACC0000819", "dst": "ACC0003658", "amount": 950.96, "timestamp": 1769041486, "cross_border": 0, "importance": 0.23808}, {"tx_id": "TX000028710", "src": "ACC0000819", "dst": "ACC0000129", "amount": 763.3, "timestamp": 1769059609, "cross_border": 0, "importance": 0.08932}], "top_features": [{"feature": "in_deg", "weight": 0.04}, {"feature": "out_amt_mean", "weight": 0.03}, {"feature": "age_days", "weight": 0.03}]},
#|  "ACC0003621": {"account_id": "ACC0003621", "risk_score": 0.11, "threshold": 0.62, "reporting_threshold": 10000, "model": "gatv2", "method": "GNNExplainer", "nodes": [{"account_id": "ACC0003621", "account_type": "individual", "country": "FR"}, {"account_id": "ACC0004329", "account_type": "business", "country": "KE"}, {"account_id": "ACC0003429", "account_type": "individual", "country": "FR"}, {"account_id": "ACC0003386", "account_type": "individual", "country": "IN"}, {"account_id": "ACC0000959", "account_type": "individual", "country": "FR"}, {"account_id": "ACC0004693", "account_type": "individual", "country": "FR"}, {"account_id": "ACC0002898", "account_type": "individual", "country": "FR"}, {"account_id": "ACC0003385", "account_type": "individual", "country": "FR"}, {"account_id": "ACC0002118", "account_type": "individual", "country": "FR"}, {"account_id": "ACC0003685", "account_type": "individual", "country": "US"}], "edges": [{"tx_id": "TX000000052", "src": "ACC0003621", "dst": "ACC0004329", "amount": 103.83, "timestamp": 1767229062, "cross_border": 1, "importance": 0.24003}, {"tx_id": "TX000002052", "src": "ACC0003621", "dst": "ACC0003429", "amount": 41.45, "timestamp": 1767360534, "cross_border": 0, "importance": 0.22644}, {"tx_id": "TX000005830", "src": "ACC0003386", "dst": "ACC0003621", "amount": 84.14, "timestamp": 1767599450, "cross_border": 1, "importance": 0.17071}, {"tx_id": "TX000007202", "src": "ACC0000959", "dst": "ACC0003621", "amount": 1291.5, "timestamp": 1767686541, "cross_border": 0, "importance": 0.13429}, {"tx_id": "TX000007700", "src": "ACC0003621", "dst": "ACC0004693", "amount": 1281.53, "timestamp": 1767717299, "cross_border": 0, "importance": 0.07077}, {"tx_id": "TX000009766", "src": "ACC0003621", "dst": "ACC0002898", "amount": 24.28, "timestamp": 1767846682, "cross_border": 0, "importance": 0.05774}, {"tx_id": "TX000015363", "src": "ACC0003621", "dst": "ACC0003385", "amount": 428.85, "timestamp": 1768201568, "cross_border": 0, "importance": 0.24254}, {"tx_id": "TX000019043", "src": "ACC0002118", "dst": "ACC0003621", "amount": 1126.88, "timestamp": 1768435886, "cross_border": 0, "importance": 0.09768}, {"tx_id": "TX000033543", "src": "ACC0003685", "dst": "ACC0003621", "amount": 221.22, "timestamp": 1769364217, "cross_border": 1, "importance": 0.19092}], "top_features": [{"feature": "in_deg", "weight": 0.04}, {"feature": "out_amt_mean", "weight": 0.03}, {"feature": "age_days", "weight": 0.03}]}
#|};
#|
#|export interface DemoNarrative {
#|  text: string;
#|  typology: string;
#|  sha256: string;
#|}
#|
#|export const DEMO_NARRATIVES: Record<string, DemoNarrative> = {
#| "ACC0001449": {
#|  "text": "Account ACC0001449 was flagged by the graph model with a risk score of 97% and is connected to 14 accounts in activity consistent with structuring (smurfing). The reviewed network contains 24 transactions totaling $207,975, occurring between 2026-01-02 and 2026-01-03 over about 31 hours. 23 of these transfers fell just below the $10,000 reporting threshold, which is consistent with structuring. Key model indicators include a high share of transfers just below the reporting threshold, an unusual burst of transactions within a short window and funds passing through with little balance retained, and the activity is referred for compliance officer review and potential Suspicious Activity Report filing.",
#|  "typology": "smurfing",
#|  "sha256": "db946f2404c25ba5b5a9f19bb03a7dc71e8e7547918e2fe2cab5f44a5a223039"
#| },
#| "ACC0000528": {
#|  "text": "Account ACC0000528 was flagged by the graph model with a risk score of 96% and is connected to 5 accounts in activity consistent with layering through shell entities. The reviewed network contains 4 transactions totaling $862,483, occurring between 2026-01-21 and 2026-01-22 over about 43 hours. 3 of the involved accounts are shell entities through which funds were passed in a layered chain. Key model indicators include funds passing through with little balance retained, an elevated share of cross-border transfers and shell or offshore account characteristics, and the activity is referred for compliance officer review and potential Suspicious Activity Report filing.",
#|  "typology": "shell_company",
#|  "sha256": "0ae0e004303a6bd43c967ea909f466a42b28ff1ff613aeea589968f6551c9be5"
#| },
#| "ACC0000283": {
#|  "text": "Account ACC0000283 was flagged by the graph model with a risk score of 91% and is connected to 5 accounts in activity consistent with circular fund flow. The reviewed network contains 15 transactions totaling $2,130,312, occurring between 2026-01-10 and 2026-01-15 over about 121 hours. Funds returned to the originating accounts through a closed loop, consistent with circular movement. Key model indicators include funds passing through with little balance retained, unusual counterparty fan-in and unusual counterparty fan-out, and the activity is referred for compliance officer review and potential Suspicious Activity Report filing.",
#|  "typology": "cyclic_loop",
#|  "sha256": "380c61695a209c6fae1ccda8e0a0ef1d9502202de63d2827a3dd189cffc25120"
#| },
#| "ACC0003271": {
#|  "text": "Account ACC0003271 was flagged by the graph model with a risk score of 93% and is connected to 13 accounts in activity consistent with scatter-gather layering. The reviewed network contains 22 transactions totaling $773,828, occurring between 2026-01-03 and 2026-01-05 over about 32 hours. Funds fanned out from a single source to several intermediaries and were then consolidated into one destination account. Key model indicators include unusual counterparty fan-out, funds passing through with little balance retained and unusual counterparty fan-in, and the activity is referred for compliance officer review and potential Suspicious Activity Report filing.",
#|  "typology": "scatter_gather",
#|  "sha256": "9ee1bdd9bfc5881f5ef52e2f8f785e1eabf48f436b11f6b22098b2b8e289aa60"
#| },
#| "ACC0005473": {
#|  "text": "Account ACC0005473 was flagged by the graph model with a risk score of 88% and is connected to 33 accounts in activity consistent with high-velocity cross-border activity. The reviewed network contains 32 transactions totaling $156,866, occurring on 2026-01-07 over about 6 hours. The activity spans 11 jurisdictions (AE, BR, CA, DE, GB, IN, JP, KE, NG, US, ZA) within a short window, indicating rapid cross-border movement of funds. Key model indicators include an unusual burst of transactions within a short window, counterparties spread across many jurisdictions and an elevated share of cross-border transfers, and the activity is referred for compliance officer review and potential Suspicious Activity Report filing.",
#|  "typology": "cross_border_velocity",
#|  "sha256": "ecc24a47faf1c8a6f2cf3227f817d09cd29fd74125521291996d5130ddf86578"
#| },
#| "ACC0002825": {
#|  "text": "Account ACC0002825 was flagged by the graph model with a risk score of 4% and is connected to 10 accounts in activity consistent with an unclassified anomalous pattern. The reviewed network contains 9 transactions totaling $3,417, occurring between 2026-01-01 and 2026-01-30 over about 715 hours. 1 of the 9 transactions were cross-border and the pattern deviates from expected account behavior. Key model indicators include unusual counterparty fan-in and a recently opened account, and the activity is referred for compliance officer review and potential Suspicious Activity Report filing.",
#|  "typology": "unclassified",
#|  "sha256": "be19a0ae5b2c2039ae1f1bcc7378023c6c65087c62b7c54c4f95aea496775838"
#| },
#| "ACC0000819": {
#|  "text": "Account ACC0000819 was flagged by the graph model with a risk score of 7% and is connected to 10 accounts in activity consistent with an unclassified anomalous pattern. The reviewed network contains 9 transactions totaling $8,266, occurring between 2026-01-01 and 2026-01-22 over about 509 hours. 1 of the 9 transactions were cross-border and the pattern deviates from expected account behavior. Key model indicators include unusual counterparty fan-in and a recently opened account, and the activity is referred for compliance officer review and potential Suspicious Activity Report filing.",
#|  "typology": "unclassified",
#|  "sha256": "3bf34cc8f0a895d5cff298b051b17c8e80c57abd4f1f90b4232a89e8d66c8d8c"
#| },
#| "ACC0003621": {
#|  "text": "Account ACC0003621 was flagged by the graph model with a risk score of 11% and is connected to 10 accounts in activity consistent with an unclassified anomalous pattern. The reviewed network contains 9 transactions totaling $4,604, occurring between 2026-01-01 and 2026-01-25 over about 593 hours. 3 of the 9 transactions were cross-border and the pattern deviates from expected account behavior. Key model indicators include unusual counterparty fan-in and a recently opened account, and the activity is referred for compliance officer review and potential Suspicious Activity Report filing.",
#|  "typology": "unclassified",
#|  "sha256": "43f04087f8fdf1e67600bea18770d092499a8509ed46909a848f15b7f46c4b0e"
#| }
#|};
#|@@@@END
#|@@@@FILE: demo/entry.tsx
#|/** Entry point of the offline demo bundle: mock backend + the real page component. */
#|import { createRoot } from "react-dom/client";
#|import Home from "@/app/page";
#|import { installMockApi } from "./mockApi";
#|
#|installMockApi();
#|const el = document.getElementById("root");
#|if (el) createRoot(el).render(<Home />);
#|@@@@END
#|@@@@FILE: demo/mockApi.ts
#|/**
#| * Offline demo backend. `installMockApi()` replaces window.fetch for /api/* so the real UI code
#| * (src/app/page.tsx and components) can run in a browser with no Docker, Postgres or GNN service.
#| *
#| * Shapes are checked against src/types.ts, so the mock cannot drift from what the real backend returns.
#| */
#|import type { Alert, AlertStatus, ModelInfo, SampleAccounts, ScanResponse } from "@/types";
#|import { DEMO_ACCOUNTS, DEMO_EXPLANATIONS, DEMO_NARRATIVES } from "./demoData";
#|
#|const THRESHOLD = 0.62;
#|const T0 = Date.now();
#|const iso = (minutesAgo: number): string => new Date(T0 - minutesAgo * 60_000).toISOString();
#|
#|const suspicious = DEMO_ACCOUNTS.filter((a) => a.label !== "benign");
#|const benign = DEMO_ACCOUNTS.filter((a) => a.label === "benign");
#|
#|const byLabel = (label: string) => DEMO_ACCOUNTS.find((a) => a.label === label);
#|
#|function makeAlert(
#|  id: number,
#|  accountId: string,
#|  status: AlertStatus,
#|  minutesAgo: number,
#|  withNarrative: boolean,
#|  reviewer: string | null = null,
#|): Alert {
#|  const account = DEMO_ACCOUNTS.find((a) => a.id === accountId);
#|  const narrative = withNarrative ? DEMO_NARRATIVES[accountId] : undefined;
#|  return {
#|    id,
#|    account_id: accountId,
#|    risk_score: account?.score ?? 0.9,
#|    status,
#|    typology: narrative?.typology ?? null,
#|    narrative: narrative?.text ?? null,
#|    narrative_source: narrative ? "template" : null,
#|    narrative_sha256: narrative?.sha256 ?? null,
#|    reviewed_by: reviewer,
#|    review_note: reviewer ? "" : null,
#|    created_at: iso(minutesAgo),
#|    reviewed_at: reviewer ? iso(Math.max(minutesAgo - 20, 1)) : null,
#|  };
#|}
#|
#|const alerts: Alert[] = [];
#|const seed = (label: string, status: AlertStatus, mins: number, narr: boolean, who: string | null = null): void => {
#|  const a = byLabel(label);
#|  if (a) alerts.push(makeAlert(alerts.length + 1, a.id, status, mins, narr, who));
#|};
#|seed("smurfing", "open", 42, true);
#|seed("shell_company", "open", 95, false);
#|seed("cyclic_loop", "confirmed", 610, true, "M. Wanjiru");
#|seed("cross_border_velocity", "dismissed", 1300, true, "M. Wanjiru");
#|let nextId = alerts.length + 1;
#|
#|const MODEL_INFO: ModelInfo = {
#|  model: "gatv2",
#|  hparams: { hidden: 64, num_layers: 2 },
#|  threshold: THRESHOLD,
#|  metrics: {
#|    val_auc_roc: 0.981, test_auc_roc: 0.983, test_pr_auc: 0.94, test_precision: 0.91, test_recall: 0.94,
#|    threshold: THRESHOLD, auc_target: 0.87, meets_auc_target: true,
#|  },
#|  targets: { auc_roc: 0.87, latency_ms: 100 },
#|  n_nodes: 5488,
#|  n_features: 26,
#|};
#|
#|interface Reply {
#|  status: number;
#|  body: unknown;
#|  delay: number;
#|}
#|const ok = (body: unknown, delay = 220): Reply => ({ status: 200, body, delay });
#|const fail = (status: number, detail: string, delay = 220): Reply => ({ status, body: { detail }, delay });
#|
#|function route(method: string, path: string, data: Record<string, unknown>): Reply {
#|  if (method === "GET" && path === "/alerts") {
#|    return ok([...alerts].sort((a, b) => b.created_at.localeCompare(a.created_at)));
#|  }
#|  if (method === "GET" && path === "/accounts/sample") {
#|    const body: SampleAccounts = { suspicious: suspicious.map((a) => a.id), benign: benign.map((a) => a.id) };
#|    return ok(body, 120);
#|  }
#|  if (method === "GET" && path === "/model/info") return ok(MODEL_INFO, 150);
#|
#|  if (method === "POST" && path === "/alerts/scan") {
#|    const id = String(data.account_id ?? "").trim();
#|    const account = DEMO_ACCOUNTS.find((a) => a.id === id);
#|    if (!account) {
#|      return fail(404, `unknown account_id '${id}'. In demo mode try one of the "Try:" accounts.`);
#|    }
#|    const flagged = account.score >= THRESHOLD;
#|    let alert: Alert | null = null;
#|    if (flagged) {
#|      alert = makeAlert(nextId++, account.id, "open", 0, false);
#|      alerts.push(alert);
#|    }
#|    const body: ScanResponse = {
#|      score: account.score,
#|      flagged,
#|      threshold: THRESHOLD,
#|      latency_ms: Math.round((6 + Math.random() * 12) * 10) / 10,
#|      alert,
#|    };
#|    return ok(body, 260);
#|  }
#|
#|  let m = path.match(/^\/accounts\/([^/]+)\/explain$/);
#|  if (method === "GET" && m) {
#|    const id = decodeURIComponent(m[1]);
#|    const explanation = DEMO_EXPLANATIONS[id];
#|    return explanation ? ok(explanation, 500) : fail(404, `unknown account_id '${id}'`);
#|  }
#|
#|  m = path.match(/^\/alerts\/(\d+)\/narrative$/);
#|  if (method === "POST" && m) {
#|    const alert = alerts.find((a) => a.id === Number(m?.[1]));
#|    if (!alert) return fail(404, "alert not found");
#|    const narrative = DEMO_NARRATIVES[alert.account_id];
#|    if (!narrative) return fail(502, "no narrative available for this account in demo mode");
#|    alert.narrative = narrative.text;
#|    alert.narrative_source = "template"; // the demo has no GPU/LLM, so this is the validated template path
#|    alert.narrative_sha256 = narrative.sha256;
#|    alert.typology = narrative.typology;
#|    return ok(alert, 1300); // pretend the LLM takes a moment
#|  }
#|
#|  m = path.match(/^\/alerts\/(\d+)\/decision$/);
#|  if (method === "POST" && m) {
#|    const alert = alerts.find((a) => a.id === Number(m?.[1]));
#|    if (!alert) return fail(404, "alert not found");
#|    if (alert.status !== "open") return fail(409, `alert already ${alert.status}`);
#|    const officer = String(data.officer ?? "").trim();
#|    const decision = data.decision;
#|    if (officer.length < 2) return fail(422, "officer name is required");
#|    if (decision !== "confirmed" && decision !== "dismissed") return fail(422, "invalid decision");
#|    alert.status = decision;
#|    alert.reviewed_by = officer;
#|    alert.review_note = String(data.note ?? "");
#|    alert.reviewed_at = new Date().toISOString();
#|    return ok(alert, 450);
#|  }
#|
#|  return fail(404, `Not found: ${method} ${path}`, 80);
#|}
#|
#|export function installMockApi(): void {
#|  const realFetch = window.fetch.bind(window);
#|  window.fetch = async (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
#|    const url = typeof input === "string" ? input : input instanceof URL ? input.pathname : input.url;
#|    if (!url.startsWith("/api/")) return realFetch(input, init);
#|
#|    let data: Record<string, unknown> = {};
#|    if (typeof init?.body === "string") {
#|      try {
#|        data = JSON.parse(init.body) as Record<string, unknown>;
#|      } catch {
#|        data = {};
#|      }
#|    }
#|    const reply = route((init?.method ?? "GET").toUpperCase(), url.slice(4), data);
#|    await new Promise<void>((resolve) => setTimeout(resolve, reply.delay));
#|    return new Response(JSON.stringify(reply.body), {
#|      status: reply.status,
#|      headers: { "Content-Type": "application/json" },
#|    });
#|  };
#|}
#|@@@@END
#|@@@@FILE: preview/demo.html
#|<!doctype html>
#|<html lang="en">
#|<head>
#|<meta charset="utf-8">
#|<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
#|<meta name="theme-color" content="#030712">
#|<title>XAI-AMLBench | Compliance Console (demo)</title>
#|<style>/* ============================ XAI-AMLBench  ·  deep-blue glass theme ============================ */
#|:root {
#|  --bg0: #030712; --bg1: #061029; --bg2: #0a1a44;
#|  --panel: rgba(12, 26, 60, .58); --panel-hi: rgba(22, 42, 92, .62);
#|  --line: rgba(96, 165, 250, .18); --line-hi: rgba(96, 165, 250, .42);
#|  --text: #e8f0ff; --muted: #8ea3cc; --dim: #5f7299;
#|  --blue: #3b82f6; --sky: #60a5fa; --cyan: #22d3ee; --violet: #818cf8;
#|  --red: #fb7185; --green: #34d399; --amber: #fbbf24;
#|  --radius: 18px;
#|  --ease: cubic-bezier(.22, 1, .36, 1);
#|  --spring: cubic-bezier(.34, 1.56, .64, 1);
#|}
#|* { box-sizing: border-box; }
#|html, body { min-height: 100%; }
#|body {
#|  margin: 0; color: var(--text); overflow-x: hidden;
#|  font: 15px/1.55 Inter, ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
#|  -webkit-font-smoothing: antialiased;
#|  background:
#|    radial-gradient(1100px 700px at 12% -8%, rgba(37, 99, 235, .28), transparent 60%),
#|    radial-gradient(900px 600px at 105% 8%, rgba(99, 102, 241, .22), transparent 58%),
#|    linear-gradient(180deg, var(--bg1), var(--bg0) 70%);
#|  background-attachment: fixed;
#|}
#|h1, h2, h3, p { margin: 0; }
#|h2 { font-size: 22px; margin-bottom: 4px; }
#|h3 { font-size: 15px; font-weight: 650; letter-spacing: .02em; margin-bottom: 14px; color: #cfe0ff; }
#|button { font: inherit; }
#|::selection { background: rgba(59, 130, 246, .45); }
#|::-webkit-scrollbar { width: 10px; height: 10px; }
#|::-webkit-scrollbar-thumb { background: rgba(96, 165, 250, .25); border-radius: 8px; }
#|
#|/* ------------------------------------------------------------------ backdrop */
#|.bg { position: fixed; inset: 0; z-index: 0; pointer-events: none; overflow: hidden; }
#|.bg-canvas { position: absolute; inset: 0; }
#|.orbs i { position: absolute; border-radius: 50%; filter: blur(80px); opacity: .55; }
#|.orbs i:nth-child(1) { width: 520px; height: 520px; left: -140px; top: 8%; background: #1d4ed8; animation: drift1 22s ease-in-out infinite; }
#|.orbs i:nth-child(2) { width: 460px; height: 460px; right: -120px; top: 30%; background: #4f46e5; animation: drift2 26s ease-in-out infinite; }
#|.orbs i:nth-child(3) { width: 420px; height: 420px; left: 35%; bottom: -160px; background: #0891b2; opacity: .35; animation: drift3 30s ease-in-out infinite; }
#|.grid-floor {
#|  position: absolute; inset: 0; opacity: .8;
#|  background-image:
#|    linear-gradient(rgba(96, 165, 250, .07) 1px, transparent 1px),
#|    linear-gradient(90deg, rgba(96, 165, 250, .07) 1px, transparent 1px);
#|  background-size: 52px 52px;
#|  -webkit-mask-image: radial-gradient(ellipse at 50% 40%, #000 25%, transparent 75%);
#|  mask-image: radial-gradient(ellipse at 50% 40%, #000 25%, transparent 75%);
#|  animation: gridMove 9s linear infinite;
#|}
#|@keyframes drift1 { 50% { transform: translate(140px, 90px) scale(1.15); } }
#|@keyframes drift2 { 50% { transform: translate(-160px, -70px) scale(1.1); } }
#|@keyframes drift3 { 50% { transform: translate(-120px, -110px) scale(1.2); } }
#|@keyframes gridMove { to { background-position: 52px 52px; } }
#|
#|/* --------------------------------------------------------------------- shell */
#|.app { position: relative; z-index: 1; max-width: 1120px; margin: 0 auto; padding: 22px 18px 150px; }
#|.topbar { display: flex; align-items: center; justify-content: space-between; margin-bottom: 26px; }
#|.brand { display: flex; align-items: center; gap: 12px; }
#|.logo {
#|  position: relative; width: 42px; height: 42px; border-radius: 13px; display: grid; place-items: center;
#|  background: linear-gradient(135deg, #2563eb, #06b6d4); box-shadow: 0 0 28px rgba(37, 99, 235, .6);
#|}
#|.logo-ring { position: absolute; inset: -4px; border-radius: 16px; border: 1.5px solid transparent;
#|  background: conic-gradient(from 0deg, transparent, rgba(34, 211, 238, .9), transparent 40%) border-box;
#|  -webkit-mask: linear-gradient(#000 0 0) padding-box, linear-gradient(#000 0 0);
#|  -webkit-mask-composite: xor; mask-composite: exclude; animation: spin 4s linear infinite; }
#|.brand-name { font-weight: 800; font-size: 18px; letter-spacing: .02em; }
#|.brand-name span { color: var(--cyan); margin: 0 1px; }
#|.brand-sub { font-size: 12px; color: var(--muted); letter-spacing: .12em; text-transform: uppercase; }
#|.status { display: flex; align-items: center; gap: 8px; font-size: 13px; color: var(--muted); padding: 7px 13px;
#|  border-radius: 999px; background: var(--panel); border: 1px solid var(--line); backdrop-filter: blur(10px); }
#|.status i { width: 8px; height: 8px; border-radius: 50%; background: var(--amber); }
#|.status.on i { background: var(--green); box-shadow: 0 0 0 0 rgba(52, 211, 153, .7); animation: ping 2s infinite; }
#|.status.off i { background: var(--red); }
#|@keyframes ping { 70% { box-shadow: 0 0 0 9px rgba(52, 211, 153, 0); } 100% { box-shadow: 0 0 0 0 rgba(52, 211, 153, 0); } }
#|@keyframes spin { to { transform: rotate(360deg); } }
#|
#|.view { animation: viewIn .55s var(--ease); display: flex; flex-direction: column; gap: 18px; }
#|@keyframes viewIn { from { opacity: 0; transform: translateY(16px) scale(.99); filter: blur(6px); } }
#|.rise { animation: rise .6s var(--ease) both; }
#|@keyframes rise { from { opacity: 0; transform: translateY(18px); } }
#|
#|/* --------------------------------------------------------------------- glass */
#|.glass {
#|  position: relative; overflow: hidden; padding: 20px; border-radius: var(--radius);
#|  background: linear-gradient(160deg, var(--panel-hi), var(--panel));
#|  border: 1px solid var(--line); backdrop-filter: blur(16px) saturate(140%); -webkit-backdrop-filter: blur(16px) saturate(140%);
#|  box-shadow: 0 10px 40px rgba(2, 6, 23, .5), inset 0 1px 0 rgba(255, 255, 255, .05);
#|  transition: border-color .3s, transform .3s var(--ease);
#|}
#|.glass::before {
#|  content: ""; position: absolute; inset: 0; border-radius: inherit; pointer-events: none; opacity: 0; transition: opacity .3s;
#|  background: radial-gradient(420px circle at var(--mx, 50%) var(--my, 0%), rgba(59, 130, 246, .16), transparent 60%);
#|}
#|.glass:hover { border-color: var(--line-hi); }
#|.glass:hover::before { opacity: 1; }
#|.row { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
#|.row.between { justify-content: space-between; }
#|.grow { flex: 1; min-width: 180px; }
#|.muted { color: var(--muted); } .small { font-size: 12.5px; }
#|.ok-text { color: var(--green); }
#|
#|/* ---------------------------------------------------------------------- hero */
#|.hero { padding: 34px 6px 10px; }
#|.eyebrow { display: inline-block; font-size: 12px; letter-spacing: .14em; text-transform: uppercase; color: var(--cyan);
#|  padding: 5px 12px; border: 1px solid rgba(34, 211, 238, .3); border-radius: 999px; background: rgba(34, 211, 238, .07); }
#|.hero h1 { font-size: clamp(34px, 6vw, 58px); line-height: 1.07; font-weight: 800; letter-spacing: -.02em; margin: 18px 0 14px; }
#|.grad-text {
#|  background: linear-gradient(90deg, #fff, #93c5fd, #22d3ee, #93c5fd, #fff); background-size: 300% 100%;
#|  -webkit-background-clip: text; background-clip: text; color: transparent; animation: shimmer 7s linear infinite;
#|}
#|@keyframes shimmer { to { background-position: -300% 0; } }
#|.lead { max-width: 640px; color: var(--muted); font-size: 17px; margin-bottom: 22px; }
#|
#|/* ------------------------------------------------------------------- buttons */
#|.btn {
#|  position: relative; overflow: hidden; border: 0; cursor: pointer; color: #fff; font-weight: 650; padding: 10px 18px; border-radius: 12px;
#|  background: linear-gradient(135deg, #2563eb, #0ea5e9); box-shadow: 0 6px 22px rgba(37, 99, 235, .45);
#|  transition: transform .25s var(--spring), box-shadow .25s, filter .25s;
#|}
#|.btn:hover:not(:disabled) { transform: translateY(-2px); box-shadow: 0 10px 30px rgba(37, 99, 235, .6); }
#|.btn:active:not(:disabled) { transform: translateY(0) scale(.97); }
#|.btn:disabled { opacity: .55; cursor: wait; }
#|.btn::after { content: ""; position: absolute; inset: 0; transform: translateX(-120%);
#|  background: linear-gradient(100deg, transparent 30%, rgba(255, 255, 255, .35), transparent 70%); }
#|.btn:hover::after, .btn.shine:not(:disabled)::after { transform: translateX(120%); transition: transform .8s var(--ease); }
#|.btn.shine:not(:disabled)::after { animation: sweep 3.2s ease-in-out infinite; }
#|@keyframes sweep { 0%, 60% { transform: translateX(-120%); } 100% { transform: translateX(120%); } }
#|.btn.ghost { background: rgba(59, 130, 246, .1); border: 1px solid var(--line-hi); color: #bcd3ff; box-shadow: none; }
#|.btn.ghost:hover:not(:disabled) { background: rgba(59, 130, 246, .2); box-shadow: 0 0 22px rgba(59, 130, 246, .3); }
#|.btn.red { background: linear-gradient(135deg, #e11d48, #fb7185); box-shadow: 0 6px 22px rgba(225, 29, 72, .4); }
#|.btn.green { background: linear-gradient(135deg, #059669, #34d399); color: #03251b; box-shadow: 0 6px 22px rgba(5, 150, 105, .4); }
#|.link { background: none; border: 0; color: var(--sky); cursor: pointer; }
#|.link:hover { color: var(--cyan); }
#|.input {
#|  background: rgba(3, 9, 30, .6); color: var(--text); border: 1px solid var(--line); border-radius: 12px; padding: 10px 14px; min-width: 200px; outline: none;
#|  transition: border-color .25s, box-shadow .25s;
#|}
#|.input::placeholder { color: var(--dim); }
#|.input:focus { border-color: var(--cyan); box-shadow: 0 0 0 4px rgba(34, 211, 238, .14), 0 0 26px rgba(34, 211, 238, .18); }
#|.chips { margin-top: 12px; }
#|.chip { background: rgba(59, 130, 246, .08); border: 1px solid var(--line); color: var(--muted); border-radius: 999px; padding: 3px 12px; font-size: 12.5px; cursor: pointer; transition: all .2s; }
#|.chip:hover { color: #fff; border-color: var(--line-hi); transform: translateY(-1px); }
#|.chip.bad { border-color: rgba(251, 113, 133, .35); color: #fda4af; }
#|.seg { display: inline-flex; background: rgba(3, 9, 30, .55); border: 1px solid var(--line); border-radius: 12px; padding: 4px; }
#|.seg button { background: none; border: 0; color: var(--muted); text-transform: capitalize; padding: 6px 14px; border-radius: 9px; cursor: pointer; transition: all .25s; }
#|.seg button.on { background: linear-gradient(135deg, #2563eb, #0ea5e9); color: #fff; box-shadow: 0 4px 14px rgba(37, 99, 235, .45); }
#|
#|/* --------------------------------------------------------------------- stats */
#|.stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 16px; }
#|.stat { padding: 18px 20px; }
#|.stat-label { color: var(--muted); font-size: 13px; letter-spacing: .05em; text-transform: uppercase; }
#|.stat-value { font-size: 40px; font-weight: 800; line-height: 1.15; margin: 6px 0 2px; font-variant-numeric: tabular-nums; }
#|.stat-hint { color: var(--dim); font-size: 12.5px; }
#|.stat-spark { position: absolute; right: -30px; bottom: -30px; width: 120px; height: 120px; border-radius: 50%; filter: blur(30px); opacity: .5; animation: breathe 4s ease-in-out infinite; }
#|.tone-blue .stat-value { color: #93c5fd; } .tone-blue .stat-spark { background: #2563eb; }
#|.tone-amber .stat-value { color: #fcd34d; } .tone-amber .stat-spark { background: #d97706; }
#|.tone-red .stat-value { color: #fda4af; } .tone-red .stat-spark { background: #e11d48; }
#|.tone-cyan .stat-value { color: #67e8f9; } .tone-cyan .stat-spark { background: #0891b2; }
#|@keyframes breathe { 50% { transform: scale(1.35); opacity: .3; } }
#|.two { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 16px; }
#|
#|/* ------------------------------------------------------------------ pipeline */
#|.pipe { display: flex; align-items: flex-start; overflow-x: auto; padding: 18px 16px 8px; margin: -6px -4px 0; }
#|.pipe-item { display: flex; flex-direction: column; align-items: center; gap: 8px; min-width: 88px; font-size: 12.5px; color: var(--muted); text-align: center; }
#|.pipe-node { width: 44px; height: 44px; border-radius: 14px; display: grid; place-items: center; color: #fff; font-size: 16px;
#|  background: linear-gradient(135deg, rgba(37, 99, 235, .8), rgba(6, 182, 212, .7)); box-shadow: 0 0 0 0 rgba(34, 211, 238, .5); animation: nodePulse 3s ease-in-out infinite; }
#|@keyframes nodePulse { 50% { box-shadow: 0 0 0 10px rgba(34, 211, 238, 0); transform: translateY(-3px); } }
#|.pipe-link { flex: 1; min-width: 26px; height: 3px; margin-top: 20px; border-radius: 3px; background: linear-gradient(90deg, rgba(96, 165, 250, .1), var(--cyan), rgba(96, 165, 250, .1)); background-size: 200% 100%; animation: flowX 1.8s linear infinite; }
#|@keyframes flowX { to { background-position: -200% 0; } }
#|
#|/* ------------------------------------------------------- model card & alerts */
#|.auc-row { display: flex; align-items: center; gap: 12px; }
#|.auc-row b { font-size: 34px; font-variant-numeric: tabular-nums; }
#|.auc-row > span:first-child { color: var(--muted); }
#|.auc-bar { position: relative; height: 10px; border-radius: 8px; background: rgba(148, 163, 184, .15); margin: 12px 0 16px; overflow: visible; }
#|.auc-fill { height: 100%; border-radius: 8px; background: linear-gradient(90deg, #2563eb, #22d3ee); box-shadow: 0 0 16px rgba(34, 211, 238, .6); animation: grow 1.3s var(--ease) both; transform-origin: left; }
#|@keyframes grow { from { transform: scaleX(0); } }
#|.auc-target { position: absolute; top: -5px; width: 2px; height: 20px; background: #fff; opacity: .8; border-radius: 2px; }
#|.mini-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; }
#|.mini-grid div { background: rgba(3, 9, 30, .45); border: 1px solid var(--line); border-radius: 12px; padding: 9px 10px; }
#|.mini-grid span { display: block; font-size: 11.5px; color: var(--dim); text-transform: uppercase; letter-spacing: .06em; }
#|.mini-grid b { font-size: 16px; }
#|.mini-alert { display: flex; align-items: center; gap: 10px; padding: 10px 8px; border-radius: 12px; cursor: pointer; transition: background .2s, transform .2s; }
#|.mini-alert:hover { background: rgba(59, 130, 246, .12); transform: translateX(4px); }
#|.dot { width: 9px; height: 9px; border-radius: 50%; flex: none; }
#|.dot.open { background: var(--amber); box-shadow: 0 0 10px var(--amber); animation: blink 1.6s infinite; }
#|.dot.confirmed { background: var(--red); } .dot.dismissed { background: var(--green); }
#|@keyframes blink { 50% { opacity: .35; } }
#|.risk { font-variant-numeric: tabular-nums; font-weight: 700; }
#|.pill { font-size: 12px; font-weight: 650; padding: 3px 10px; border-radius: 999px; border: 1px solid var(--line); text-transform: capitalize; }
#|.pill.neutral { color: var(--muted); } .pill.ok { color: var(--green); border-color: rgba(52, 211, 153, .4); background: rgba(52, 211, 153, .08); }
#|.pill.warn { color: var(--amber); border-color: rgba(251, 191, 36, .4); }
#|.pill.s-open { color: var(--amber); border-color: rgba(251, 191, 36, .4); background: rgba(251, 191, 36, .08); }
#|.pill.s-confirmed { color: var(--red); border-color: rgba(251, 113, 133, .4); background: rgba(251, 113, 133, .08); }
#|.pill.s-dismissed { color: var(--green); border-color: rgba(52, 211, 153, .4); background: rgba(52, 211, 153, .08); }
#|.pill.t-shell { color: var(--red); } .pill.t-business { color: var(--cyan); } .pill.t-individual { color: var(--sky); }
#|.alert-card { padding: 18px 20px; }
#|.alert-id { color: var(--dim); font-weight: 700; }
#|.bar { width: 130px; height: 7px; border-radius: 6px; background: rgba(148, 163, 184, .15); overflow: hidden; }
#|.bar > div { height: 100%; border-radius: 6px; background: linear-gradient(90deg, var(--amber), var(--red)); animation: grow 1s var(--ease) both; transform-origin: left; }
#|.narr { margin: 14px 0 6px; padding: 13px 16px; border-radius: 12px; background: rgba(3, 9, 30, .55); border-left: 3px solid var(--cyan); line-height: 1.65; }
#|.narr-meta { margin-top: 8px; font-size: 12px; color: var(--dim); }
#|.caret { display: inline-block; width: 8px; height: 1.1em; margin-left: 2px; vertical-align: text-bottom; background: var(--cyan); animation: blink .8s steps(2) infinite; }
#|.actions { margin-top: 12px; }
#|
#|/* ---------------------------------------------------------------------- scan */
#|.scan-card h2 { font-size: 24px; }
#|.scan-card .row { margin-top: 14px; }
#|.scan-wait { display: flex; flex-direction: column; align-items: center; gap: 12px; padding: 30px 0 10px; }
#|.radar { position: relative; width: 130px; height: 130px; border-radius: 50%; border: 1px solid var(--line-hi); overflow: hidden;
#|  background: repeating-radial-gradient(circle, transparent 0 20px, rgba(96, 165, 250, .2) 21px 22px); box-shadow: 0 0 46px rgba(34, 211, 238, .28); }
#|.radar::before, .radar::after { content: ""; position: absolute; }
#|.radar::before { left: 50%; top: 0; bottom: 0; width: 1px; background: rgba(96, 165, 250, .25); }
#|.radar::after { inset: 0; border-radius: 50%; background: conic-gradient(from 0deg, rgba(34, 211, 238, .65), transparent 38%); animation: spin 1.3s linear infinite; }
#|.scan-result { display: flex; gap: 28px; align-items: center; flex-wrap: wrap; margin-top: 26px; padding-top: 22px; border-top: 1px solid var(--line); }
#|.scan-meta { flex: 1; min-width: 240px; display: flex; flex-direction: column; gap: 10px; }
#|.verdict { font-size: 22px; font-weight: 800; }
#|.verdict.bad { color: var(--red); text-shadow: 0 0 24px rgba(251, 113, 133, .5); } .verdict.good { color: var(--green); }
#|.gauge { position: relative; width: 190px; height: 190px; flex: none; }
#|.gauge svg { width: 100%; height: 100%; }
#|.gauge-arc { transition: stroke-dashoffset 1.4s var(--ease); filter: drop-shadow(0 0 8px rgba(34, 211, 238, .55)); }
#|.gauge.bad .gauge-arc { filter: drop-shadow(0 0 10px rgba(251, 113, 133, .65)); }
#|.gauge-text { position: absolute; inset: 0; display: flex; flex-direction: column; align-items: center; justify-content: center; }
#|.gauge-text b { font-size: 42px; font-weight: 800; line-height: 1; font-variant-numeric: tabular-nums; }
#|.gauge-text small { font-size: 20px; color: var(--muted); }
#|.gauge-text span { font-size: 12px; letter-spacing: .14em; text-transform: uppercase; color: var(--muted); margin-top: 4px; }
#|
#|/* ------------------------------------------------------------------- network */
#|.net-layout { display: grid; grid-template-columns: minmax(0, 2.1fr) minmax(260px, 1fr); gap: 16px; }
#|.net-card { padding: 10px; }
#|.net { width: 100%; height: auto; display: block; border-radius: 12px;
#|  background: radial-gradient(ellipse at center, rgba(37, 99, 235, .1), transparent 70%); }
#|.edge { fill: none; stroke-linecap: round; transition: opacity .3s; }
#|.edge.flow { stroke-dasharray: 5 9; animation: dashflow 1.2s linear infinite; }
#|.edge.dim { opacity: .07 !important; }
#|@keyframes dashflow { to { stroke-dashoffset: -14; } }
#|.node { cursor: pointer; animation: nodeIn .7s var(--spring) both; }
#|.node circle { transition: r .25s; }
#|.node:hover circle:nth-of-type(2) { opacity: .28; }
#|.node.sel circle:nth-of-type(3) { stroke: #fff; stroke-width: 3; }
#|.node-label { fill: #c7d8ff; font-size: 11px; pointer-events: none; paint-order: stroke; stroke: rgba(3, 7, 18, .85); stroke-width: 3px; }
#|.pulse { transform-box: fill-box; transform-origin: center; animation: pulseRing 2.2s ease-out infinite; }
#|@keyframes pulseRing { from { transform: scale(1); opacity: .8; } to { transform: scale(2.7); opacity: 0; } }
#|@keyframes nodeIn { from { opacity: 0; } }
#|.legend { display: flex; flex-wrap: wrap; gap: 6px 16px; padding: 10px 14px 6px; font-size: 12.5px; color: var(--muted); }
#|.legend i { display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 6px; }
#|.legend i.ln { width: 16px; height: 3px; border-radius: 2px; vertical-align: middle; }
#|.legend .hint { margin-left: auto; color: var(--dim); }
#|.net-side { display: flex; flex-direction: column; gap: 16px; }
#|.score-line { display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 12px; color: var(--muted); }
#|.score-line b { font-size: 28px; color: var(--red); }
#|.feat { margin-bottom: 10px; }
#|.feat-top { display: flex; justify-content: space-between; font-size: 13px; gap: 8px; }
#|.feat-top em { font-style: normal; color: var(--cyan); font-variant-numeric: tabular-nums; }
#|.feat-bar { height: 6px; border-radius: 5px; background: rgba(148, 163, 184, .15); margin-top: 5px; overflow: hidden; }
#|.feat-bar div { height: 100%; border-radius: 5px; background: linear-gradient(90deg, #2563eb, #22d3ee); animation: grow 1s var(--ease) both; transform-origin: left; }
#|.kv { display: flex; justify-content: space-between; gap: 10px; padding: 5px 0; border-bottom: 1px dashed var(--line); }
#|.kv em { font-style: normal; color: var(--dim); font-weight: 400; }
#|.node-detail .row { margin: 6px 0 10px; }
#|
#|/* --------------------------------------------------------------------- toasts */
#|.toasts { position: fixed; top: 18px; right: 18px; z-index: 50; display: flex; flex-direction: column; gap: 10px; max-width: min(380px, calc(100vw - 36px)); }
#|.toast { padding: 12px 16px; border-radius: 14px; font-size: 14px; backdrop-filter: blur(14px); border: 1px solid var(--line-hi);
#|  background: rgba(8, 20, 52, .85); box-shadow: 0 12px 34px rgba(0, 0, 0, .5); animation: toastIn .5s var(--spring), toastOut .4s ease 4.7s forwards; }
#|.toast.ok { border-color: rgba(52, 211, 153, .55); } .toast.err { border-color: rgba(251, 113, 133, .6); color: #fecdd3; } .toast.warn { border-color: rgba(251, 191, 36, .55); }
#|@keyframes toastIn { from { opacity: 0; transform: translateX(40px) scale(.95); } }
#|@keyframes toastOut { to { opacity: 0; transform: translateX(30px); } }
#|
#|/* ----------------------------------------------------------- floating dock nav */
#|.dock {
#|  position: fixed; left: 0; right: 0; bottom: 22px; margin: 0 auto; z-index: 40;
#|  width: min(540px, calc(100vw - 24px)); padding: 7px; border-radius: 24px;
#|  display: grid; grid-template-columns: repeat(var(--n), 1fr);
#|  background: rgba(8, 20, 52, .62); border: 1px solid var(--line-hi);
#|  backdrop-filter: blur(22px) saturate(160%); -webkit-backdrop-filter: blur(22px) saturate(160%);
#|  box-shadow: 0 18px 50px rgba(2, 6, 23, .7), 0 0 0 1px rgba(255, 255, 255, .03) inset;
#|  animation: dockFloat 6s ease-in-out infinite;
#|}
#|@keyframes dockFloat { 50% { transform: translateY(-7px); } }
#|.dock-glow { position: absolute; inset: -1px; border-radius: 24px; z-index: -1; filter: blur(22px); opacity: .45;
#|  background: linear-gradient(90deg, #2563eb, #22d3ee, #6366f1, #2563eb); background-size: 300% 100%; animation: shimmer 8s linear infinite; }
#|.dock-indicator {
#|  position: absolute; top: 7px; bottom: 7px; left: 7px; border-radius: 18px;
#|  width: calc((100% - 14px) / var(--n)); transform: translateX(calc(var(--i) * 100%));
#|  transition: transform .55s var(--spring);
#|  background: linear-gradient(135deg, rgba(37, 99, 235, .95), rgba(14, 165, 233, .85)); box-shadow: 0 6px 24px rgba(37, 99, 235, .65);
#|}
#|.dock-tab { position: relative; z-index: 1; display: flex; flex-direction: column; align-items: center; gap: 2px; padding: 9px 4px 8px; background: none; border: 0;
#|  color: var(--muted); cursor: pointer; border-radius: 18px; transition: color .25s, transform .3s var(--spring); }
#|.dock-tab:hover { color: #fff; transform: translateY(-3px); }
#|.dock-tab.on { color: #fff; }
#|.dock-tab.on .dock-ico { transform: scale(1.12); }
#|.dock-ico { display: grid; transition: transform .35s var(--spring); }
#|.dock-label { font-size: 11.5px; font-weight: 650; letter-spacing: .04em; }
#|.dock-badge { position: absolute; top: 4px; right: calc(50% - 26px); min-width: 19px; height: 19px; padding: 0 5px; display: grid; place-items: center;
#|  font-size: 11px; font-weight: 800; color: #fff; border-radius: 999px; background: linear-gradient(135deg, #f43f5e, #fb923c); box-shadow: 0 0 14px rgba(244, 63, 94, .8); animation: badgePop 2.4s ease-in-out infinite; }
#|@keyframes badgePop { 50% { transform: scale(1.18); } }
#|
#|/* ------------------------------------------------------------------ demo mode */
#|.app.demo { padding-top: 48px; }
#|.demo-badge { position: fixed; top: 8px; left: 50%; transform: translateX(-50%); z-index: 60;
#|  font: 600 11.5px/1 system-ui, sans-serif; letter-spacing: .06em; text-transform: uppercase; white-space: nowrap; color: #fde68a;
#|  padding: 6px 12px; border-radius: 999px; background: rgba(120, 53, 15, .55); border: 1px solid rgba(251, 191, 36, .45); backdrop-filter: blur(8px); }
#|@media (max-width: 560px) { .demo-badge { font-size: 10px; padding: 5px 9px; } }
#|
#|/* ---------------------------------------------------------------- responsive */
#|@media (max-width: 860px) { .net-layout { grid-template-columns: 1fr; } .mini-grid { grid-template-columns: repeat(2, 1fr); } }
#|@media (max-width: 560px) {
#|  .app { padding: 16px 12px 140px; } .status { display: none; } .bar { width: 90px; }
#|  .gauge { width: 160px; height: 160px; } .stat-value { font-size: 34px; }
#|}
#|@media (prefers-reduced-motion: reduce) {
#|  *, *::before, *::after { animation-duration: .001ms !important; animation-iteration-count: 1 !important; transition-duration: .001ms !important; }
#|}
#|
#|:root { color-scheme: dark; box-sizing: border-box; padding-top: env(safe-area-inset-top, 0px); padding-bottom: env(safe-area-inset-bottom, 0px); }
#|html { scroll-padding-top: env(safe-area-inset-top, 0px); }
#|.dock { bottom: calc(22px + env(safe-area-inset-bottom, 0px)); }
#|.app { padding-top: 48px; }
#|</style>
#|</head>
#|<body>
#|<div class="demo-badge">Demo mode &middot; sample data &middot; no live backend</div>
#|<div id="root"></div>
#|<script src="https://cdnjs.cloudflare.com/ajax/libs/react/18.3.1/umd/react.production.min.js"></script>
#|<script src="https://cdnjs.cloudflare.com/ajax/libs/react-dom/18.3.1/umd/react-dom.production.min.js"></script>
#|<script>
#|if (!window.React || !window.ReactDOM) {
#|  document.getElementById("root").innerHTML = '<p style="padding:40px;color:#e8f0ff;font-family:system-ui">Could not load React from cdnjs.cloudflare.com. Check your internet connection and reload.</p>';
#|} else {
#|(function () {
#|  var defs = {}, cache = {};
#|  window.__d = function (id, fn) { defs[id] = fn; };
#|  function norm(p) { var out = []; p.split("/").forEach(function (s) { if (s === "..") out.pop(); else if (s && s !== ".") out.push(s); }); return out.join("/"); }
#|  function dirname(id) { return id.split("/").slice(0, -1).join("/"); }
#|  function load(id) {
#|    if (cache[id]) return cache[id].exports;
#|    if (!defs[id]) throw new Error("Module not found: " + id);
#|    var m = (cache[id] = { exports: {} });
#|    defs[id].call(m.exports, m.exports, function (r) { return resolve(id, r); }, m);
#|    return m.exports;
#|  }
#|  function resolve(from, r) {
#|    if (r === "react") return window.React;
#|    if (r === "react-dom" || r === "react-dom/client") return window.ReactDOM;
#|    if (r.indexOf("@/") === 0) return load(norm("src/" + r.slice(2)));
#|    if (r.charAt(0) === ".") return load(norm(dirname(from) + "/" + r));
#|    throw new Error("Unbundled module: " + r);
#|  }
#|  window.__r = load;
#|})();
#|__d("src/app/page", function (exports, require, module) {
#|"use strict";
#|"use client";
#|var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
#|    if (k2 === undefined) k2 = k;
#|    var desc = Object.getOwnPropertyDescriptor(m, k);
#|    if (!desc || ("get" in desc ? !m.__esModule : desc.writable || desc.configurable)) {
#|      desc = { enumerable: true, get: function() { return m[k]; } };
#|    }
#|    Object.defineProperty(o, k2, desc);
#|}) : (function(o, m, k, k2) {
#|    if (k2 === undefined) k2 = k;
#|    o[k2] = m[k];
#|}));
#|var __setModuleDefault = (this && this.__setModuleDefault) || (Object.create ? (function(o, v) {
#|    Object.defineProperty(o, "default", { enumerable: true, value: v });
#|}) : function(o, v) {
#|    o["default"] = v;
#|});
#|var __importStar = (this && this.__importStar) || (function () {
#|    var ownKeys = function(o) {
#|        ownKeys = Object.getOwnPropertyNames || function (o) {
#|            var ar = [];
#|            for (var k in o) if (Object.prototype.hasOwnProperty.call(o, k)) ar[ar.length] = k;
#|            return ar;
#|        };
#|        return ownKeys(o);
#|    };
#|    return function (mod) {
#|        if (mod && mod.__esModule) return mod;
#|        var result = {};
#|        if (mod != null) for (var k = ownKeys(mod), i = 0; i < k.length; i++) if (k[i] !== "default") __createBinding(result, mod, k[i]);
#|        __setModuleDefault(result, mod);
#|        return result;
#|    };
#|})();
#|var __importDefault = (this && this.__importDefault) || function (mod) {
#|    return (mod && mod.__esModule) ? mod : { "default": mod };
#|};
#|Object.defineProperty(exports, "__esModule", { value: true });
#|exports.default = Home;
#|const react_1 = require("react");
#|const Background_1 = __importDefault(require("@/components/Background"));
#|const FloatingDock_1 = __importDefault(require("@/components/FloatingDock"));
#|const Glass_1 = __importDefault(require("@/components/Glass"));
#|const Icons_1 = require("@/components/Icons");
#|const NetworkGraph_1 = __importDefault(require("@/components/NetworkGraph"));
#|const RiskGauge_1 = __importDefault(require("@/components/RiskGauge"));
#|const hooks_1 = require("@/components/hooks");
#|const api_1 = require("@/lib/api");
#|const demo_1 = require("@/lib/demo");
#|const niceType = (t) => (t ? t.replaceAll("_", " ") : "");
#|const FILTERS = ["all", "open", "confirmed", "dismissed"];
#|function Stat({ label, value, tone, hint, suffix = "" }) {
#|    const v = (0, hooks_1.useCountUp)(value);
#|    return (React.createElement(Glass_1.default, { as: "div", className: `stat tone-${tone}` },
#|        React.createElement("div", { className: "stat-label" }, label),
#|        React.createElement("div", { className: "stat-value" },
#|            v.toFixed(0),
#|            suffix),
#|        React.createElement("div", { className: "stat-hint" }, hint),
#|        React.createElement("span", { className: "stat-spark" })));
#|}
#|const PIPE = ["Transactions", "Kafka stream", "GNN detection", "Explainability", "SAR narrative", "immudb ledger"];
#|function Pipeline() {
#|    return (React.createElement("div", { className: "pipe" }, PIPE.map((s, i) => (React.createElement(react_1.Fragment, { key: s },
#|        React.createElement("div", { className: "pipe-item" },
#|            React.createElement("div", { className: "pipe-node", style: { animationDelay: `${i * 0.4}s` } },
#|                React.createElement("b", null, i + 1)),
#|            React.createElement("span", null, s)),
#|        i < PIPE.length - 1 && React.createElement("i", { className: "pipe-link", style: { animationDelay: `${i * 0.25}s` } }))))));
#|}
#|function Narrative({ text, source, sha256, animate, onDone }) {
#|    const shown = (0, hooks_1.useTypewriter)(text, animate, onDone);
#|    return (React.createElement("div", { className: "narr" },
#|        React.createElement("span", null, shown),
#|        animate && shown.length < text.length && React.createElement("i", { className: "caret" }),
#|        React.createElement("div", { className: "narr-meta" },
#|            "source: ", source !== null && source !== void 0 ? source : "n/a",
#|            " \u00B7 sha256 ",
#|            sha256 ? `${sha256.slice(0, 12)}…` : "n/a")));
#|}
#|function ModelCard({ info }) {
#|    var _a, _b;
#|    const auc = (_a = info === null || info === void 0 ? void 0 : info.metrics.test_auc_roc) !== null && _a !== void 0 ? _a : 0;
#|    const target = (_b = info === null || info === void 0 ? void 0 : info.targets.auc_roc) !== null && _b !== void 0 ? _b : 0.87;
#|    const aucAnim = (0, hooks_1.useCountUp)(auc * 100, 1100);
#|    return (React.createElement(Glass_1.default, { className: "model-card" },
#|        React.createElement("h3", null, "Detection model"),
#|        info === null && React.createElement("p", { className: "muted" }, "Model not loaded yet. Train one (see README), then refresh."),
#|        info !== null && (React.createElement(React.Fragment, null,
#|            React.createElement("div", { className: "auc-row" },
#|                React.createElement("span", null, "AUC-ROC"),
#|                React.createElement("b", null, (aucAnim / 100).toFixed(3)),
#|                React.createElement("span", { className: `pill ${info.metrics.meets_auc_target ? "ok" : "warn"}` }, info.metrics.meets_auc_target ? `meets ≥ ${target}` : `below ${target}`)),
#|            React.createElement("div", { className: "auc-bar" },
#|                React.createElement("div", { className: "auc-fill", style: { width: `${Math.min(100, auc * 100)}%` } }),
#|                React.createElement("i", { className: "auc-target", style: { left: `${target * 100}%` } })),
#|            React.createElement("div", { className: "mini-grid" },
#|                React.createElement("div", null,
#|                    React.createElement("span", null, "Model"),
#|                    React.createElement("b", null, info.model.toUpperCase())),
#|                React.createElement("div", null,
#|                    React.createElement("span", null, "Precision"),
#|                    React.createElement("b", null, (0, api_1.pct)(info.metrics.test_precision, 0))),
#|                React.createElement("div", null,
#|                    React.createElement("span", null, "Recall"),
#|                    React.createElement("b", null, (0, api_1.pct)(info.metrics.test_recall, 0))),
#|                React.createElement("div", null,
#|                    React.createElement("span", null, "Threshold"),
#|                    React.createElement("b", null, (0, api_1.pct)(info.threshold, 0))))))));
#|}
#|/* ----------------------------------------------------------------------- page */
#|function Home() {
#|    var _a;
#|    const demoMode = (0, demo_1.isDemo)();
#|    // In demo mode we must install the mock backend BEFORE the first request, so wait for it.
#|    const [ready, setReady] = (0, react_1.useState)(!demoMode);
#|    const [tab, setTab] = (0, react_1.useState)("overview");
#|    const [alerts, setAlerts] = (0, react_1.useState)([]);
#|    const [online, setOnline] = (0, react_1.useState)(null);
#|    const [samples, setSamples] = (0, react_1.useState)(null);
#|    const [modelInfo, setModelInfo] = (0, react_1.useState)(null);
#|    const [accountId, setAccountId] = (0, react_1.useState)("");
#|    const [scan, setScan] = (0, react_1.useState)(null);
#|    const [officer, setOfficer] = (0, react_1.useState)("");
#|    const [filter, setFilter] = (0, react_1.useState)("all");
#|    const [busy, setBusy] = (0, react_1.useState)("");
#|    const [fresh, setFresh] = (0, react_1.useState)(() => new Set());
#|    const [netId, setNetId] = (0, react_1.useState)("");
#|    const [graph, setGraph] = (0, react_1.useState)(null);
#|    const [toasts, setToasts] = (0, react_1.useState)([]);
#|    const notify = (0, react_1.useCallback)((kind, msg) => {
#|        const id = Date.now() + Math.random();
#|        setToasts((t) => [...t, { id, kind, msg }]);
#|        setTimeout(() => setToasts((t) => t.filter((x) => x.id !== id)), 5200);
#|    }, []);
#|    const load = (0, react_1.useCallback)(async () => {
#|        try {
#|            setAlerts(await (0, api_1.api)("/alerts"));
#|            setOnline(true);
#|        }
#|        catch {
#|            setOnline(false);
#|        }
#|    }, []);
#|    (0, react_1.useEffect)(() => {
#|        if (!demoMode)
#|            return;
#|        void Promise.resolve().then(() => __importStar(require("../../demo/mockApi"))).then((m) => {
#|            m.installMockApi();
#|            setReady(true);
#|        });
#|    }, [demoMode]);
#|    (0, react_1.useEffect)(() => {
#|        if (!ready)
#|            return undefined;
#|        void load();
#|        (0, api_1.api)("/accounts/sample").then(setSamples).catch(() => undefined);
#|        (0, api_1.api)("/model/info").then(setModelInfo).catch(() => undefined);
#|        const id = setInterval(() => void load(), 15000);
#|        return () => clearInterval(id);
#|    }, [load, ready]);
#|    const stats = (0, react_1.useMemo)(() => {
#|        const open = alerts.filter((a) => a.status === "open").length;
#|        const confirmed = alerts.filter((a) => a.status === "confirmed").length;
#|        const avg = alerts.length ? alerts.reduce((s, a) => s + a.risk_score, 0) / alerts.length : 0;
#|        return { total: alerts.length, open, confirmed, avg };
#|    }, [alerts]);
#|    /* ---- actions ---- */
#|    const doScan = async () => {
#|        const id = accountId.trim();
#|        if (!id)
#|            return;
#|        setBusy("scan");
#|        setScan(null);
#|        try {
#|            const [res] = await Promise.all([
#|                (0, api_1.api)("/alerts/scan", { method: "POST", body: JSON.stringify({ account_id: id }) }),
#|                (0, api_1.sleep)(900),
#|            ]);
#|            setScan({ ...res, account_id: id });
#|            if (res.alert) {
#|                notify("warn", `Alert #${res.alert.id} created for ${id}`);
#|                void load();
#|            }
#|        }
#|        catch (e) {
#|            notify("err", (0, api_1.errorMessage)(e));
#|        }
#|        finally {
#|            setBusy("");
#|        }
#|    };
#|    const doNarrative = async (id) => {
#|        var _a;
#|        setBusy(`n${id}`);
#|        try {
#|            const updated = await (0, api_1.api)(`/alerts/${id}/narrative`, { method: "POST" });
#|            setFresh((s) => new Set(s).add(id));
#|            setAlerts((list) => list.map((x) => (x.id === id ? updated : x)));
#|            notify("ok", `SAR narrative ready (${(_a = updated.narrative_source) !== null && _a !== void 0 ? _a : "unknown"})`);
#|        }
#|        catch (e) {
#|            notify("err", (0, api_1.errorMessage)(e));
#|        }
#|        finally {
#|            setBusy("");
#|        }
#|    };
#|    const doDecision = async (id, decision) => {
#|        if (!officer.trim()) {
#|            notify("err", "Enter the reviewing officer's name first.");
#|            return;
#|        }
#|        setBusy(`d${id}`);
#|        try {
#|            const updated = await (0, api_1.api)(`/alerts/${id}/decision`, {
#|                method: "POST",
#|                body: JSON.stringify({ officer: officer.trim(), decision, note: "" }),
#|            });
#|            setAlerts((list) => list.map((x) => (x.id === id ? updated : x)));
#|            notify("ok", `Alert #${id} ${decision} and written to the audit ledger`);
#|        }
#|        catch (e) {
#|            notify("err", (0, api_1.errorMessage)(e));
#|        }
#|        finally {
#|            setBusy("");
#|        }
#|    };
#|    const doExplain = async (id) => {
#|        const acct = (id !== null && id !== void 0 ? id : netId).trim();
#|        if (!acct)
#|            return;
#|        setNetId(acct);
#|        setTab("network");
#|        setBusy("explain");
#|        setGraph(null);
#|        try {
#|            const [data] = await Promise.all([(0, api_1.api)(`/accounts/${encodeURIComponent(acct)}/explain`), (0, api_1.sleep)(600)]);
#|            setGraph(data);
#|        }
#|        catch (e) {
#|            notify("err", (0, api_1.errorMessage)(e));
#|        }
#|        finally {
#|            setBusy("");
#|        }
#|    };
#|    const clearFresh = (id) => setFresh((s) => {
#|        const next = new Set(s);
#|        next.delete(id);
#|        return next;
#|    });
#|    const tabs = [
#|        { id: "overview", label: "Overview", icon: React.createElement(Icons_1.IconHome, null) },
#|        { id: "scan", label: "Scan", icon: React.createElement(Icons_1.IconRadar, null) },
#|        { id: "alerts", label: "Alerts", icon: React.createElement(Icons_1.IconBell, null), badge: stats.open },
#|        { id: "network", label: "Network", icon: React.createElement(Icons_1.IconGraph, null) },
#|    ];
#|    const shown = alerts.filter((a) => filter === "all" || a.status === filter);
#|    const latencyTarget = (_a = modelInfo === null || modelInfo === void 0 ? void 0 : modelInfo.targets.latency_ms) !== null && _a !== void 0 ? _a : 100;
#|    /* ---- views ---- */
#|    const overview = (React.createElement(React.Fragment, null,
#|        React.createElement("section", { className: "hero" },
#|            React.createElement("span", { className: "eyebrow" }, "Explainable AML \u00B7 FinCEN-ready audit trail"),
#|            React.createElement("h1", null,
#|                React.createElement("span", { className: "grad-text" }, "Follow the money."),
#|                React.createElement("br", null),
#|                "Explain every flag."),
#|            React.createElement("p", { className: "lead" }, "Graph neural networks spot laundering typologies across multi-hop transaction networks, and every alert comes with a human-readable SAR narrative and a tamper-proof audit record."),
#|            React.createElement("div", { className: "row" },
#|                React.createElement("button", { className: "btn", onClick: () => setTab("scan") }, "Scan an account"),
#|                React.createElement("button", { className: "btn ghost", onClick: () => setTab("alerts") }, "Review alerts"))),
#|        React.createElement("div", { className: "stats" },
#|            React.createElement(Stat, { label: "Total alerts", value: stats.total, tone: "blue", hint: "all time" }),
#|            React.createElement(Stat, { label: "Awaiting review", value: stats.open, tone: "amber", hint: "open cases" }),
#|            React.createElement(Stat, { label: "Confirmed", value: stats.confirmed, tone: "red", hint: "marked suspicious" }),
#|            React.createElement(Stat, { label: "Average risk", value: stats.avg * 100, suffix: "%", tone: "cyan", hint: "across alerts" })),
#|        React.createElement(Glass_1.default, null,
#|            React.createElement("h3", null, "Detection pipeline"),
#|            React.createElement(Pipeline, null)),
#|        React.createElement("div", { className: "two" },
#|            React.createElement(ModelCard, { info: modelInfo }),
#|            React.createElement(Glass_1.default, null,
#|                React.createElement("div", { className: "row between" },
#|                    React.createElement("h3", null, "Latest alerts"),
#|                    React.createElement("button", { className: "link", onClick: () => setTab("alerts") }, "View all \u2192")),
#|                alerts.length === 0 && React.createElement("p", { className: "muted" }, "Nothing yet. Scan a suspicious account to create the first alert."),
#|                alerts.slice(0, 4).map((a, i) => (React.createElement("div", { className: "mini-alert rise", key: a.id, style: { animationDelay: `${i * 70}ms` }, onClick: () => setTab("alerts") },
#|                    React.createElement("span", { className: `dot ${a.status}` }),
#|                    React.createElement("b", null, a.account_id),
#|                    React.createElement("span", { className: "muted" }, niceType(a.typology) || "no narrative yet"),
#|                    React.createElement("span", { className: "grow" }),
#|                    React.createElement("span", { className: "risk" }, (0, api_1.pct)(a.risk_score, 0)))))))));
#|    const scanView = (React.createElement(Glass_1.default, { className: "scan-card" },
#|        React.createElement("h2", null, "Scan an account"),
#|        React.createElement("p", { className: "muted" }, "The model scores the account from its 2-hop transaction neighbourhood."),
#|        React.createElement("div", { className: "row" },
#|            React.createElement("input", { className: "input grow", placeholder: "Account ID, e.g. ACC0000123", value: accountId, onChange: (e) => setAccountId(e.target.value), onKeyDown: (e) => {
#|                    if (e.key === "Enter")
#|                        void doScan();
#|                } }),
#|            React.createElement("button", { className: "btn shine", disabled: !accountId.trim() || busy === "scan", onClick: () => void doScan() }, busy === "scan" ? "Scanning…" : "Scan")),
#|        samples !== null && (React.createElement("div", { className: "row chips" },
#|            React.createElement("span", { className: "muted" }, "Try:"),
#|            samples.suspicious.slice(0, 3).map((a) => (React.createElement("button", { key: a, className: "chip bad", onClick: () => setAccountId(a) }, a))),
#|            samples.benign.slice(0, 2).map((a) => (React.createElement("button", { key: a, className: "chip", onClick: () => setAccountId(a) }, a))))),
#|        busy === "scan" && (React.createElement("div", { className: "scan-wait" },
#|            React.createElement("div", { className: "radar" }),
#|            React.createElement("p", { className: "muted" }, "Running graph inference\u2026"))),
#|        scan !== null && busy !== "scan" && (React.createElement("div", { className: "scan-result rise" },
#|            React.createElement(RiskGauge_1.default, { key: `${scan.account_id}-${scan.score}`, score: scan.score, threshold: scan.threshold }),
#|            React.createElement("div", { className: "scan-meta" },
#|                React.createElement("div", { className: `verdict ${scan.flagged ? "bad" : "good"}` }, scan.flagged ? "Flagged: suspicious pattern" : "Below alert threshold"),
#|                React.createElement("p", { className: "muted" },
#|                    scan.account_id,
#|                    " \u00B7 threshold ",
#|                    (0, api_1.pct)(scan.threshold),
#|                    " \u00B7 inference ",
#|                    scan.latency_ms,
#|                    " ms",
#|                    scan.latency_ms <= latencyTarget && React.createElement("span", { className: "ok-text" },
#|                        " \u2713 within ",
#|                        latencyTarget,
#|                        " ms target")),
#|                React.createElement("div", { className: "row" },
#|                    React.createElement("button", { className: "btn ghost", onClick: () => void doExplain(scan.account_id) }, "Explain network"),
#|                    scan.alert !== null && (React.createElement("button", { className: "btn", onClick: () => setTab("alerts") },
#|                        "Open alert #",
#|                        scan.alert.id))))))));
#|    const alertsView = (React.createElement(React.Fragment, null,
#|        React.createElement(Glass_1.default, null,
#|            React.createElement("div", { className: "row between" },
#|                React.createElement("div", { className: "seg" }, FILTERS.map((f) => (React.createElement("button", { key: f, className: filter === f ? "on" : "", onClick: () => setFilter(f) }, f)))),
#|                React.createElement("div", { className: "row" },
#|                    React.createElement("input", { className: "input", placeholder: "Reviewing officer name", value: officer, onChange: (e) => setOfficer(e.target.value) }),
#|                    React.createElement("button", { className: "btn ghost", onClick: () => void load() }, "Refresh")))),
#|        shown.length === 0 && (React.createElement(Glass_1.default, null,
#|            React.createElement("p", { className: "muted" }, "No alerts in this view."))),
#|        shown.map((a, i) => (React.createElement(Glass_1.default, { className: "alert-card rise", key: a.id, style: { animationDelay: `${Math.min(i, 8) * 60}ms` } },
#|            React.createElement("div", { className: "row between" },
#|                React.createElement("div", { className: "row" },
#|                    React.createElement("span", { className: "alert-id" },
#|                        "#",
#|                        a.id),
#|                    React.createElement("b", null, a.account_id),
#|                    React.createElement("span", { className: `pill s-${a.status}` }, a.status),
#|                    a.typology && React.createElement("span", { className: "pill neutral" }, niceType(a.typology))),
#|                React.createElement("div", { className: "row" },
#|                    React.createElement("div", { className: "bar" },
#|                        React.createElement("div", { style: { width: `${Math.round(a.risk_score * 100)}%` } })),
#|                    React.createElement("b", { className: "risk" }, (0, api_1.pct)(a.risk_score, 0)))),
#|            a.narrative && (React.createElement(Narrative, { text: a.narrative, source: a.narrative_source, sha256: a.narrative_sha256, animate: fresh.has(a.id), onDone: () => clearFresh(a.id) })),
#|            React.createElement("div", { className: "row actions" },
#|                React.createElement("button", { className: "btn ghost", disabled: busy === `n${a.id}`, onClick: () => void doNarrative(a.id) }, busy === `n${a.id}` ? "Generating…" : a.narrative ? "Regenerate narrative" : "Generate SAR narrative"),
#|                React.createElement("button", { className: "btn ghost", onClick: () => void doExplain(a.account_id) }, "View network"),
#|                a.status === "open" && a.narrative && (React.createElement(React.Fragment, null,
#|                    React.createElement("button", { className: "btn red", disabled: busy === `d${a.id}`, onClick: () => void doDecision(a.id, "confirmed") }, "Confirm suspicious"),
#|                    React.createElement("button", { className: "btn green", disabled: busy === `d${a.id}`, onClick: () => void doDecision(a.id, "dismissed") }, "Dismiss"))),
#|                a.reviewed_by && React.createElement("span", { className: "muted" },
#|                    "reviewed by ",
#|                    a.reviewed_by)))))));
#|    const networkView = (React.createElement(React.Fragment, null,
#|        React.createElement(Glass_1.default, null,
#|            React.createElement("div", { className: "row" },
#|                React.createElement("input", { className: "input grow", placeholder: "Account ID to explain, e.g. ACC0000123", value: netId, onChange: (e) => setNetId(e.target.value), onKeyDown: (e) => {
#|                        if (e.key === "Enter")
#|                            void doExplain();
#|                    } }),
#|                React.createElement("button", { className: "btn shine", disabled: !netId.trim() || busy === "explain", onClick: () => void doExplain() }, busy === "explain" ? "Explaining…" : "Explain")),
#|            samples !== null && (React.createElement("div", { className: "row chips" },
#|                React.createElement("span", { className: "muted" }, "Try:"),
#|                samples.suspicious.slice(0, 3).map((a) => (React.createElement("button", { key: a, className: "chip bad", onClick: () => setNetId(a) }, a)))))),
#|        busy === "explain" && (React.createElement(Glass_1.default, null,
#|            React.createElement("div", { className: "scan-wait" },
#|                React.createElement("div", { className: "radar" }),
#|                React.createElement("p", { className: "muted" }, "Running GNNExplainer\u2026")))),
#|        graph !== null && busy !== "explain" && (React.createElement("div", { className: "rise" },
#|            React.createElement(NetworkGraph_1.default, { key: graph.account_id, data: graph }))),
#|        graph === null && busy !== "explain" && (React.createElement(Glass_1.default, null,
#|            React.createElement("p", { className: "muted" }, "Explain an account to see the transaction network the model reasoned over.")))));
#|    return (React.createElement(React.Fragment, null,
#|        React.createElement(Background_1.default, null),
#|        demoMode && React.createElement("div", { className: "demo-badge" }, "Demo mode \u00B7 sample data \u00B7 no live backend"),
#|        React.createElement("div", { className: "toasts" }, toasts.map((t) => (React.createElement("div", { key: t.id, className: `toast ${t.kind}` }, t.msg)))),
#|        React.createElement("div", { className: `app${demoMode ? " demo" : ""}` },
#|            React.createElement("header", { className: "topbar" },
#|                React.createElement("div", { className: "brand" },
#|                    React.createElement("div", { className: "logo" },
#|                        React.createElement("span", { className: "logo-ring" }),
#|                        React.createElement(Icons_1.IconShield, null)),
#|                    React.createElement("div", null,
#|                        React.createElement("div", { className: "brand-name" },
#|                            "XAI",
#|                            React.createElement("span", null, "\u00B7"),
#|                            "AMLBench"),
#|                        React.createElement("div", { className: "brand-sub" }, "Compliance Console"))),
#|                React.createElement("div", { className: `status ${online === false ? "off" : online ? "on" : ""}` },
#|                    React.createElement("i", null),
#|                    online === null ? "Connecting…" : online ? "Backend online" : "Backend offline")),
#|            React.createElement("main", { key: tab, className: "view" },
#|                tab === "overview" && overview,
#|                tab === "scan" && scanView,
#|                tab === "alerts" && alertsView,
#|                tab === "network" && networkView)),
#|        React.createElement(FloatingDock_1.default, { tabs: tabs, active: tab, onChange: setTab })));
#|}
#|
#|});
#|__d("src/components/Background", function (exports, require, module) {
#|"use strict";
#|"use client";
#|Object.defineProperty(exports, "__esModule", { value: true });
#|exports.default = Background;
#|const react_1 = require("react");
#|/** Animated backdrop: drifting glow orbs, a moving grid and a cursor-reactive particle network. */
#|function Background() {
#|    const ref = (0, react_1.useRef)(null);
#|    (0, react_1.useEffect)(() => {
#|        const canvas = ref.current;
#|        if (!canvas)
#|            return undefined;
#|        const ctx = canvas.getContext("2d");
#|        if (!ctx)
#|            return undefined;
#|        const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
#|        const dpr = Math.min(window.devicePixelRatio || 1, 2);
#|        const mouse = { x: -9999, y: -9999 };
#|        let w = 0;
#|        let h = 0;
#|        let pts = [];
#|        let raf = 0;
#|        const resize = () => {
#|            w = window.innerWidth;
#|            h = window.innerHeight;
#|            canvas.width = w * dpr;
#|            canvas.height = h * dpr;
#|            canvas.style.width = `${w}px`;
#|            canvas.style.height = `${h}px`;
#|            ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
#|            const n = Math.round(Math.min(95, (w * h) / 15000));
#|            pts = Array.from({ length: n }, () => ({
#|                x: Math.random() * w,
#|                y: Math.random() * h,
#|                vx: (Math.random() - 0.5) * 0.35,
#|                vy: (Math.random() - 0.5) * 0.35,
#|                r: Math.random() * 1.5 + 0.6,
#|            }));
#|        };
#|        const draw = () => {
#|            ctx.clearRect(0, 0, w, h);
#|            if (!reduce) {
#|                for (const p of pts) {
#|                    p.x += p.vx;
#|                    p.y += p.vy;
#|                    if (p.x < -10)
#|                        p.x = w + 10;
#|                    else if (p.x > w + 10)
#|                        p.x = -10;
#|                    if (p.y < -10)
#|                        p.y = h + 10;
#|                    else if (p.y > h + 10)
#|                        p.y = -10;
#|                }
#|            }
#|            for (let i = 0; i < pts.length; i++) {
#|                const a = pts[i];
#|                for (let j = i + 1; j < pts.length; j++) {
#|                    const b = pts[j];
#|                    const d = Math.hypot(a.x - b.x, a.y - b.y);
#|                    if (d < 130) {
#|                        ctx.strokeStyle = `rgba(96,165,250,${(1 - d / 130) * 0.32})`;
#|                        ctx.lineWidth = 1;
#|                        ctx.beginPath();
#|                        ctx.moveTo(a.x, a.y);
#|                        ctx.lineTo(b.x, b.y);
#|                        ctx.stroke();
#|                    }
#|                }
#|                const dm = Math.hypot(a.x - mouse.x, a.y - mouse.y);
#|                if (dm < 170) {
#|                    ctx.strokeStyle = `rgba(34,211,238,${(1 - dm / 170) * 0.7})`;
#|                    ctx.beginPath();
#|                    ctx.moveTo(a.x, a.y);
#|                    ctx.lineTo(mouse.x, mouse.y);
#|                    ctx.stroke();
#|                }
#|                ctx.fillStyle = "rgba(147,197,253,0.85)";
#|                ctx.beginPath();
#|                ctx.arc(a.x, a.y, a.r, 0, Math.PI * 2);
#|                ctx.fill();
#|            }
#|            if (!reduce)
#|                raf = requestAnimationFrame(draw);
#|        };
#|        const onMove = (e) => {
#|            mouse.x = e.clientX;
#|            mouse.y = e.clientY;
#|        };
#|        const onLeave = () => {
#|            mouse.x = -9999;
#|            mouse.y = -9999;
#|        };
#|        resize();
#|        draw();
#|        window.addEventListener("resize", resize);
#|        window.addEventListener("mousemove", onMove);
#|        window.addEventListener("mouseleave", onLeave);
#|        return () => {
#|            cancelAnimationFrame(raf);
#|            window.removeEventListener("resize", resize);
#|            window.removeEventListener("mousemove", onMove);
#|            window.removeEventListener("mouseleave", onLeave);
#|        };
#|    }, []);
#|    return (React.createElement("div", { className: "bg", "aria-hidden": "true" },
#|        React.createElement("div", { className: "orbs" },
#|            React.createElement("i", null),
#|            React.createElement("i", null),
#|            React.createElement("i", null)),
#|        React.createElement("div", { className: "grid-floor" }),
#|        React.createElement("canvas", { ref: ref, className: "bg-canvas" })));
#|}
#|
#|});
#|__d("src/components/FloatingDock", function (exports, require, module) {
#|"use strict";
#|"use client";
#|Object.defineProperty(exports, "__esModule", { value: true });
#|exports.default = FloatingDock;
#|/** Floating pill navigation with a sliding, springy active indicator. */
#|function FloatingDock({ tabs, active, onChange }) {
#|    const idx = Math.max(0, tabs.findIndex((t) => t.id === active));
#|    // CSS custom properties are not part of CSSProperties, so cast once here.
#|    const vars = { "--n": tabs.length, "--i": idx };
#|    return (React.createElement("nav", { className: "dock", style: vars, "aria-label": "Sections" },
#|        React.createElement("span", { className: "dock-glow" }),
#|        React.createElement("span", { className: "dock-indicator" }),
#|        tabs.map((t) => (React.createElement("button", { key: t.id, type: "button", className: `dock-tab ${t.id === active ? "on" : ""}`, onClick: () => onChange(t.id), "aria-current": t.id === active ? "page" : undefined },
#|            React.createElement("span", { className: "dock-ico" }, t.icon),
#|            React.createElement("span", { className: "dock-label" }, t.label),
#|            t.badge !== undefined && t.badge > 0 && React.createElement("span", { className: "dock-badge" }, t.badge))))));
#|}
#|
#|});
#|__d("src/components/Glass", function (exports, require, module) {
#|"use strict";
#|"use client";
#|Object.defineProperty(exports, "__esModule", { value: true });
#|exports.default = Glass;
#|/** Glass panel with a soft spotlight that follows the cursor. */
#|function Glass({ as: Tag = "section", className = "", children, ...rest }) {
#|    const onMove = (e) => {
#|        const el = e.currentTarget;
#|        const r = el.getBoundingClientRect();
#|        el.style.setProperty("--mx", `${e.clientX - r.left}px`);
#|        el.style.setProperty("--my", `${e.clientY - r.top}px`);
#|    };
#|    return (React.createElement(Tag, { className: `glass ${className}`, onMouseMove: onMove, ...rest }, children));
#|}
#|
#|});
#|__d("src/components/Icons", function (exports, require, module) {
#|"use strict";
#|Object.defineProperty(exports, "__esModule", { value: true });
#|exports.IconShield = exports.IconGraph = exports.IconBell = exports.IconRadar = exports.IconHome = void 0;
#|const base = {
#|    width: 22,
#|    height: 22,
#|    viewBox: "0 0 24 24",
#|    fill: "none",
#|    stroke: "currentColor",
#|    strokeWidth: 1.8,
#|    strokeLinecap: "round",
#|    strokeLinejoin: "round",
#|};
#|const IconHome = () => (React.createElement("svg", { ...base },
#|    React.createElement("path", { d: "M3 11.5 12 4l9 7.5" }),
#|    React.createElement("path", { d: "M5 10v9a1 1 0 0 0 1 1h4v-6h4v6h4a1 1 0 0 0 1-1v-9" })));
#|exports.IconHome = IconHome;
#|const IconRadar = () => (React.createElement("svg", { ...base },
#|    React.createElement("circle", { cx: "12", cy: "12", r: "9" }),
#|    React.createElement("circle", { cx: "12", cy: "12", r: "4.5" }),
#|    React.createElement("path", { d: "M12 12 19 6.5" })));
#|exports.IconRadar = IconRadar;
#|const IconBell = () => (React.createElement("svg", { ...base },
#|    React.createElement("path", { d: "M6 9a6 6 0 1 1 12 0c0 6 2.5 7.5 2.5 7.5h-17S6 15 6 9Z" }),
#|    React.createElement("path", { d: "M10 20a2 2 0 0 0 4 0" })));
#|exports.IconBell = IconBell;
#|const IconGraph = () => (React.createElement("svg", { ...base },
#|    React.createElement("circle", { cx: "6", cy: "7", r: "2.3" }),
#|    React.createElement("circle", { cx: "18", cy: "6", r: "2.3" }),
#|    React.createElement("circle", { cx: "12", cy: "18", r: "2.3" }),
#|    React.createElement("path", { d: "M8.2 7.3 15.8 6.5M7.2 9.1 10.9 15.9M16.9 8.1 13.1 15.9" })));
#|exports.IconGraph = IconGraph;
#|const IconShield = () => (React.createElement("svg", { width: "22", height: "22", viewBox: "0 0 24 24", fill: "none", stroke: "#fff", strokeWidth: "1.9", strokeLinecap: "round", strokeLinejoin: "round" },
#|    React.createElement("path", { d: "M12 3 4.5 6v5.5c0 4.6 3.1 8.2 7.5 9.5 4.4-1.3 7.5-4.9 7.5-9.5V6L12 3Z" }),
#|    React.createElement("path", { d: "m9 12 2.2 2.2L15.5 10" })));
#|exports.IconShield = IconShield;
#|
#|});
#|__d("src/components/NetworkGraph", function (exports, require, module) {
#|"use strict";
#|"use client";
#|var __importDefault = (this && this.__importDefault) || function (mod) {
#|    return (mod && mod.__esModule) ? mod : { "default": mod };
#|};
#|Object.defineProperty(exports, "__esModule", { value: true });
#|exports.default = NetworkGraph;
#|const react_1 = require("react");
#|const api_1 = require("@/lib/api");
#|const Glass_1 = __importDefault(require("./Glass"));
#|const W = 900;
#|const H = 560;
#|const TYPE_COLOR = {
#|    individual: "#60a5fa",
#|    business: "#22d3ee",
#|    shell: "#fb7185",
#|    unknown: "#94a3b8",
#|};
#|const FEATURE_LABELS = {
#|    near_thr_ratio: "Transfers just below reporting threshold",
#|    near_thr_cnt: "Transfers just below reporting threshold",
#|    xb_out_ratio: "Cross-border outflow share",
#|    xb_in_ratio: "Cross-border inflow share",
#|    n_cp_countries: "Counterparty countries",
#|    burst_6h: "Burst of activity in 6 hours",
#|    flow_ratio: "Inflow vs outflow balance",
#|    retained_frac: "Pass-through (little retained)",
#|    out_uniq: "Distinct recipients (fan-out)",
#|    in_uniq: "Distinct senders (fan-in)",
#|    out_deg: "Outgoing transactions",
#|    in_deg: "Incoming transactions",
#|    out_amt_sum: "Total sent",
#|    in_amt_sum: "Total received",
#|    out_amt_mean: "Average sent",
#|    in_amt_mean: "Average received",
#|    out_amt_max: "Largest sent",
#|    in_amt_max: "Largest received",
#|    out_amt_std: "Sent amount variability",
#|    in_amt_std: "Received amount variability",
#|    age_days: "Account age",
#|    active_span_h: "Active time span",
#|    is_offshore: "Offshore jurisdiction",
#|    type_shell: "Shell entity",
#|    type_business: "Business account",
#|    type_individual: "Individual account",
#|};
#|const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
#|/** Fruchterman-Reingold layout (deterministic, so the picture never jumps between renders). */
#|function computeLayout(nodes, edges, subject) {
#|    let seed = 7;
#|    const rnd = () => {
#|        seed = (seed * 16807) % 2147483647;
#|        return seed / 2147483647;
#|    };
#|    const idx = new Map();
#|    nodes.forEach((n, i) => idx.set(n.account_id, i));
#|    const pos = nodes.map((n, i) => {
#|        const angle = (i / Math.max(nodes.length, 1)) * 2 * Math.PI;
#|        const r = n.account_id === subject ? 0 : 170 + rnd() * 90;
#|        return { x: W / 2 + Math.cos(angle) * r, y: H / 2 + Math.sin(angle) * r };
#|    });
#|    const links = [];
#|    for (const e of edges) {
#|        const a = idx.get(e.src);
#|        const b = idx.get(e.dst);
#|        if (a !== undefined && b !== undefined && a !== b)
#|            links.push([a, b]);
#|    }
#|    const k = Math.sqrt((W * H) / Math.max(nodes.length, 1)) * 0.85;
#|    const pinned = idx.get(subject); // the flagged account stays in the centre
#|    const ITER = 320;
#|    for (let it = 0; it < ITER; it++) {
#|        const cool = 1 - it / ITER;
#|        const disp = pos.map(() => ({ x: 0, y: 0 }));
#|        for (let i = 0; i < pos.length; i++) {
#|            for (let j = i + 1; j < pos.length; j++) {
#|                const dx = pos[i].x - pos[j].x;
#|                const dy = pos[i].y - pos[j].y;
#|                const d = Math.hypot(dx, dy) || 0.01;
#|                const f = (k * k) / d;
#|                disp[i].x += (dx / d) * f;
#|                disp[i].y += (dy / d) * f;
#|                disp[j].x -= (dx / d) * f;
#|                disp[j].y -= (dy / d) * f;
#|            }
#|        }
#|        for (const [a, b] of links) {
#|            const dx = pos[a].x - pos[b].x;
#|            const dy = pos[a].y - pos[b].y;
#|            const d = Math.hypot(dx, dy) || 0.01;
#|            const f = (d * d) / k;
#|            disp[a].x -= (dx / d) * f;
#|            disp[a].y -= (dy / d) * f;
#|            disp[b].x += (dx / d) * f;
#|            disp[b].y += (dy / d) * f;
#|        }
#|        const lim = 46 * cool + 1;
#|        for (let i = 0; i < pos.length; i++) {
#|            if (i === pinned) {
#|                pos[i].x = W / 2;
#|                pos[i].y = H / 2;
#|                continue;
#|            }
#|            disp[i].x += (W / 2 - pos[i].x) * 0.05;
#|            disp[i].y += (H / 2 - pos[i].y) * 0.05;
#|            const d = Math.hypot(disp[i].x, disp[i].y) || 1;
#|            pos[i].x = clamp(pos[i].x + (disp[i].x / d) * Math.min(d, lim), 44, W - 44);
#|            pos[i].y = clamp(pos[i].y + (disp[i].y / d) * Math.min(d, lim), 44, H - 44);
#|        }
#|    }
#|    // Collision pass: keep nodes (and the subject's long id label) from overlapping.
#|    for (let pass = 0; pass < 80; pass++) {
#|        let moved = false;
#|        for (let i = 0; i < pos.length; i++) {
#|            for (let j = i + 1; j < pos.length; j++) {
#|                const minD = i === pinned || j === pinned ? 84 : 46;
#|                let dx = pos[j].x - pos[i].x;
#|                let dy = pos[j].y - pos[i].y;
#|                let d = Math.hypot(dx, dy);
#|                if (d >= minD)
#|                    continue;
#|                if (d < 0.01) {
#|                    dx = Math.cos(i + j + 1);
#|                    dy = Math.sin(i + j + 1);
#|                    d = 1;
#|                }
#|                const push = minD - d + 0.5;
#|                const wi = i === pinned ? 0 : j === pinned ? 1 : 0.5;
#|                pos[i].x -= (dx / d) * push * wi;
#|                pos[i].y -= (dy / d) * push * wi;
#|                pos[j].x += (dx / d) * push * (1 - wi);
#|                pos[j].y += (dy / d) * push * (1 - wi);
#|                moved = true;
#|            }
#|        }
#|        for (const p of pos) {
#|            p.x = clamp(p.x, 44, W - 44);
#|            p.y = clamp(p.y, 44, H - 44);
#|        }
#|        if (!moved)
#|            break;
#|    }
#|    return pos;
#|}
#|/** Curved edge that stops at the rim of the destination node so the arrowhead stays visible. */
#|function edgePath(p1, p2, r2, k) {
#|    const dx = p2.x - p1.x;
#|    const dy = p2.y - p1.y;
#|    const d = Math.hypot(dx, dy) || 1;
#|    const off = 14 + k * 12;
#|    const cx = (p1.x + p2.x) / 2 + (-dy / d) * off;
#|    const cy = (p1.y + p2.y) / 2 + (dx / d) * off;
#|    const ex = p2.x - cx;
#|    const ey = p2.y - cy;
#|    const el = Math.hypot(ex, ey) || 1;
#|    const tx = p2.x - (ex / el) * (r2 + 5);
#|    const ty = p2.y - (ey / el) * (r2 + 5);
#|    return `M${p1.x.toFixed(1)},${p1.y.toFixed(1)} Q${cx.toFixed(1)},${cy.toFixed(1)} ${tx.toFixed(1)},${ty.toFixed(1)}`;
#|}
#|function NetworkGraph({ data }) {
#|    const [sel, setSel] = (0, react_1.useState)(null);
#|    const subject = data.account_id;
#|    const model = (0, react_1.useMemo)(() => {
#|        var _a;
#|        const nodes = data.nodes;
#|        const idx = new Map();
#|        nodes.forEach((n, i) => idx.set(n.account_id, i));
#|        const pos = computeLayout(nodes, data.edges, subject);
#|        const maxImp = Math.max(1e-9, ...data.edges.map((e) => e.importance));
#|        const pairs = new Map();
#|        const edges = [];
#|        for (const e of data.edges) {
#|            const a = idx.get(e.src);
#|            const b = idx.get(e.dst);
#|            if (a === undefined || b === undefined)
#|                continue;
#|            const key = [e.src, e.dst].sort().join("|");
#|            const k = (_a = pairs.get(key)) !== null && _a !== void 0 ? _a : 0;
#|            pairs.set(key, k + 1);
#|            edges.push({ ...e, a, b, k, w: e.importance / maxImp });
#|        }
#|        const deg = nodes.map(() => 0);
#|        for (const e of edges) {
#|            deg[e.a] += 1;
#|            deg[e.b] += 1;
#|        }
#|        const radius = nodes.map((n, i) => n.account_id === subject ? 17 : Math.min(15, 8 + Math.sqrt(deg[i]) * 1.4));
#|        return { nodes, pos, edges, radius };
#|    }, [data, subject]);
#|    const detail = (0, react_1.useMemo)(() => {
#|        if (sel === null)
#|            return null;
#|        const node = model.nodes[sel];
#|        const out = model.edges.filter((e) => e.src === node.account_id);
#|        const inn = model.edges.filter((e) => e.dst === node.account_id);
#|        const sum = (list) => list.reduce((s, e) => s + e.amount, 0);
#|        const top = [...out, ...inn].sort((x, y) => y.amount - x.amount).slice(0, 5);
#|        return { node, out: out.length, inn: inn.length, sent: sum(out), recv: sum(inn), top };
#|    }, [sel, model]);
#|    const isConnected = (e) => sel !== null && (e.a === sel || e.b === sel);
#|    return (React.createElement("div", { className: "net-layout" },
#|        React.createElement(Glass_1.default, { className: "net-card" },
#|            React.createElement("svg", { viewBox: `0 0 ${W} ${H}`, className: "net", role: "img", "aria-label": "Transaction network of the flagged account" },
#|                React.createElement("defs", null,
#|                    React.createElement("marker", { id: "arrD", markerUnits: "userSpaceOnUse", markerWidth: "10", markerHeight: "10", refX: "8", refY: "5", orient: "auto" },
#|                        React.createElement("path", { d: "M0,0 L10,5 L0,10 Z", fill: "#22d3ee" })),
#|                    React.createElement("marker", { id: "arrX", markerUnits: "userSpaceOnUse", markerWidth: "10", markerHeight: "10", refX: "8", refY: "5", orient: "auto" },
#|                        React.createElement("path", { d: "M0,0 L10,5 L0,10 Z", fill: "#fbbf24" })),
#|                    React.createElement("filter", { id: "glow", x: "-50%", y: "-50%", width: "200%", height: "200%" },
#|                        React.createElement("feGaussianBlur", { stdDeviation: "4", result: "b" }),
#|                        React.createElement("feMerge", null,
#|                            React.createElement("feMergeNode", { in: "b" }),
#|                            React.createElement("feMergeNode", { in: "SourceGraphic" })))),
#|                model.edges.map((e, i) => (React.createElement("path", { key: `${e.tx_id}-${i}`, d: edgePath(model.pos[e.a], model.pos[e.b], model.radius[e.b], e.k), className: `edge flow ${sel !== null && !isConnected(e) ? "dim" : ""}`, stroke: e.cross_border ? "#fbbf24" : "#22d3ee", strokeWidth: 1 + e.w * 2.6, opacity: 0.35 + e.w * 0.65, markerEnd: `url(#${e.cross_border ? "arrX" : "arrD"})`, style: { animationDuration: `${1.6 - e.w * 0.9}s` } },
#|                    React.createElement("title", null, `${e.src} → ${e.dst} · ${(0, api_1.money)(e.amount)}${e.cross_border ? " · cross-border" : ""}`)))),
#|                model.nodes.map((n, i) => {
#|                    var _a;
#|                    const isSubject = n.account_id === subject;
#|                    const color = (_a = TYPE_COLOR[n.account_type]) !== null && _a !== void 0 ? _a : TYPE_COLOR.unknown;
#|                    const r = model.radius[i];
#|                    return (React.createElement("g", { key: n.account_id, className: `node ${sel === i ? "sel" : ""}`, transform: `translate(${model.pos[i].x.toFixed(1)},${model.pos[i].y.toFixed(1)})`, onClick: () => setSel(sel === i ? null : i), style: { animationDelay: `${(i % 12) * 60}ms` } },
#|                        isSubject && React.createElement("circle", { className: "pulse", r: r, fill: "none", stroke: "#e8f0ff", strokeWidth: "2" }),
#|                        React.createElement("circle", { r: r + 6, fill: color, opacity: "0.13" }),
#|                        React.createElement("circle", { r: r, fill: color, filter: isSubject || n.account_type === "shell" ? "url(#glow)" : undefined, stroke: isSubject ? "#ffffff" : "rgba(3,7,18,.7)", strokeWidth: isSubject ? 3 : 1.5 }),
#|                        React.createElement("text", { y: r + 15, textAnchor: "middle", className: "node-label" }, isSubject ? n.account_id : n.account_id.slice(-4))));
#|                })),
#|            React.createElement("div", { className: "legend" },
#|                React.createElement("span", null,
#|                    React.createElement("i", { style: { background: TYPE_COLOR.individual } }),
#|                    "Individual"),
#|                React.createElement("span", null,
#|                    React.createElement("i", { style: { background: TYPE_COLOR.business } }),
#|                    "Business"),
#|                React.createElement("span", null,
#|                    React.createElement("i", { style: { background: TYPE_COLOR.shell } }),
#|                    "Shell"),
#|                React.createElement("span", null,
#|                    React.createElement("i", { className: "ln", style: { background: "#22d3ee" } }),
#|                    "Domestic"),
#|                React.createElement("span", null,
#|                    React.createElement("i", { className: "ln", style: { background: "#fbbf24" } }),
#|                    "Cross-border"),
#|                React.createElement("span", { className: "hint" }, "Click a node \u00B7 thicker = more influential to the model"))),
#|        React.createElement("div", { className: "net-side" },
#|            React.createElement(Glass_1.default, null,
#|                React.createElement("h3", null, "Why the model flagged it"),
#|                React.createElement("div", { className: "score-line" },
#|                    React.createElement("span", null, "Risk score"),
#|                    React.createElement("b", null,
#|                        (data.risk_score * 100).toFixed(1),
#|                        "%")),
#|                data.top_features.length === 0 && React.createElement("p", { className: "muted" }, "No feature attribution returned."),
#|                data.top_features.map((f, i) => {
#|                    var _a;
#|                    return (React.createElement("div", { className: "feat", key: f.feature },
#|                        React.createElement("div", { className: "feat-top" },
#|                            React.createElement("span", null, (_a = FEATURE_LABELS[f.feature]) !== null && _a !== void 0 ? _a : f.feature),
#|                            React.createElement("em", null,
#|                                (f.weight * 100).toFixed(0),
#|                                "%")),
#|                        React.createElement("div", { className: "feat-bar" },
#|                            React.createElement("div", { style: { width: `${Math.min(100, f.weight * 260)}%`, animationDelay: `${i * 90}ms` } }))));
#|                }),
#|                React.createElement("p", { className: "muted small" },
#|                    data.method,
#|                    " \u00B7 ",
#|                    data.model,
#|                    " \u00B7 ",
#|                    data.edges.length,
#|                    " transactions shown")),
#|            React.createElement(Glass_1.default, { className: "node-detail" },
#|                detail === null && React.createElement("p", { className: "muted" }, "Select a node to inspect its flows."),
#|                detail !== null && (React.createElement(React.Fragment, null,
#|                    React.createElement("h3", null, detail.node.account_id),
#|                    React.createElement("div", { className: "row" },
#|                        React.createElement("span", { className: `pill t-${detail.node.account_type}` }, detail.node.account_type),
#|                        React.createElement("span", { className: "pill neutral" }, detail.node.country)),
#|                    React.createElement("div", { className: "kv" },
#|                        React.createElement("span", null, "Sent"),
#|                        React.createElement("b", null,
#|                            (0, api_1.money)(detail.sent),
#|                            " ",
#|                            React.createElement("em", null,
#|                                "(",
#|                                detail.out,
#|                                " tx)"))),
#|                    React.createElement("div", { className: "kv" },
#|                        React.createElement("span", null, "Received"),
#|                        React.createElement("b", null,
#|                            (0, api_1.money)(detail.recv),
#|                            " ",
#|                            React.createElement("em", null,
#|                                "(",
#|                                detail.inn,
#|                                " tx)"))),
#|                    React.createElement("div", { className: "muted small", style: { marginTop: 8 } }, "Largest transfers"),
#|                    detail.top.map((e) => (React.createElement("div", { className: "kv small", key: e.tx_id },
#|                        React.createElement("span", null,
#|                            e.src.slice(-4),
#|                            " \u2192 ",
#|                            e.dst.slice(-4),
#|                            e.cross_border ? " ✈" : ""),
#|                        React.createElement("b", null, (0, api_1.money)(e.amount)))))))))));
#|}
#|
#|});
#|__d("src/components/RiskGauge", function (exports, require, module) {
#|"use strict";
#|"use client";
#|Object.defineProperty(exports, "__esModule", { value: true });
#|exports.default = RiskGauge;
#|const react_1 = require("react");
#|const hooks_1 = require("./hooks");
#|/** 270-degree animated risk gauge. Give it a `key` that changes per scan to replay the sweep. */
#|function RiskGauge({ score, threshold }) {
#|    const [on, setOn] = (0, react_1.useState)(false);
#|    const shown = (0, hooks_1.useCountUp)(score * 100, 1200);
#|    (0, react_1.useEffect)(() => {
#|        const id = requestAnimationFrame(() => setOn(true));
#|        return () => cancelAnimationFrame(id);
#|    }, []);
#|    const r = 70;
#|    const c = 2 * Math.PI * r;
#|    const arc = c * 0.75;
#|    const flagged = score >= threshold;
#|    const tickAngle = ((135 + 270 * threshold) * Math.PI) / 180;
#|    const tx1 = 90 + Math.cos(tickAngle) * (r - 11);
#|    const ty1 = 90 + Math.sin(tickAngle) * (r - 11);
#|    const tx2 = 90 + Math.cos(tickAngle) * (r + 11);
#|    const ty2 = 90 + Math.sin(tickAngle) * (r + 11);
#|    return (React.createElement("div", { className: `gauge ${flagged ? "bad" : "good"}` },
#|        React.createElement("svg", { viewBox: "0 0 180 180" },
#|            React.createElement("defs", null,
#|                React.createElement("linearGradient", { id: "gaugeBad", x1: "0", y1: "0", x2: "1", y2: "1" },
#|                    React.createElement("stop", { offset: "0%", stopColor: "#fbbf24" }),
#|                    React.createElement("stop", { offset: "100%", stopColor: "#fb7185" })),
#|                React.createElement("linearGradient", { id: "gaugeGood", x1: "0", y1: "0", x2: "1", y2: "1" },
#|                    React.createElement("stop", { offset: "0%", stopColor: "#22d3ee" }),
#|                    React.createElement("stop", { offset: "100%", stopColor: "#3b82f6" }))),
#|            React.createElement("circle", { cx: "90", cy: "90", r: r, fill: "none", stroke: "rgba(148,163,184,.14)", strokeWidth: "12", strokeLinecap: "round", strokeDasharray: `${arc} ${c}`, transform: "rotate(135 90 90)" }),
#|            React.createElement("circle", { className: "gauge-arc", cx: "90", cy: "90", r: r, fill: "none", strokeWidth: "12", strokeLinecap: "round", stroke: flagged ? "url(#gaugeBad)" : "url(#gaugeGood)", strokeDasharray: `${arc} ${c}`, strokeDashoffset: on ? arc * (1 - score) : arc, transform: "rotate(135 90 90)" }),
#|            React.createElement("line", { x1: tx1, y1: ty1, x2: tx2, y2: ty2, stroke: "#e8f0ff", strokeWidth: "2", strokeLinecap: "round", opacity: ".8" })),
#|        React.createElement("div", { className: "gauge-text" },
#|            React.createElement("b", null,
#|                shown.toFixed(0),
#|                React.createElement("small", null, "%")),
#|            React.createElement("span", null, "risk score"))));
#|}
#|
#|});
#|__d("src/components/hooks", function (exports, require, module) {
#|"use strict";
#|"use client";
#|Object.defineProperty(exports, "__esModule", { value: true });
#|exports.useCountUp = useCountUp;
#|exports.useTypewriter = useTypewriter;
#|const react_1 = require("react");
#|/** Smoothly animates a number from its previous value to `target`. */
#|function useCountUp(target, duration = 900) {
#|    const [val, setVal] = (0, react_1.useState)(0);
#|    const from = (0, react_1.useRef)(0);
#|    (0, react_1.useEffect)(() => {
#|        const begin = performance.now();
#|        const startVal = from.current;
#|        let raf = 0;
#|        const tick = (now) => {
#|            const t = Math.min(1, (now - begin) / duration);
#|            const eased = 1 - Math.pow(1 - t, 3);
#|            const v = startVal + (target - startVal) * eased;
#|            from.current = v;
#|            setVal(v);
#|            if (t < 1)
#|                raf = requestAnimationFrame(tick);
#|        };
#|        raf = requestAnimationFrame(tick);
#|        return () => cancelAnimationFrame(raf);
#|    }, [target, duration]);
#|    return val;
#|}
#|/** Types `text` out character by character while `enabled`; calls `onDone` when finished. */
#|function useTypewriter(text, enabled, onDone) {
#|    const [out, setOut] = (0, react_1.useState)(enabled ? "" : text);
#|    const doneRef = (0, react_1.useRef)(onDone);
#|    doneRef.current = onDone;
#|    (0, react_1.useEffect)(() => {
#|        if (!enabled) {
#|            setOut(text);
#|            return undefined;
#|        }
#|        let i = 0;
#|        setOut("");
#|        const id = setInterval(() => {
#|            var _a;
#|            i += 2;
#|            setOut(text.slice(0, i));
#|            if (i >= text.length) {
#|                clearInterval(id);
#|                (_a = doneRef.current) === null || _a === void 0 ? void 0 : _a.call(doneRef);
#|            }
#|        }, 14);
#|        return () => clearInterval(id);
#|    }, [text, enabled]);
#|    return out;
#|}
#|
#|});
#|__d("src/lib/api", function (exports, require, module) {
#|"use strict";
#|Object.defineProperty(exports, "__esModule", { value: true });
#|exports.errorMessage = exports.money = exports.pct = exports.sleep = void 0;
#|exports.api = api;
#|/** Thin typed wrapper around fetch for the /api routes (served by Traefik or the Next.js rewrite). */
#|async function api(path, options) {
#|    const res = await fetch(`/api${path}`, {
#|        headers: { "Content-Type": "application/json" },
#|        ...options,
#|    });
#|    const text = await res.text();
#|    let data;
#|    try {
#|        data = JSON.parse(text);
#|    }
#|    catch {
#|        data = { detail: text };
#|    }
#|    if (!res.ok) {
#|        const detail = data.detail;
#|        throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
#|    }
#|    return data;
#|}
#|const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
#|exports.sleep = sleep;
#|const pct = (x, digits = 1) => `${(x * 100).toFixed(digits)}%`;
#|exports.pct = pct;
#|const money = (n) => `$${Math.round(n).toLocaleString("en-US")}`;
#|exports.money = money;
#|const errorMessage = (e) => (e instanceof Error ? e.message : String(e));
#|exports.errorMessage = errorMessage;
#|
#|});
#|__d("src/lib/demo", function (exports, require, module) {
#|"use strict";
#|Object.defineProperty(exports, "__esModule", { value: true });
#|exports.isDemo = void 0;
#|/**
#| * Demo mode: run the real UI with built-in sample data and NO backend.
#| *
#| *   npm run dev:demo      ->  http://localhost:3000
#| *
#| * next.config.js inlines NEXT_PUBLIC_DEMO at build time ("1" only for the dev:demo script), so in a normal
#| * production build this is a constant `false` and the mock backend code is dropped from the bundle.
#| * The try/catch keeps it safe in plain-browser bundles (demo/build-demo.cjs) where `process` does not exist.
#| */
#|const isDemo = () => {
#|    try {
#|        return process.env.NEXT_PUBLIC_DEMO === "1";
#|    }
#|    catch {
#|        return false;
#|    }
#|};
#|exports.isDemo = isDemo;
#|
#|});
#|__d("src/types", function (exports, require, module) {
#|"use strict";
#|/** Types shared between the UI and the backend JSON (see backend/main.py and gnn_aml_core/main.py). */
#|Object.defineProperty(exports, "__esModule", { value: true });
#|
#|});
#|__d("demo/demoData", function (exports, require, module) {
#|"use strict";
#|Object.defineProperty(exports, "__esModule", { value: true });
#|exports.DEMO_NARRATIVES = exports.DEMO_EXPLANATIONS = exports.DEMO_ACCOUNTS = void 0;
#|exports.DEMO_ACCOUNTS = [
#|    {
#|        "id": "ACC0001449",
#|        "score": 0.97,
#|        "label": "smurfing"
#|    },
#|    {
#|        "id": "ACC0000528",
#|        "score": 0.96,
#|        "label": "shell_company"
#|    },
#|    {
#|        "id": "ACC0000283",
#|        "score": 0.91,
#|        "label": "cyclic_loop"
#|    },
#|    {
#|        "id": "ACC0003271",
#|        "score": 0.93,
#|        "label": "scatter_gather"
#|    },
#|    {
#|        "id": "ACC0005473",
#|        "score": 0.88,
#|        "label": "cross_border_velocity"
#|    },
#|    {
#|        "id": "ACC0002825",
#|        "score": 0.04,
#|        "label": "benign"
#|    },
#|    {
#|        "id": "ACC0000819",
#|        "score": 0.07,
#|        "label": "benign"
#|    },
#|    {
#|        "id": "ACC0003621",
#|        "score": 0.11,
#|        "label": "benign"
#|    }
#|];
#|/** One explained account per line. Each entry is what POST /explain returns. */
#|exports.DEMO_EXPLANATIONS = {
#|    "ACC0001449": { "account_id": "ACC0001449", "risk_score": 0.97, "threshold": 0.62, "reporting_threshold": 10000, "model": "gatv2", "method": "GNNExplainer", "nodes": [{ "account_id": "ACC0001449", "account_type": "individual", "country": "FR" }, { "account_id": "ACC0005009", "account_type": "individual", "country": "NG" }, { "account_id": "ACC0005002", "account_type": "individual", "country": "IN" }, { "account_id": "ACC0005004", "account_type": "individual", "country": "AE" }, { "account_id": "ACC0005008", "account_type": "individual", "country": "IN" }, { "account_id": "ACC0005003", "account_type": "individual", "country": "FR" }, { "account_id": "ACC0005007", "account_type": "individual", "country": "US" }, { "account_id": "ACC0005005", "account_type": "individual", "country": "NG" }, { "account_id": "ACC0005010", "account_type": "individual", "country": "BR" }, { "account_id": "ACC0005006", "account_type": "individual", "country": "US" }, { "account_id": "ACC0005011", "account_type": "individual", "country": "GB" }, { "account_id": "ACC0005000", "account_type": "business", "country": "DE" }, { "account_id": "ACC0005001", "account_type": "individual", "country": "DE" }, { "account_id": "ACC0005012", "account_type": "individual", "country": "AE" }], "edges": [{ "tx_id": "TX000001531", "src": "ACC0001449", "dst": "ACC0005009", "amount": 8748.4, "timestamp": 1767327013, "cross_border": 1, "importance": 0.73095 }, { "tx_id": "TX000001747", "src": "ACC0001449", "dst": "ACC0005002", "amount": 9327.9, "timestamp": 1767341273, "cross_border": 1, "importance": 0.77391 }, { "tx_id": "TX000001810", "src": "ACC0001449", "dst": "ACC0005004", "amount": 8961.11, "timestamp": 1767345429, "cross_border": 1, "importance": 0.91968 }, { "tx_id": "TX000001882", "src": "ACC0001449", "dst": "ACC0005008", "amount": 9315.29, "timestamp": 1767349570, "cross_border": 1, "importance": 0.73626 }, { "tx_id": "TX000001972", "src": "ACC0001449", "dst": "ACC0005003", "amount": 8456.88, "timestamp": 1767355171, "cross_border": 0, "importance": 0.75314 }, { "tx_id": "TX000001995", "src": "ACC0001449", "dst": "ACC0005007", "amount": 9432.58, "timestamp": 1767356663, "cross_border": 1, "importance": 0.78495 }, { "tx_id": "TX000002000", "src": "ACC0001449", "dst": "ACC0005005", "amount": 8277.01, "timestamp": 1767357071, "cross_border": 1, "importance": 0.62386 }, { "tx_id": "TX000002157", "src": "ACC0001449", "dst": "ACC0005010", "amount": 8227.41, "timestamp": 1767367070, "cross_border": 1, "importance": 0.75476 }, { "tx_id": "TX000002170", "src": "ACC0001449", "dst": "ACC0005006", "amount": 8972.81, "timestamp": 1767367768, "cross_border": 1, "importance": 0.80195 }, { "tx_id": "TX000002186", "src": "ACC0001449", "dst": "ACC0005011", "amount": 8648.84, "timestamp": 1767368766, "cross_border": 1, "importance": 0.86719 }, { "tx_id": "TX000002188", "src": "ACC0005002", "dst": "ACC0005000", "amount": 9111.48, "timestamp": 1767368864, "cross_border": 1, "importance": 0.15577 }, { "tx_id": "TX000002225", "src": "ACC0005005", "dst": "ACC0005000", "amount": 7952.88, "timestamp": 1767371500, "cross_border": 1, "importance": 0.23529 }, { "tx_id": "TX000002248", "src": "ACC0001449", "dst": "ACC0005001", "amount": 8466.42, "timestamp": 1767372983, "cross_border": 1, "importance": 0.58627 }, { "tx_id": "TX000002314", "src": "ACC0005009", "dst": "ACC0005000", "amount": 8622.65, "timestamp": 1767377716, "cross_border": 1, "importance": 0.42766 }, { "tx_id": "TX000002323", "src": "ACC0005004", "dst": "ACC0005000", "amount": 8618.88, "timestamp": 1767378465, "cross_border": 1, "importance": 0.38351 }, { "tx_id": "TX000002326", "src": "ACC0005008", "dst": "ACC0005000", "amount": 8863.72, "timestamp": 1767378731, "cross_border": 1, "importance": 0.13591 }, { "tx_id": "TX000002435", "src": "ACC0005011", "dst": "ACC0005000", "amount": 8556.03, "timestamp": 1767384990, "cross_border": 1, "importance": 0.49323 }, { "tx_id": "TX000002562", "src": "ACC0001449", "dst": "ACC0005012", "amount": 8760.89, "timestamp": 1767393627, "cross_border": 1, "importance": 0.9359 }, { "tx_id": "TX000002743", "src": "ACC0005003", "dst": "ACC0005000", "amount": 8346.14, "timestamp": 1767404671, "cross_border": 1, "importance": 0.36849 }, { "tx_id": "TX000002792", "src": "ACC0005007", "dst": "ACC0005000", "amount": 9041.85, "timestamp": 1767407852, "cross_border": 1, "importance": 0.35391 }, { "tx_id": "TX000002879", "src": "ACC0005006", "dst": "ACC0005000", "amount": 8727.65, "timestamp": 1767412307, "cross_border": 1, "importance": 0.17985 }, { "tx_id": "TX000003084", "src": "ACC0005010", "dst": "ACC0005000", "amount": 8003.49, "timestamp": 1767425230, "cross_border": 1, "importance": 0.1257 }, { "tx_id": "TX000003143", "src": "ACC0005001", "dst": "ACC0005000", "amount": 8120.17, "timestamp": 1767429258, "cross_border": 0, "importance": 0.32078 }, { "tx_id": "TX000003286", "src": "ACC0005012", "dst": "ACC0005000", "amount": 8414.65, "timestamp": 1767438922, "cross_border": 1, "importance": 0.14263 }], "top_features": [{ "feature": "near_thr_ratio", "weight": 0.31 }, { "feature": "burst_6h", "weight": 0.19 }, { "feature": "flow_ratio", "weight": 0.15 }, { "feature": "out_uniq", "weight": 0.12 }, { "feature": "retained_frac", "weight": 0.09 }] },
#|    "ACC0000528": { "account_id": "ACC0000528", "risk_score": 0.96, "threshold": 0.62, "reporting_threshold": 10000, "model": "gatv2", "method": "GNNExplainer", "nodes": [{ "account_id": "ACC0000528", "account_type": "individual", "country": "CA" }, { "account_id": "ACC0005398", "account_type": "shell", "country": "BZ" }, { "account_id": "ACC0005399", "account_type": "shell", "country": "KY" }, { "account_id": "ACC0005400", "account_type": "shell", "country": "BZ" }, { "account_id": "ACC0005397", "account_type": "business", "country": "US" }], "edges": [{ "tx_id": "TX000027163", "src": "ACC0000528", "dst": "ACC0005398", "amount": 217855.54, "timestamp": 1768956500, "cross_border": 1, "importance": 0.62608 }, { "tx_id": "TX000028488", "src": "ACC0005398", "dst": "ACC0005399", "amount": 217086.63, "timestamp": 1769045035, "cross_border": 1, "importance": 0.21194 }, { "tx_id": "TX000028796", "src": "ACC0005399", "dst": "ACC0005400", "amount": 214467.46, "timestamp": 1769065773, "cross_border": 1, "importance": 0.13143 }, { "tx_id": "TX000029551", "src": "ACC0005400", "dst": "ACC0005397", "amount": 213073.76, "timestamp": 1769111464, "cross_border": 1, "importance": 0.2963 }], "top_features": [{ "feature": "retained_frac", "weight": 0.28 }, { "feature": "xb_out_ratio", "weight": 0.21 }, { "feature": "out_amt_max", "weight": 0.16 }, { "feature": "flow_ratio", "weight": 0.11 }, { "feature": "is_offshore", "weight": 0.08 }] },
#|    "ACC0000283": { "account_id": "ACC0000283", "risk_score": 0.91, "threshold": 0.62, "reporting_threshold": 10000, "model": "gatv2", "method": "GNNExplainer", "nodes": [{ "account_id": "ACC0000283", "account_type": "individual", "country": "US" }, { "account_id": "ACC0005353", "account_type": "business", "country": "IN" }, { "account_id": "ACC0005354", "account_type": "business", "country": "FR" }, { "account_id": "ACC0005355", "account_type": "business", "country": "CA" }, { "account_id": "ACC0005356", "account_type": "business", "country": "JP" }], "edges": [{ "tx_id": "TX000013336", "src": "ACC0000283", "dst": "ACC0005353", "amount": 151105.12, "timestamp": 1768072206, "cross_border": 1, "importance": 0.72621 }, { "tx_id": "TX000014109", "src": "ACC0005353", "dst": "ACC0005354", "amount": 149983.67, "timestamp": 1768121519, "cross_border": 1, "importance": 0.44012 }, { "tx_id": "TX000014598", "src": "ACC0005354", "dst": "ACC0005355", "amount": 149218.98, "timestamp": 1768152246, "cross_border": 1, "importance": 0.31727 }, { "tx_id": "TX000015124", "src": "ACC0005355", "dst": "ACC0005356", "amount": 147847.55, "timestamp": 1768186008, "cross_border": 1, "importance": 0.36331 }, { "tx_id": "TX000015365", "src": "ACC0005356", "dst": "ACC0000283", "amount": 146890.18, "timestamp": 1768201732, "cross_border": 1, "importance": 0.74991 }, { "tx_id": "TX000016122", "src": "ACC0000283", "dst": "ACC0005353", "amount": 144730.38, "timestamp": 1768248500, "cross_border": 1, "importance": 0.81498 }, { "tx_id": "TX000016671", "src": "ACC0005353", "dst": "ACC0005354", "amount": 142846.04, "timestamp": 1768285861, "cross_border": 1, "importance": 0.29379 }, { "tx_id": "TX000017109", "src": "ACC0005354", "dst": "ACC0005355", "amount": 142476.01, "timestamp": 1768313013, "cross_border": 1, "importance": 0.2257 }, { "tx_id": "TX000017871", "src": "ACC0005355", "dst": "ACC0005356", "amount": 141335.35, "timestamp": 1768359826, "cross_border": 1, "importance": 0.49911 }, { "tx_id": "TX000018411", "src": "ACC0005356", "dst": "ACC0000283", "amount": 139235.95, "timestamp": 1768393477, "cross_border": 1, "importance": 0.94828 }, { "tx_id": "TX000018671", "src": "ACC0000283", "dst": "ACC0005353", "amount": 137530.53, "timestamp": 1768411021, "cross_border": 1, "importance": 0.88609 }, { "tx_id": "TX000019432", "src": "ACC0005353", "dst": "ACC0005354", "amount": 135807.83, "timestamp": 1768460709, "cross_border": 1, "importance": 0.38897 }, { "tx_id": "TX000020036", "src": "ACC0005354", "dst": "ACC0005355", "amount": 134856.89, "timestamp": 1768498137, "cross_border": 1, "importance": 0.23981 }, { "tx_id": "TX000020120", "src": "ACC0005355", "dst": "ACC0005356", "amount": 133424.12, "timestamp": 1768503032, "cross_border": 1, "importance": 0.20727 }, { "tx_id": "TX000020214", "src": "ACC0005356", "dst": "ACC0000283", "amount": 133023.3, "timestamp": 1768508693, "cross_border": 1, "importance": 0.66562 }], "top_features": [{ "feature": "flow_ratio", "weight": 0.3 }, { "feature": "retained_frac", "weight": 0.22 }, { "feature": "in_deg", "weight": 0.14 }, { "feature": "out_deg", "weight": 0.12 }, { "feature": "out_amt_sum", "weight": 0.09 }] },
#|    "ACC0003271": { "account_id": "ACC0003271", "risk_score": 0.93, "threshold": 0.62, "reporting_threshold": 10000, "model": "gatv2", "method": "GNNExplainer", "nodes": [{ "account_id": "ACC0003271", "account_type": "business", "country": "AE" }, { "account_id": "ACC0005218", "account_type": "business", "country": "US" }, { "account_id": "ACC0005212", "account_type": "business", "country": "US" }, { "account_id": "ACC0005219", "account_type": "individual", "country": "US" }, { "account_id": "ACC0005211", "account_type": "individual", "country": "AE" }, { "account_id": "ACC0005209", "account_type": "business", "country": "FR" }, { "account_id": "ACC0005214", "account_type": "individual", "country": "CA" }, { "account_id": "ACC0005215", "account_type": "individual", "country": "IN" }, { "account_id": "ACC0005217", "account_type": "business", "country": "US" }, { "account_id": "ACC0005213", "account_type": "business", "country": "US" }, { "account_id": "ACC0005216", "account_type": "business", "country": "NG" }, { "account_id": "ACC0005208", "account_type": "business", "country": "US" }, { "account_id": "ACC0005210", "account_type": "business", "country": "AE" }], "edges": [{ "tx_id": "TX000003945", "src": "ACC0003271", "dst": "ACC0005218", "amount": 48495.35, "timestamp": 1767480177, "cross_border": 1, "importance": 0.57809 }, { "tx_id": "TX000003977", "src": "ACC0003271", "dst": "ACC0005212", "amount": 54295.48, "timestamp": 1767482199, "cross_border": 1, "importance": 0.85652 }, { "tx_id": "TX000004070", "src": "ACC0003271", "dst": "ACC0005219", "amount": 30337.61, "timestamp": 1767488286, "cross_border": 1, "importance": 0.71016 }, { "tx_id": "TX000004099", "src": "ACC0003271", "dst": "ACC0005211", "amount": 82803.15, "timestamp": 1767490082, "cross_border": 0, "importance": 0.88863 }, { "tx_id": "TX000004103", "src": "ACC0003271", "dst": "ACC0005209", "amount": 50731.07, "timestamp": 1767490542, "cross_border": 1, "importance": 0.70461 }, { "tx_id": "TX000004214", "src": "ACC0003271", "dst": "ACC0005214", "amount": 1834.39, "timestamp": 1767498393, "cross_border": 1, "importance": 0.93322 }, { "tx_id": "TX000004216", "src": "ACC0003271", "dst": "ACC0005215", "amount": 57615.79, "timestamp": 1767498510, "cross_border": 1, "importance": 0.88892 }, { "tx_id": "TX000004221", "src": "ACC0003271", "dst": "ACC0005217", "amount": 11.26, "timestamp": 1767498693, "cross_border": 1, "importance": 0.55022 }, { "tx_id": "TX000004255", "src": "ACC0003271", "dst": "ACC0005213", "amount": 2052.36, "timestamp": 1767500829, "cross_border": 1, "importance": 0.63389 }, { "tx_id": "TX000004271", "src": "ACC0003271", "dst": "ACC0005216", "amount": 22207.16, "timestamp": 1767501857, "cross_border": 1, "importance": 0.91411 }, { "tx_id": "TX000004312", "src": "ACC0005218", "dst": "ACC0005208", "amount": 47313.63, "timestamp": 1767504065, "cross_border": 0, "importance": 0.2986 }, { "tx_id": "TX000004434", "src": "ACC0003271", "dst": "ACC0005210", "amount": 39873.45, "timestamp": 1767512052, "cross_border": 0, "importance": 0.94214 }, { "tx_id": "TX000004629", "src": "ACC0005217", "dst": "ACC0005208", "amount": 11.16, "timestamp": 1767524551, "cross_border": 0, "importance": 0.27102 }, { "tx_id": "TX000004962", "src": "ACC0005210", "dst": "ACC0005208", "amount": 39574.94, "timestamp": 1767545578, "cross_border": 1, "importance": 0.14775 }, { "tx_id": "TX000005108", "src": "ACC0005219", "dst": "ACC0005208", "amount": 30060.4, "timestamp": 1767554651, "cross_border": 0, "importance": 0.35919 }, { "tx_id": "TX000005234", "src": "ACC0005212", "dst": "ACC0005208", "amount": 53712.63, "timestamp": 1767562515, "cross_border": 0, "importance": 0.41583 }, { "tx_id": "TX000005265", "src": "ACC0005213", "dst": "ACC0005208", "amount": 1993.67, "timestamp": 1767564310, "cross_border": 0, "importance": 0.22251 }, { "tx_id": "TX000005279", "src": "ACC0005214", "dst": "ACC0005208", "amount": 1803.59, "timestamp": 1767565459, "cross_border": 1, "importance": 0.15311 }, { "tx_id": "TX000005371", "src": "ACC0005215", "dst": "ACC0005208", "amount": 56100.39, "timestamp": 1767570369, "cross_border": 1, "importance": 0.24638 }, { "tx_id": "TX000005578", "src": "ACC0005211", "dst": "ACC0005208", "amount": 81876.18, "timestamp": 1767583924, "cross_border": 1, "importance": 0.48635 }, { "tx_id": "TX000005757", "src": "ACC0005216", "dst": "ACC0005208", "amount": 21851.32, "timestamp": 1767594746, "cross_border": 1, "importance": 0.40806 }, { "tx_id": "TX000005768", "src": "ACC0005209", "dst": "ACC0005208", "amount": 49272.75, "timestamp": 1767595122, "cross_border": 1, "importance": 0.16484 }], "top_features": [{ "feature": "out_uniq", "weight": 0.29 }, { "feature": "flow_ratio", "weight": 0.2 }, { "feature": "retained_frac", "weight": 0.16 }, { "feature": "in_uniq", "weight": 0.1 }, { "feature": "out_amt_sum", "weight": 0.08 }] },
#|    "ACC0005473": { "account_id": "ACC0005473", "risk_score": 0.88, "threshold": 0.62, "reporting_threshold": 10000, "model": "gatv2", "method": "GNNExplainer", "nodes": [{ "account_id": "ACC0005473", "account_type": "individual", "country": "IN" }, { "account_id": "ACC0002541", "account_type": "business", "country": "DE" }, { "account_id": "ACC0001809", "account_type": "individual", "country": "CA" }, { "account_id": "ACC0003808", "account_type": "business", "country": "ZA" }, { "account_id": "ACC0004402", "account_type": "individual", "country": "JP" }, { "account_id": "ACC0003165", "account_type": "individual", "country": "US" }, { "account_id": "ACC0000997", "account_type": "individual", "country": "GB" }, { "account_id": "ACC0000684", "account_type": "individual", "country": "BR" }, { "account_id": "ACC0003376", "account_type": "business", "country": "GB" }, { "account_id": "ACC0001724", "account_type": "individual", "country": "US" }, { "account_id": "ACC0000935", "account_type": "individual", "country": "GB" }, { "account_id": "ACC0004078", "account_type": "individual", "country": "CA" }, { "account_id": "ACC0000613", "account_type": "individual", "country": "US" }, { "account_id": "ACC0001181", "account_type": "individual", "country": "GB" }, { "account_id": "ACC0001600", "account_type": "individual", "country": "US" }, { "account_id": "ACC0003326", "account_type": "business", "country": "US" }, { "account_id": "ACC0002048", "account_type": "individual", "country": "NG" }, { "account_id": "ACC0003861", "account_type": "individual", "country": "JP" }, { "account_id": "ACC0000864", "account_type": "individual", "country": "US" }, { "account_id": "ACC0002464", "account_type": "individual", "country": "ZA" }, { "account_id": "ACC0001986", "account_type": "individual", "country": "GB" }, { "account_id": "ACC0004445", "account_type": "individual", "country": "US" }, { "account_id": "ACC0001045", "account_type": "individual", "country": "BR" }, { "account_id": "ACC0001517", "account_type": "individual", "country": "US" }, { "account_id": "ACC0001679", "account_type": "individual", "country": "JP" }, { "account_id": "ACC0004858", "account_type": "individual", "country": "US" }, { "account_id": "ACC0002258", "account_type": "individual", "country": "GB" }, { "account_id": "ACC0001539", "account_type": "individual", "country": "DE" }, { "account_id": "ACC0002329", "account_type": "business", "country": "US" }, { "account_id": "ACC0000929", "account_type": "individual", "country": "CA" }, { "account_id": "ACC0001379", "account_type": "individual", "country": "ZA" }, { "account_id": "ACC0002718", "account_type": "individual", "country": "AE" }, { "account_id": "ACC0001762", "account_type": "individual", "country": "KE" }], "edges": [{ "tx_id": "TX000008371", "src": "ACC0005473", "dst": "ACC0002541", "amount": 2109.37, "timestamp": 1767760385, "cross_border": 1, "importance": 0.64856 }, { "tx_id": "TX000008383", "src": "ACC0005473", "dst": "ACC0001809", "amount": 4420.42, "timestamp": 1767761079, "cross_border": 1, "importance": 0.59042 }, { "tx_id": "TX000008389", "src": "ACC0003808", "dst": "ACC0005473", "amount": 2716.85, "timestamp": 1767761541, "cross_border": 1, "importance": 0.57396 }, { "tx_id": "TX000008408", "src": "ACC0005473", "dst": "ACC0004402", "amount": 7495.99, "timestamp": 1767762580, "cross_border": 1, "importance": 0.86881 }, { "tx_id": "TX000008438", "src": "ACC0003165", "dst": "ACC0005473", "amount": 3692.16, "timestamp": 1767764052, "cross_border": 1, "importance": 0.62107 }, { "tx_id": "TX000008464", "src": "ACC0005473", "dst": "ACC0000997", "amount": 5408.19, "timestamp": 1767765338, "cross_border": 1, "importance": 0.77372 }, { "tx_id": "TX000008479", "src": "ACC0000684", "dst": "ACC0005473", "amount": 4058.83, "timestamp": 1767766057, "cross_border": 1, "importance": 0.72897 }, { "tx_id": "TX000008489", "src": "ACC0005473", "dst": "ACC0003376", "amount": 6973.2, "timestamp": 1767766744, "cross_border": 1, "importance": 0.62627 }, { "tx_id": "TX000008492", "src": "ACC0001724", "dst": "ACC0005473", "amount": 5724.02, "timestamp": 1767767013, "cross_border": 1, "importance": 0.84276 }, { "tx_id": "TX000008524", "src": "ACC0000935", "dst": "ACC0005473", "amount": 5319.81, "timestamp": 1767768497, "cross_border": 1, "importance": 0.60239 }, { "tx_id": "TX000008531", "src": "ACC0004078", "dst": "ACC0005473", "amount": 9343.06, "timestamp": 1767768854, "cross_border": 1, "importance": 0.80749 }, { "tx_id": "TX000008551", "src": "ACC0000613", "dst": "ACC0005473", "amount": 3027.92, "timestamp": 1767770045, "cross_border": 1, "importance": 0.5966 }, { "tx_id": "TX000008552", "src": "ACC0005473", "dst": "ACC0001181", "amount": 3945.88, "timestamp": 1767770057, "cross_border": 1, "importance": 0.7183 }, { "tx_id": "TX000008558", "src": "ACC0001600", "dst": "ACC0005473", "amount": 3587.28, "timestamp": 1767770512, "cross_border": 1, "importance": 0.63515 }, { "tx_id": "TX000008559", "src": "ACC0005473", "dst": "ACC0003326", "amount": 8430.75, "timestamp": 1767770523, "cross_border": 1, "importance": 0.65792 }, { "tx_id": "TX000008564", "src": "ACC0002048", "dst": "ACC0005473", "amount": 3747.78, "timestamp": 1767770688, "cross_border": 1, "importance": 0.93837 }, { "tx_id": "TX000008567", "src": "ACC0005473", "dst": "ACC0003861", "amount": 2576.04, "timestamp": 1767770884, "cross_border": 1, "importance": 0.87136 }, { "tx_id": "TX000008571", "src": "ACC0005473", "dst": "ACC0000864", "amount": 3189.61, "timestamp": 1767771211, "cross_border": 1, "importance": 0.67166 }, { "tx_id": "TX000008576", "src": "ACC0002464", "dst": "ACC0005473", "amount": 5301.43, "timestamp": 1767771476, "cross_border": 1, "importance": 0.90395 }, { "tx_id": "TX000008577", "src": "ACC0001986", "dst": "ACC0005473", "amount": 4895.67, "timestamp": 1767771490, "cross_border": 1, "importance": 0.63428 }, { "tx_id": "TX000008609", "src": "ACC0005473", "dst": "ACC0004445", "amount": 6728.51, "timestamp": 1767772901, "cross_border": 1, "importance": 0.70771 }, { "tx_id": "TX000008611", "src": "ACC0001045", "dst": "ACC0005473", "amount": 3635.64, "timestamp": 1767772959, "cross_border": 1, "importance": 0.89175 }, { "tx_id": "TX000008625", "src": "ACC0001517", "dst": "ACC0005473", "amount": 5391.47, "timestamp": 1767773867, "cross_border": 1, "importance": 0.80673 }, { "tx_id": "TX000008650", "src": "ACC0005473", "dst": "ACC0001679", "amount": 5779.57, "timestamp": 1767775708, "cross_border": 1, "importance": 0.59013 }, { "tx_id": "TX000008655", "src": "ACC0005473", "dst": "ACC0004858", "amount": 7549.63, "timestamp": 1767775992, "cross_border": 1, "importance": 0.94572 }, { "tx_id": "TX000008678", "src": "ACC0002258", "dst": "ACC0005473", "amount": 1060.81, "timestamp": 1767777498, "cross_border": 1, "importance": 0.6353 }, { "tx_id": "TX000008689", "src": "ACC0005473", "dst": "ACC0001539", "amount": 1301.13, "timestamp": 1767778245, "cross_border": 1, "importance": 0.65331 }, { "tx_id": "TX000008692", "src": "ACC0005473", "dst": "ACC0002329", "amount": 4024.56, "timestamp": 1767778346, "cross_border": 1, "importance": 0.85908 }, { "tx_id": "TX000008707", "src": "ACC0005473", "dst": "ACC0000929", "amount": 4863.41, "timestamp": 1767779553, "cross_border": 1, "importance": 0.68158 }, { "tx_id": "TX000008720", "src": "ACC0001379", "dst": "ACC0005473", "amount": 6754.46, "timestamp": 1767780479, "cross_border": 1, "importance": 0.66853 }, { "tx_id": "TX000008738", "src": "ACC0002718", "dst": "ACC0005473", "amount": 7812.01, "timestamp": 1767781164, "cross_border": 1, "importance": 0.57936 }, { "tx_id": "TX000008740", "src": "ACC0005473", "dst": "ACC0001762", "amount": 6000.96, "timestamp": 1767781360, "cross_border": 1, "importance": 0.58605 }], "top_features": [{ "feature": "burst_6h", "weight": 0.3 }, { "feature": "n_cp_countries", "weight": 0.24 }, { "feature": "xb_in_ratio", "weight": 0.16 }, { "feature": "xb_out_ratio", "weight": 0.12 }, { "feature": "in_deg", "weight": 0.07 }] },
#|    "ACC0002825": { "account_id": "ACC0002825", "risk_score": 0.04, "threshold": 0.62, "reporting_threshold": 10000, "model": "gatv2", "method": "GNNExplainer", "nodes": [{ "account_id": "ACC0002825", "account_type": "individual", "country": "IN" }, { "account_id": "ACC0004648", "account_type": "business", "country": "IN" }, { "account_id": "ACC0001936", "account_type": "business", "country": "IN" }, { "account_id": "ACC0003589", "account_type": "individual", "country": "IN" }, { "account_id": "ACC0000200", "account_type": "individual", "country": "IN" }, { "account_id": "ACC0002494", "account_type": "individual", "country": "IN" }, { "account_id": "ACC0003900", "account_type": "individual", "country": "IN" }, { "account_id": "ACC0003609", "account_type": "individual", "country": "KE" }, { "account_id": "ACC0003059", "account_type": "individual", "country": "IN" }, { "account_id": "ACC0003525", "account_type": "individual", "country": "IN" }], "edges": [{ "tx_id": "TX000000026", "src": "ACC0002825", "dst": "ACC0004648", "amount": 117.26, "timestamp": 1767227880, "cross_border": 0, "importance": 0.16655 }, { "tx_id": "TX000004197", "src": "ACC0001936", "dst": "ACC0002825", "amount": 20.17, "timestamp": 1767497395, "cross_border": 0, "importance": 0.0986 }, { "tx_id": "TX000010238", "src": "ACC0003589", "dst": "ACC0002825", "amount": 2032.54, "timestamp": 1767876696, "cross_border": 0, "importance": 0.17026 }, { "tx_id": "TX000024404", "src": "ACC0002825", "dst": "ACC0000200", "amount": 154.27, "timestamp": 1768781859, "cross_border": 0, "importance": 0.12434 }, { "tx_id": "TX000024434", "src": "ACC0002494", "dst": "ACC0002825", "amount": 257.12, "timestamp": 1768783062, "cross_border": 0, "importance": 0.14064 }, { "tx_id": "TX000028586", "src": "ACC0002825", "dst": "ACC0003900", "amount": 188.96, "timestamp": 1769050881, "cross_border": 0, "importance": 0.24183 }, { "tx_id": "TX000028904", "src": "ACC0003609", "dst": "ACC0002825", "amount": 102.28, "timestamp": 1769073156, "cross_border": 1, "importance": 0.14674 }, { "tx_id": "TX000037327", "src": "ACC0002825", "dst": "ACC0003059", "amount": 436.02, "timestamp": 1769603960, "cross_border": 0, "importance": 0.16491 }, { "tx_id": "TX000040372", "src": "ACC0003525", "dst": "ACC0002825", "amount": 108.03, "timestamp": 1769802950, "cross_border": 0, "importance": 0.22331 }], "top_features": [{ "feature": "in_deg", "weight": 0.04 }, { "feature": "out_amt_mean", "weight": 0.03 }, { "feature": "age_days", "weight": 0.03 }] },
#|    "ACC0000819": { "account_id": "ACC0000819", "risk_score": 0.07, "threshold": 0.62, "reporting_threshold": 10000, "model": "gatv2", "method": "GNNExplainer", "nodes": [{ "account_id": "ACC0000819", "account_type": "business", "country": "US" }, { "account_id": "ACC0001988", "account_type": "individual", "country": "US" }, { "account_id": "ACC0004466", "account_type": "individual", "country": "US" }, { "account_id": "ACC0000831", "account_type": "individual", "country": "JP" }, { "account_id": "ACC0001879", "account_type": "individual", "country": "US" }, { "account_id": "ACC0001202", "account_type": "business", "country": "US" }, { "account_id": "ACC0002204", "account_type": "individual", "country": "US" }, { "account_id": "ACC0002194", "account_type": "individual", "country": "US" }, { "account_id": "ACC0003658", "account_type": "individual", "country": "US" }, { "account_id": "ACC0000129", "account_type": "individual", "country": "US" }], "edges": [{ "tx_id": "TX000000036", "src": "ACC0000819", "dst": "ACC0001988", "amount": 469.02, "timestamp": 1767228443, "cross_border": 0, "importance": 0.08657 }, { "tx_id": "TX000001079", "src": "ACC0004466", "dst": "ACC0000819", "amount": 4021.45, "timestamp": 1767298455, "cross_border": 0, "importance": 0.08083 }, { "tx_id": "TX000002564", "src": "ACC0000831", "dst": "ACC0000819", "amount": 503.17, "timestamp": 1767393810, "cross_border": 1, "importance": 0.23168 }, { "tx_id": "TX000011129", "src": "ACC0000819", "dst": "ACC0001879", "amount": 300.19, "timestamp": 1767931660, "cross_border": 0, "importance": 0.21356 }, { "tx_id": "TX000023073", "src": "ACC0001202", "dst": "ACC0000819", "amount": 526.39, "timestamp": 1768692869, "cross_border": 0, "importance": 0.0999 }, { "tx_id": "TX000026226", "src": "ACC0000819", "dst": "ACC0002204", "amount": 573.68, "timestamp": 1768896614, "cross_border": 0, "importance": 0.08796 }, { "tx_id": "TX000027159", "src": "ACC0000819", "dst": "ACC0002194", "amount": 158.27, "timestamp": 1768956137, "cross_border": 0, "importance": 0.19788 }, { "tx_id": "TX000028441", "src": "ACC0000819", "dst": "ACC0003658", "amount": 950.96, "timestamp": 1769041486, "cross_border": 0, "importance": 0.23808 }, { "tx_id": "TX000028710", "src": "ACC0000819", "dst": "ACC0000129", "amount": 763.3, "timestamp": 1769059609, "cross_border": 0, "importance": 0.08932 }], "top_features": [{ "feature": "in_deg", "weight": 0.04 }, { "feature": "out_amt_mean", "weight": 0.03 }, { "feature": "age_days", "weight": 0.03 }] },
#|    "ACC0003621": { "account_id": "ACC0003621", "risk_score": 0.11, "threshold": 0.62, "reporting_threshold": 10000, "model": "gatv2", "method": "GNNExplainer", "nodes": [{ "account_id": "ACC0003621", "account_type": "individual", "country": "FR" }, { "account_id": "ACC0004329", "account_type": "business", "country": "KE" }, { "account_id": "ACC0003429", "account_type": "individual", "country": "FR" }, { "account_id": "ACC0003386", "account_type": "individual", "country": "IN" }, { "account_id": "ACC0000959", "account_type": "individual", "country": "FR" }, { "account_id": "ACC0004693", "account_type": "individual", "country": "FR" }, { "account_id": "ACC0002898", "account_type": "individual", "country": "FR" }, { "account_id": "ACC0003385", "account_type": "individual", "country": "FR" }, { "account_id": "ACC0002118", "account_type": "individual", "country": "FR" }, { "account_id": "ACC0003685", "account_type": "individual", "country": "US" }], "edges": [{ "tx_id": "TX000000052", "src": "ACC0003621", "dst": "ACC0004329", "amount": 103.83, "timestamp": 1767229062, "cross_border": 1, "importance": 0.24003 }, { "tx_id": "TX000002052", "src": "ACC0003621", "dst": "ACC0003429", "amount": 41.45, "timestamp": 1767360534, "cross_border": 0, "importance": 0.22644 }, { "tx_id": "TX000005830", "src": "ACC0003386", "dst": "ACC0003621", "amount": 84.14, "timestamp": 1767599450, "cross_border": 1, "importance": 0.17071 }, { "tx_id": "TX000007202", "src": "ACC0000959", "dst": "ACC0003621", "amount": 1291.5, "timestamp": 1767686541, "cross_border": 0, "importance": 0.13429 }, { "tx_id": "TX000007700", "src": "ACC0003621", "dst": "ACC0004693", "amount": 1281.53, "timestamp": 1767717299, "cross_border": 0, "importance": 0.07077 }, { "tx_id": "TX000009766", "src": "ACC0003621", "dst": "ACC0002898", "amount": 24.28, "timestamp": 1767846682, "cross_border": 0, "importance": 0.05774 }, { "tx_id": "TX000015363", "src": "ACC0003621", "dst": "ACC0003385", "amount": 428.85, "timestamp": 1768201568, "cross_border": 0, "importance": 0.24254 }, { "tx_id": "TX000019043", "src": "ACC0002118", "dst": "ACC0003621", "amount": 1126.88, "timestamp": 1768435886, "cross_border": 0, "importance": 0.09768 }, { "tx_id": "TX000033543", "src": "ACC0003685", "dst": "ACC0003621", "amount": 221.22, "timestamp": 1769364217, "cross_border": 1, "importance": 0.19092 }], "top_features": [{ "feature": "in_deg", "weight": 0.04 }, { "feature": "out_amt_mean", "weight": 0.03 }, { "feature": "age_days", "weight": 0.03 }] }
#|};
#|exports.DEMO_NARRATIVES = {
#|    "ACC0001449": {
#|        "text": "Account ACC0001449 was flagged by the graph model with a risk score of 97% and is connected to 14 accounts in activity consistent with structuring (smurfing). The reviewed network contains 24 transactions totaling $207,975, occurring between 2026-01-02 and 2026-01-03 over about 31 hours. 23 of these transfers fell just below the $10,000 reporting threshold, which is consistent with structuring. Key model indicators include a high share of transfers just below the reporting threshold, an unusual burst of transactions within a short window and funds passing through with little balance retained, and the activity is referred for compliance officer review and potential Suspicious Activity Report filing.",
#|        "typology": "smurfing",
#|        "sha256": "db946f2404c25ba5b5a9f19bb03a7dc71e8e7547918e2fe2cab5f44a5a223039"
#|    },
#|    "ACC0000528": {
#|        "text": "Account ACC0000528 was flagged by the graph model with a risk score of 96% and is connected to 5 accounts in activity consistent with layering through shell entities. The reviewed network contains 4 transactions totaling $862,483, occurring between 2026-01-21 and 2026-01-22 over about 43 hours. 3 of the involved accounts are shell entities through which funds were passed in a layered chain. Key model indicators include funds passing through with little balance retained, an elevated share of cross-border transfers and shell or offshore account characteristics, and the activity is referred for compliance officer review and potential Suspicious Activity Report filing.",
#|        "typology": "shell_company",
#|        "sha256": "0ae0e004303a6bd43c967ea909f466a42b28ff1ff613aeea589968f6551c9be5"
#|    },
#|    "ACC0000283": {
#|        "text": "Account ACC0000283 was flagged by the graph model with a risk score of 91% and is connected to 5 accounts in activity consistent with circular fund flow. The reviewed network contains 15 transactions totaling $2,130,312, occurring between 2026-01-10 and 2026-01-15 over about 121 hours. Funds returned to the originating accounts through a closed loop, consistent with circular movement. Key model indicators include funds passing through with little balance retained, unusual counterparty fan-in and unusual counterparty fan-out, and the activity is referred for compliance officer review and potential Suspicious Activity Report filing.",
#|        "typology": "cyclic_loop",
#|        "sha256": "380c61695a209c6fae1ccda8e0a0ef1d9502202de63d2827a3dd189cffc25120"
#|    },
#|    "ACC0003271": {
#|        "text": "Account ACC0003271 was flagged by the graph model with a risk score of 93% and is connected to 13 accounts in activity consistent with scatter-gather layering. The reviewed network contains 22 transactions totaling $773,828, occurring between 2026-01-03 and 2026-01-05 over about 32 hours. Funds fanned out from a single source to several intermediaries and were then consolidated into one destination account. Key model indicators include unusual counterparty fan-out, funds passing through with little balance retained and unusual counterparty fan-in, and the activity is referred for compliance officer review and potential Suspicious Activity Report filing.",
#|        "typology": "scatter_gather",
#|        "sha256": "9ee1bdd9bfc5881f5ef52e2f8f785e1eabf48f436b11f6b22098b2b8e289aa60"
#|    },
#|    "ACC0005473": {
#|        "text": "Account ACC0005473 was flagged by the graph model with a risk score of 88% and is connected to 33 accounts in activity consistent with high-velocity cross-border activity. The reviewed network contains 32 transactions totaling $156,866, occurring on 2026-01-07 over about 6 hours. The activity spans 11 jurisdictions (AE, BR, CA, DE, GB, IN, JP, KE, NG, US, ZA) within a short window, indicating rapid cross-border movement of funds. Key model indicators include an unusual burst of transactions within a short window, counterparties spread across many jurisdictions and an elevated share of cross-border transfers, and the activity is referred for compliance officer review and potential Suspicious Activity Report filing.",
#|        "typology": "cross_border_velocity",
#|        "sha256": "ecc24a47faf1c8a6f2cf3227f817d09cd29fd74125521291996d5130ddf86578"
#|    },
#|    "ACC0002825": {
#|        "text": "Account ACC0002825 was flagged by the graph model with a risk score of 4% and is connected to 10 accounts in activity consistent with an unclassified anomalous pattern. The reviewed network contains 9 transactions totaling $3,417, occurring between 2026-01-01 and 2026-01-30 over about 715 hours. 1 of the 9 transactions were cross-border and the pattern deviates from expected account behavior. Key model indicators include unusual counterparty fan-in and a recently opened account, and the activity is referred for compliance officer review and potential Suspicious Activity Report filing.",
#|        "typology": "unclassified",
#|        "sha256": "be19a0ae5b2c2039ae1f1bcc7378023c6c65087c62b7c54c4f95aea496775838"
#|    },
#|    "ACC0000819": {
#|        "text": "Account ACC0000819 was flagged by the graph model with a risk score of 7% and is connected to 10 accounts in activity consistent with an unclassified anomalous pattern. The reviewed network contains 9 transactions totaling $8,266, occurring between 2026-01-01 and 2026-01-22 over about 509 hours. 1 of the 9 transactions were cross-border and the pattern deviates from expected account behavior. Key model indicators include unusual counterparty fan-in and a recently opened account, and the activity is referred for compliance officer review and potential Suspicious Activity Report filing.",
#|        "typology": "unclassified",
#|        "sha256": "3bf34cc8f0a895d5cff298b051b17c8e80c57abd4f1f90b4232a89e8d66c8d8c"
#|    },
#|    "ACC0003621": {
#|        "text": "Account ACC0003621 was flagged by the graph model with a risk score of 11% and is connected to 10 accounts in activity consistent with an unclassified anomalous pattern. The reviewed network contains 9 transactions totaling $4,604, occurring between 2026-01-01 and 2026-01-25 over about 593 hours. 3 of the 9 transactions were cross-border and the pattern deviates from expected account behavior. Key model indicators include unusual counterparty fan-in and a recently opened account, and the activity is referred for compliance officer review and potential Suspicious Activity Report filing.",
#|        "typology": "unclassified",
#|        "sha256": "43f04087f8fdf1e67600bea18770d092499a8509ed46909a848f15b7f46c4b0e"
#|    }
#|};
#|
#|});
#|__d("demo/entry", function (exports, require, module) {
#|"use strict";
#|var __importDefault = (this && this.__importDefault) || function (mod) {
#|    return (mod && mod.__esModule) ? mod : { "default": mod };
#|};
#|Object.defineProperty(exports, "__esModule", { value: true });
#|/** Entry point of the offline demo bundle: mock backend + the real page component. */
#|const client_1 = require("react-dom/client");
#|const page_1 = __importDefault(require("@/app/page"));
#|const mockApi_1 = require("./mockApi");
#|(0, mockApi_1.installMockApi)();
#|const el = document.getElementById("root");
#|if (el)
#|    (0, client_1.createRoot)(el).render(React.createElement(page_1.default, null));
#|
#|});
#|__d("demo/mockApi", function (exports, require, module) {
#|"use strict";
#|Object.defineProperty(exports, "__esModule", { value: true });
#|exports.installMockApi = installMockApi;
#|const demoData_1 = require("./demoData");
#|const THRESHOLD = 0.62;
#|const T0 = Date.now();
#|const iso = (minutesAgo) => new Date(T0 - minutesAgo * 60000).toISOString();
#|const suspicious = demoData_1.DEMO_ACCOUNTS.filter((a) => a.label !== "benign");
#|const benign = demoData_1.DEMO_ACCOUNTS.filter((a) => a.label === "benign");
#|const byLabel = (label) => demoData_1.DEMO_ACCOUNTS.find((a) => a.label === label);
#|function makeAlert(id, accountId, status, minutesAgo, withNarrative, reviewer = null) {
#|    var _a, _b, _c, _d;
#|    const account = demoData_1.DEMO_ACCOUNTS.find((a) => a.id === accountId);
#|    const narrative = withNarrative ? demoData_1.DEMO_NARRATIVES[accountId] : undefined;
#|    return {
#|        id,
#|        account_id: accountId,
#|        risk_score: (_a = account === null || account === void 0 ? void 0 : account.score) !== null && _a !== void 0 ? _a : 0.9,
#|        status,
#|        typology: (_b = narrative === null || narrative === void 0 ? void 0 : narrative.typology) !== null && _b !== void 0 ? _b : null,
#|        narrative: (_c = narrative === null || narrative === void 0 ? void 0 : narrative.text) !== null && _c !== void 0 ? _c : null,
#|        narrative_source: narrative ? "template" : null,
#|        narrative_sha256: (_d = narrative === null || narrative === void 0 ? void 0 : narrative.sha256) !== null && _d !== void 0 ? _d : null,
#|        reviewed_by: reviewer,
#|        review_note: reviewer ? "" : null,
#|        created_at: iso(minutesAgo),
#|        reviewed_at: reviewer ? iso(Math.max(minutesAgo - 20, 1)) : null,
#|    };
#|}
#|const alerts = [];
#|const seed = (label, status, mins, narr, who = null) => {
#|    const a = byLabel(label);
#|    if (a)
#|        alerts.push(makeAlert(alerts.length + 1, a.id, status, mins, narr, who));
#|};
#|seed("smurfing", "open", 42, true);
#|seed("shell_company", "open", 95, false);
#|seed("cyclic_loop", "confirmed", 610, true, "M. Wanjiru");
#|seed("cross_border_velocity", "dismissed", 1300, true, "M. Wanjiru");
#|let nextId = alerts.length + 1;
#|const MODEL_INFO = {
#|    model: "gatv2",
#|    hparams: { hidden: 64, num_layers: 2 },
#|    threshold: THRESHOLD,
#|    metrics: {
#|        val_auc_roc: 0.981, test_auc_roc: 0.983, test_pr_auc: 0.94, test_precision: 0.91, test_recall: 0.94,
#|        threshold: THRESHOLD, auc_target: 0.87, meets_auc_target: true,
#|    },
#|    targets: { auc_roc: 0.87, latency_ms: 100 },
#|    n_nodes: 5488,
#|    n_features: 26,
#|};
#|const ok = (body, delay = 220) => ({ status: 200, body, delay });
#|const fail = (status, detail, delay = 220) => ({ status, body: { detail }, delay });
#|function route(method, path, data) {
#|    var _a, _b, _c;
#|    if (method === "GET" && path === "/alerts") {
#|        return ok([...alerts].sort((a, b) => b.created_at.localeCompare(a.created_at)));
#|    }
#|    if (method === "GET" && path === "/accounts/sample") {
#|        const body = { suspicious: suspicious.map((a) => a.id), benign: benign.map((a) => a.id) };
#|        return ok(body, 120);
#|    }
#|    if (method === "GET" && path === "/model/info")
#|        return ok(MODEL_INFO, 150);
#|    if (method === "POST" && path === "/alerts/scan") {
#|        const id = String((_a = data.account_id) !== null && _a !== void 0 ? _a : "").trim();
#|        const account = demoData_1.DEMO_ACCOUNTS.find((a) => a.id === id);
#|        if (!account) {
#|            return fail(404, `unknown account_id '${id}'. In demo mode try one of the "Try:" accounts.`);
#|        }
#|        const flagged = account.score >= THRESHOLD;
#|        let alert = null;
#|        if (flagged) {
#|            alert = makeAlert(nextId++, account.id, "open", 0, false);
#|            alerts.push(alert);
#|        }
#|        const body = {
#|            score: account.score,
#|            flagged,
#|            threshold: THRESHOLD,
#|            latency_ms: Math.round((6 + Math.random() * 12) * 10) / 10,
#|            alert,
#|        };
#|        return ok(body, 260);
#|    }
#|    let m = path.match(/^\/accounts\/([^/]+)\/explain$/);
#|    if (method === "GET" && m) {
#|        const id = decodeURIComponent(m[1]);
#|        const explanation = demoData_1.DEMO_EXPLANATIONS[id];
#|        return explanation ? ok(explanation, 500) : fail(404, `unknown account_id '${id}'`);
#|    }
#|    m = path.match(/^\/alerts\/(\d+)\/narrative$/);
#|    if (method === "POST" && m) {
#|        const alert = alerts.find((a) => a.id === Number(m === null || m === void 0 ? void 0 : m[1]));
#|        if (!alert)
#|            return fail(404, "alert not found");
#|        const narrative = demoData_1.DEMO_NARRATIVES[alert.account_id];
#|        if (!narrative)
#|            return fail(502, "no narrative available for this account in demo mode");
#|        alert.narrative = narrative.text;
#|        alert.narrative_source = "template"; // the demo has no GPU/LLM, so this is the validated template path
#|        alert.narrative_sha256 = narrative.sha256;
#|        alert.typology = narrative.typology;
#|        return ok(alert, 1300); // pretend the LLM takes a moment
#|    }
#|    m = path.match(/^\/alerts\/(\d+)\/decision$/);
#|    if (method === "POST" && m) {
#|        const alert = alerts.find((a) => a.id === Number(m === null || m === void 0 ? void 0 : m[1]));
#|        if (!alert)
#|            return fail(404, "alert not found");
#|        if (alert.status !== "open")
#|            return fail(409, `alert already ${alert.status}`);
#|        const officer = String((_b = data.officer) !== null && _b !== void 0 ? _b : "").trim();
#|        const decision = data.decision;
#|        if (officer.length < 2)
#|            return fail(422, "officer name is required");
#|        if (decision !== "confirmed" && decision !== "dismissed")
#|            return fail(422, "invalid decision");
#|        alert.status = decision;
#|        alert.reviewed_by = officer;
#|        alert.review_note = String((_c = data.note) !== null && _c !== void 0 ? _c : "");
#|        alert.reviewed_at = new Date().toISOString();
#|        return ok(alert, 450);
#|    }
#|    return fail(404, `Not found: ${method} ${path}`, 80);
#|}
#|function installMockApi() {
#|    const realFetch = window.fetch.bind(window);
#|    window.fetch = async (input, init) => {
#|        var _a;
#|        const url = typeof input === "string" ? input : input instanceof URL ? input.pathname : input.url;
#|        if (!url.startsWith("/api/"))
#|            return realFetch(input, init);
#|        let data = {};
#|        if (typeof (init === null || init === void 0 ? void 0 : init.body) === "string") {
#|            try {
#|                data = JSON.parse(init.body);
#|            }
#|            catch {
#|                data = {};
#|            }
#|        }
#|        const reply = route(((_a = init === null || init === void 0 ? void 0 : init.method) !== null && _a !== void 0 ? _a : "GET").toUpperCase(), url.slice(4), data);
#|        await new Promise((resolve) => setTimeout(resolve, reply.delay));
#|        return new Response(JSON.stringify(reply.body), {
#|            status: reply.status,
#|            headers: { "Content-Type": "application/json" },
#|        });
#|    };
#|}
#|
#|});
#|__r("demo/entry");
#|
#|}
#|</script>
#|</body>
#|</html>
#|@@@@END
#|@@@@FILE: src/app/globals.css
#|/* ============================ XAI-AMLBench  ·  deep-blue glass theme ============================ */
#|:root {
#|  --bg0: #030712; --bg1: #061029; --bg2: #0a1a44;
#|  --panel: rgba(12, 26, 60, .58); --panel-hi: rgba(22, 42, 92, .62);
#|  --line: rgba(96, 165, 250, .18); --line-hi: rgba(96, 165, 250, .42);
#|  --text: #e8f0ff; --muted: #8ea3cc; --dim: #5f7299;
#|  --blue: #3b82f6; --sky: #60a5fa; --cyan: #22d3ee; --violet: #818cf8;
#|  --red: #fb7185; --green: #34d399; --amber: #fbbf24;
#|  --radius: 18px;
#|  --ease: cubic-bezier(.22, 1, .36, 1);
#|  --spring: cubic-bezier(.34, 1.56, .64, 1);
#|}
#|* { box-sizing: border-box; }
#|html, body { min-height: 100%; }
#|body {
#|  margin: 0; color: var(--text); overflow-x: hidden;
#|  font: 15px/1.55 Inter, ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
#|  -webkit-font-smoothing: antialiased;
#|  background:
#|    radial-gradient(1100px 700px at 12% -8%, rgba(37, 99, 235, .28), transparent 60%),
#|    radial-gradient(900px 600px at 105% 8%, rgba(99, 102, 241, .22), transparent 58%),
#|    linear-gradient(180deg, var(--bg1), var(--bg0) 70%);
#|  background-attachment: fixed;
#|}
#|h1, h2, h3, p { margin: 0; }
#|h2 { font-size: 22px; margin-bottom: 4px; }
#|h3 { font-size: 15px; font-weight: 650; letter-spacing: .02em; margin-bottom: 14px; color: #cfe0ff; }
#|button { font: inherit; }
#|::selection { background: rgba(59, 130, 246, .45); }
#|::-webkit-scrollbar { width: 10px; height: 10px; }
#|::-webkit-scrollbar-thumb { background: rgba(96, 165, 250, .25); border-radius: 8px; }
#|
#|/* ------------------------------------------------------------------ backdrop */
#|.bg { position: fixed; inset: 0; z-index: 0; pointer-events: none; overflow: hidden; }
#|.bg-canvas { position: absolute; inset: 0; }
#|.orbs i { position: absolute; border-radius: 50%; filter: blur(80px); opacity: .55; }
#|.orbs i:nth-child(1) { width: 520px; height: 520px; left: -140px; top: 8%; background: #1d4ed8; animation: drift1 22s ease-in-out infinite; }
#|.orbs i:nth-child(2) { width: 460px; height: 460px; right: -120px; top: 30%; background: #4f46e5; animation: drift2 26s ease-in-out infinite; }
#|.orbs i:nth-child(3) { width: 420px; height: 420px; left: 35%; bottom: -160px; background: #0891b2; opacity: .35; animation: drift3 30s ease-in-out infinite; }
#|.grid-floor {
#|  position: absolute; inset: 0; opacity: .8;
#|  background-image:
#|    linear-gradient(rgba(96, 165, 250, .07) 1px, transparent 1px),
#|    linear-gradient(90deg, rgba(96, 165, 250, .07) 1px, transparent 1px);
#|  background-size: 52px 52px;
#|  -webkit-mask-image: radial-gradient(ellipse at 50% 40%, #000 25%, transparent 75%);
#|  mask-image: radial-gradient(ellipse at 50% 40%, #000 25%, transparent 75%);
#|  animation: gridMove 9s linear infinite;
#|}
#|@keyframes drift1 { 50% { transform: translate(140px, 90px) scale(1.15); } }
#|@keyframes drift2 { 50% { transform: translate(-160px, -70px) scale(1.1); } }
#|@keyframes drift3 { 50% { transform: translate(-120px, -110px) scale(1.2); } }
#|@keyframes gridMove { to { background-position: 52px 52px; } }
#|
#|/* --------------------------------------------------------------------- shell */
#|.app { position: relative; z-index: 1; max-width: 1120px; margin: 0 auto; padding: 22px 18px 150px; }
#|.topbar { display: flex; align-items: center; justify-content: space-between; margin-bottom: 26px; }
#|.brand { display: flex; align-items: center; gap: 12px; }
#|.logo {
#|  position: relative; width: 42px; height: 42px; border-radius: 13px; display: grid; place-items: center;
#|  background: linear-gradient(135deg, #2563eb, #06b6d4); box-shadow: 0 0 28px rgba(37, 99, 235, .6);
#|}
#|.logo-ring { position: absolute; inset: -4px; border-radius: 16px; border: 1.5px solid transparent;
#|  background: conic-gradient(from 0deg, transparent, rgba(34, 211, 238, .9), transparent 40%) border-box;
#|  -webkit-mask: linear-gradient(#000 0 0) padding-box, linear-gradient(#000 0 0);
#|  -webkit-mask-composite: xor; mask-composite: exclude; animation: spin 4s linear infinite; }
#|.brand-name { font-weight: 800; font-size: 18px; letter-spacing: .02em; }
#|.brand-name span { color: var(--cyan); margin: 0 1px; }
#|.brand-sub { font-size: 12px; color: var(--muted); letter-spacing: .12em; text-transform: uppercase; }
#|.status { display: flex; align-items: center; gap: 8px; font-size: 13px; color: var(--muted); padding: 7px 13px;
#|  border-radius: 999px; background: var(--panel); border: 1px solid var(--line); backdrop-filter: blur(10px); }
#|.status i { width: 8px; height: 8px; border-radius: 50%; background: var(--amber); }
#|.status.on i { background: var(--green); box-shadow: 0 0 0 0 rgba(52, 211, 153, .7); animation: ping 2s infinite; }
#|.status.off i { background: var(--red); }
#|@keyframes ping { 70% { box-shadow: 0 0 0 9px rgba(52, 211, 153, 0); } 100% { box-shadow: 0 0 0 0 rgba(52, 211, 153, 0); } }
#|@keyframes spin { to { transform: rotate(360deg); } }
#|
#|.view { animation: viewIn .55s var(--ease); display: flex; flex-direction: column; gap: 18px; }
#|@keyframes viewIn { from { opacity: 0; transform: translateY(16px) scale(.99); filter: blur(6px); } }
#|.rise { animation: rise .6s var(--ease) both; }
#|@keyframes rise { from { opacity: 0; transform: translateY(18px); } }
#|
#|/* --------------------------------------------------------------------- glass */
#|.glass {
#|  position: relative; overflow: hidden; padding: 20px; border-radius: var(--radius);
#|  background: linear-gradient(160deg, var(--panel-hi), var(--panel));
#|  border: 1px solid var(--line); backdrop-filter: blur(16px) saturate(140%); -webkit-backdrop-filter: blur(16px) saturate(140%);
#|  box-shadow: 0 10px 40px rgba(2, 6, 23, .5), inset 0 1px 0 rgba(255, 255, 255, .05);
#|  transition: border-color .3s, transform .3s var(--ease);
#|}
#|.glass::before {
#|  content: ""; position: absolute; inset: 0; border-radius: inherit; pointer-events: none; opacity: 0; transition: opacity .3s;
#|  background: radial-gradient(420px circle at var(--mx, 50%) var(--my, 0%), rgba(59, 130, 246, .16), transparent 60%);
#|}
#|.glass:hover { border-color: var(--line-hi); }
#|.glass:hover::before { opacity: 1; }
#|.row { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
#|.row.between { justify-content: space-between; }
#|.grow { flex: 1; min-width: 180px; }
#|.muted { color: var(--muted); } .small { font-size: 12.5px; }
#|.ok-text { color: var(--green); }
#|
#|/* ---------------------------------------------------------------------- hero */
#|.hero { padding: 34px 6px 10px; }
#|.eyebrow { display: inline-block; font-size: 12px; letter-spacing: .14em; text-transform: uppercase; color: var(--cyan);
#|  padding: 5px 12px; border: 1px solid rgba(34, 211, 238, .3); border-radius: 999px; background: rgba(34, 211, 238, .07); }
#|.hero h1 { font-size: clamp(34px, 6vw, 58px); line-height: 1.07; font-weight: 800; letter-spacing: -.02em; margin: 18px 0 14px; }
#|.grad-text {
#|  background: linear-gradient(90deg, #fff, #93c5fd, #22d3ee, #93c5fd, #fff); background-size: 300% 100%;
#|  -webkit-background-clip: text; background-clip: text; color: transparent; animation: shimmer 7s linear infinite;
#|}
#|@keyframes shimmer { to { background-position: -300% 0; } }
#|.lead { max-width: 640px; color: var(--muted); font-size: 17px; margin-bottom: 22px; }
#|
#|/* ------------------------------------------------------------------- buttons */
#|.btn {
#|  position: relative; overflow: hidden; border: 0; cursor: pointer; color: #fff; font-weight: 650; padding: 10px 18px; border-radius: 12px;
#|  background: linear-gradient(135deg, #2563eb, #0ea5e9); box-shadow: 0 6px 22px rgba(37, 99, 235, .45);
#|  transition: transform .25s var(--spring), box-shadow .25s, filter .25s;
#|}
#|.btn:hover:not(:disabled) { transform: translateY(-2px); box-shadow: 0 10px 30px rgba(37, 99, 235, .6); }
#|.btn:active:not(:disabled) { transform: translateY(0) scale(.97); }
#|.btn:disabled { opacity: .55; cursor: wait; }
#|.btn::after { content: ""; position: absolute; inset: 0; transform: translateX(-120%);
#|  background: linear-gradient(100deg, transparent 30%, rgba(255, 255, 255, .35), transparent 70%); }
#|.btn:hover::after, .btn.shine:not(:disabled)::after { transform: translateX(120%); transition: transform .8s var(--ease); }
#|.btn.shine:not(:disabled)::after { animation: sweep 3.2s ease-in-out infinite; }
#|@keyframes sweep { 0%, 60% { transform: translateX(-120%); } 100% { transform: translateX(120%); } }
#|.btn.ghost { background: rgba(59, 130, 246, .1); border: 1px solid var(--line-hi); color: #bcd3ff; box-shadow: none; }
#|.btn.ghost:hover:not(:disabled) { background: rgba(59, 130, 246, .2); box-shadow: 0 0 22px rgba(59, 130, 246, .3); }
#|.btn.red { background: linear-gradient(135deg, #e11d48, #fb7185); box-shadow: 0 6px 22px rgba(225, 29, 72, .4); }
#|.btn.green { background: linear-gradient(135deg, #059669, #34d399); color: #03251b; box-shadow: 0 6px 22px rgba(5, 150, 105, .4); }
#|.link { background: none; border: 0; color: var(--sky); cursor: pointer; }
#|.link:hover { color: var(--cyan); }
#|.input {
#|  background: rgba(3, 9, 30, .6); color: var(--text); border: 1px solid var(--line); border-radius: 12px; padding: 10px 14px; min-width: 200px; outline: none;
#|  transition: border-color .25s, box-shadow .25s;
#|}
#|.input::placeholder { color: var(--dim); }
#|.input:focus { border-color: var(--cyan); box-shadow: 0 0 0 4px rgba(34, 211, 238, .14), 0 0 26px rgba(34, 211, 238, .18); }
#|.chips { margin-top: 12px; }
#|.chip { background: rgba(59, 130, 246, .08); border: 1px solid var(--line); color: var(--muted); border-radius: 999px; padding: 3px 12px; font-size: 12.5px; cursor: pointer; transition: all .2s; }
#|.chip:hover { color: #fff; border-color: var(--line-hi); transform: translateY(-1px); }
#|.chip.bad { border-color: rgba(251, 113, 133, .35); color: #fda4af; }
#|.seg { display: inline-flex; background: rgba(3, 9, 30, .55); border: 1px solid var(--line); border-radius: 12px; padding: 4px; }
#|.seg button { background: none; border: 0; color: var(--muted); text-transform: capitalize; padding: 6px 14px; border-radius: 9px; cursor: pointer; transition: all .25s; }
#|.seg button.on { background: linear-gradient(135deg, #2563eb, #0ea5e9); color: #fff; box-shadow: 0 4px 14px rgba(37, 99, 235, .45); }
#|
#|/* --------------------------------------------------------------------- stats */
#|.stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 16px; }
#|.stat { padding: 18px 20px; }
#|.stat-label { color: var(--muted); font-size: 13px; letter-spacing: .05em; text-transform: uppercase; }
#|.stat-value { font-size: 40px; font-weight: 800; line-height: 1.15; margin: 6px 0 2px; font-variant-numeric: tabular-nums; }
#|.stat-hint { color: var(--dim); font-size: 12.5px; }
#|.stat-spark { position: absolute; right: -30px; bottom: -30px; width: 120px; height: 120px; border-radius: 50%; filter: blur(30px); opacity: .5; animation: breathe 4s ease-in-out infinite; }
#|.tone-blue .stat-value { color: #93c5fd; } .tone-blue .stat-spark { background: #2563eb; }
#|.tone-amber .stat-value { color: #fcd34d; } .tone-amber .stat-spark { background: #d97706; }
#|.tone-red .stat-value { color: #fda4af; } .tone-red .stat-spark { background: #e11d48; }
#|.tone-cyan .stat-value { color: #67e8f9; } .tone-cyan .stat-spark { background: #0891b2; }
#|@keyframes breathe { 50% { transform: scale(1.35); opacity: .3; } }
#|.two { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 16px; }
#|
#|/* ------------------------------------------------------------------ pipeline */
#|.pipe { display: flex; align-items: flex-start; overflow-x: auto; padding: 18px 16px 8px; margin: -6px -4px 0; }
#|.pipe-item { display: flex; flex-direction: column; align-items: center; gap: 8px; min-width: 88px; font-size: 12.5px; color: var(--muted); text-align: center; }
#|.pipe-node { width: 44px; height: 44px; border-radius: 14px; display: grid; place-items: center; color: #fff; font-size: 16px;
#|  background: linear-gradient(135deg, rgba(37, 99, 235, .8), rgba(6, 182, 212, .7)); box-shadow: 0 0 0 0 rgba(34, 211, 238, .5); animation: nodePulse 3s ease-in-out infinite; }
#|@keyframes nodePulse { 50% { box-shadow: 0 0 0 10px rgba(34, 211, 238, 0); transform: translateY(-3px); } }
#|.pipe-link { flex: 1; min-width: 26px; height: 3px; margin-top: 20px; border-radius: 3px; background: linear-gradient(90deg, rgba(96, 165, 250, .1), var(--cyan), rgba(96, 165, 250, .1)); background-size: 200% 100%; animation: flowX 1.8s linear infinite; }
#|@keyframes flowX { to { background-position: -200% 0; } }
#|
#|/* ------------------------------------------------------- model card & alerts */
#|.auc-row { display: flex; align-items: center; gap: 12px; }
#|.auc-row b { font-size: 34px; font-variant-numeric: tabular-nums; }
#|.auc-row > span:first-child { color: var(--muted); }
#|.auc-bar { position: relative; height: 10px; border-radius: 8px; background: rgba(148, 163, 184, .15); margin: 12px 0 16px; overflow: visible; }
#|.auc-fill { height: 100%; border-radius: 8px; background: linear-gradient(90deg, #2563eb, #22d3ee); box-shadow: 0 0 16px rgba(34, 211, 238, .6); animation: grow 1.3s var(--ease) both; transform-origin: left; }
#|@keyframes grow { from { transform: scaleX(0); } }
#|.auc-target { position: absolute; top: -5px; width: 2px; height: 20px; background: #fff; opacity: .8; border-radius: 2px; }
#|.mini-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; }
#|.mini-grid div { background: rgba(3, 9, 30, .45); border: 1px solid var(--line); border-radius: 12px; padding: 9px 10px; }
#|.mini-grid span { display: block; font-size: 11.5px; color: var(--dim); text-transform: uppercase; letter-spacing: .06em; }
#|.mini-grid b { font-size: 16px; }
#|.mini-alert { display: flex; align-items: center; gap: 10px; padding: 10px 8px; border-radius: 12px; cursor: pointer; transition: background .2s, transform .2s; }
#|.mini-alert:hover { background: rgba(59, 130, 246, .12); transform: translateX(4px); }
#|.dot { width: 9px; height: 9px; border-radius: 50%; flex: none; }
#|.dot.open { background: var(--amber); box-shadow: 0 0 10px var(--amber); animation: blink 1.6s infinite; }
#|.dot.confirmed { background: var(--red); } .dot.dismissed { background: var(--green); }
#|@keyframes blink { 50% { opacity: .35; } }
#|.risk { font-variant-numeric: tabular-nums; font-weight: 700; }
#|.pill { font-size: 12px; font-weight: 650; padding: 3px 10px; border-radius: 999px; border: 1px solid var(--line); text-transform: capitalize; }
#|.pill.neutral { color: var(--muted); } .pill.ok { color: var(--green); border-color: rgba(52, 211, 153, .4); background: rgba(52, 211, 153, .08); }
#|.pill.warn { color: var(--amber); border-color: rgba(251, 191, 36, .4); }
#|.pill.s-open { color: var(--amber); border-color: rgba(251, 191, 36, .4); background: rgba(251, 191, 36, .08); }
#|.pill.s-confirmed { color: var(--red); border-color: rgba(251, 113, 133, .4); background: rgba(251, 113, 133, .08); }
#|.pill.s-dismissed { color: var(--green); border-color: rgba(52, 211, 153, .4); background: rgba(52, 211, 153, .08); }
#|.pill.t-shell { color: var(--red); } .pill.t-business { color: var(--cyan); } .pill.t-individual { color: var(--sky); }
#|.alert-card { padding: 18px 20px; }
#|.alert-id { color: var(--dim); font-weight: 700; }
#|.bar { width: 130px; height: 7px; border-radius: 6px; background: rgba(148, 163, 184, .15); overflow: hidden; }
#|.bar > div { height: 100%; border-radius: 6px; background: linear-gradient(90deg, var(--amber), var(--red)); animation: grow 1s var(--ease) both; transform-origin: left; }
#|.narr { margin: 14px 0 6px; padding: 13px 16px; border-radius: 12px; background: rgba(3, 9, 30, .55); border-left: 3px solid var(--cyan); line-height: 1.65; }
#|.narr-meta { margin-top: 8px; font-size: 12px; color: var(--dim); }
#|.caret { display: inline-block; width: 8px; height: 1.1em; margin-left: 2px; vertical-align: text-bottom; background: var(--cyan); animation: blink .8s steps(2) infinite; }
#|.actions { margin-top: 12px; }
#|
#|/* ---------------------------------------------------------------------- scan */
#|.scan-card h2 { font-size: 24px; }
#|.scan-card .row { margin-top: 14px; }
#|.scan-wait { display: flex; flex-direction: column; align-items: center; gap: 12px; padding: 30px 0 10px; }
#|.radar { position: relative; width: 130px; height: 130px; border-radius: 50%; border: 1px solid var(--line-hi); overflow: hidden;
#|  background: repeating-radial-gradient(circle, transparent 0 20px, rgba(96, 165, 250, .2) 21px 22px); box-shadow: 0 0 46px rgba(34, 211, 238, .28); }
#|.radar::before, .radar::after { content: ""; position: absolute; }
#|.radar::before { left: 50%; top: 0; bottom: 0; width: 1px; background: rgba(96, 165, 250, .25); }
#|.radar::after { inset: 0; border-radius: 50%; background: conic-gradient(from 0deg, rgba(34, 211, 238, .65), transparent 38%); animation: spin 1.3s linear infinite; }
#|.scan-result { display: flex; gap: 28px; align-items: center; flex-wrap: wrap; margin-top: 26px; padding-top: 22px; border-top: 1px solid var(--line); }
#|.scan-meta { flex: 1; min-width: 240px; display: flex; flex-direction: column; gap: 10px; }
#|.verdict { font-size: 22px; font-weight: 800; }
#|.verdict.bad { color: var(--red); text-shadow: 0 0 24px rgba(251, 113, 133, .5); } .verdict.good { color: var(--green); }
#|.gauge { position: relative; width: 190px; height: 190px; flex: none; }
#|.gauge svg { width: 100%; height: 100%; }
#|.gauge-arc { transition: stroke-dashoffset 1.4s var(--ease); filter: drop-shadow(0 0 8px rgba(34, 211, 238, .55)); }
#|.gauge.bad .gauge-arc { filter: drop-shadow(0 0 10px rgba(251, 113, 133, .65)); }
#|.gauge-text { position: absolute; inset: 0; display: flex; flex-direction: column; align-items: center; justify-content: center; }
#|.gauge-text b { font-size: 42px; font-weight: 800; line-height: 1; font-variant-numeric: tabular-nums; }
#|.gauge-text small { font-size: 20px; color: var(--muted); }
#|.gauge-text span { font-size: 12px; letter-spacing: .14em; text-transform: uppercase; color: var(--muted); margin-top: 4px; }
#|
#|/* ------------------------------------------------------------------- network */
#|.net-layout { display: grid; grid-template-columns: minmax(0, 2.1fr) minmax(260px, 1fr); gap: 16px; }
#|.net-card { padding: 10px; }
#|.net { width: 100%; height: auto; display: block; border-radius: 12px;
#|  background: radial-gradient(ellipse at center, rgba(37, 99, 235, .1), transparent 70%); }
#|.edge { fill: none; stroke-linecap: round; transition: opacity .3s; }
#|.edge.flow { stroke-dasharray: 5 9; animation: dashflow 1.2s linear infinite; }
#|.edge.dim { opacity: .07 !important; }
#|@keyframes dashflow { to { stroke-dashoffset: -14; } }
#|.node { cursor: pointer; animation: nodeIn .7s var(--spring) both; }
#|.node circle { transition: r .25s; }
#|.node:hover circle:nth-of-type(2) { opacity: .28; }
#|.node.sel circle:nth-of-type(3) { stroke: #fff; stroke-width: 3; }
#|.node-label { fill: #c7d8ff; font-size: 11px; pointer-events: none; paint-order: stroke; stroke: rgba(3, 7, 18, .85); stroke-width: 3px; }
#|.pulse { transform-box: fill-box; transform-origin: center; animation: pulseRing 2.2s ease-out infinite; }
#|@keyframes pulseRing { from { transform: scale(1); opacity: .8; } to { transform: scale(2.7); opacity: 0; } }
#|@keyframes nodeIn { from { opacity: 0; } }
#|.legend { display: flex; flex-wrap: wrap; gap: 6px 16px; padding: 10px 14px 6px; font-size: 12.5px; color: var(--muted); }
#|.legend i { display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 6px; }
#|.legend i.ln { width: 16px; height: 3px; border-radius: 2px; vertical-align: middle; }
#|.legend .hint { margin-left: auto; color: var(--dim); }
#|.net-side { display: flex; flex-direction: column; gap: 16px; }
#|.score-line { display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 12px; color: var(--muted); }
#|.score-line b { font-size: 28px; color: var(--red); }
#|.feat { margin-bottom: 10px; }
#|.feat-top { display: flex; justify-content: space-between; font-size: 13px; gap: 8px; }
#|.feat-top em { font-style: normal; color: var(--cyan); font-variant-numeric: tabular-nums; }
#|.feat-bar { height: 6px; border-radius: 5px; background: rgba(148, 163, 184, .15); margin-top: 5px; overflow: hidden; }
#|.feat-bar div { height: 100%; border-radius: 5px; background: linear-gradient(90deg, #2563eb, #22d3ee); animation: grow 1s var(--ease) both; transform-origin: left; }
#|.kv { display: flex; justify-content: space-between; gap: 10px; padding: 5px 0; border-bottom: 1px dashed var(--line); }
#|.kv em { font-style: normal; color: var(--dim); font-weight: 400; }
#|.node-detail .row { margin: 6px 0 10px; }
#|
#|/* --------------------------------------------------------------------- toasts */
#|.toasts { position: fixed; top: 18px; right: 18px; z-index: 50; display: flex; flex-direction: column; gap: 10px; max-width: min(380px, calc(100vw - 36px)); }
#|.toast { padding: 12px 16px; border-radius: 14px; font-size: 14px; backdrop-filter: blur(14px); border: 1px solid var(--line-hi);
#|  background: rgba(8, 20, 52, .85); box-shadow: 0 12px 34px rgba(0, 0, 0, .5); animation: toastIn .5s var(--spring), toastOut .4s ease 4.7s forwards; }
#|.toast.ok { border-color: rgba(52, 211, 153, .55); } .toast.err { border-color: rgba(251, 113, 133, .6); color: #fecdd3; } .toast.warn { border-color: rgba(251, 191, 36, .55); }
#|@keyframes toastIn { from { opacity: 0; transform: translateX(40px) scale(.95); } }
#|@keyframes toastOut { to { opacity: 0; transform: translateX(30px); } }
#|
#|/* ----------------------------------------------------------- floating dock nav */
#|.dock {
#|  position: fixed; left: 0; right: 0; bottom: 22px; margin: 0 auto; z-index: 40;
#|  width: min(540px, calc(100vw - 24px)); padding: 7px; border-radius: 24px;
#|  display: grid; grid-template-columns: repeat(var(--n), 1fr);
#|  background: rgba(8, 20, 52, .62); border: 1px solid var(--line-hi);
#|  backdrop-filter: blur(22px) saturate(160%); -webkit-backdrop-filter: blur(22px) saturate(160%);
#|  box-shadow: 0 18px 50px rgba(2, 6, 23, .7), 0 0 0 1px rgba(255, 255, 255, .03) inset;
#|  animation: dockFloat 6s ease-in-out infinite;
#|}
#|@keyframes dockFloat { 50% { transform: translateY(-7px); } }
#|.dock-glow { position: absolute; inset: -1px; border-radius: 24px; z-index: -1; filter: blur(22px); opacity: .45;
#|  background: linear-gradient(90deg, #2563eb, #22d3ee, #6366f1, #2563eb); background-size: 300% 100%; animation: shimmer 8s linear infinite; }
#|.dock-indicator {
#|  position: absolute; top: 7px; bottom: 7px; left: 7px; border-radius: 18px;
#|  width: calc((100% - 14px) / var(--n)); transform: translateX(calc(var(--i) * 100%));
#|  transition: transform .55s var(--spring);
#|  background: linear-gradient(135deg, rgba(37, 99, 235, .95), rgba(14, 165, 233, .85)); box-shadow: 0 6px 24px rgba(37, 99, 235, .65);
#|}
#|.dock-tab { position: relative; z-index: 1; display: flex; flex-direction: column; align-items: center; gap: 2px; padding: 9px 4px 8px; background: none; border: 0;
#|  color: var(--muted); cursor: pointer; border-radius: 18px; transition: color .25s, transform .3s var(--spring); }
#|.dock-tab:hover { color: #fff; transform: translateY(-3px); }
#|.dock-tab.on { color: #fff; }
#|.dock-tab.on .dock-ico { transform: scale(1.12); }
#|.dock-ico { display: grid; transition: transform .35s var(--spring); }
#|.dock-label { font-size: 11.5px; font-weight: 650; letter-spacing: .04em; }
#|.dock-badge { position: absolute; top: 4px; right: calc(50% - 26px); min-width: 19px; height: 19px; padding: 0 5px; display: grid; place-items: center;
#|  font-size: 11px; font-weight: 800; color: #fff; border-radius: 999px; background: linear-gradient(135deg, #f43f5e, #fb923c); box-shadow: 0 0 14px rgba(244, 63, 94, .8); animation: badgePop 2.4s ease-in-out infinite; }
#|@keyframes badgePop { 50% { transform: scale(1.18); } }
#|
#|/* ------------------------------------------------------------------ demo mode */
#|.app.demo { padding-top: 48px; }
#|.demo-badge { position: fixed; top: 8px; left: 50%; transform: translateX(-50%); z-index: 60;
#|  font: 600 11.5px/1 system-ui, sans-serif; letter-spacing: .06em; text-transform: uppercase; white-space: nowrap; color: #fde68a;
#|  padding: 6px 12px; border-radius: 999px; background: rgba(120, 53, 15, .55); border: 1px solid rgba(251, 191, 36, .45); backdrop-filter: blur(8px); }
#|@media (max-width: 560px) { .demo-badge { font-size: 10px; padding: 5px 9px; } }
#|
#|/* ---------------------------------------------------------------- responsive */
#|@media (max-width: 860px) { .net-layout { grid-template-columns: 1fr; } .mini-grid { grid-template-columns: repeat(2, 1fr); } }
#|@media (max-width: 560px) {
#|  .app { padding: 16px 12px 140px; } .status { display: none; } .bar { width: 90px; }
#|  .gauge { width: 160px; height: 160px; } .stat-value { font-size: 34px; }
#|}
#|@media (prefers-reduced-motion: reduce) {
#|  *, *::before, *::after { animation-duration: .001ms !important; animation-iteration-count: 1 !important; transition-duration: .001ms !important; }
#|}
#|@@@@END
#|@@@@FILE: src/app/layout.tsx
#|import type { Metadata, Viewport } from "next";
#|import type { ReactNode } from "react";
#|import "./globals.css";
#|
#|export const metadata: Metadata = {
#|  title: "XAI-AMLBench | Compliance Console",
#|  description: "Explainable graph-based AML alert review",
#|};
#|
#|export const viewport: Viewport = {
#|  themeColor: "#030712",
#|  width: "device-width",
#|  initialScale: 1,
#|};
#|
#|export default function RootLayout({ children }: { children: ReactNode }) {
#|  return (
#|    <html lang="en">
#|      <body>{children}</body>
#|    </html>
#|  );
#|}
#|@@@@END
#|@@@@FILE: src/app/page.tsx
#|"use client";
#|
#|import { Fragment, useCallback, useEffect, useMemo, useState } from "react";
#|import type { ReactElement } from "react";
#|import Background from "@/components/Background";
#|import FloatingDock from "@/components/FloatingDock";
#|import type { DockTab } from "@/components/FloatingDock";
#|import Glass from "@/components/Glass";
#|import { IconBell, IconGraph, IconHome, IconRadar, IconShield } from "@/components/Icons";
#|import NetworkGraph from "@/components/NetworkGraph";
#|import RiskGauge from "@/components/RiskGauge";
#|import { useCountUp, useTypewriter } from "@/components/hooks";
#|import { api, errorMessage, pct, sleep } from "@/lib/api";
#|import { isDemo } from "@/lib/demo";
#|import type {
#|  Alert,
#|  AlertStatus,
#|  Explanation,
#|  ModelInfo,
#|  SampleAccounts,
#|  ScanResponse,
#|  ScanResult,
#|  TabId,
#|  Toast,
#|  ToastKind,
#|} from "@/types";
#|
#|const niceType = (t: string | null): string => (t ? t.replaceAll("_", " ") : "");
#|
#|type Tone = "blue" | "amber" | "red" | "cyan";
#|type Filter = "all" | AlertStatus;
#|const FILTERS: Filter[] = ["all", "open", "confirmed", "dismissed"];
#|
#|/* ----------------------------------------------------------------- small pieces */
#|interface StatProps {
#|  label: string;
#|  value: number;
#|  tone: Tone;
#|  hint: string;
#|  suffix?: string;
#|}
#|
#|function Stat({ label, value, tone, hint, suffix = "" }: StatProps) {
#|  const v = useCountUp(value);
#|  return (
#|    <Glass as="div" className={`stat tone-${tone}`}>
#|      <div className="stat-label">{label}</div>
#|      <div className="stat-value">
#|        {v.toFixed(0)}
#|        {suffix}
#|      </div>
#|      <div className="stat-hint">{hint}</div>
#|      <span className="stat-spark" />
#|    </Glass>
#|  );
#|}
#|
#|const PIPE: string[] = ["Transactions", "Kafka stream", "GNN detection", "Explainability", "SAR narrative", "immudb ledger"];
#|
#|function Pipeline() {
#|  return (
#|    <div className="pipe">
#|      {PIPE.map((s, i) => (
#|        <Fragment key={s}>
#|          <div className="pipe-item">
#|            <div className="pipe-node" style={{ animationDelay: `${i * 0.4}s` }}>
#|              <b>{i + 1}</b>
#|            </div>
#|            <span>{s}</span>
#|          </div>
#|          {i < PIPE.length - 1 && <i className="pipe-link" style={{ animationDelay: `${i * 0.25}s` }} />}
#|        </Fragment>
#|      ))}
#|    </div>
#|  );
#|}
#|
#|interface NarrativeProps {
#|  text: string;
#|  source: string | null;
#|  sha256: string | null;
#|  animate: boolean;
#|  onDone: () => void;
#|}
#|
#|function Narrative({ text, source, sha256, animate, onDone }: NarrativeProps) {
#|  const shown = useTypewriter(text, animate, onDone);
#|  return (
#|    <div className="narr">
#|      <span>{shown}</span>
#|      {animate && shown.length < text.length && <i className="caret" />}
#|      <div className="narr-meta">
#|        source: {source ?? "n/a"} · sha256 {sha256 ? `${sha256.slice(0, 12)}…` : "n/a"}
#|      </div>
#|    </div>
#|  );
#|}
#|
#|function ModelCard({ info }: { info: ModelInfo | null }) {
#|  const auc = info?.metrics.test_auc_roc ?? 0;
#|  const target = info?.targets.auc_roc ?? 0.87;
#|  const aucAnim = useCountUp(auc * 100, 1100);
#|  return (
#|    <Glass className="model-card">
#|      <h3>Detection model</h3>
#|      {info === null && <p className="muted">Model not loaded yet. Train one (see README), then refresh.</p>}
#|      {info !== null && (
#|        <>
#|          <div className="auc-row">
#|            <span>AUC-ROC</span>
#|            <b>{(aucAnim / 100).toFixed(3)}</b>
#|            <span className={`pill ${info.metrics.meets_auc_target ? "ok" : "warn"}`}>
#|              {info.metrics.meets_auc_target ? `meets ≥ ${target}` : `below ${target}`}
#|            </span>
#|          </div>
#|          <div className="auc-bar">
#|            <div className="auc-fill" style={{ width: `${Math.min(100, auc * 100)}%` }} />
#|            <i className="auc-target" style={{ left: `${target * 100}%` }} />
#|          </div>
#|          <div className="mini-grid">
#|            <div><span>Model</span><b>{info.model.toUpperCase()}</b></div>
#|            <div><span>Precision</span><b>{pct(info.metrics.test_precision, 0)}</b></div>
#|            <div><span>Recall</span><b>{pct(info.metrics.test_recall, 0)}</b></div>
#|            <div><span>Threshold</span><b>{pct(info.threshold, 0)}</b></div>
#|          </div>
#|        </>
#|      )}
#|    </Glass>
#|  );
#|}
#|
#|/* ----------------------------------------------------------------------- page */
#|export default function Home(): ReactElement {
#|  const demoMode = isDemo();
#|  // In demo mode we must install the mock backend BEFORE the first request, so wait for it.
#|  const [ready, setReady] = useState<boolean>(!demoMode);
#|  const [tab, setTab] = useState<TabId>("overview");
#|  const [alerts, setAlerts] = useState<Alert[]>([]);
#|  const [online, setOnline] = useState<boolean | null>(null);
#|  const [samples, setSamples] = useState<SampleAccounts | null>(null);
#|  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null);
#|  const [accountId, setAccountId] = useState<string>("");
#|  const [scan, setScan] = useState<ScanResult | null>(null);
#|  const [officer, setOfficer] = useState<string>("");
#|  const [filter, setFilter] = useState<Filter>("all");
#|  const [busy, setBusy] = useState<string>("");
#|  const [fresh, setFresh] = useState<Set<number>>(() => new Set<number>());
#|  const [netId, setNetId] = useState<string>("");
#|  const [graph, setGraph] = useState<Explanation | null>(null);
#|  const [toasts, setToasts] = useState<Toast[]>([]);
#|
#|  const notify = useCallback((kind: ToastKind, msg: string): void => {
#|    const id = Date.now() + Math.random();
#|    setToasts((t) => [...t, { id, kind, msg }]);
#|    setTimeout(() => setToasts((t) => t.filter((x) => x.id !== id)), 5200);
#|  }, []);
#|
#|  const load = useCallback(async (): Promise<void> => {
#|    try {
#|      setAlerts(await api<Alert[]>("/alerts"));
#|      setOnline(true);
#|    } catch {
#|      setOnline(false);
#|    }
#|  }, []);
#|
#|  useEffect(() => {
#|    if (!demoMode) return;
#|    void import("../../demo/mockApi").then((m) => {
#|      m.installMockApi();
#|      setReady(true);
#|    });
#|  }, [demoMode]);
#|
#|  useEffect(() => {
#|    if (!ready) return undefined;
#|    void load();
#|    api<SampleAccounts>("/accounts/sample").then(setSamples).catch(() => undefined);
#|    api<ModelInfo>("/model/info").then(setModelInfo).catch(() => undefined);
#|    const id = setInterval(() => void load(), 15000);
#|    return () => clearInterval(id);
#|  }, [load, ready]);
#|
#|  const stats = useMemo(() => {
#|    const open = alerts.filter((a) => a.status === "open").length;
#|    const confirmed = alerts.filter((a) => a.status === "confirmed").length;
#|    const avg = alerts.length ? alerts.reduce((s, a) => s + a.risk_score, 0) / alerts.length : 0;
#|    return { total: alerts.length, open, confirmed, avg };
#|  }, [alerts]);
#|
#|  /* ---- actions ---- */
#|  const doScan = async (): Promise<void> => {
#|    const id = accountId.trim();
#|    if (!id) return;
#|    setBusy("scan");
#|    setScan(null);
#|    try {
#|      const [res] = await Promise.all([
#|        api<ScanResponse>("/alerts/scan", { method: "POST", body: JSON.stringify({ account_id: id }) }),
#|        sleep(900),
#|      ]);
#|      setScan({ ...res, account_id: id });
#|      if (res.alert) {
#|        notify("warn", `Alert #${res.alert.id} created for ${id}`);
#|        void load();
#|      }
#|    } catch (e) {
#|      notify("err", errorMessage(e));
#|    } finally {
#|      setBusy("");
#|    }
#|  };
#|
#|  const doNarrative = async (id: number): Promise<void> => {
#|    setBusy(`n${id}`);
#|    try {
#|      const updated = await api<Alert>(`/alerts/${id}/narrative`, { method: "POST" });
#|      setFresh((s) => new Set(s).add(id));
#|      setAlerts((list) => list.map((x) => (x.id === id ? updated : x)));
#|      notify("ok", `SAR narrative ready (${updated.narrative_source ?? "unknown"})`);
#|    } catch (e) {
#|      notify("err", errorMessage(e));
#|    } finally {
#|      setBusy("");
#|    }
#|  };
#|
#|  const doDecision = async (id: number, decision: "confirmed" | "dismissed"): Promise<void> => {
#|    if (!officer.trim()) {
#|      notify("err", "Enter the reviewing officer's name first.");
#|      return;
#|    }
#|    setBusy(`d${id}`);
#|    try {
#|      const updated = await api<Alert>(`/alerts/${id}/decision`, {
#|        method: "POST",
#|        body: JSON.stringify({ officer: officer.trim(), decision, note: "" }),
#|      });
#|      setAlerts((list) => list.map((x) => (x.id === id ? updated : x)));
#|      notify("ok", `Alert #${id} ${decision} and written to the audit ledger`);
#|    } catch (e) {
#|      notify("err", errorMessage(e));
#|    } finally {
#|      setBusy("");
#|    }
#|  };
#|
#|  const doExplain = async (id?: string): Promise<void> => {
#|    const acct = (id ?? netId).trim();
#|    if (!acct) return;
#|    setNetId(acct);
#|    setTab("network");
#|    setBusy("explain");
#|    setGraph(null);
#|    try {
#|      const [data] = await Promise.all([api<Explanation>(`/accounts/${encodeURIComponent(acct)}/explain`), sleep(600)]);
#|      setGraph(data);
#|    } catch (e) {
#|      notify("err", errorMessage(e));
#|    } finally {
#|      setBusy("");
#|    }
#|  };
#|
#|  const clearFresh = (id: number): void =>
#|    setFresh((s) => {
#|      const next = new Set(s);
#|      next.delete(id);
#|      return next;
#|    });
#|
#|  const tabs: DockTab[] = [
#|    { id: "overview", label: "Overview", icon: <IconHome /> },
#|    { id: "scan", label: "Scan", icon: <IconRadar /> },
#|    { id: "alerts", label: "Alerts", icon: <IconBell />, badge: stats.open },
#|    { id: "network", label: "Network", icon: <IconGraph /> },
#|  ];
#|
#|  const shown = alerts.filter((a) => filter === "all" || a.status === filter);
#|  const latencyTarget = modelInfo?.targets.latency_ms ?? 100;
#|
#|  /* ---- views ---- */
#|  const overview = (
#|    <>
#|      <section className="hero">
#|        <span className="eyebrow">Explainable AML · FinCEN-ready audit trail</span>
#|        <h1>
#|          <span className="grad-text">Follow the money.</span>
#|          <br />
#|          Explain every flag.
#|        </h1>
#|        <p className="lead">
#|          Graph neural networks spot laundering typologies across multi-hop transaction networks, and every alert
#|          comes with a human-readable SAR narrative and a tamper-proof audit record.
#|        </p>
#|        <div className="row">
#|          <button className="btn" onClick={() => setTab("scan")}>Scan an account</button>
#|          <button className="btn ghost" onClick={() => setTab("alerts")}>Review alerts</button>
#|        </div>
#|      </section>
#|
#|      <div className="stats">
#|        <Stat label="Total alerts" value={stats.total} tone="blue" hint="all time" />
#|        <Stat label="Awaiting review" value={stats.open} tone="amber" hint="open cases" />
#|        <Stat label="Confirmed" value={stats.confirmed} tone="red" hint="marked suspicious" />
#|        <Stat label="Average risk" value={stats.avg * 100} suffix="%" tone="cyan" hint="across alerts" />
#|      </div>
#|
#|      <Glass>
#|        <h3>Detection pipeline</h3>
#|        <Pipeline />
#|      </Glass>
#|
#|      <div className="two">
#|        <ModelCard info={modelInfo} />
#|        <Glass>
#|          <div className="row between">
#|            <h3>Latest alerts</h3>
#|            <button className="link" onClick={() => setTab("alerts")}>View all →</button>
#|          </div>
#|          {alerts.length === 0 && <p className="muted">Nothing yet. Scan a suspicious account to create the first alert.</p>}
#|          {alerts.slice(0, 4).map((a, i) => (
#|            <div className="mini-alert rise" key={a.id} style={{ animationDelay: `${i * 70}ms` }} onClick={() => setTab("alerts")}>
#|              <span className={`dot ${a.status}`} />
#|              <b>{a.account_id}</b>
#|              <span className="muted">{niceType(a.typology) || "no narrative yet"}</span>
#|              <span className="grow" />
#|              <span className="risk">{pct(a.risk_score, 0)}</span>
#|            </div>
#|          ))}
#|        </Glass>
#|      </div>
#|    </>
#|  );
#|
#|  const scanView = (
#|    <Glass className="scan-card">
#|      <h2>Scan an account</h2>
#|      <p className="muted">The model scores the account from its 2-hop transaction neighbourhood.</p>
#|      <div className="row">
#|        <input
#|          className="input grow"
#|          placeholder="Account ID, e.g. ACC0000123"
#|          value={accountId}
#|          onChange={(e) => setAccountId(e.target.value)}
#|          onKeyDown={(e) => {
#|            if (e.key === "Enter") void doScan();
#|          }}
#|        />
#|        <button className="btn shine" disabled={!accountId.trim() || busy === "scan"} onClick={() => void doScan()}>
#|          {busy === "scan" ? "Scanning…" : "Scan"}
#|        </button>
#|      </div>
#|      {samples !== null && (
#|        <div className="row chips">
#|          <span className="muted">Try:</span>
#|          {samples.suspicious.slice(0, 3).map((a) => (
#|            <button key={a} className="chip bad" onClick={() => setAccountId(a)}>{a}</button>
#|          ))}
#|          {samples.benign.slice(0, 2).map((a) => (
#|            <button key={a} className="chip" onClick={() => setAccountId(a)}>{a}</button>
#|          ))}
#|        </div>
#|      )}
#|
#|      {busy === "scan" && (
#|        <div className="scan-wait">
#|          <div className="radar" />
#|          <p className="muted">Running graph inference…</p>
#|        </div>
#|      )}
#|      {scan !== null && busy !== "scan" && (
#|        <div className="scan-result rise">
#|          <RiskGauge key={`${scan.account_id}-${scan.score}`} score={scan.calibrated_score ?? scan.score} threshold={scan.calibrated_threshold ?? scan.threshold} />
#|          <div className="scan-meta">
#|            <div className={`verdict ${scan.flagged ? "bad" : "good"}`}>
#|              {scan.flagged ? "Flagged: suspicious pattern" : "Below alert threshold"}
#|            </div>
#|            <p className="muted">
#|              {scan.account_id} · threshold {pct(scan.calibrated_threshold ?? scan.threshold)} · inference {scan.latency_ms} ms
#|              {scan.latency_ms <= latencyTarget && <span className="ok-text"> ✓ within {latencyTarget} ms target</span>}
#|            </p>
#|            <div className="row">
#|              <button className="btn ghost" onClick={() => void doExplain(scan.account_id)}>Explain network</button>
#|              {scan.alert !== null && (
#|                <button className="btn" onClick={() => setTab("alerts")}>Open alert #{scan.alert.id}</button>
#|              )}
#|            </div>
#|          </div>
#|        </div>
#|      )}
#|    </Glass>
#|  );
#|
#|  const alertsView = (
#|    <>
#|      <Glass>
#|        <div className="row between">
#|          <div className="seg">
#|            {FILTERS.map((f) => (
#|              <button key={f} className={filter === f ? "on" : ""} onClick={() => setFilter(f)}>{f}</button>
#|            ))}
#|          </div>
#|          <div className="row">
#|            <input className="input" placeholder="Reviewing officer name" value={officer} onChange={(e) => setOfficer(e.target.value)} />
#|            <button className="btn ghost" onClick={() => void load()}>Refresh</button>
#|          </div>
#|        </div>
#|      </Glass>
#|
#|      {shown.length === 0 && (
#|        <Glass>
#|          <p className="muted">No alerts in this view.</p>
#|        </Glass>
#|      )}
#|      {shown.map((a, i) => (
#|        <Glass className="alert-card rise" key={a.id} style={{ animationDelay: `${Math.min(i, 8) * 60}ms` }}>
#|          <div className="row between">
#|            <div className="row">
#|              <span className="alert-id">#{a.id}</span>
#|              <b>{a.account_id}</b>
#|              <span className={`pill s-${a.status}`}>{a.status}</span>
#|              {a.typology && <span className="pill neutral">{niceType(a.typology)}</span>}
#|            </div>
#|            <div className="row">
#|              <div className="bar"><div style={{ width: `${Math.round(a.risk_score * 100)}%` }} /></div>
#|              <b className="risk">{pct(a.risk_score, 0)}</b>
#|            </div>
#|          </div>
#|
#|          {a.narrative && (
#|            <Narrative
#|              text={a.narrative}
#|              source={a.narrative_source}
#|              sha256={a.narrative_sha256}
#|              animate={fresh.has(a.id)}
#|              onDone={() => clearFresh(a.id)}
#|            />
#|          )}
#|
#|          <div className="row actions">
#|            <button className="btn ghost" disabled={busy === `n${a.id}`} onClick={() => void doNarrative(a.id)}>
#|              {busy === `n${a.id}` ? "Generating…" : a.narrative ? "Regenerate narrative" : "Generate SAR narrative"}
#|            </button>
#|            <button className="btn ghost" onClick={() => void doExplain(a.account_id)}>View network</button>
#|            {a.status === "open" && a.narrative && (
#|              <>
#|                <button className="btn red" disabled={busy === `d${a.id}`} onClick={() => void doDecision(a.id, "confirmed")}>
#|                  Confirm suspicious
#|                </button>
#|                <button className="btn green" disabled={busy === `d${a.id}`} onClick={() => void doDecision(a.id, "dismissed")}>
#|                  Dismiss
#|                </button>
#|              </>
#|            )}
#|            {a.reviewed_by && <span className="muted">reviewed by {a.reviewed_by}</span>}
#|          </div>
#|        </Glass>
#|      ))}
#|    </>
#|  );
#|
#|  const networkView = (
#|    <>
#|      <Glass>
#|        <div className="row">
#|          <input
#|            className="input grow"
#|            placeholder="Account ID to explain, e.g. ACC0000123"
#|            value={netId}
#|            onChange={(e) => setNetId(e.target.value)}
#|            onKeyDown={(e) => {
#|              if (e.key === "Enter") void doExplain();
#|            }}
#|          />
#|          <button className="btn shine" disabled={!netId.trim() || busy === "explain"} onClick={() => void doExplain()}>
#|            {busy === "explain" ? "Explaining…" : "Explain"}
#|          </button>
#|        </div>
#|        {samples !== null && (
#|          <div className="row chips">
#|            <span className="muted">Try:</span>
#|            {samples.suspicious.slice(0, 3).map((a) => (
#|              <button key={a} className="chip bad" onClick={() => setNetId(a)}>{a}</button>
#|            ))}
#|          </div>
#|        )}
#|      </Glass>
#|      {busy === "explain" && (
#|        <Glass>
#|          <div className="scan-wait">
#|            <div className="radar" />
#|            <p className="muted">Running GNNExplainer…</p>
#|          </div>
#|        </Glass>
#|      )}
#|      {graph !== null && busy !== "explain" && (
#|        <div className="rise">
#|          <NetworkGraph key={graph.account_id} data={graph} />
#|        </div>
#|      )}
#|      {graph === null && busy !== "explain" && (
#|        <Glass>
#|          <p className="muted">Explain an account to see the transaction network the model reasoned over.</p>
#|        </Glass>
#|      )}
#|    </>
#|  );
#|
#|  return (
#|    <>
#|      <Background />
#|      {demoMode && <div className="demo-badge">Demo mode &middot; sample data &middot; no live backend</div>}
#|      <div className="toasts">
#|        {toasts.map((t) => (
#|          <div key={t.id} className={`toast ${t.kind}`}>{t.msg}</div>
#|        ))}
#|      </div>
#|
#|      <div className={`app${demoMode ? " demo" : ""}`}>
#|        <header className="topbar">
#|          <div className="brand">
#|            <div className="logo">
#|              <span className="logo-ring" />
#|              <IconShield />
#|            </div>
#|            <div>
#|              <div className="brand-name">XAI<span>·</span>AMLBench</div>
#|              <div className="brand-sub">Compliance Console</div>
#|            </div>
#|          </div>
#|          <div className={`status ${online === false ? "off" : online ? "on" : ""}`}>
#|            <i />
#|            {online === null ? "Connecting…" : online ? "Backend online" : "Backend offline"}
#|          </div>
#|        </header>
#|
#|        <main key={tab} className="view">
#|          {tab === "overview" && overview}
#|          {tab === "scan" && scanView}
#|          {tab === "alerts" && alertsView}
#|          {tab === "network" && networkView}
#|        </main>
#|      </div>
#|
#|      <FloatingDock tabs={tabs} active={tab} onChange={setTab} />
#|    </>
#|  );
#|}
#|@@@@END
#|@@@@FILE: src/components/Background.tsx
#|"use client";
#|
#|import { useEffect, useRef } from "react";
#|
#|interface Particle {
#|  x: number;
#|  y: number;
#|  vx: number;
#|  vy: number;
#|  r: number;
#|}
#|
#|/** Animated backdrop: drifting glow orbs, a moving grid and a cursor-reactive particle network. */
#|export default function Background() {
#|  const ref = useRef<HTMLCanvasElement>(null);
#|
#|  useEffect(() => {
#|    const canvas = ref.current;
#|    if (!canvas) return undefined;
#|    const ctx = canvas.getContext("2d");
#|    if (!ctx) return undefined;
#|
#|    const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
#|    const dpr = Math.min(window.devicePixelRatio || 1, 2);
#|    const mouse = { x: -9999, y: -9999 };
#|    let w = 0;
#|    let h = 0;
#|    let pts: Particle[] = [];
#|    let raf = 0;
#|
#|    const resize = (): void => {
#|      w = window.innerWidth;
#|      h = window.innerHeight;
#|      canvas.width = w * dpr;
#|      canvas.height = h * dpr;
#|      canvas.style.width = `${w}px`;
#|      canvas.style.height = `${h}px`;
#|      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
#|      const n = Math.round(Math.min(95, (w * h) / 15000));
#|      pts = Array.from({ length: n }, (): Particle => ({
#|        x: Math.random() * w,
#|        y: Math.random() * h,
#|        vx: (Math.random() - 0.5) * 0.35,
#|        vy: (Math.random() - 0.5) * 0.35,
#|        r: Math.random() * 1.5 + 0.6,
#|      }));
#|    };
#|
#|    const draw = (): void => {
#|      ctx.clearRect(0, 0, w, h);
#|      if (!reduce) {
#|        for (const p of pts) {
#|          p.x += p.vx;
#|          p.y += p.vy;
#|          if (p.x < -10) p.x = w + 10;
#|          else if (p.x > w + 10) p.x = -10;
#|          if (p.y < -10) p.y = h + 10;
#|          else if (p.y > h + 10) p.y = -10;
#|        }
#|      }
#|      for (let i = 0; i < pts.length; i++) {
#|        const a = pts[i];
#|        for (let j = i + 1; j < pts.length; j++) {
#|          const b = pts[j];
#|          const d = Math.hypot(a.x - b.x, a.y - b.y);
#|          if (d < 130) {
#|            ctx.strokeStyle = `rgba(96,165,250,${(1 - d / 130) * 0.32})`;
#|            ctx.lineWidth = 1;
#|            ctx.beginPath();
#|            ctx.moveTo(a.x, a.y);
#|            ctx.lineTo(b.x, b.y);
#|            ctx.stroke();
#|          }
#|        }
#|        const dm = Math.hypot(a.x - mouse.x, a.y - mouse.y);
#|        if (dm < 170) {
#|          ctx.strokeStyle = `rgba(34,211,238,${(1 - dm / 170) * 0.7})`;
#|          ctx.beginPath();
#|          ctx.moveTo(a.x, a.y);
#|          ctx.lineTo(mouse.x, mouse.y);
#|          ctx.stroke();
#|        }
#|        ctx.fillStyle = "rgba(147,197,253,0.85)";
#|        ctx.beginPath();
#|        ctx.arc(a.x, a.y, a.r, 0, Math.PI * 2);
#|        ctx.fill();
#|      }
#|      if (!reduce) raf = requestAnimationFrame(draw);
#|    };
#|
#|    const onMove = (e: MouseEvent): void => {
#|      mouse.x = e.clientX;
#|      mouse.y = e.clientY;
#|    };
#|    const onLeave = (): void => {
#|      mouse.x = -9999;
#|      mouse.y = -9999;
#|    };
#|
#|    resize();
#|    draw();
#|    window.addEventListener("resize", resize);
#|    window.addEventListener("mousemove", onMove);
#|    window.addEventListener("mouseleave", onLeave);
#|    return () => {
#|      cancelAnimationFrame(raf);
#|      window.removeEventListener("resize", resize);
#|      window.removeEventListener("mousemove", onMove);
#|      window.removeEventListener("mouseleave", onLeave);
#|    };
#|  }, []);
#|
#|  return (
#|    <div className="bg" aria-hidden="true">
#|      <div className="orbs">
#|        <i />
#|        <i />
#|        <i />
#|      </div>
#|      <div className="grid-floor" />
#|      <canvas ref={ref} className="bg-canvas" />
#|    </div>
#|  );
#|}
#|@@@@END
#|@@@@FILE: src/components/FloatingDock.tsx
#|"use client";
#|
#|import type { CSSProperties, ReactNode } from "react";
#|import type { TabId } from "@/types";
#|
#|export interface DockTab {
#|  id: TabId;
#|  label: string;
#|  icon: ReactNode;
#|  badge?: number;
#|}
#|
#|interface FloatingDockProps {
#|  tabs: DockTab[];
#|  active: TabId;
#|  onChange: (id: TabId) => void;
#|}
#|
#|/** Floating pill navigation with a sliding, springy active indicator. */
#|export default function FloatingDock({ tabs, active, onChange }: FloatingDockProps) {
#|  const idx = Math.max(0, tabs.findIndex((t) => t.id === active));
#|  // CSS custom properties are not part of CSSProperties, so cast once here.
#|  const vars = { "--n": tabs.length, "--i": idx } as CSSProperties;
#|
#|  return (
#|    <nav className="dock" style={vars} aria-label="Sections">
#|      <span className="dock-glow" />
#|      <span className="dock-indicator" />
#|      {tabs.map((t) => (
#|        <button
#|          key={t.id}
#|          type="button"
#|          className={`dock-tab ${t.id === active ? "on" : ""}`}
#|          onClick={() => onChange(t.id)}
#|          aria-current={t.id === active ? "page" : undefined}
#|        >
#|          <span className="dock-ico">{t.icon}</span>
#|          <span className="dock-label">{t.label}</span>
#|          {t.badge !== undefined && t.badge > 0 && <span className="dock-badge">{t.badge}</span>}
#|        </button>
#|      ))}
#|    </nav>
#|  );
#|}
#|@@@@END
#|@@@@FILE: src/components/Glass.tsx
#|"use client";
#|
#|import type { ElementType, HTMLAttributes, MouseEvent, ReactNode } from "react";
#|
#|interface GlassProps extends HTMLAttributes<HTMLElement> {
#|  /** Element to render, e.g. "section" (default) or "div". */
#|  as?: ElementType;
#|  children?: ReactNode;
#|}
#|
#|/** Glass panel with a soft spotlight that follows the cursor. */
#|export default function Glass({ as: Tag = "section", className = "", children, ...rest }: GlassProps) {
#|  const onMove = (e: MouseEvent<HTMLElement>): void => {
#|    const el = e.currentTarget;
#|    const r = el.getBoundingClientRect();
#|    el.style.setProperty("--mx", `${e.clientX - r.left}px`);
#|    el.style.setProperty("--my", `${e.clientY - r.top}px`);
#|  };
#|  return (
#|    <Tag className={`glass ${className}`} onMouseMove={onMove} {...rest}>
#|      {children}
#|    </Tag>
#|  );
#|}
#|@@@@END
#|@@@@FILE: src/components/Icons.tsx
#|import type { SVGProps } from "react";
#|
#|const base: SVGProps<SVGSVGElement> = {
#|  width: 22,
#|  height: 22,
#|  viewBox: "0 0 24 24",
#|  fill: "none",
#|  stroke: "currentColor",
#|  strokeWidth: 1.8,
#|  strokeLinecap: "round",
#|  strokeLinejoin: "round",
#|};
#|
#|export const IconHome = () => (
#|  <svg {...base}>
#|    <path d="M3 11.5 12 4l9 7.5" />
#|    <path d="M5 10v9a1 1 0 0 0 1 1h4v-6h4v6h4a1 1 0 0 0 1-1v-9" />
#|  </svg>
#|);
#|
#|export const IconRadar = () => (
#|  <svg {...base}>
#|    <circle cx="12" cy="12" r="9" />
#|    <circle cx="12" cy="12" r="4.5" />
#|    <path d="M12 12 19 6.5" />
#|  </svg>
#|);
#|
#|export const IconBell = () => (
#|  <svg {...base}>
#|    <path d="M6 9a6 6 0 1 1 12 0c0 6 2.5 7.5 2.5 7.5h-17S6 15 6 9Z" />
#|    <path d="M10 20a2 2 0 0 0 4 0" />
#|  </svg>
#|);
#|
#|export const IconGraph = () => (
#|  <svg {...base}>
#|    <circle cx="6" cy="7" r="2.3" />
#|    <circle cx="18" cy="6" r="2.3" />
#|    <circle cx="12" cy="18" r="2.3" />
#|    <path d="M8.2 7.3 15.8 6.5M7.2 9.1 10.9 15.9M16.9 8.1 13.1 15.9" />
#|  </svg>
#|);
#|
#|export const IconShield = () => (
#|  <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#fff" strokeWidth="1.9" strokeLinecap="round" strokeLinejoin="round">
#|    <path d="M12 3 4.5 6v5.5c0 4.6 3.1 8.2 7.5 9.5 4.4-1.3 7.5-4.9 7.5-9.5V6L12 3Z" />
#|    <path d="m9 12 2.2 2.2L15.5 10" />
#|  </svg>
#|);
#|@@@@END
#|@@@@FILE: src/components/NetworkGraph.tsx
#|"use client";
#|
#|import { useMemo, useState } from "react";
#|import { money } from "@/lib/api";
#|import type { ExplainEdge, ExplainNode, Explanation } from "@/types";
#|import Glass from "./Glass";
#|
#|const W = 900;
#|const H = 560;
#|
#|interface Point {
#|  x: number;
#|  y: number;
#|}
#|
#|/** An explained edge plus the values the renderer needs. */
#|interface LaidEdge extends ExplainEdge {
#|  a: number; // index of source node
#|  b: number; // index of destination node
#|  k: number; // n-th parallel edge between the same pair (curves them apart)
#|  w: number; // importance normalised to 0..1
#|}
#|
#|const TYPE_COLOR: Record<string, string> = {
#|  individual: "#60a5fa",
#|  business: "#22d3ee",
#|  shell: "#fb7185",
#|  unknown: "#94a3b8",
#|};
#|
#|const FEATURE_LABELS: Record<string, string> = {
#|  near_thr_ratio: "Transfers just below reporting threshold",
#|  near_thr_cnt: "Transfers just below reporting threshold",
#|  xb_out_ratio: "Cross-border outflow share",
#|  xb_in_ratio: "Cross-border inflow share",
#|  n_cp_countries: "Counterparty countries",
#|  burst_6h: "Burst of activity in 6 hours",
#|  flow_ratio: "Inflow vs outflow balance",
#|  retained_frac: "Pass-through (little retained)",
#|  out_uniq: "Distinct recipients (fan-out)",
#|  in_uniq: "Distinct senders (fan-in)",
#|  out_deg: "Outgoing transactions",
#|  in_deg: "Incoming transactions",
#|  out_amt_sum: "Total sent",
#|  in_amt_sum: "Total received",
#|  out_amt_mean: "Average sent",
#|  in_amt_mean: "Average received",
#|  out_amt_max: "Largest sent",
#|  in_amt_max: "Largest received",
#|  out_amt_std: "Sent amount variability",
#|  in_amt_std: "Received amount variability",
#|  age_days: "Account age",
#|  active_span_h: "Active time span",
#|  is_offshore: "Offshore jurisdiction",
#|  type_shell: "Shell entity",
#|  type_business: "Business account",
#|  type_individual: "Individual account",
#|};
#|
#|const clamp = (v: number, lo: number, hi: number): number => Math.max(lo, Math.min(hi, v));
#|
#|/** Fruchterman-Reingold layout (deterministic, so the picture never jumps between renders). */
#|function computeLayout(nodes: ExplainNode[], edges: Pick<ExplainEdge, "src" | "dst">[], subject: string): Point[] {
#|  let seed = 7;
#|  const rnd = (): number => {
#|    seed = (seed * 16807) % 2147483647;
#|    return seed / 2147483647;
#|  };
#|
#|  const idx = new Map<string, number>();
#|  nodes.forEach((n, i) => idx.set(n.account_id, i));
#|
#|  const pos: Point[] = nodes.map((n, i) => {
#|    const angle = (i / Math.max(nodes.length, 1)) * 2 * Math.PI;
#|    const r = n.account_id === subject ? 0 : 170 + rnd() * 90;
#|    return { x: W / 2 + Math.cos(angle) * r, y: H / 2 + Math.sin(angle) * r };
#|  });
#|
#|  const links: [number, number][] = [];
#|  for (const e of edges) {
#|    const a = idx.get(e.src);
#|    const b = idx.get(e.dst);
#|    if (a !== undefined && b !== undefined && a !== b) links.push([a, b]);
#|  }
#|
#|  const k = Math.sqrt((W * H) / Math.max(nodes.length, 1)) * 0.85;
#|  const pinned = idx.get(subject); // the flagged account stays in the centre
#|  const ITER = 320;
#|
#|  for (let it = 0; it < ITER; it++) {
#|    const cool = 1 - it / ITER;
#|    const disp: Point[] = pos.map(() => ({ x: 0, y: 0 }));
#|
#|    for (let i = 0; i < pos.length; i++) {
#|      for (let j = i + 1; j < pos.length; j++) {
#|        const dx = pos[i].x - pos[j].x;
#|        const dy = pos[i].y - pos[j].y;
#|        const d = Math.hypot(dx, dy) || 0.01;
#|        const f = (k * k) / d;
#|        disp[i].x += (dx / d) * f;
#|        disp[i].y += (dy / d) * f;
#|        disp[j].x -= (dx / d) * f;
#|        disp[j].y -= (dy / d) * f;
#|      }
#|    }
#|    for (const [a, b] of links) {
#|      const dx = pos[a].x - pos[b].x;
#|      const dy = pos[a].y - pos[b].y;
#|      const d = Math.hypot(dx, dy) || 0.01;
#|      const f = (d * d) / k;
#|      disp[a].x -= (dx / d) * f;
#|      disp[a].y -= (dy / d) * f;
#|      disp[b].x += (dx / d) * f;
#|      disp[b].y += (dy / d) * f;
#|    }
#|
#|    const lim = 46 * cool + 1;
#|    for (let i = 0; i < pos.length; i++) {
#|      if (i === pinned) {
#|        pos[i].x = W / 2;
#|        pos[i].y = H / 2;
#|        continue;
#|      }
#|      disp[i].x += (W / 2 - pos[i].x) * 0.05;
#|      disp[i].y += (H / 2 - pos[i].y) * 0.05;
#|      const d = Math.hypot(disp[i].x, disp[i].y) || 1;
#|      pos[i].x = clamp(pos[i].x + (disp[i].x / d) * Math.min(d, lim), 44, W - 44);
#|      pos[i].y = clamp(pos[i].y + (disp[i].y / d) * Math.min(d, lim), 44, H - 44);
#|    }
#|  }
#|
#|  // Collision pass: keep nodes (and the subject's long id label) from overlapping.
#|  for (let pass = 0; pass < 80; pass++) {
#|    let moved = false;
#|    for (let i = 0; i < pos.length; i++) {
#|      for (let j = i + 1; j < pos.length; j++) {
#|        const minD = i === pinned || j === pinned ? 84 : 46;
#|        let dx = pos[j].x - pos[i].x;
#|        let dy = pos[j].y - pos[i].y;
#|        let d = Math.hypot(dx, dy);
#|        if (d >= minD) continue;
#|        if (d < 0.01) {
#|          dx = Math.cos(i + j + 1);
#|          dy = Math.sin(i + j + 1);
#|          d = 1;
#|        }
#|        const push = minD - d + 0.5;
#|        const wi = i === pinned ? 0 : j === pinned ? 1 : 0.5;
#|        pos[i].x -= (dx / d) * push * wi;
#|        pos[i].y -= (dy / d) * push * wi;
#|        pos[j].x += (dx / d) * push * (1 - wi);
#|        pos[j].y += (dy / d) * push * (1 - wi);
#|        moved = true;
#|      }
#|    }
#|    for (const p of pos) {
#|      p.x = clamp(p.x, 44, W - 44);
#|      p.y = clamp(p.y, 44, H - 44);
#|    }
#|    if (!moved) break;
#|  }
#|  return pos;
#|}
#|
#|/** Curved edge that stops at the rim of the destination node so the arrowhead stays visible. */
#|function edgePath(p1: Point, p2: Point, r2: number, k: number): string {
#|  const dx = p2.x - p1.x;
#|  const dy = p2.y - p1.y;
#|  const d = Math.hypot(dx, dy) || 1;
#|  const off = 14 + k * 12;
#|  const cx = (p1.x + p2.x) / 2 + (-dy / d) * off;
#|  const cy = (p1.y + p2.y) / 2 + (dx / d) * off;
#|  const ex = p2.x - cx;
#|  const ey = p2.y - cy;
#|  const el = Math.hypot(ex, ey) || 1;
#|  const tx = p2.x - (ex / el) * (r2 + 5);
#|  const ty = p2.y - (ey / el) * (r2 + 5);
#|  return `M${p1.x.toFixed(1)},${p1.y.toFixed(1)} Q${cx.toFixed(1)},${cy.toFixed(1)} ${tx.toFixed(1)},${ty.toFixed(1)}`;
#|}
#|
#|interface NodeDetail {
#|  node: ExplainNode;
#|  out: number;
#|  inn: number;
#|  sent: number;
#|  recv: number;
#|  top: LaidEdge[];
#|}
#|
#|interface NetworkGraphProps {
#|  data: Explanation;
#|}
#|
#|export default function NetworkGraph({ data }: NetworkGraphProps) {
#|  const [sel, setSel] = useState<number | null>(null);
#|  const subject = data.account_id;
#|
#|  const model = useMemo(() => {
#|    const nodes = data.nodes;
#|    const idx = new Map<string, number>();
#|    nodes.forEach((n, i) => idx.set(n.account_id, i));
#|
#|    const pos = computeLayout(nodes, data.edges, subject);
#|    const maxImp = Math.max(1e-9, ...data.edges.map((e) => e.importance));
#|    const pairs = new Map<string, number>();
#|    const edges: LaidEdge[] = [];
#|
#|    for (const e of data.edges) {
#|      const a = idx.get(e.src);
#|      const b = idx.get(e.dst);
#|      if (a === undefined || b === undefined) continue;
#|      const key = [e.src, e.dst].sort().join("|");
#|      const k = pairs.get(key) ?? 0;
#|      pairs.set(key, k + 1);
#|      edges.push({ ...e, a, b, k, w: e.importance / maxImp });
#|    }
#|
#|    const deg: number[] = nodes.map(() => 0);
#|    for (const e of edges) {
#|      deg[e.a] += 1;
#|      deg[e.b] += 1;
#|    }
#|    const radius: number[] = nodes.map((n, i) =>
#|      n.account_id === subject ? 17 : Math.min(15, 8 + Math.sqrt(deg[i]) * 1.4),
#|    );
#|    return { nodes, pos, edges, radius };
#|  }, [data, subject]);
#|
#|  const detail = useMemo<NodeDetail | null>(() => {
#|    if (sel === null) return null;
#|    const node = model.nodes[sel];
#|    const out = model.edges.filter((e) => e.src === node.account_id);
#|    const inn = model.edges.filter((e) => e.dst === node.account_id);
#|    const sum = (list: LaidEdge[]): number => list.reduce((s, e) => s + e.amount, 0);
#|    const top = [...out, ...inn].sort((x, y) => y.amount - x.amount).slice(0, 5);
#|    return { node, out: out.length, inn: inn.length, sent: sum(out), recv: sum(inn), top };
#|  }, [sel, model]);
#|
#|  const isConnected = (e: LaidEdge): boolean => sel !== null && (e.a === sel || e.b === sel);
#|
#|  return (
#|    <div className="net-layout">
#|      <Glass className="net-card">
#|        <svg viewBox={`0 0 ${W} ${H}`} className="net" role="img" aria-label="Transaction network of the flagged account">
#|          <defs>
#|            <marker id="arrD" markerUnits="userSpaceOnUse" markerWidth="10" markerHeight="10" refX="8" refY="5" orient="auto">
#|              <path d="M0,0 L10,5 L0,10 Z" fill="#22d3ee" />
#|            </marker>
#|            <marker id="arrX" markerUnits="userSpaceOnUse" markerWidth="10" markerHeight="10" refX="8" refY="5" orient="auto">
#|              <path d="M0,0 L10,5 L0,10 Z" fill="#fbbf24" />
#|            </marker>
#|            <filter id="glow" x="-50%" y="-50%" width="200%" height="200%">
#|              <feGaussianBlur stdDeviation="4" result="b" />
#|              <feMerge>
#|                <feMergeNode in="b" />
#|                <feMergeNode in="SourceGraphic" />
#|              </feMerge>
#|            </filter>
#|          </defs>
#|
#|          {model.edges.map((e, i) => (
#|            <path
#|              key={`${e.tx_id}-${i}`}
#|              d={edgePath(model.pos[e.a], model.pos[e.b], model.radius[e.b], e.k)}
#|              className={`edge flow ${sel !== null && !isConnected(e) ? "dim" : ""}`}
#|              stroke={e.cross_border ? "#fbbf24" : "#22d3ee"}
#|              strokeWidth={1 + e.w * 2.6}
#|              opacity={0.35 + e.w * 0.65}
#|              markerEnd={`url(#${e.cross_border ? "arrX" : "arrD"})`}
#|              style={{ animationDuration: `${1.6 - e.w * 0.9}s` }}
#|            >
#|              <title>{`${e.src} → ${e.dst} · ${money(e.amount)}${e.cross_border ? " · cross-border" : ""}`}</title>
#|            </path>
#|          ))}
#|
#|          {model.nodes.map((n, i) => {
#|            const isSubject = n.account_id === subject;
#|            const color = TYPE_COLOR[n.account_type] ?? TYPE_COLOR.unknown;
#|            const r = model.radius[i];
#|            return (
#|              <g
#|                key={n.account_id}
#|                className={`node ${sel === i ? "sel" : ""}`}
#|                transform={`translate(${model.pos[i].x.toFixed(1)},${model.pos[i].y.toFixed(1)})`}
#|                onClick={() => setSel(sel === i ? null : i)}
#|                style={{ animationDelay: `${(i % 12) * 60}ms` }}
#|              >
#|                {isSubject && <circle className="pulse" r={r} fill="none" stroke="#e8f0ff" strokeWidth="2" />}
#|                <circle r={r + 6} fill={color} opacity="0.13" />
#|                <circle
#|                  r={r}
#|                  fill={color}
#|                  filter={isSubject || n.account_type === "shell" ? "url(#glow)" : undefined}
#|                  stroke={isSubject ? "#ffffff" : "rgba(3,7,18,.7)"}
#|                  strokeWidth={isSubject ? 3 : 1.5}
#|                />
#|                <text y={r + 15} textAnchor="middle" className="node-label">
#|                  {isSubject ? n.account_id : n.account_id.slice(-4)}
#|                </text>
#|              </g>
#|            );
#|          })}
#|        </svg>
#|
#|        <div className="legend">
#|          <span><i style={{ background: TYPE_COLOR.individual }} />Individual</span>
#|          <span><i style={{ background: TYPE_COLOR.business }} />Business</span>
#|          <span><i style={{ background: TYPE_COLOR.shell }} />Shell</span>
#|          <span><i className="ln" style={{ background: "#22d3ee" }} />Domestic</span>
#|          <span><i className="ln" style={{ background: "#fbbf24" }} />Cross-border</span>
#|          <span className="hint">Click a node · thicker = more influential to the model</span>
#|        </div>
#|      </Glass>
#|
#|      <div className="net-side">
#|        <Glass>
#|          <h3>Why the model flagged it</h3>
#|          <div className="score-line">
#|            <span>Risk score</span>
#|            <b>{(data.risk_score * 100).toFixed(1)}%</b>
#|          </div>
#|          {data.top_features.length === 0 && <p className="muted">No feature attribution returned.</p>}
#|          {data.top_features.map((f, i) => (
#|            <div className="feat" key={f.feature}>
#|              <div className="feat-top">
#|                <span>{FEATURE_LABELS[f.feature] ?? f.feature}</span>
#|                <em>{(f.weight * 100).toFixed(0)}%</em>
#|              </div>
#|              <div className="feat-bar">
#|                <div style={{ width: `${Math.min(100, f.weight * 260)}%`, animationDelay: `${i * 90}ms` }} />
#|              </div>
#|            </div>
#|          ))}
#|          <p className="muted small">
#|            {data.method} · {data.model} · {data.edges.length} transactions shown
#|          </p>
#|        </Glass>
#|
#|        <Glass className="node-detail">
#|          {detail === null && <p className="muted">Select a node to inspect its flows.</p>}
#|          {detail !== null && (
#|            <>
#|              <h3>{detail.node.account_id}</h3>
#|              <div className="row">
#|                <span className={`pill t-${detail.node.account_type}`}>{detail.node.account_type}</span>
#|                <span className="pill neutral">{detail.node.country}</span>
#|              </div>
#|              <div className="kv">
#|                <span>Sent</span>
#|                <b>{money(detail.sent)} <em>({detail.out} tx)</em></b>
#|              </div>
#|              <div className="kv">
#|                <span>Received</span>
#|                <b>{money(detail.recv)} <em>({detail.inn} tx)</em></b>
#|              </div>
#|              <div className="muted small" style={{ marginTop: 8 }}>Largest transfers</div>
#|              {detail.top.map((e) => (
#|                <div className="kv small" key={e.tx_id}>
#|                  <span>
#|                    {e.src.slice(-4)} → {e.dst.slice(-4)}
#|                    {e.cross_border ? " ✈" : ""}
#|                  </span>
#|                  <b>{money(e.amount)}</b>
#|                </div>
#|              ))}
#|            </>
#|          )}
#|        </Glass>
#|      </div>
#|    </div>
#|  );
#|}
#|@@@@END
#|@@@@FILE: src/components/RiskGauge.tsx
#|"use client";
#|
#|import { useEffect, useState } from "react";
#|import { useCountUp } from "./hooks";
#|
#|interface RiskGaugeProps {
#|  /** 0..1 model probability */
#|  score: number;
#|  /** 0..1 alert threshold, drawn as a tick on the arc */
#|  threshold: number;
#|}
#|
#|/** 270-degree animated risk gauge. Give it a `key` that changes per scan to replay the sweep. */
#|export default function RiskGauge({ score, threshold }: RiskGaugeProps) {
#|  const [on, setOn] = useState<boolean>(false);
#|  const shown = useCountUp(score * 100, 1200);
#|
#|  useEffect(() => {
#|    const id = requestAnimationFrame(() => setOn(true));
#|    return () => cancelAnimationFrame(id);
#|  }, []);
#|
#|  const r = 70;
#|  const c = 2 * Math.PI * r;
#|  const arc = c * 0.75;
#|  const flagged = score >= threshold;
#|  const tickAngle = ((135 + 270 * threshold) * Math.PI) / 180;
#|  const tx1 = 90 + Math.cos(tickAngle) * (r - 11);
#|  const ty1 = 90 + Math.sin(tickAngle) * (r - 11);
#|  const tx2 = 90 + Math.cos(tickAngle) * (r + 11);
#|  const ty2 = 90 + Math.sin(tickAngle) * (r + 11);
#|
#|  return (
#|    <div className={`gauge ${flagged ? "bad" : "good"}`}>
#|      <svg viewBox="0 0 180 180">
#|        <defs>
#|          <linearGradient id="gaugeBad" x1="0" y1="0" x2="1" y2="1">
#|            <stop offset="0%" stopColor="#fbbf24" />
#|            <stop offset="100%" stopColor="#fb7185" />
#|          </linearGradient>
#|          <linearGradient id="gaugeGood" x1="0" y1="0" x2="1" y2="1">
#|            <stop offset="0%" stopColor="#22d3ee" />
#|            <stop offset="100%" stopColor="#3b82f6" />
#|          </linearGradient>
#|        </defs>
#|        <circle
#|          cx="90" cy="90" r={r} fill="none" stroke="rgba(148,163,184,.14)" strokeWidth="12"
#|          strokeLinecap="round" strokeDasharray={`${arc} ${c}`} transform="rotate(135 90 90)"
#|        />
#|        <circle
#|          className="gauge-arc" cx="90" cy="90" r={r} fill="none" strokeWidth="12" strokeLinecap="round"
#|          stroke={flagged ? "url(#gaugeBad)" : "url(#gaugeGood)"}
#|          strokeDasharray={`${arc} ${c}`} strokeDashoffset={on ? arc * (1 - score) : arc}
#|          transform="rotate(135 90 90)"
#|        />
#|        <line x1={tx1} y1={ty1} x2={tx2} y2={ty2} stroke="#e8f0ff" strokeWidth="2" strokeLinecap="round" opacity=".8" />
#|      </svg>
#|      <div className="gauge-text">
#|        <b>
#|          {shown.toFixed(0)}
#|          <small>%</small>
#|        </b>
#|        <span>risk score</span>
#|      </div>
#|    </div>
#|  );
#|}
#|@@@@END
#|@@@@FILE: src/components/hooks.ts
#|"use client";
#|
#|import { useEffect, useRef, useState } from "react";
#|
#|/** Smoothly animates a number from its previous value to `target`. */
#|export function useCountUp(target: number, duration = 900): number {
#|  const [val, setVal] = useState<number>(0);
#|  const from = useRef<number>(0);
#|
#|  useEffect(() => {
#|    const begin = performance.now();
#|    const startVal = from.current;
#|    let raf = 0;
#|    const tick = (now: number): void => {
#|      const t = Math.min(1, (now - begin) / duration);
#|      const eased = 1 - Math.pow(1 - t, 3);
#|      const v = startVal + (target - startVal) * eased;
#|      from.current = v;
#|      setVal(v);
#|      if (t < 1) raf = requestAnimationFrame(tick);
#|    };
#|    raf = requestAnimationFrame(tick);
#|    return () => cancelAnimationFrame(raf);
#|  }, [target, duration]);
#|
#|  return val;
#|}
#|
#|/** Types `text` out character by character while `enabled`; calls `onDone` when finished. */
#|export function useTypewriter(text: string, enabled: boolean, onDone?: () => void): string {
#|  const [out, setOut] = useState<string>(enabled ? "" : text);
#|  const doneRef = useRef<(() => void) | undefined>(onDone);
#|  doneRef.current = onDone;
#|
#|  useEffect(() => {
#|    if (!enabled) {
#|      setOut(text);
#|      return undefined;
#|    }
#|    let i = 0;
#|    setOut("");
#|    const id = setInterval(() => {
#|      i += 2;
#|      setOut(text.slice(0, i));
#|      if (i >= text.length) {
#|        clearInterval(id);
#|        doneRef.current?.();
#|      }
#|    }, 14);
#|    return () => clearInterval(id);
#|  }, [text, enabled]);
#|
#|  return out;
#|}
#|@@@@END
#|@@@@FILE: src/lib/api.ts
#|/** Thin typed wrapper around fetch for the /api routes (served by Traefik or the Next.js rewrite). */
#|export async function api<T>(path: string, options?: RequestInit): Promise<T> {
#|  const res = await fetch(`/api${path}`, {
#|    headers: { "Content-Type": "application/json" },
#|    ...options,
#|  });
#|  const text = await res.text();
#|
#|  let data: unknown;
#|  try {
#|    data = JSON.parse(text);
#|  } catch {
#|    data = { detail: text };
#|  }
#|
#|  if (!res.ok) {
#|    const detail = (data as { detail?: unknown }).detail;
#|    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
#|  }
#|  return data as T;
#|}
#|
#|export const sleep = (ms: number): Promise<void> => new Promise((resolve) => setTimeout(resolve, ms));
#|
#|export const pct = (x: number, digits = 1): string => `${(x * 100).toFixed(digits)}%`;
#|
#|export const money = (n: number): string => `$${Math.round(n).toLocaleString("en-US")}`;
#|
#|export const errorMessage = (e: unknown): string => (e instanceof Error ? e.message : String(e));
#|@@@@END
#|@@@@FILE: src/lib/demo.ts
#|/**
#| * Demo mode: run the real UI with built-in sample data and NO backend.
#| *
#| *   npm run dev:demo      ->  http://localhost:3000
#| *
#| * next.config.js inlines NEXT_PUBLIC_DEMO at build time ("1" only for the dev:demo script), so in a normal
#| * production build this is a constant `false` and the mock backend code is dropped from the bundle.
#| * The try/catch keeps it safe in plain-browser bundles (demo/build-demo.cjs) where `process` does not exist.
#| */
#|export const isDemo = (): boolean => {
#|  try {
#|    return process.env.NEXT_PUBLIC_DEMO === "1";
#|  } catch {
#|    return false;
#|  }
#|};
#|@@@@END
#|@@@@FILE: src/types.ts
#|/** Types shared between the UI and the backend JSON (see backend/main.py and gnn_aml_core/main.py). */
#|
#|export type AlertStatus = "open" | "confirmed" | "dismissed";
#|export type TabId = "overview" | "scan" | "alerts" | "network";
#|export type ToastKind = "ok" | "err" | "warn";
#|
#|export interface Alert {
#|  id: number;
#|  account_id: string;
#|  risk_score: number;
#|  status: AlertStatus;
#|  typology: string | null;
#|  narrative: string | null;
#|  narrative_source: string | null;
#|  narrative_sha256: string | null;
#|  reviewed_by: string | null;
#|  review_note: string | null;
#|  created_at: string;
#|  reviewed_at: string | null;
#|}
#|
#|export interface ScanResponse {
#|  score: number;
#|  calibrated_score?: number;
#|  flagged: boolean;
#|  threshold: number;
#|  calibrated_threshold?: number;
#|  latency_ms: number;
#|  alert: Alert | null;
#|}
#|
#|export interface ScanResult extends ScanResponse {
#|  account_id: string;
#|}
#|
#|export interface SampleAccounts {
#|  suspicious: string[];
#|  benign: string[];
#|}
#|
#|export interface ModelMetrics {
#|  val_auc_roc: number;
#|  test_auc_roc: number;
#|  test_pr_auc: number;
#|  test_precision: number;
#|  test_recall: number;
#|  threshold: number;
#|  auc_target: number;
#|  meets_auc_target: boolean;
#|}
#|
#|export interface ModelInfo {
#|  model: string;
#|  hparams: Record<string, number>;
#|  threshold: number;
#|  metrics: ModelMetrics;
#|  targets: { auc_roc: number; latency_ms: number };
#|  n_nodes: number;
#|  n_features: number;
#|}
#|
#|/* ---- GNNExplainer payload returned by POST /explain ---- */
#|export interface ExplainNode {
#|  account_id: string;
#|  account_type: string; // individual | business | shell
#|  country: string;
#|}
#|
#|export interface ExplainEdge {
#|  tx_id: string;
#|  src: string;
#|  dst: string;
#|  amount: number;
#|  timestamp: number;
#|  cross_border: number; // 0 | 1
#|  importance: number;
#|}
#|
#|export interface TopFeature {
#|  feature: string;
#|  weight: number;
#|}
#|
#|export interface Explanation {
#|  account_id: string;
#|  risk_score: number;
#|  raw_risk_score?: number;
#|  threshold: number;
#|  reporting_threshold: number;
#|  model: string;
#|  method: string;
#|  nodes: ExplainNode[];
#|  edges: ExplainEdge[];
#|  top_features: TopFeature[];
#|}
#|
#|export interface Toast {
#|  id: number;
#|  kind: ToastKind;
#|  msg: string;
#|}
#|@@@@END