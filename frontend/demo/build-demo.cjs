#!/usr/bin/env node
/**
 * Compiles the real TypeScript sources (src/**, demo/**) into ONE self-contained HTML page that runs
 * in any browser: React 18 comes from cdnjs, everything else is inlined. No Next.js, Docker or backend.
 *
 *   npm install            # provides `typescript`
 *   node demo/build-demo.cjs        ->  preview/demo.html
 */
const fs = require("fs");
const path = require("path");
const ts = require("typescript");

const ROOT = path.resolve(__dirname, "..");
const REACT = "https://cdnjs.cloudflare.com/ajax/libs/react/18.3.1/umd/react.production.min.js";
const REACT_DOM = "https://cdnjs.cloudflare.com/ajax/libs/react-dom/18.3.1/umd/react-dom.production.min.js";

const walk = (dir) =>
  fs.readdirSync(dir, { withFileTypes: true }).flatMap((e) =>
    e.isDirectory() ? walk(path.join(dir, e.name)) : [path.join(dir, e.name)]);

/** Tiny CommonJS-style module registry, so `import x from "@/..."` keeps working inside one file. */
const RUNTIME = `(function () {
  var defs = {}, cache = {};
  window.__d = function (id, fn) { defs[id] = fn; };
  function norm(p) { var out = []; p.split("/").forEach(function (s) { if (s === "..") out.pop(); else if (s && s !== ".") out.push(s); }); return out.join("/"); }
  function dirname(id) { return id.split("/").slice(0, -1).join("/"); }
  function load(id) {
    if (cache[id]) return cache[id].exports;
    if (!defs[id]) throw new Error("Module not found: " + id);
    var m = (cache[id] = { exports: {} });
    defs[id].call(m.exports, m.exports, function (r) { return resolve(id, r); }, m);
    return m.exports;
  }
  function resolve(from, r) {
    if (r === "react") return window.React;
    if (r === "react-dom" || r === "react-dom/client") return window.ReactDOM;
    if (r.indexOf("@/") === 0) return load(norm("src/" + r.slice(2)));
    if (r.charAt(0) === ".") return load(norm(dirname(from) + "/" + r));
    throw new Error("Unbundled module: " + r);
  }
  window.__r = load;
})();
`;

function buildBundle() {
  const files = [...walk(path.join(ROOT, "src")), ...walk(path.join(ROOT, "demo"))].filter(
    (f) => /\.(ts|tsx)$/.test(f) && !f.endsWith(".d.ts") && !f.endsWith(path.join("app", "layout.tsx")));

  let mods = "";
  for (const f of files) {
    const id = path.relative(ROOT, f).split(path.sep).join("/").replace(/\.(ts|tsx)$/, "");
    const out = ts.transpileModule(fs.readFileSync(f, "utf8"), {
      fileName: f,
      reportDiagnostics: true,
      compilerOptions: {
        module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2019, jsx: ts.JsxEmit.React, esModuleInterop: true,
      },
    });
    if (out.diagnostics && out.diagnostics.length) {
      throw new Error(id + ": " + out.diagnostics.map((d) => ts.flattenDiagnosticMessageText(d.messageText, "\n")).join("; "));
    }
    mods += `__d(${JSON.stringify(id)}, function (exports, require, module) {\n${out.outputText}\n});\n`;
  }
  const js = (RUNTIME + mods + `__r("demo/entry");\n`).replace(/<\/script/gi, "<\\/script");
  const css = fs.readFileSync(path.join(ROOT, "src/app/globals.css"), "utf8");
  return { js, css, modules: files.length };
}

const DEMO_CSS = `
:root { color-scheme: dark; box-sizing: border-box; padding-top: env(safe-area-inset-top, 0px); padding-bottom: env(safe-area-inset-bottom, 0px); }
html { scroll-padding-top: env(safe-area-inset-top, 0px); }
.dock { bottom: calc(22px + env(safe-area-inset-bottom, 0px)); }
.app { padding-top: 48px; }
`;

function page({ js, css }, scripts) {
  return `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<meta name="theme-color" content="#030712">
<title>XAI-AMLBench | Compliance Console (demo)</title>
<style>${css}${DEMO_CSS}</style>
</head>
<body>
<div class="demo-badge">Demo mode &middot; sample data &middot; no live backend</div>
<div id="root"></div>
${scripts}
<script>
if (!window.React || !window.ReactDOM) {
  document.getElementById("root").innerHTML = '<p style="padding:40px;color:#e8f0ff;font-family:system-ui">Could not load React from cdnjs.cloudflare.com. Check your internet connection and reload.</p>';
} else {
${js}
}
</script>
</body>
</html>
`;
}

module.exports = { buildBundle, page, REACT, REACT_DOM };

if (require.main === module) {
  const bundle = buildBundle();
  const html = page(bundle, `<script src="${REACT}"></script>\n<script src="${REACT_DOM}"></script>`);
  const out = path.join(ROOT, "preview", "demo.html");
  fs.mkdirSync(path.dirname(out), { recursive: true });
  fs.writeFileSync(out, html);
  console.log(`built ${out}  (${bundle.modules} TypeScript modules, ${(html.length / 1024).toFixed(0)} KB)`);
}
