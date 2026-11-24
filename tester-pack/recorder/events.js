(() => {
  if (window.__xaiEv) return; window.__xaiEv = true;
  const label = (t) => { if (!t) return ""; return (t.innerText || t.value || t.getAttribute("aria-label") || t.getAttribute("title") || t.id || t.name || t.tagName || "").trim().replace(/\s+/g," ").slice(0,80); };
  const send = (o) => { try { window.xai_event(Object.assign({ url: location.href }, o)); } catch (e) {} };
  document.addEventListener("click", (e) => {
    let t = e.target; if (t.closest && t.closest("#xai-root")) return;
    t = t.closest ? (t.closest("button,a,input,select,[role=button],[role=tab]") || t) : t;
    send({ type: "click", tag: t.tagName, text: label(t) });
  }, true);
  document.addEventListener("change", (e) => {
    const t = e.target; if (!t || (t.closest && t.closest("#xai-root"))) return;
    send({ type: "field", tag: t.tagName, field: t.name || t.id || t.placeholder || t.type, chars: t.type === "password" ? "hidden" : (t.value || "").length });
  }, true);

  // Watches every call the page makes to the server. Only measurements and verdicts leave this function, never request or response contents.
  const WATCH = ["password","password_hash","hashed_password","totp_secret","id_number","passkey","invite_token","secret"];
  const walk = (o, found, d) => { if (!o || typeof o !== "object" || d > 6) return; for (const k of Object.keys(o)) { if (WATCH.includes(k.toLowerCase())) found.add(k.toLowerCase()); walk(o[k], found, d + 1); } };
  const TRACE = /Traceback \(most recent|File "\/|psycopg|sqlalchemy|at [\w.$<>]+ \(.*:\d+:\d+\)|\/app\/|\/usr\/lib\//;
  const report = async (method, url, status, ms, resp) => {
    try {
      const u = new URL(url, location.href);
      if (!u.pathname.startsWith("/api/")) return;
      const o = { type: "api", m: method, p: u.pathname, s: status, ms: Math.round(ms), found: [], trace: false };
      let text = ""; try { text = await resp.clone().text(); } catch (e) {}
      if (status >= 400 && TRACE.test(text)) o.trace = true;
      let body = null; try { body = JSON.parse(text); } catch (e) {}
      if (body) {
        const f = new Set(); walk(body, f, 0); o.found = [...f];
        if (/\/api\/(login|auth\/login|admin\/login)$/.test(u.pathname) && body.token) { o.token = body.token; o.bank = body.bank || null; o.role = body.role || null; }
        if (typeof body.latency_ms === "number") o.latency = body.latency_ms;
      }
      window.xai_api(o);
    } catch (e) {}
  };
  const f0 = window.fetch;
  window.fetch = async function (input, init) {
    const t0 = performance.now(); const url = typeof input === "string" ? input : (input.url || String(input));
    const method = ((init && init.method) || (input && input.method) || "GET").toUpperCase();
    try { const r = await f0.apply(this, arguments); report(method, url, r.status, performance.now() - t0, r); return r; }
    catch (e) { try { window.xai_api({ type: "api", m: method, p: new URL(url, location.href).pathname, s: 0, ms: Math.round(performance.now() - t0), found: [], trace: false }); } catch (x) {} throw e; }
  };
})();
