(() => {
  if (window.__xaiV) return;
  const V = window.__xaiV = { lcp: null, cls: 0, inp: null, ttfb: null };
  try {
    const nav = performance.getEntriesByType("navigation")[0];
    if (nav) V.ttfb = Math.round(nav.responseStart);
    new PerformanceObserver(l => { const e = l.getEntries(); V.lcp = Math.round(e[e.length - 1].startTime); }).observe({ type: "largest-contentful-paint", buffered: true });
    new PerformanceObserver(l => { for (const e of l.getEntries()) if (!e.hadRecentInput) V.cls += e.value; }).observe({ type: "layout-shift", buffered: true });
    const durs = [];
    new PerformanceObserver(l => { for (const e of l.getEntries()) if (e.interactionId) { durs.push(e.duration); durs.sort((a, b) => b - a); V.inp = Math.round(durs.length > 50 ? durs[Math.floor(durs.length / 50)] : durs[0]); } })
      .observe({ type: "event", durationThreshold: 16, buffered: true });
  } catch (e) {}
})();
