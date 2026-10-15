"use client";

import { useMemo, useState } from "react";
import { money } from "@/lib/api";
import type { ExplainEdge, ExplainNode, Explanation } from "@/types";
import Glass from "./Glass";

const W = 900;
const H = 560;

interface Point {
  x: number;
  y: number;
}

/** An explained edge plus the values the renderer needs. */
interface LaidEdge extends ExplainEdge {
  a: number; // index of source node
  b: number; // index of destination node
  k: number; // n-th parallel edge between the same pair (curves them apart)
  w: number; // importance normalised to 0..1
}

const TYPE_COLOR: Record<string, string> = {
  individual: "#60a5fa",
  business: "#22d3ee",
  shell: "#fb7185",
  unknown: "#94a3b8",
};

const FEATURE_LABELS: Record<string, string> = {
  near_thr_ratio: "Transfers just below reporting threshold",
  near_thr_cnt: "Transfers just below reporting threshold",
  xb_out_ratio: "Cross-border outflow share",
  xb_in_ratio: "Cross-border inflow share",
  n_cp_countries: "Counterparty countries",
  burst_6h: "Burst of activity in 6 hours",
  flow_ratio: "Inflow vs outflow balance",
  retained_frac: "Pass-through (little retained)",
  out_uniq: "Distinct recipients (fan-out)",
  in_uniq: "Distinct senders (fan-in)",
  out_deg: "Outgoing transactions",
  in_deg: "Incoming transactions",
  out_amt_sum: "Total sent",
  in_amt_sum: "Total received",
  out_amt_mean: "Average sent",
  in_amt_mean: "Average received",
  out_amt_max: "Largest sent",
  in_amt_max: "Largest received",
  out_amt_std: "Sent amount variability",
  in_amt_std: "Received amount variability",
  age_days: "Account age",
  active_span_h: "Active time span",
  is_offshore: "Offshore jurisdiction",
  type_shell: "Shell entity",
  type_business: "Business account",
  type_individual: "Individual account",
};

const clamp = (v: number, lo: number, hi: number): number => Math.max(lo, Math.min(hi, v));

/** Fruchterman-Reingold layout (deterministic, so the picture never jumps between renders). */
function computeLayout(nodes: ExplainNode[], edges: Pick<ExplainEdge, "src" | "dst">[], subject: string): Point[] {
  let seed = 7;
  const rnd = (): number => {
    seed = (seed * 16807) % 2147483647;
    return seed / 2147483647;
  };

  const idx = new Map<string, number>();
  nodes.forEach((n, i) => idx.set(n.account_id, i));

  const pos: Point[] = nodes.map((n, i) => {
    const angle = (i / Math.max(nodes.length, 1)) * 2 * Math.PI;
    const r = n.account_id === subject ? 0 : 170 + rnd() * 90;
    return { x: W / 2 + Math.cos(angle) * r, y: H / 2 + Math.sin(angle) * r };
  });

  const links: [number, number][] = [];
  for (const e of edges) {
    const a = idx.get(e.src);
    const b = idx.get(e.dst);
    if (a !== undefined && b !== undefined && a !== b) links.push([a, b]);
  }

  const k = Math.sqrt((W * H) / Math.max(nodes.length, 1)) * 0.85;
  const pinned = idx.get(subject); // the flagged account stays in the centre
  const ITER = 320;

  for (let it = 0; it < ITER; it++) {
    const cool = 1 - it / ITER;
    const disp: Point[] = pos.map(() => ({ x: 0, y: 0 }));

    for (let i = 0; i < pos.length; i++) {
      for (let j = i + 1; j < pos.length; j++) {
        const dx = pos[i].x - pos[j].x;
        const dy = pos[i].y - pos[j].y;
        const d = Math.hypot(dx, dy) || 0.01;
        const f = (k * k) / d;
        disp[i].x += (dx / d) * f;
        disp[i].y += (dy / d) * f;
        disp[j].x -= (dx / d) * f;
        disp[j].y -= (dy / d) * f;
      }
    }
    for (const [a, b] of links) {
      const dx = pos[a].x - pos[b].x;
      const dy = pos[a].y - pos[b].y;
      const d = Math.hypot(dx, dy) || 0.01;
      const f = (d * d) / k;
      disp[a].x -= (dx / d) * f;
      disp[a].y -= (dy / d) * f;
      disp[b].x += (dx / d) * f;
      disp[b].y += (dy / d) * f;
    }

    const lim = 46 * cool + 1;
    for (let i = 0; i < pos.length; i++) {
      if (i === pinned) {
        pos[i].x = W / 2;
        pos[i].y = H / 2;
        continue;
      }
      disp[i].x += (W / 2 - pos[i].x) * 0.05;
      disp[i].y += (H / 2 - pos[i].y) * 0.05;
      const d = Math.hypot(disp[i].x, disp[i].y) || 1;
      pos[i].x = clamp(pos[i].x + (disp[i].x / d) * Math.min(d, lim), 44, W - 44);
      pos[i].y = clamp(pos[i].y + (disp[i].y / d) * Math.min(d, lim), 44, H - 44);
    }
  }

  // Collision pass: keep nodes (and the subject's long id label) from overlapping.
  for (let pass = 0; pass < 80; pass++) {
    let moved = false;
    for (let i = 0; i < pos.length; i++) {
      for (let j = i + 1; j < pos.length; j++) {
        const minD = i === pinned || j === pinned ? 84 : 46;
        let dx = pos[j].x - pos[i].x;
        let dy = pos[j].y - pos[i].y;
        let d = Math.hypot(dx, dy);
        if (d >= minD) continue;
        if (d < 0.01) {
          dx = Math.cos(i + j + 1);
          dy = Math.sin(i + j + 1);
          d = 1;
        }
        const push = minD - d + 0.5;
        const wi = i === pinned ? 0 : j === pinned ? 1 : 0.5;
        pos[i].x -= (dx / d) * push * wi;
        pos[i].y -= (dy / d) * push * wi;
        pos[j].x += (dx / d) * push * (1 - wi);
        pos[j].y += (dy / d) * push * (1 - wi);
        moved = true;
      }
    }
    for (const p of pos) {
      p.x = clamp(p.x, 44, W - 44);
      p.y = clamp(p.y, 44, H - 44);
    }
    if (!moved) break;
  }
  return pos;
}

/** Curved edge that stops at the rim of the destination node so the arrowhead stays visible. */
function edgePath(p1: Point, p2: Point, r2: number, k: number): string {
  const dx = p2.x - p1.x;
  const dy = p2.y - p1.y;
  const d = Math.hypot(dx, dy) || 1;
  const off = 14 + k * 12;
  const cx = (p1.x + p2.x) / 2 + (-dy / d) * off;
  const cy = (p1.y + p2.y) / 2 + (dx / d) * off;
  const ex = p2.x - cx;
  const ey = p2.y - cy;
  const el = Math.hypot(ex, ey) || 1;
  const tx = p2.x - (ex / el) * (r2 + 5);
  const ty = p2.y - (ey / el) * (r2 + 5);
  return `M${p1.x.toFixed(1)},${p1.y.toFixed(1)} Q${cx.toFixed(1)},${cy.toFixed(1)} ${tx.toFixed(1)},${ty.toFixed(1)}`;
}

interface NodeDetail {
  node: ExplainNode;
  out: number;
  inn: number;
  sent: number;
  recv: number;
  top: LaidEdge[];
}

interface NetworkGraphProps {
  data: Explanation;
}

export default function NetworkGraph({ data }: NetworkGraphProps) {
  const [sel, setSel] = useState<number | null>(null);
  const subject = data.account_id;

  const model = useMemo(() => {
    const nodes = data.nodes;
    const idx = new Map<string, number>();
    nodes.forEach((n, i) => idx.set(n.account_id, i));

    const pos = computeLayout(nodes, data.edges, subject);
    const maxImp = Math.max(1e-9, ...data.edges.map((e) => e.importance));
    const pairs = new Map<string, number>();
    const edges: LaidEdge[] = [];

    for (const e of data.edges) {
      const a = idx.get(e.src);
      const b = idx.get(e.dst);
      if (a === undefined || b === undefined) continue;
      const key = [e.src, e.dst].sort().join("|");
      const k = pairs.get(key) ?? 0;
      pairs.set(key, k + 1);
      edges.push({ ...e, a, b, k, w: e.importance / maxImp });
    }

    const deg: number[] = nodes.map(() => 0);
    for (const e of edges) {
      deg[e.a] += 1;
      deg[e.b] += 1;
    }
    const radius: number[] = nodes.map((n, i) =>
      n.account_id === subject ? 17 : Math.min(15, 8 + Math.sqrt(deg[i]) * 1.4),
    );
    return { nodes, pos, edges, radius };
  }, [data, subject]);

  const detail = useMemo<NodeDetail | null>(() => {
    if (sel === null) return null;
    const node = model.nodes[sel];
    const out = model.edges.filter((e) => e.src === node.account_id);
    const inn = model.edges.filter((e) => e.dst === node.account_id);
    const sum = (list: LaidEdge[]): number => list.reduce((s, e) => s + e.amount, 0);
    const top = [...out, ...inn].sort((x, y) => y.amount - x.amount).slice(0, 5);
    return { node, out: out.length, inn: inn.length, sent: sum(out), recv: sum(inn), top };
  }, [sel, model]);

  const isConnected = (e: LaidEdge): boolean => sel !== null && (e.a === sel || e.b === sel);

  return (
    <div className="net-layout">
      <Glass className="net-card">
        <svg viewBox={`0 0 ${W} ${H}`} className="net" role="img" aria-label="Transaction network of the flagged account">
          <defs>
            <marker id="arrD" markerUnits="userSpaceOnUse" markerWidth="10" markerHeight="10" refX="8" refY="5" orient="auto">
              <path d="M0,0 L10,5 L0,10 Z" fill="#22d3ee" />
            </marker>
            <marker id="arrX" markerUnits="userSpaceOnUse" markerWidth="10" markerHeight="10" refX="8" refY="5" orient="auto">
              <path d="M0,0 L10,5 L0,10 Z" fill="#fbbf24" />
            </marker>
            <filter id="glow" x="-50%" y="-50%" width="200%" height="200%">
              <feGaussianBlur stdDeviation="4" result="b" />
              <feMerge>
                <feMergeNode in="b" />
                <feMergeNode in="SourceGraphic" />
              </feMerge>
            </filter>
          </defs>

          {model.edges.map((e, i) => (
            <path
              key={`${e.tx_id}-${i}`}
              d={edgePath(model.pos[e.a], model.pos[e.b], model.radius[e.b], e.k)}
              className={`edge flow ${sel !== null && !isConnected(e) ? "dim" : ""}`}
              stroke={e.cross_border ? "#fbbf24" : "#22d3ee"}
              strokeWidth={1 + e.w * 2.6}
              opacity={0.35 + e.w * 0.65}
              markerEnd={`url(#${e.cross_border ? "arrX" : "arrD"})`}
              style={{ animationDuration: `${1.6 - e.w * 0.9}s` }}
            >
              <title>{`${e.src} → ${e.dst} · ${money(e.amount)}${e.cross_border ? " · cross-border" : ""}`}</title>
            </path>
          ))}

          {model.nodes.map((n, i) => {
            const isSubject = n.account_id === subject;
            const color = TYPE_COLOR[n.account_type] ?? TYPE_COLOR.unknown;
            const r = model.radius[i];
            return (
              <g
                key={n.account_id}
                className={`node ${sel === i ? "sel" : ""}`}
                transform={`translate(${model.pos[i].x.toFixed(1)},${model.pos[i].y.toFixed(1)})`}
                onClick={() => setSel(sel === i ? null : i)}
                style={{ animationDelay: `${(i % 12) * 60}ms` }}
              >
                {isSubject && <circle className="pulse" r={r} fill="none" stroke="#e8f0ff" strokeWidth="2" />}
                <circle r={r + 6} fill={color} opacity="0.13" />
                <circle
                  r={r}
                  fill={color}
                  filter={isSubject || n.account_type === "shell" ? "url(#glow)" : undefined}
                  stroke={isSubject ? "#ffffff" : "rgba(3,7,18,.7)"}
                  strokeWidth={isSubject ? 3 : 1.5}
                />
                <text y={r + 15} textAnchor="middle" className="node-label">
                  {isSubject ? n.account_id : n.account_id.slice(-4)}
                </text>
              </g>
            );
          })}
        </svg>

        <div className="legend">
          <span><i style={{ background: TYPE_COLOR.individual }} />Individual</span>
          <span><i style={{ background: TYPE_COLOR.business }} />Business</span>
          <span><i style={{ background: TYPE_COLOR.shell }} />Shell</span>
          <span><i className="ln" style={{ background: "#22d3ee" }} />Domestic</span>
          <span><i className="ln" style={{ background: "#fbbf24" }} />Cross-border</span>
          <span className="hint">Click a node · thicker = more influential to the model</span>
        </div>
      </Glass>

      <div className="net-side">
        <Glass>
          <h3>Why the model flagged it</h3>
          <div className="score-line">
            <span>Risk score</span>
            <b>{(data.risk_score * 100).toFixed(1)}%</b>
          </div>
          {data.top_features.length === 0 && <p className="muted">No feature attribution returned.</p>}
          {data.top_features.map((f, i) => (
            <div className="feat" key={f.feature}>
              <div className="feat-top">
                <span>{FEATURE_LABELS[f.feature] ?? f.feature}</span>
                <em>{(f.weight * 100).toFixed(0)}%</em>
              </div>
              <div className="feat-bar">
                <div style={{ width: `${Math.min(100, f.weight * 260)}%`, animationDelay: `${i * 90}ms` }} />
              </div>
            </div>
          ))}
          <p className="muted small">
            {data.method} · {data.model} · {data.edges.length} transactions shown
          </p>
        </Glass>

        <Glass className="node-detail">
          {detail === null && <p className="muted">Select a node to inspect its flows.</p>}
          {detail !== null && (
            <>
              <h3>{detail.node.account_id}</h3>
              <div className="row">
                <span className={`pill t-${detail.node.account_type}`}>{detail.node.account_type}</span>
                <span className="pill neutral">{detail.node.country}</span>
              </div>
              <div className="kv">
                <span>Sent</span>
                <b>{money(detail.sent)} <em>({detail.out} tx)</em></b>
              </div>
              <div className="kv">
                <span>Received</span>
                <b>{money(detail.recv)} <em>({detail.inn} tx)</em></b>
              </div>
              <div className="muted small" style={{ marginTop: 8 }}>Largest transfers</div>
              {detail.top.map((e) => (
                <div className="kv small" key={e.tx_id}>
                  <span>
                    {e.src.slice(-4)} → {e.dst.slice(-4)}
                    {e.cross_border ? " ✈" : ""}
                  </span>
                  <b>{money(e.amount)}</b>
                </div>
              ))}
            </>
          )}
        </Glass>
      </div>
    </div>
  );
}
