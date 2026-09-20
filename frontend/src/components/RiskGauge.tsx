"use client";

import { useEffect, useState } from "react";
import { useCountUp } from "./hooks";

interface RiskGaugeProps {
  /** 0..1 model probability */
  score: number;
  /** 0..1 alert threshold, drawn as a tick on the arc */
  threshold: number;
}

/** 270-degree animated risk gauge. Give it a `key` that changes per scan to replay the sweep. */
export default function RiskGauge({ score, threshold }: RiskGaugeProps) {
  const [on, setOn] = useState<boolean>(false);
  const shown = useCountUp(score * 100, 1200);

  useEffect(() => {
    const id = requestAnimationFrame(() => setOn(true));
    return () => cancelAnimationFrame(id);
  }, []);

  const r = 70;
  const c = 2 * Math.PI * r;
  const arc = c * 0.75;
  const flagged = score >= threshold;
  const tickAngle = ((135 + 270 * threshold) * Math.PI) / 180;
  const tx1 = 90 + Math.cos(tickAngle) * (r - 11);
  const ty1 = 90 + Math.sin(tickAngle) * (r - 11);
  const tx2 = 90 + Math.cos(tickAngle) * (r + 11);
  const ty2 = 90 + Math.sin(tickAngle) * (r + 11);

  return (
    <div className={`gauge ${flagged ? "bad" : "good"}`}>
      <svg viewBox="0 0 180 180">
        <defs>
          <linearGradient id="gaugeBad" x1="0" y1="0" x2="1" y2="1">
            <stop offset="0%" stopColor="#fbbf24" />
            <stop offset="100%" stopColor="#fb7185" />
          </linearGradient>
          <linearGradient id="gaugeGood" x1="0" y1="0" x2="1" y2="1">
            <stop offset="0%" stopColor="#22d3ee" />
            <stop offset="100%" stopColor="#3b82f6" />
          </linearGradient>
        </defs>
        <circle
          cx="90" cy="90" r={r} fill="none" stroke="rgba(148,163,184,.14)" strokeWidth="12"
          strokeLinecap="round" strokeDasharray={`${arc} ${c}`} transform="rotate(135 90 90)"
        />
        <circle
          className="gauge-arc" cx="90" cy="90" r={r} fill="none" strokeWidth="12" strokeLinecap="round"
          stroke={flagged ? "url(#gaugeBad)" : "url(#gaugeGood)"}
          strokeDasharray={`${arc} ${c}`} strokeDashoffset={on ? arc * (1 - score) : arc}
          transform="rotate(135 90 90)"
        />
        <line x1={tx1} y1={ty1} x2={tx2} y2={ty2} stroke="#e8f0ff" strokeWidth="2" strokeLinecap="round" opacity=".8" />
      </svg>
      <div className="gauge-text">
        <b>
          {shown.toFixed(0)}
          <small>%</small>
        </b>
        <span>risk score</span>
      </div>
    </div>
  );
}
