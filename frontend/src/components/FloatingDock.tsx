"use client";

import type { CSSProperties, ReactNode } from "react";
import type { TabId } from "@/types";

export interface DockTab {
  id: TabId;
  label: string;
  icon: ReactNode;
  badge?: number;
}

interface FloatingDockProps {
  tabs: DockTab[];
  active: TabId;
  onChange: (id: TabId) => void;
}

/** Floating pill navigation with a sliding, springy active indicator. */
export default function FloatingDock({ tabs, active, onChange }: FloatingDockProps) {
  const idx = Math.max(0, tabs.findIndex((t) => t.id === active));
  // CSS custom properties are not part of CSSProperties, so cast once here.
  const vars = { "--n": tabs.length, "--i": idx } as CSSProperties;

  return (
    <nav className="dock" style={vars} aria-label="Sections">
      <span className="dock-glow" />
      <span className="dock-indicator" />
      {tabs.map((t) => (
        <button
          key={t.id}
          type="button"
          className={`dock-tab ${t.id === active ? "on" : ""}`}
          onClick={() => onChange(t.id)}
          aria-current={t.id === active ? "page" : undefined}
        >
          <span className="dock-ico">{t.icon}</span>
          <span className="dock-label">{t.label}</span>
          {t.badge !== undefined && t.badge > 0 && <span className="dock-badge">{t.badge}</span>}
        </button>
      ))}
    </nav>
  );
}
