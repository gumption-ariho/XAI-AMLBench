"use client";

import type { ReactNode } from "react";
import type { TabId } from "@/types";

export interface DockTab {
  id: TabId;
  label: string;
  icon: ReactNode;
  /** One-line description shown under the label, e.g. "Live counts & system pulse". */
  desc?: string;
  /** Section this item sits under in the sidebar, e.g. "Monitoring". Items without a group render at the top. */
  group?: string;
  badge?: number;
}

interface SidebarProps {
  tabs: DockTab[];
  active: TabId;
  onChange: (id: TabId) => void;
}

/** Fixed left-hand navigation, grouped into labelled sections with a short description under each item --
 * still a static list (nothing slides or animates in), just organised the way an analyst actually thinks
 * about the console: Monitoring / Detection / Investigation / System. */
export default function Sidebar({ tabs, active, onChange }: SidebarProps) {
  const groups: { name: string | null; items: DockTab[] }[] = [];
  for (const t of tabs) {
    const name = t.group ?? null;
    const last = groups[groups.length - 1];
    if (last && last.name === name) last.items.push(t);
    else groups.push({ name, items: [t] });
  }

  return (
    <nav className="sidebar" aria-label="Sections">
      {groups.map((g, gi) => (
        <div className="sidebar-group" key={g.name ?? `g${gi}`}>
          {g.name && <div className="sidebar-group-label">{g.name}</div>}
          {g.items.map((t) => (
            <button
              key={t.id}
              type="button"
              className={`sidebar-tab ${t.id === active ? "on" : ""}`}
              onClick={() => onChange(t.id)}
              aria-current={t.id === active ? "page" : undefined}
            >
              <span className="sidebar-ico">{t.icon}</span>
              <span className="sidebar-text">
                <span className="sidebar-row">
                  <span className="sidebar-label">{t.label}</span>
                  {t.badge !== undefined && t.badge > 0 && <span className="sidebar-badge">{t.badge}</span>}
                </span>
                {t.desc && <span className="sidebar-desc">{t.desc}</span>}
              </span>
            </button>
          ))}
        </div>
      ))}
    </nav>
  );
}
