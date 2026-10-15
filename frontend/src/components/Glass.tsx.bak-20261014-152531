"use client";

import type { ElementType, HTMLAttributes, MouseEvent, ReactNode } from "react";

interface GlassProps extends HTMLAttributes<HTMLElement> {
  /** Element to render, e.g. "section" (default) or "div". */
  as?: ElementType;
  children?: ReactNode;
}

/** Glass panel with a soft spotlight that follows the cursor. */
export default function Glass({ as: Tag = "section", className = "", children, ...rest }: GlassProps) {
  const onMove = (e: MouseEvent<HTMLElement>): void => {
    const el = e.currentTarget;
    const r = el.getBoundingClientRect();
    el.style.setProperty("--mx", `${e.clientX - r.left}px`);
    el.style.setProperty("--my", `${e.clientY - r.top}px`);
  };
  return (
    <Tag className={`glass ${className}`} onMouseMove={onMove} {...rest}>
      {children}
    </Tag>
  );
}
