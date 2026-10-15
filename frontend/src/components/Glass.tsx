"use client";

import type { ElementType, HTMLAttributes, ReactNode } from "react";

interface GlassProps extends HTMLAttributes<HTMLElement> {
  /** Element to render, e.g. "section" (default) or "div". */
  as?: ElementType;
  children?: ReactNode;
}

/** Plain bordered card -- flat, no blur, no cursor-follow glow. Keeps the name "Glass" and the same props so
 * every existing <Glass> usage across the app keeps working unchanged; only the look underneath it changed. */
export default function Glass({ as: Tag = "section", className = "", children, ...rest }: GlassProps) {
  return (
    <Tag className={`glass ${className}`} {...rest}>
      {children}
    </Tag>
  );
}
