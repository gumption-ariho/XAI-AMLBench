"use client";

import { useEffect, useRef, useState } from "react";

/** Quick ease-out count-up, capped short (at most ~550ms) regardless of the duration a call site asks for --
 * a brief sweep on the overview stats, AUC bar and risk gauge, not the slow multi-second animation this had
 * before. Jumps straight to `target` if the value hasn't changed, so it never re-animates on an unrelated
 * re-render. */
export function useCountUp(target: number, duration = 900): number {
  const capped = Math.min(duration, 550);
  const [shown, setShown] = useState(target);
  const prev = useRef(target);

  useEffect(() => {
    if (prev.current === target) return;
    const from = prev.current;
    const start = performance.now();
    let raf = 0;
    const tick = (now: number) => {
      const t = Math.min(1, (now - start) / capped);
      const eased = 1 - Math.pow(1 - t, 3);
      setShown(from + (target - from) * eased);
      if (t < 1) raf = requestAnimationFrame(tick);
      else prev.current = target;
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [target, capped]);

  return shown;
}

/** Character-by-character reveal for the SAR narrative -- a few characters per tick so it stays brisk (a long
 * narrative finishes in well under a second) rather than dragging out, with `onDone` firing once the full text
 * is shown. If `enabled` is false, the text appears immediately with no animation. */
export function useTypewriter(text: string, enabled: boolean, onDone?: () => void): string {
  const [shown, setShown] = useState(enabled ? "" : text);
  const firedFor = useRef<string | null>(null);

  useEffect(() => {
    if (!enabled) {
      setShown(text);
      if (firedFor.current !== text) {
        firedFor.current = text;
        onDone?.();
      }
      return;
    }
    setShown("");
    let i = 0;
    const step = Math.max(1, Math.ceil(text.length / 120));
    const id = setInterval(() => {
      i += step;
      if (i >= text.length) {
        setShown(text);
        clearInterval(id);
        if (firedFor.current !== text) {
          firedFor.current = text;
          onDone?.();
        }
      } else {
        setShown(text.slice(0, i));
      }
    }, 12);
    return () => clearInterval(id);
  }, [text, enabled, onDone]);

  return shown;
}
