"use client";

import { useEffect, useRef, useState } from "react";

/** Smoothly animates a number from its previous value to `target`. */
export function useCountUp(target: number, duration = 900): number {
  const [val, setVal] = useState<number>(0);
  const from = useRef<number>(0);

  useEffect(() => {
    const begin = performance.now();
    const startVal = from.current;
    let raf = 0;
    const tick = (now: number): void => {
      const t = Math.min(1, (now - begin) / duration);
      const eased = 1 - Math.pow(1 - t, 3);
      const v = startVal + (target - startVal) * eased;
      from.current = v;
      setVal(v);
      if (t < 1) raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [target, duration]);

  return val;
}

/** Types `text` out character by character while `enabled`; calls `onDone` when finished. */
export function useTypewriter(text: string, enabled: boolean, onDone?: () => void): string {
  const [out, setOut] = useState<string>(enabled ? "" : text);
  const doneRef = useRef<(() => void) | undefined>(onDone);
  doneRef.current = onDone;

  useEffect(() => {
    if (!enabled) {
      setOut(text);
      return undefined;
    }
    let i = 0;
    setOut("");
    const id = setInterval(() => {
      i += 2;
      setOut(text.slice(0, i));
      if (i >= text.length) {
        clearInterval(id);
        doneRef.current?.();
      }
    }, 14);
    return () => clearInterval(id);
  }, [text, enabled]);

  return out;
}
