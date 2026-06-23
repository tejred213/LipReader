/**
 * Buttery smooth scrolling via Lenis. Mounted once at the app root.
 *
 * Lenis takes over the page's scroll so every gesture (wheel, trackpad,
 * `scrollIntoView`) decays through a single easing curve. Premium SaaS sites
 * (Linear, Vercel) use this to make scroll feel fluid rather than steppy.
 */

import { useEffect, useRef } from "react";
import Lenis from "lenis";

export function useSmoothScroll() {
  const lenisRef = useRef<Lenis | null>(null);

  useEffect(() => {
    const lenis = new Lenis({
      duration: 1.1,                                     // seconds
      easing: (t: number) => Math.min(1, 1.001 - 2 ** (-10 * t)), // expo-out
      smoothWheel: true,
      lerp: 0.1,
    });
    lenisRef.current = lenis;

    let rafId = 0;
    const raf = (time: number) => {
      lenis.raf(time);
      rafId = requestAnimationFrame(raf);
    };
    rafId = requestAnimationFrame(raf);

    return () => {
      cancelAnimationFrame(rafId);
      lenis.destroy();
      lenisRef.current = null;
    };
  }, []);

  return lenisRef;
}
