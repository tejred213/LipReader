import { AnimatePresence, motion } from "framer-motion";
import { useCallback, useEffect, useState } from "react";
import Navbar from "./components/Navbar";
import Hero from "./components/Hero";
import HowItWorks from "./components/HowItWorks";
import Demo from "./components/Demo";
import Features from "./components/Features";
import Footer from "./components/Footer";
import { useSmoothScroll } from "./lib/useSmoothScroll";

type Mode = "landing" | "app";

/** Sync the mode with the URL hash so refresh + browser-back work. */
function readModeFromHash(): Mode {
  return window.location.hash === "#app" ? "app" : "landing";
}

export default function App() {
  useSmoothScroll();
  const [mode, setMode] = useState<Mode>(readModeFromHash);

  // Keep URL hash in sync with mode (deep-linkable, supports browser back/forward).
  useEffect(() => {
    const desired = mode === "app" ? "#app" : "";
    if (window.location.hash !== desired) {
      history.replaceState(null, "", desired || window.location.pathname);
    }
  }, [mode]);

  useEffect(() => {
    const onPop = () => setMode(readModeFromHash());
    window.addEventListener("popstate", onPop);
    return () => window.removeEventListener("popstate", onPop);
  }, []);

  const enterApp = useCallback(() => {
    window.scrollTo({ top: 0, behavior: "auto" });
    setMode("app");
  }, []);

  const leaveApp = useCallback(() => {
    window.scrollTo({ top: 0, behavior: "auto" });
    setMode("landing");
  }, []);

  return (
    <div id="top" className="app-bg relative min-h-screen text-slate-900 dark:text-slate-100">
      {/* Animated mesh-gradient backdrop (fixed, behind everything) */}
      <div className="mesh-bg" aria-hidden="true">
        <div className="grain" />
      </div>

      {/* All real content sits above the mesh */}
      <div className="relative z-10">
        <Navbar mode={mode} onEnterApp={enterApp} onLeaveApp={leaveApp} />

        <AnimatePresence mode="wait" initial={false}>
          {mode === "landing" ? (
            <motion.main
              key="landing"
              initial={{ opacity: 0, scale: 0.985, y: -8 }}
              animate={{ opacity: 1, scale: 1, y: 0 }}
              exit={{ opacity: 0, scale: 0.965, y: -12 }}
              transition={{ duration: 0.45, ease: [0.22, 1, 0.36, 1] }}
            >
              <Hero onEnterApp={enterApp} />
              <HowItWorks />
              <Features />
              <Footer />
            </motion.main>
          ) : (
            <motion.main
              key="app"
              initial={{ opacity: 0, scale: 1.04, y: 12 }}
              animate={{ opacity: 1, scale: 1, y: 0 }}
              exit={{ opacity: 0, scale: 1.02 }}
              transition={{ duration: 0.5, ease: [0.22, 1, 0.36, 1] }}
              className="min-h-[80vh] pt-8 sm:pt-12"
            >
              <Demo showHeading={false} />
              <Footer />
            </motion.main>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
}
