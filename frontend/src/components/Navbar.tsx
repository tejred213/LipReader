import { motion } from "framer-motion";
import ThemeToggle from "./ThemeToggle";

interface NavbarProps {
  /** "landing" shows nav links + Try-it CTA; "app" shows a Back button. */
  mode: "landing" | "app";
  onEnterApp: () => void;
  onLeaveApp: () => void;
}

function scrollTo(id: string) {
  document.getElementById(id)?.scrollIntoView({ behavior: "smooth", block: "start" });
}

export default function Navbar({ mode, onEnterApp, onLeaveApp }: NavbarProps) {
  const isApp = mode === "app";

  return (
    <motion.header
      initial={{ opacity: 0, y: -8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4 }}
      className="sticky top-0 z-30 border-b border-transparent backdrop-blur-md"
    >
      <div className="mx-auto flex max-w-6xl items-center justify-between gap-4 px-4 py-3.5 sm:px-6">
        {/* LEFT — Back button (app) or Brand (landing) */}
        {isApp ? (
          <button
            onClick={onLeaveApp}
            className="group inline-flex items-center gap-2 rounded-full border border-slate-200/70 bg-white/60 px-3.5 py-1.5 text-sm font-medium text-slate-700 backdrop-blur transition hover:bg-white dark:border-white/10 dark:bg-white/[0.04] dark:text-slate-200 dark:hover:bg-white/[0.08]"
          >
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" className="transition-transform group-hover:-translate-x-0.5">
              <path d="M19 12H5M12 5l-7 7 7 7" />
            </svg>
            Back to landing
          </button>
        ) : (
          <a
            href="#top"
            onClick={(e) => {
              e.preventDefault();
              window.scrollTo({ top: 0, behavior: "smooth" });
            }}
            className="group flex items-center gap-2.5"
          >
            <span className="grid h-8 w-8 place-items-center rounded-xl bg-gradient-to-br from-sky-500 to-cyan-400 shadow-md shadow-sky-500/25 transition group-hover:scale-[1.04]">
              <span className="serif text-xl leading-none text-white" aria-hidden="true">L</span>
            </span>
            <span className="font-display text-lg font-bold tracking-tight text-slate-900 dark:text-white">LipReader</span>
          </a>
        )}

        {/* CENTER — landing-only nav links */}
        {!isApp && (
          <nav className="hidden items-center gap-1 sm:flex">
            <button onClick={() => scrollTo("how")} className="pill-link">How it works</button>
            <button onClick={() => scrollTo("features")} className="pill-link">Features</button>
          </nav>
        )}

        {/* RIGHT — brand-while-in-app, theme toggle, CTA */}
        <div className="flex items-center gap-2">
          {isApp && (
            <span className="hidden items-center gap-2 pr-1 sm:flex">
              <span className="grid h-7 w-7 place-items-center rounded-lg bg-gradient-to-br from-sky-500 to-cyan-400 shadow-sm shadow-sky-500/20">
                <span className="serif text-base leading-none text-white" aria-hidden="true">L</span>
              </span>
              <span className="font-display text-base font-bold tracking-tight text-slate-900 dark:text-white">LipReader</span>
            </span>
          )}
          <ThemeToggle />
          {!isApp && (
            <button
              onClick={onEnterApp}
              className="hidden rounded-full bg-slate-900 px-4 py-1.5 text-sm font-semibold text-white transition hover:bg-slate-800 sm:inline-block dark:bg-white dark:text-slate-900 dark:hover:bg-slate-100"
            >
              Try it
            </button>
          )}
        </div>
      </div>
    </motion.header>
  );
}
