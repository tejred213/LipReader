import { motion } from "framer-motion";
import ThemeToggle from "./ThemeToggle";

function scrollTo(id: string) {
  document.getElementById(id)?.scrollIntoView({ behavior: "smooth", block: "start" });
}

export default function Navbar() {
  return (
    <motion.header
      initial={{ opacity: 0, y: -8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4 }}
      className="sticky top-0 z-30 border-b border-transparent backdrop-blur-md"
    >
      <div className="mx-auto flex max-w-6xl items-center justify-between gap-4 px-4 py-3.5 sm:px-6">
        {/* Brand */}
        <a href="#top" onClick={(e) => { e.preventDefault(); window.scrollTo({ top: 0, behavior: "smooth" }); }} className="flex items-center gap-2.5 group">
          <span className="grid h-8 w-8 place-items-center rounded-xl bg-gradient-to-br from-sky-500 to-cyan-400 shadow-md shadow-sky-500/25 transition group-hover:scale-[1.04]">
            <svg viewBox="0 0 24 24" className="h-4 w-4" fill="white" aria-hidden="true">
              <path d="M4 11.3 C 6.6 8.7, 9.1 8.7, 10.7 10.7 Q 12 12.1, 13.3 10.7 C 14.9 8.7, 17.4 8.7, 20 11.3 C 16 12, 14 12.1, 12 12.1 C 10 12.1, 8 12, 4 11.3 Z" />
              <path d="M4.4 12.9 C 8 13.3, 10 13.4, 12 13.4 C 14 13.4, 16 13.3, 19.6 12.9 C 17.8 16.9, 15 18.7, 12 18.7 C 9 18.7, 6.2 16.9, 4.4 12.9 Z" />
            </svg>
          </span>
          <span className="font-display text-lg font-bold tracking-tight text-slate-900 dark:text-white">LipReader</span>
        </a>

        {/* Center links — hidden on small */}
        <nav className="hidden items-center gap-1 sm:flex">
          <button onClick={() => scrollTo("how")} className="pill-link">How it works</button>
          <button onClick={() => scrollTo("demo")} className="pill-link">Demo</button>
        </nav>

        {/* Actions */}
        <div className="flex items-center gap-2">
          <ThemeToggle />
          <button
            onClick={() => scrollTo("demo")}
            className="hidden rounded-full bg-slate-900 px-4 py-1.5 text-sm font-semibold text-white transition hover:bg-slate-800 sm:inline-block dark:bg-white dark:text-slate-900 dark:hover:bg-slate-100"
          >
            Try it
          </button>
        </div>
      </div>
    </motion.header>
  );
}
