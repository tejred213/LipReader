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
            <span className="serif text-xl leading-none text-white" aria-hidden="true">L</span>
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
