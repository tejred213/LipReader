import { motion } from "framer-motion";

function scrollTo(id: string) {
  document.getElementById(id)?.scrollIntoView({ behavior: "smooth", block: "start" });
}

/** Decorative "result card" mock-up — sells the product visually. */
function ResultCardMock() {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20, rotate: -2 }}
      animate={{ opacity: 1, y: 0, rotate: -2 }}
      transition={{ delay: 0.35, duration: 0.8, ease: [0.22, 1, 0.36, 1] }}
      className="card-elevated relative w-full max-w-md p-6 sm:p-7"
    >
      <div className="flex items-center gap-2 text-[10px] font-medium uppercase tracking-wider text-slate-500 dark:text-slate-400">
        <span className="grid h-1.5 w-1.5 place-items-center">
          <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-emerald-400" />
        </span>
        Live transcript
      </div>

      <div className="mt-4 flex items-center gap-4">
        {/* Video clip placeholder — abstract "frame" with a play glyph. */}
        <div className="relative h-14 w-32 shrink-0 overflow-hidden rounded-md bg-gradient-to-br from-slate-700 to-slate-900 ring-1 ring-white/10">
          {/* Subtle "scanline" texture */}
          <div
            className="absolute inset-0 opacity-25"
            style={{
              backgroundImage:
                "repeating-linear-gradient(180deg, rgba(255,255,255,0.10) 0 1px, transparent 1px 4px)",
            }}
          />
          {/* Soft sky glow behind the play glyph */}
          <div
            className="absolute inset-0"
            style={{
              background:
                "radial-gradient(circle at 50% 50%, rgba(56,189,248,0.22), transparent 60%)",
            }}
          />
          <svg
            viewBox="0 0 24 24"
            className="absolute left-1/2 top-1/2 h-5 w-5 -translate-x-1/2 -translate-y-1/2 text-white/90"
            fill="currentColor"
            aria-hidden="true"
          >
            <path d="M9 6.5v11l9-5.5z" />
          </svg>
        </div>
        <div className="min-w-0 flex-1">
          <p className="font-display text-xl font-semibold leading-tight text-slate-900 dark:text-white">
            read my lips.
          </p>
          <p className="mt-1 text-xs text-slate-500 dark:text-slate-400">3 seconds · 75 frames</p>
        </div>
      </div>

      <div className="mt-5">
        <div className="mb-1 flex items-center justify-between text-[10px] font-medium uppercase tracking-wider text-slate-500 dark:text-slate-400">
          <span>Confidence</span>
          <span>93%</span>
        </div>
        <div className="h-1.5 w-full overflow-hidden rounded-full bg-slate-200 dark:bg-white/10">
          <motion.div
            initial={{ width: 0 }}
            animate={{ width: "93%" }}
            transition={{ delay: 1.1, duration: 0.9, ease: [0.22, 1, 0.36, 1] }}
            className="h-full rounded-full bg-gradient-to-r from-emerald-400 to-sky-400"
          />
        </div>
      </div>

      {/* Glow */}
      <div
        aria-hidden
        className="pointer-events-none absolute -inset-px -z-10 rounded-3xl opacity-60 blur-2xl"
        style={{
          background:
            "radial-gradient(closest-side, rgba(14,165,233,0.25), transparent 70%)",
        }}
      />
    </motion.div>
  );
}

export default function Hero() {
  return (
    <section className="relative pt-12 pb-20 sm:pt-20 sm:pb-28">
      <div className="mx-auto grid max-w-6xl items-center gap-12 px-4 sm:px-6 lg:grid-cols-[1.15fr_1fr] lg:gap-16">
        {/* LEFT — copy */}
        <div>
          <motion.div
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.4 }}
            className="eyebrow"
          >
            <span className="h-1.5 w-1.5 rounded-full bg-emerald-400" />
            Open-vocabulary visual speech recognition
          </motion.div>

          <motion.h1
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.55, delay: 0.05 }}
            className="mt-5 font-display text-5xl font-bold leading-[1.05] tracking-tight text-slate-900 sm:text-6xl lg:text-7xl dark:text-white"
          >
            See what they're{" "}
            <span className="serif text-gradient">saying.</span>
          </motion.h1>

          <motion.p
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.15 }}
            className="mt-6 max-w-xl text-lg leading-relaxed text-slate-600 dark:text-slate-300"
          >
            A modern lip-reading model that transcribes natural speech from{" "}
            <span className="text-slate-900 dark:text-white">silent video</span>.
            Upload a clip, record yourself, and read what was said — no audio required.
          </motion.p>

          <motion.div
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.25 }}
            className="mt-9 flex flex-wrap items-center gap-3"
          >
            <motion.button
              whileHover={{ y: -1 }}
              whileTap={{ scale: 0.98 }}
              onClick={() => scrollTo("demo")}
              className="group inline-flex items-center gap-2 rounded-full bg-slate-900 px-5 py-3 text-sm font-semibold text-white shadow-lg shadow-sky-500/10 transition hover:bg-slate-800 dark:bg-white dark:text-slate-900 dark:hover:bg-slate-100"
            >
              Try the demo
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" className="transition-transform group-hover:translate-x-0.5">
                <path d="M5 12h14M13 5l7 7-7 7" />
              </svg>
            </motion.button>
            <motion.button
              whileHover={{ y: -1 }}
              whileTap={{ scale: 0.98 }}
              onClick={() => scrollTo("how")}
              className="inline-flex items-center gap-2 rounded-full border border-slate-200/80 bg-white/60 px-5 py-3 text-sm font-semibold text-slate-700 backdrop-blur transition hover:bg-white dark:border-white/15 dark:bg-white/[0.04] dark:text-slate-200 dark:hover:bg-white/[0.08]"
            >
              How it works
            </motion.button>
          </motion.div>

          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.6, delay: 0.4 }}
            className="mt-10 flex flex-wrap items-center gap-x-5 gap-y-2 text-xs text-slate-500 dark:text-slate-500"
          >
            <span className="uppercase tracking-wider">Built on</span>
            <span>Auto-AVSR</span>
            <span className="opacity-40">·</span>
            <span>PyTorch</span>
            <span className="opacity-40">·</span>
            <span>MediaPipe</span>
            <span className="opacity-40">·</span>
            <span>FastAPI</span>
            <span className="opacity-40">·</span>
            <span>React</span>
          </motion.div>
        </div>

        {/* RIGHT — product mock */}
        <div className="relative flex justify-center lg:justify-end">
          <ResultCardMock />
        </div>
      </div>
    </section>
  );
}
