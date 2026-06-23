import { AnimatePresence, motion } from "framer-motion";
import { useState } from "react";
import UploadTab from "./UploadTab";
import WebcamTab from "./WebcamTab";
import ResultPanel from "./ResultPanel";
import type { PredictResult } from "../api";

type TabKey = "upload" | "webcam";

const TABS: { key: TabKey; label: string }[] = [
  { key: "upload", label: "Upload" },
  { key: "webcam", label: "Webcam" },
];

interface DemoProps {
  /** Hide the marketing eyebrow/heading when used as the app's own view. */
  showHeading?: boolean;
}

export default function Demo({ showHeading = true }: DemoProps) {
  const [tab, setTab] = useState<TabKey>("upload");
  const [result, setResult] = useState<PredictResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function run(factory: () => Promise<PredictResult>) {
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      setResult(await factory());
    } catch (e) {
      setError(e instanceof Error ? e.message : "Something went wrong.");
    } finally {
      setLoading(false);
    }
  }

  function switchTab(key: TabKey) {
    setTab(key);
    setResult(null);
    setError(null);
  }

  return (
    <section id="demo" className="relative py-20 sm:py-28 scroll-mt-16">
      <div className="mx-auto max-w-6xl px-4 sm:px-6">
        {showHeading && (
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, margin: "-100px" }}
            transition={{ duration: 0.45 }}
            className="max-w-2xl"
          >
            <span className="eyebrow">
              <span className="h-1.5 w-1.5 rounded-full bg-sky-500" />
              Live
            </span>
            <h2 className="mt-4 font-display text-3xl font-bold tracking-tight text-slate-900 sm:text-4xl dark:text-white">
              Try it. <span className="serif text-slate-500 dark:text-slate-400">Right here.</span>
            </h2>
            <p className="mt-4 text-base leading-relaxed text-slate-600 dark:text-slate-400">
              Upload a short, front-facing clip — or record yourself with your camera —
              and watch the model transcribe.
            </p>
          </motion.div>
        )}

        <motion.div
          initial={{ opacity: 0, y: 18 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: "-80px" }}
          transition={{ duration: 0.55, delay: 0.05 }}
          className={`card-elevated overflow-hidden p-4 sm:p-6 ${showHeading ? "mt-10" : ""}`}
        >
          <div className="grid gap-6 lg:grid-cols-2">
            {/* INPUT */}
            <div>
              <div className="relative mb-5 inline-flex rounded-xl bg-slate-100 p-1 dark:bg-white/[0.06]">
                {TABS.map((t) => (
                  <button
                    key={t.key}
                    onClick={() => switchTab(t.key)}
                    className={`relative z-10 rounded-lg px-4 py-1.5 text-sm font-medium transition-colors ${
                      tab === t.key
                        ? "text-sky-600 dark:text-white"
                        : "text-slate-500 hover:text-slate-700 dark:text-slate-400 dark:hover:text-slate-200"
                    }`}
                  >
                    {tab === t.key && (
                      <motion.span
                        layoutId="demo-tab-pill"
                        className="absolute inset-0 -z-10 rounded-lg bg-white shadow-sm dark:bg-white/10"
                        transition={{ type: "spring", stiffness: 380, damping: 30 }}
                      />
                    )}
                    {t.label}
                  </button>
                ))}
              </div>

              <AnimatePresence mode="wait">
                <motion.div
                  key={tab}
                  initial={{ opacity: 0, x: 6 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: -6 }}
                  transition={{ duration: 0.18 }}
                >
                  {tab === "upload" && <UploadTab run={run} loading={loading} />}
                  {tab === "webcam" && <WebcamTab run={run} loading={loading} />}
                </motion.div>
              </AnimatePresence>
            </div>

            {/* OUTPUT */}
            <div className="min-h-[20rem] rounded-2xl border border-slate-200/60 bg-white/40 p-5 sm:p-6 dark:border-white/5 dark:bg-white/[0.02]">
              <AnimatePresence mode="wait">
                {loading ? (
                  <motion.div
                    key="loading"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    className="flex h-full min-h-[18rem] flex-col items-center justify-center gap-3 text-slate-500 dark:text-slate-400"
                  >
                    <div className="h-8 w-8 animate-spin rounded-full border-2 border-slate-300 border-t-sky-500" />
                    <p className="text-sm">Detecting mouth &amp; reading lips…</p>
                  </motion.div>
                ) : error ? (
                  <motion.div
                    key="error"
                    initial={{ opacity: 0, y: 4 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0 }}
                    className="flex h-full min-h-[18rem] flex-col items-center justify-center gap-2 text-center"
                  >
                    <span className="text-3xl">😕</span>
                    <p className="font-medium text-rose-600 dark:text-rose-400">{error}</p>
                  </motion.div>
                ) : result ? (
                  <ResultPanel key="result" result={result} />
                ) : (
                  <motion.div
                    key="empty"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    className="flex h-full min-h-[18rem] flex-col items-center justify-center gap-2 text-center text-slate-400"
                  >
                    <span className="text-3xl">🎬</span>
                    <p className="text-sm">Your transcript will appear here.</p>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          </div>
        </motion.div>
      </div>
    </section>
  );
}
