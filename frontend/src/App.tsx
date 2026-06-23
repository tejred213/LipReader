import { useState } from "react";
import ThemeToggle from "./components/ThemeToggle";
import UploadTab from "./components/UploadTab";
import WebcamTab from "./components/WebcamTab";
import ResultPanel from "./components/ResultPanel";
import type { PredictResult } from "./api";

type TabKey = "upload" | "webcam";

const TABS: { key: TabKey; label: string }[] = [
  { key: "upload", label: "Upload" },
  { key: "webcam", label: "Webcam" },
];

export default function App() {
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
    <div className="app-bg min-h-screen text-slate-900 dark:text-slate-100">
      <div className="mx-auto max-w-6xl px-4 py-8 sm:px-6 lg:py-12">
        {/* Header */}
        <header className="mb-8 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="grid h-11 w-11 place-items-center rounded-2xl bg-gradient-to-br from-sky-500 to-cyan-400 shadow-lg shadow-sky-500/25">
              <svg viewBox="0 0 24 24" className="h-6 w-6" fill="white" aria-hidden="true">
                <path d="M4 11.3 C 6.6 8.7, 9.1 8.7, 10.7 10.7 Q 12 12.1, 13.3 10.7 C 14.9 8.7, 17.4 8.7, 20 11.3 C 16 12, 14 12.1, 12 12.1 C 10 12.1, 8 12, 4 11.3 Z" />
                <path d="M4.4 12.9 C 8 13.3, 10 13.4, 12 13.4 C 14 13.4, 16 13.3, 19.6 12.9 C 17.8 16.9, 15 18.7, 12 18.7 C 9 18.7, 6.2 16.9, 4.4 12.9 Z" />
              </svg>
            </div>
            <div>
              <h1 className="text-xl font-bold tracking-tight sm:text-2xl">LipReader</h1>
              <p className="text-xs text-slate-500 dark:text-slate-400">Reading speech from silent video</p>
            </div>
          </div>
          <ThemeToggle />
        </header>

        {/* Usage tip */}
        <div className="card mb-6 flex gap-3 p-4 text-sm text-slate-600 dark:text-slate-300">
          <span className="text-lg leading-none">💡</span>
          <p>
            For best results, use a <strong>clear, front-facing</strong> clip with good lighting and speak
            naturally. The model reads lips only — no audio is used.
          </p>
        </div>

        <div className="grid gap-6 lg:grid-cols-2">
          {/* Left: input */}
          <section className="card p-5 sm:p-6">
            <div className="mb-5 inline-flex rounded-xl bg-slate-100 p-1 dark:bg-white/5">
              {TABS.map((t) => (
                <button
                  key={t.key}
                  onClick={() => switchTab(t.key)}
                  className={`rounded-lg px-4 py-1.5 text-sm font-medium transition ${
                    tab === t.key
                      ? "bg-white text-sky-600 shadow-sm dark:bg-white/10 dark:text-white"
                      : "text-slate-500 hover:text-slate-700 dark:text-slate-400 dark:hover:text-slate-200"
                  }`}
                >
                  {t.label}
                </button>
              ))}
            </div>

            {tab === "upload" && <UploadTab run={run} loading={loading} />}
            {tab === "webcam" && <WebcamTab run={run} loading={loading} />}
          </section>

          {/* Right: results */}
          <section className="card min-h-[20rem] p-5 sm:p-6">
            {loading ? (
              <div className="flex h-full min-h-[18rem] flex-col items-center justify-center gap-3 text-slate-500 dark:text-slate-400">
                <div className="h-8 w-8 animate-spin rounded-full border-2 border-slate-300 border-t-sky-500" />
                <p className="text-sm">Detecting mouth &amp; reading lips…</p>
              </div>
            ) : error ? (
              <div className="flex h-full min-h-[18rem] flex-col items-center justify-center gap-2 text-center">
                <span className="text-3xl">😕</span>
                <p className="font-medium text-rose-600 dark:text-rose-400">{error}</p>
              </div>
            ) : result ? (
              <ResultPanel result={result} />
            ) : (
              <div className="flex h-full min-h-[18rem] flex-col items-center justify-center gap-2 text-center text-slate-400">
                <span className="text-3xl">🎬</span>
                <p className="text-sm">Your prediction will appear here.</p>
              </div>
            )}
          </section>
        </div>

        <footer className="mt-10 text-center text-xs text-slate-400 dark:text-slate-500">
          Auto-AVSR (open-vocabulary VSR) · PyTorch · MediaPipe · FastAPI · React
        </footer>
      </div>
    </div>
  );
}
