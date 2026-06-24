import { motion } from "framer-motion";
import { useEffect, useState } from "react";
import type { PredictResult } from "../api";

function confColor(p: number): string {
  if (p >= 0.7) return "bg-emerald-500";
  if (p >= 0.4) return "bg-amber-500";
  return "bg-rose-500";
}

/** Reveal `text` one character at a time. ~22ms per char feels readable. */
function useTypewriter(text: string, charMs = 22): string {
  const [shown, setShown] = useState("");
  useEffect(() => {
    setShown("");
    if (!text) return;
    let i = 0;
    const id = window.setInterval(() => {
      i += 1;
      setShown(text.slice(0, i));
      if (i >= text.length) window.clearInterval(id);
    }, charMs);
    return () => window.clearInterval(id);
  }, [text, charMs]);
  return shown;
}

export default function ResultPanel({ result }: { result: PredictResult }) {
  const pct = Math.round(result.confidence * 100);
  const transcript = result.text.trim();
  const typed = useTypewriter(transcript);
  const isTyping = typed.length < transcript.length;

  // Animate the confidence bar fill from 0 → pct on result change.
  const [conf, setConf] = useState(0);
  useEffect(() => {
    setConf(0);
    const id = window.setTimeout(() => setConf(pct), 50);
    return () => window.clearTimeout(id);
  }, [pct]);

  return (
    <motion.div
      initial={{ opacity: 0, y: 6 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
      className="space-y-5"
    >
      <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.05 }}>
        <p className="mb-2 text-xs font-medium uppercase tracking-wide text-slate-500 dark:text-slate-400">
          What the model sees
        </p>
        <img
          src={result.mouth_gif}
          alt="Detected mouth region"
          className="w-44 rounded-xl border border-slate-200/70 shadow-sm dark:border-white/10"
          style={{ imageRendering: "pixelated" }}
        />
      </motion.div>

      <div>
        <p className="mb-1 text-xs font-medium uppercase tracking-wide text-slate-500 dark:text-slate-400">
          Transcript
        </p>
        <p
          className={`font-display text-3xl font-semibold leading-tight text-slate-900 dark:text-white ${
            isTyping ? "caret" : ""
          }`}
        >
          {transcript ? typed : <span className="text-slate-400">(no speech decoded)</span>}
        </p>
      </div>

      <div>
        <div className="mb-1 flex items-center justify-between text-xs font-medium uppercase tracking-wide text-slate-500 dark:text-slate-400">
          <span>Confidence</span>
          <motion.span
            key={pct}
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.5 }}
          >
            {pct}%
          </motion.span>
        </div>
        <div className="h-2.5 w-full overflow-hidden rounded-full bg-slate-200 dark:bg-white/10">
          <motion.div
            className={`h-full ${confColor(result.confidence)}`}
            initial={{ width: 0 }}
            animate={{ width: `${conf}%` }}
            transition={{ duration: 0.7, ease: [0.22, 1, 0.36, 1] }}
          />
        </div>
      </div>

      <p className="text-xs text-slate-400 dark:text-slate-500">
        {result.frame_count} mouth frames analysed
      </p>
    </motion.div>
  );
}
