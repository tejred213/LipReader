import { motion } from "framer-motion";
import type { ReactNode } from "react";

interface Feature {
  title: string;
  desc: string;
  icon: ReactNode;
}

const FEATURES: Feature[] = [
  {
    title: "Open vocabulary",
    desc: "Trained on LRS3 — not a fixed grid. Real English, decoded from real faces.",
    icon: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round">
        <path d="M4 19V6a2 2 0 0 1 2-2h12a2 2 0 0 1 2 2v13" />
        <path d="M8 8h6M8 12h8M8 16h5" />
      </svg>
    ),
  },
  {
    title: "In-browser tracking",
    desc: "MediaPipe FaceLandmarker runs client-side. You see what the model will see, live.",
    icon: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round">
        <rect x="3" y="6" width="14" height="12" rx="2" />
        <path d="M21 8.5v7l-4-2v-3l4-2z" />
      </svg>
    ),
  },
  {
    title: "Runs locally",
    desc: "CPU-only PyTorch inference, ~2 s per clip. Your video never leaves your machine.",
    icon: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round">
        <rect x="3" y="5" width="18" height="12" rx="2" />
        <path d="M8 21h8M12 17v4" />
      </svg>
    ),
  },
  {
    title: "Open source",
    desc: "Apache-2.0 throughout. Auto-AVSR engine vendored, swap models freely.",
    icon: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round">
        <path d="M9 19c-4 1-4-2-6-2M15 22v-3.5a3 3 0 0 0-1-2.3c3-.3 6-1.4 6-6 0-1.2-.5-2.4-1.3-3.3.2-1 .3-2.1-.2-3 0 0-1-.3-3.5 1.3a12 12 0 0 0-6 0C6.5 3.3 5.5 3.6 5.5 3.6c-.5.9-.4 2 -.2 3A4.7 4.7 0 0 0 4 9.9c0 4.6 3 5.7 6 6a3 3 0 0 0-1 2.3V22" />
      </svg>
    ),
  },
];

export default function Features() {
  return (
    <section id="features" className="relative scroll-mt-16 py-20 sm:py-24">
      <div className="mx-auto max-w-6xl px-4 sm:px-6">
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: "-100px" }}
          transition={{ duration: 0.45 }}
          className="max-w-2xl"
        >
          <span className="eyebrow">Why this matters</span>
          <h2 className="mt-4 font-display text-3xl font-bold tracking-tight text-slate-900 sm:text-4xl dark:text-white">
            A serious lip-reader, built for the open web.
          </h2>
        </motion.div>

        <div className="mt-12 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          {FEATURES.map((f, i) => (
            <motion.div
              key={f.title}
              initial={{ opacity: 0, y: 12 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true, margin: "-80px" }}
              transition={{ duration: 0.4, delay: i * 0.07 }}
              className="card p-5"
            >
              <div className="grid h-9 w-9 place-items-center rounded-lg bg-slate-100 text-slate-700 dark:bg-white/[0.06] dark:text-slate-200">
                <span className="h-5 w-5">{f.icon}</span>
              </div>
              <h3 className="mt-4 text-base font-semibold text-slate-900 dark:text-white">{f.title}</h3>
              <p className="mt-1.5 text-sm leading-relaxed text-slate-600 dark:text-slate-400">{f.desc}</p>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
}
