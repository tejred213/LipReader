import { motion } from "framer-motion";
import type { ReactNode } from "react";

interface Step {
  num: string;
  title: string;
  desc: string;
  icon: ReactNode;
}

const STEPS: Step[] = [
  {
    num: "01",
    title: "Detect",
    desc: "MediaPipe FaceLandmarker locates the speaker's face and tracks 478 landmarks in real time — right in your browser, before anything reaches the server.",
    icon: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
        <circle cx="12" cy="12" r="8.5" />
        <circle cx="12" cy="12" r="3" />
        <path d="M3.5 12h2M18.5 12h2M12 3.5v2M12 18.5v2" />
      </svg>
    ),
  },
  {
    num: "02",
    title: "Crop",
    desc: "A stable 75-frame mouth region is extracted, normalized, and aligned to the model's expected geometry — the same recipe used during training.",
    icon: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
        <path d="M6 3v15a3 3 0 0 0 3 3h12" />
        <path d="M3 6h15a3 3 0 0 1 3 3v12" />
      </svg>
    ),
  },
  {
    num: "03",
    title: "Read",
    desc: "An Auto-AVSR transformer (3D-CNN → Conformer → CTC + attention) decodes the lip motion into open-vocabulary English. Confidence scored per token.",
    icon: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
        <path d="M4 7h16M4 12h10M4 17h13" />
      </svg>
    ),
  },
];

export default function HowItWorks() {
  return (
    <section id="how" className="relative py-20 sm:py-28">
      <div className="mx-auto max-w-6xl px-4 sm:px-6">
        <motion.div
          initial={{ opacity: 0, y: 12 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: "-100px" }}
          transition={{ duration: 0.5 }}
          className="max-w-2xl"
        >
          <span className="eyebrow">The pipeline</span>
          <h2 className="mt-4 font-display text-3xl font-bold tracking-tight text-slate-900 sm:text-4xl dark:text-white">
            From a silent clip to{" "}
            <span className="serif text-gradient">readable text.</span>
          </h2>
          <p className="mt-4 text-base leading-relaxed text-slate-600 dark:text-slate-400">
            Three stages, end to end. Detection happens in your browser; the heavy
            lifting runs locally on your machine — your video never leaves the box.
          </p>
        </motion.div>

        <div className="mt-12 grid gap-5 sm:grid-cols-2 lg:grid-cols-3">
          {STEPS.map((s, i) => (
            <motion.div
              key={s.num}
              initial={{ opacity: 0, y: 14 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true, margin: "-80px" }}
              transition={{ duration: 0.45, delay: i * 0.08 }}
              className="card-elevated relative overflow-hidden p-6"
            >
              <div className="flex items-start justify-between">
                <div className="grid h-10 w-10 place-items-center rounded-xl bg-gradient-to-br from-sky-500/15 to-cyan-400/10 text-sky-500 ring-1 ring-inset ring-sky-500/20 dark:text-sky-300">
                  <span className="h-5 w-5">{s.icon}</span>
                </div>
                <span className="font-mono text-xs text-slate-400 dark:text-slate-500">{s.num}</span>
              </div>
              <h3 className="mt-5 text-lg font-semibold text-slate-900 dark:text-white">{s.title}</h3>
              <p className="mt-2 text-sm leading-relaxed text-slate-600 dark:text-slate-400">{s.desc}</p>

              {/* subtle bottom-edge accent */}
              <div className="pointer-events-none absolute inset-x-0 bottom-0 h-px bg-gradient-to-r from-transparent via-sky-500/30 to-transparent" />
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
}
