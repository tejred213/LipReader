import type { PredictResult } from "../api";

function confColor(p: number): string {
  if (p >= 0.7) return "bg-emerald-500";
  if (p >= 0.4) return "bg-amber-500";
  return "bg-rose-500";
}

export default function ResultPanel({ result }: { result: PredictResult }) {
  const pct = Math.round(result.confidence * 100);

  return (
    <div className="animate-fade-in space-y-5">
      <div>
        <p className="mb-2 text-xs font-medium uppercase tracking-wide text-slate-500 dark:text-slate-400">
          What the model sees
        </p>
        <img
          src={result.mouth_gif}
          alt="Detected mouth region"
          className="w-44 rounded-xl border border-slate-200/70 dark:border-white/10"
          style={{ imageRendering: "pixelated" }}
        />
      </div>

      <div>
        <p className="mb-1 text-xs font-medium uppercase tracking-wide text-slate-500 dark:text-slate-400">
          Transcript
        </p>
        <p className="text-2xl font-semibold leading-snug text-slate-900 dark:text-white">
          {result.text.trim() || <span className="text-slate-400">(no speech decoded)</span>}
        </p>
      </div>

      <div>
        <div className="mb-1 flex items-center justify-between text-xs font-medium uppercase tracking-wide text-slate-500 dark:text-slate-400">
          <span>Confidence</span>
          <span>{pct}%</span>
        </div>
        <div className="h-2.5 w-full overflow-hidden rounded-full bg-slate-200 dark:bg-white/10">
          <div className={`h-full ${confColor(result.confidence)} transition-all`} style={{ width: `${pct}%` }} />
        </div>
      </div>

      <p className="text-xs text-slate-400 dark:text-slate-500">
        {result.frame_count} mouth frames analysed
      </p>
    </div>
  );
}
