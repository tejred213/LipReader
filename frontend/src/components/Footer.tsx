export default function Footer() {
  return (
    <footer className="relative border-t border-slate-200/60 dark:border-white/5">
      <div className="mx-auto flex max-w-6xl flex-col items-start justify-between gap-6 px-4 py-10 sm:flex-row sm:items-center sm:px-6">
        <div className="flex items-center gap-3">
          <div className="grid h-8 w-8 place-items-center rounded-xl bg-gradient-to-br from-sky-500 to-cyan-400 shadow-md shadow-sky-500/20">
            <span className="serif text-xl leading-none text-white" aria-hidden="true">L</span>
          </div>
          <div className="text-xs text-slate-500 dark:text-slate-500">
            <p className="font-medium text-slate-700 dark:text-slate-300">LipReader</p>
            <p>Open-source visual speech recognition · Apache-2.0</p>
          </div>
        </div>

        <div className="flex flex-wrap items-center gap-x-5 gap-y-2 text-xs text-slate-500 dark:text-slate-500">
          <a
            href="https://github.com/mpc001/auto_avsr"
            target="_blank"
            rel="noopener noreferrer"
            className="transition hover:text-slate-900 dark:hover:text-white"
          >
            Auto-AVSR
          </a>
          <span className="opacity-30">·</span>
          <a
            href="https://github.com/amanvirparhar/chaplin"
            target="_blank"
            rel="noopener noreferrer"
            className="transition hover:text-slate-900 dark:hover:text-white"
          >
            Chaplin (packaging)
          </a>
          <span className="opacity-30">·</span>
          <a
            href="https://github.com/google-ai-edge/mediapipe"
            target="_blank"
            rel="noopener noreferrer"
            className="transition hover:text-slate-900 dark:hover:text-white"
          >
            MediaPipe
          </a>
        </div>
      </div>
    </footer>
  );
}
