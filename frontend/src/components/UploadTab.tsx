import { useRef, useState } from "react";
import { predictUpload, type PredictResult } from "../api";

interface Props {
  run: (factory: () => Promise<PredictResult>) => void;
  loading: boolean;
}

export default function UploadTab({ run, loading }: Props) {
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  function choose(f: File | undefined) {
    if (!f) return;
    setFile(f);
    setPreviewUrl((old) => {
      if (old) URL.revokeObjectURL(old);
      return URL.createObjectURL(f);
    });
  }

  return (
    <div className="space-y-4">
      <div
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragging(false);
          choose(e.dataTransfer.files?.[0]);
        }}
        onClick={() => inputRef.current?.click()}
        className={`flex cursor-pointer flex-col items-center justify-center gap-2 rounded-2xl border-2 border-dashed
          px-6 py-10 text-center transition
          ${
            dragging
              ? "border-sky-500 bg-sky-50/60 dark:bg-sky-500/10"
              : "border-slate-300 hover:border-sky-400 dark:border-white/15"
          }`}
      >
        <svg width="34" height="34" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" className="text-sky-500">
          <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
          <path d="M17 8l-5-5-5 5M12 3v12" />
        </svg>
        <p className="font-medium text-slate-700 dark:text-slate-200">
          {file ? file.name : "Drag & drop a video, or click to browse"}
        </p>
        <p className="text-xs text-slate-500 dark:text-slate-400">MP4, MOV, AVI, WEBM · a few seconds, front-facing</p>
        <input
          ref={inputRef}
          type="file"
          accept="video/*"
          className="hidden"
          onChange={(e) => choose(e.target.files?.[0] ?? undefined)}
        />
      </div>

      {previewUrl && (
        <video src={previewUrl} controls className="w-full rounded-xl border border-slate-200/70 dark:border-white/10" />
      )}

      <button
        disabled={!file || loading}
        onClick={() => file && run(() => predictUpload(file, file.name))}
        className="w-full rounded-xl bg-sky-600 py-3 font-semibold text-white transition
                   hover:bg-sky-500 disabled:cursor-not-allowed disabled:opacity-50"
      >
        {loading ? "Reading lips…" : "Read lips"}
      </button>
    </div>
  );
}
