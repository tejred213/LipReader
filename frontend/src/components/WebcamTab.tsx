import { useEffect, useRef, useState } from "react";
import { predictUpload, type PredictResult } from "../api";

interface Props {
  run: (factory: () => Promise<PredictResult>) => void;
  loading: boolean;
}

export default function WebcamTab({ run, loading }: Props) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const streamRef = useRef<MediaStream | null>(null);

  const [ready, setReady] = useState(false);
  const [recording, setRecording] = useState(false);
  const [clip, setClip] = useState<Blob | null>(null);
  const [clipUrl, setClipUrl] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    return () => streamRef.current?.getTracks().forEach((t) => t.stop());
  }, []);

  async function enable() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
      streamRef.current = stream;
      if (videoRef.current) videoRef.current.srcObject = stream;
      setReady(true);
      setError(null);
    } catch {
      setError("Could not access the camera. Check browser permissions.");
    }
  }

  function startRec() {
    if (!streamRef.current) return;
    chunksRef.current = [];
    const rec = new MediaRecorder(streamRef.current, { mimeType: "video/webm" });
    rec.ondataavailable = (e) => e.data.size && chunksRef.current.push(e.data);
    rec.onstop = () => {
      const blob = new Blob(chunksRef.current, { type: "video/webm" });
      setClip(blob);
      setClipUrl((old) => {
        if (old) URL.revokeObjectURL(old);
        return URL.createObjectURL(blob);
      });
    };
    recorderRef.current = rec;
    rec.start();
    setRecording(true);
  }

  function stopRec() {
    recorderRef.current?.stop();
    setRecording(false);
  }

  return (
    <div className="space-y-4">
      {!ready ? (
        <button
          onClick={enable}
          className="w-full rounded-xl bg-sky-600 py-3 font-semibold text-white transition hover:bg-sky-500"
        >
          Enable camera
        </button>
      ) : (
        <>
          <video
            ref={videoRef}
            autoPlay
            muted
            playsInline
            className="w-full rounded-xl border border-slate-200/70 bg-black dark:border-white/10"
          />
          <div className="flex gap-3">
            {!recording ? (
              <button
                onClick={startRec}
                className="flex flex-1 items-center justify-center gap-2 rounded-xl bg-rose-600 py-3 font-semibold text-white transition hover:bg-rose-500"
              >
                <span className="h-2.5 w-2.5 rounded-full bg-white" /> Record
              </button>
            ) : (
              <button
                onClick={stopRec}
                className="flex flex-1 items-center justify-center gap-2 rounded-xl bg-slate-700 py-3 font-semibold text-white transition hover:bg-slate-600"
              >
                <span className="h-2.5 w-2.5 rounded-sm bg-white" /> Stop
              </button>
            )}
          </div>
        </>
      )}

      {error && <p className="text-sm text-rose-600 dark:text-rose-400">{error}</p>}

      {clipUrl && (
        <>
          <p className="text-xs font-medium uppercase tracking-wide text-slate-500 dark:text-slate-400">
            Recorded clip
          </p>
          <video src={clipUrl} controls className="w-full rounded-xl border border-slate-200/70 dark:border-white/10" />
          <button
            disabled={!clip || loading}
            onClick={() => clip && run(() => predictUpload(clip, "webcam.webm"))}
            className="w-full rounded-xl bg-sky-600 py-3 font-semibold text-white transition
                       hover:bg-sky-500 disabled:cursor-not-allowed disabled:opacity-50"
          >
            {loading ? "Reading lips…" : "Read lips"}
          </button>
        </>
      )}
    </div>
  );
}
