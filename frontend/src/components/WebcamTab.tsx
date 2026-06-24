import { motion, AnimatePresence } from "framer-motion";
import { useEffect, useRef, useState } from "react";
import { predictUpload, type PredictResult } from "../api";
import { detectMouthBox, warmupMouthTracker, type MouthBox } from "../lib/mouthTracker";

interface Props {
  run: (factory: () => Promise<PredictResult>) => void;
  loading: boolean;
}

export default function WebcamTab({ run, loading }: Props) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const streamRef = useRef<MediaStream | null>(null);
  const rafRef = useRef<number | null>(null);
  const lastBoxRef = useRef<MouthBox | null>(null);

  const [ready, setReady] = useState(false);
  const [recording, setRecording] = useState(false);
  const [faceFound, setFaceFound] = useState(false);
  const [clip, setClip] = useState<Blob | null>(null);
  const [clipUrl, setClipUrl] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Preload the tracker model on mount so the first detection isn't slow.
  useEffect(() => {
    warmupMouthTracker();
  }, []);

  // Tear down camera + tracking loop on unmount.
  useEffect(() => {
    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
      streamRef.current?.getTracks().forEach((t) => t.stop());
    };
  }, []);

  function drawOverlay(box: MouthBox | null) {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas) return;

    // Match canvas pixel grid to the rendered video size.
    const rect = video.getBoundingClientRect();
    if (canvas.width !== rect.width || canvas.height !== rect.height) {
      canvas.width = rect.width;
      canvas.height = rect.height;
    }
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (!box || !video.videoWidth) return;

    // Scale from video pixels to displayed CSS pixels.
    const sx = rect.width / video.videoWidth;
    const sy = rect.height / video.videoHeight;
    const x = box.x * sx;
    const y = box.y * sy;
    const w = box.width * sx;
    const h = box.height * sy;

    // Glowy rounded box.
    ctx.save();
    ctx.strokeStyle = recording ? "rgb(244 63 94)" : "rgb(56 189 248)";
    ctx.shadowColor = ctx.strokeStyle;
    ctx.shadowBlur = 12;
    ctx.lineWidth = 2.5;
    const r = 10;
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.arcTo(x + w, y, x + w, y + h, r);
    ctx.arcTo(x + w, y + h, x, y + h, r);
    ctx.arcTo(x, y + h, x, y, r);
    ctx.arcTo(x, y, x + w, y, r);
    ctx.closePath();
    ctx.stroke();
    ctx.restore();
  }

  function startTrackingLoop() {
    const tick = async () => {
      const video = videoRef.current;
      if (!video) return;
      try {
        const box = await detectMouthBox(video, performance.now());
        lastBoxRef.current = box;
        setFaceFound(!!box);
        drawOverlay(box);
      } catch {
        /* MediaPipe occasionally throws on torn-down streams — ignore. */
      }
      rafRef.current = requestAnimationFrame(tick);
    };
    rafRef.current = requestAnimationFrame(tick);
  }

  async function enable() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user" }, audio: false });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play().catch(() => {});
      }
      setReady(true);
      setError(null);
      startTrackingLoop();
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
        <motion.button
          whileHover={{ scale: 1.01 }}
          whileTap={{ scale: 0.98 }}
          onClick={enable}
          className="w-full rounded-xl bg-sky-600 py-3 font-semibold text-white shadow-md shadow-sky-500/20 transition hover:bg-sky-500"
        >
          Enable camera
        </motion.button>
      ) : (
        <>
          {/* Live video with mouth-tracking overlay */}
          <div className="relative overflow-hidden rounded-xl border border-slate-200/70 bg-black dark:border-white/10">
            <video
              ref={videoRef}
              autoPlay
              muted
              playsInline
              className="block w-full"
              style={{ transform: "scaleX(-1)" }}
            />
            <canvas
              ref={canvasRef}
              className="pointer-events-none absolute inset-0 h-full w-full"
              style={{ transform: "scaleX(-1)" }}
            />

            {/* Status pill */}
            <div className="pointer-events-none absolute left-3 top-3">
              <AnimatePresence mode="wait">
                <motion.div
                  key={recording ? "rec" : faceFound ? "ok" : "no"}
                  initial={{ opacity: 0, y: -4 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -4 }}
                  transition={{ duration: 0.18 }}
                  className={`flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-medium backdrop-blur ${
                    recording
                      ? "bg-rose-500/90 text-white"
                      : faceFound
                        ? "bg-emerald-500/90 text-white"
                        : "bg-amber-500/90 text-white"
                  }`}
                >
                  <span
                    className={`h-1.5 w-1.5 rounded-full bg-white ${
                      recording ? "animate-pulse" : ""
                    }`}
                  />
                  {recording ? "Recording" : faceFound ? "Face tracked" : "No face"}
                </motion.div>
              </AnimatePresence>
            </div>
          </div>

          <div className="flex gap-3">
            {!recording ? (
              <motion.button
                whileHover={{ scale: 1.01 }}
                whileTap={{ scale: 0.97 }}
                onClick={startRec}
                className="flex flex-1 items-center justify-center gap-2 rounded-xl bg-rose-600 py-3 font-semibold text-white shadow-md shadow-rose-500/25 transition hover:bg-rose-500"
              >
                <span className="h-2.5 w-2.5 rounded-full bg-white" /> Record
              </motion.button>
            ) : (
              <motion.button
                whileHover={{ scale: 1.01 }}
                whileTap={{ scale: 0.97 }}
                onClick={stopRec}
                className="flex flex-1 items-center justify-center gap-2 rounded-xl bg-slate-700 py-3 font-semibold text-white transition hover:bg-slate-600"
              >
                <span className="h-2.5 w-2.5 rounded-sm bg-white" /> Stop
              </motion.button>
            )}
          </div>
        </>
      )}

      {error && <p className="text-sm text-rose-600 dark:text-rose-400">{error}</p>}

      <AnimatePresence>
        {clipUrl && (
          <motion.div
            key="clip"
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 8 }}
            transition={{ duration: 0.25 }}
            className="space-y-3"
          >
            <p className="text-xs font-medium uppercase tracking-wide text-slate-500 dark:text-slate-400">
              Recorded clip
            </p>
            <video
              src={clipUrl}
              controls
              className="w-full rounded-xl border border-slate-200/70 dark:border-white/10"
              style={{ transform: "scaleX(-1)" }}
            />
            <motion.button
              whileHover={{ scale: 1.01 }}
              whileTap={{ scale: 0.98 }}
              disabled={!clip || loading}
              onClick={() => clip && run(() => predictUpload(clip, "webcam.webm"))}
              className="w-full rounded-xl bg-sky-600 py-3 font-semibold text-white shadow-md shadow-sky-500/20 transition
                         hover:bg-sky-500 disabled:cursor-not-allowed disabled:opacity-50"
            >
              {loading ? "Reading lips…" : "Read lips"}
            </motion.button>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
