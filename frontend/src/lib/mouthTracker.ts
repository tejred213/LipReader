/**
 * In-browser mouth tracker using MediaPipe FaceLandmarker.
 *
 * Loads the model once (cached), exposes a `detectMouthBox(video, now)` that
 * returns a normalized lip bounding box (or null when no face is found).
 * The box is in video-pixel coordinates: { x, y, width, height }.
 */

import {
  FaceLandmarker,
  FilesetResolver,
  type NormalizedLandmark,
} from "@mediapipe/tasks-vision";

// MediaPipe lip-landmark indices (outer + inner ring of the mouth).
const LIP_INDICES = [
  61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
  78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
  185, 40, 39, 37, 0, 267, 269, 270, 409,
];

export interface MouthBox {
  x: number;
  y: number;
  width: number;
  height: number;
}

let landmarkerPromise: Promise<FaceLandmarker> | null = null;

async function getLandmarker(): Promise<FaceLandmarker> {
  if (!landmarkerPromise) {
    landmarkerPromise = (async () => {
      const fileset = await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.35/wasm",
      );
      return FaceLandmarker.createFromOptions(fileset, {
        baseOptions: {
          modelAssetPath:
            "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
          delegate: "GPU",
        },
        runningMode: "VIDEO",
        numFaces: 1,
      });
    })();
  }
  return landmarkerPromise;
}

/** Warm up the model so the first detection isn't slow. */
export function warmupMouthTracker(): void {
  void getLandmarker();
}

/**
 * Run one detection. `now` should be a monotonic ms timestamp (e.g.
 * performance.now()) — MediaPipe requires it to be strictly increasing.
 */
export async function detectMouthBox(
  video: HTMLVideoElement,
  now: number,
): Promise<MouthBox | null> {
  if (video.readyState < 2 || video.videoWidth === 0) return null;

  const lm = await getLandmarker();
  const result = lm.detectForVideo(video, now);
  const face: NormalizedLandmark[] | undefined = result.faceLandmarks?.[0];
  if (!face) return null;

  let minX = 1,
    minY = 1,
    maxX = 0,
    maxY = 0;
  for (const i of LIP_INDICES) {
    const p = face[i];
    if (!p) continue;
    if (p.x < minX) minX = p.x;
    if (p.x > maxX) maxX = p.x;
    if (p.y < minY) minY = p.y;
    if (p.y > maxY) maxY = p.y;
  }
  if (maxX <= minX || maxY <= minY) return null;

  // Pad horizontally to roughly match the backend's GRID-calibrated crop
  // (so the on-screen box reflects "what the model will see").
  const lipW = maxX - minX;
  const cx = (minX + maxX) / 2 + lipW * -0.05; // slight leftward bias
  const cy = (minY + maxY) / 2;
  const cropW = lipW * 2.6; // wider than the bare lips for context
  const cropH = cropW * (46 / 140); // GRID aspect ratio

  const W = video.videoWidth;
  const H = video.videoHeight;
  return {
    x: Math.max(0, (cx - cropW / 2) * W),
    y: Math.max(0, (cy - cropH / 2) * H),
    width: Math.min(W, cropW * W),
    height: Math.min(H, cropH * H),
  };
}
