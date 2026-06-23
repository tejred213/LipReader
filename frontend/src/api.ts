export interface PredictResult {
  text: string;
  confidence: number;
  mouth_gif: string;
  frame_count: number;
}

async function asError(res: Response): Promise<never> {
  let detail = `Request failed (${res.status})`;
  try {
    const body = await res.json();
    if (body?.detail) detail = body.detail;
  } catch {
    /* ignore non-JSON bodies */
  }
  throw new Error(detail);
}

export async function predictUpload(file: Blob, name = "clip.mp4"): Promise<PredictResult> {
  const form = new FormData();
  form.append("video", file, name);
  const res = await fetch("/api/predict", { method: "POST", body: form });
  if (!res.ok) return asError(res);
  return res.json();
}
