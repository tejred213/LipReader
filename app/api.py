"""FastAPI backend for LipReader.

Open-vocabulary visual speech recognition (Auto-AVSR / LRS3) behind a small
HTTP API the React frontend talks to. Runs on CPU in the torch venv (.venv-vsr).

Local dev:
    uvicorn api:app --reload --port 8000      (from the app/ directory)

Production (Docker / HF Spaces):
    The same process serves both the API (/api/*) and the built SPA
    (`frontend/dist`) — single origin, no CORS needed. Port comes from the
    PORT env var (default 7860 on Hugging Face Spaces).
"""

import os
import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

import vsr_infer

REPO_ROOT = Path(__file__).resolve().parent.parent

app = FastAPI(title="LipReader API")

# Allowed CORS origins. The SPA in production is served from the SAME origin
# as the API, so CORS is a no-op there. In local dev the Vite dev server runs
# on :5173 and needs to be allowed. Override via env for custom deploys.
_DEFAULT_ORIGINS = "http://localhost:5173,http://127.0.0.1:5173"
_ALLOWED_ORIGINS = [
    o.strip()
    for o in os.environ.get("LIPREADER_ALLOWED_ORIGINS", _DEFAULT_ORIGINS).split(",")
    if o.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

_TMP = Path(tempfile.gettempdir()) / "lipreader_api"
_TMP.mkdir(exist_ok=True)


@app.on_event("startup")
def _warm_up() -> None:
    """Load the model once at startup so the first request isn't slow."""
    if vsr_infer.weights_available():
        vsr_infer.get_pipeline()


@app.get("/api/health")
def health() -> dict:
    return {"ok": True, "model_loaded": vsr_infer.weights_available()}


@app.post("/api/predict")
async def predict_upload(video: UploadFile = File(...)) -> dict:
    """Transcribe an uploaded or webcam-recorded clip."""
    if not vsr_infer.weights_available():
        raise HTTPException(503, "Model weights not available. Run download_vsr_weights.py.")

    suffix = Path(video.filename or "clip.mp4").suffix or ".mp4"
    tmp_path = _TMP / f"upload_{next(tempfile._get_candidate_names())}{suffix}"
    with tmp_path.open("wb") as f:
        shutil.copyfileobj(video.file, f)

    try:
        return vsr_infer.predict_video(str(tmp_path))
    except ValueError as e:  # no face/mouth detected
        raise HTTPException(422, str(e))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, f"Could not process video: {e}")
    finally:
        tmp_path.unlink(missing_ok=True)


# Serve the built frontend (frontend/dist) in production, if present.
_DIST = REPO_ROOT / "frontend" / "dist"
if _DIST.exists():
    app.mount("/", StaticFiles(directory=str(_DIST), html=True), name="frontend")
