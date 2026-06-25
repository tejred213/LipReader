"""Modern open-vocabulary VSR engine (Auto-AVSR / LRS3) wrapper.

Wraps the vendored `vsr/pipelines` (Apache-2.0, Pingchuan Ma / auto_avsr,
packaged by the Chaplin project) behind a single `predict_video()` call that
returns transcript + confidence + a mouth-crop preview GIF.

Runs on CPU (no GPU required). Requires the torch venv (.venv-vsr).
"""

import base64
import io
import math
import os
import re
import sys
import tempfile
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = APP_DIR.parent
VSR_DIR = REPO_ROOT / "vsr"
CONFIG_TEMPLATE = VSR_DIR / "configs" / "LRS3_V_WER19.1.ini"
CKPT_INDEX = VSR_DIR / "benchmarks" / "LRS3" / "models" / "LRS3_V_WER19.1" / "model.pth"

# Make the vendored engine importable (`import pipelines...`, `import espnet...`).
if str(VSR_DIR) not in sys.path:
    sys.path.insert(0, str(VSR_DIR))

_pipeline = None


def weights_available() -> bool:
    return CKPT_INDEX.exists()


def _install_read_video_shim() -> None:
    """torchvision>=0.20 removed `torchvision.io.read_video`; the vendored
    engine still calls it. Provide a cv2-based replacement (returns RGB)."""
    import cv2
    import numpy as np
    import torch
    import torchvision

    if hasattr(torchvision.io, "read_video"):
        return

    def _read_video(path, pts_unit="sec", **_):
        cap = cv2.VideoCapture(path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        video = torch.from_numpy(np.stack(frames)) if frames else torch.empty(0)
        return video, torch.empty(0), {}

    torchvision.io.read_video = _read_video


# Beam width for CTC/attention decoding. The vendored config ships 40, but
# measured output is byte-for-byte identical down to beam ~5 while decoding is
# meaningfully cheaper — so we default to 10 (free speedup, no accuracy loss)
# and expose an env override so it's tunable from the deploy without a rebuild.
DEFAULT_BEAM_SIZE = 10


def _beam_size() -> int:
    raw = os.environ.get("LIPREADER_BEAM_SIZE", str(DEFAULT_BEAM_SIZE))
    try:
        return max(1, int(raw))
    except ValueError:
        return DEFAULT_BEAM_SIZE


def _absolute_config() -> str:
    """Rewrite the template config's relative checkpoint paths to absolute, apply
    the beam-size override, and write it to a temp file (CWD-independent)."""
    bench = VSR_DIR / "benchmarks"
    text = CONFIG_TEMPLATE.read_text().replace("benchmarks/", f"{bench}/")
    text = re.sub(r"beam_size\s*=\s*\d+", f"beam_size={_beam_size()}", text)
    out = Path(tempfile.gettempdir()) / "lipreader_vsr_config.ini"
    out.write_text(text)
    return str(out)


def get_pipeline():
    """Lazily build and cache the inference pipeline (CPU, MediaPipe)."""
    global _pipeline
    if _pipeline is None:
        _install_read_video_shim()
        from pipelines.pipeline import InferencePipeline

        _pipeline = InferencePipeline(
            _absolute_config(), detector="mediapipe", face_track=True, device="cpu"
        )
    return _pipeline


def _mouth_gif_data_url(crop, fps: int = 16) -> str:
    """Render the (T, H, W) grayscale mouth crop to a base64 GIF data URL."""
    import imageio
    import numpy as np

    frames = np.asarray(crop)
    frames = np.clip(frames, 0, 255).astype(np.uint8)
    if frames.ndim == 3:  # (T, H, W) -> (T, H, W, 3)
        frames = np.repeat(frames[..., None], 3, axis=-1)
    buf = io.BytesIO()
    imageio.mimsave(buf, frames, format="GIF", fps=fps)
    return "data:image/gif;base64," + base64.b64encode(buf.getvalue()).decode()


def predict_video(path: str) -> dict:
    """Transcribe a video file. Returns text, confidence, mouth GIF, frames."""
    import torch
    from espnet.asr.asr_utils import add_results_to_json

    pipe = get_pipeline()

    landmarks = pipe.landmarks_detector(path)
    if landmarks is None:
        raise ValueError("No face/mouth detected. Use a clear, front-facing clip.")

    raw = pipe.dataloader.load_video(path)               # (T, H, W, 3) RGB
    crop = pipe.dataloader.video_process(raw, landmarks)  # (T, 96, 96) grayscale
    if crop is None or len(crop) == 0:
        raise ValueError("Could not isolate the mouth region from this video.")

    crop_tensor = torch.tensor(crop)
    data = pipe.dataloader.video_transform(crop_tensor)

    model_obj = pipe.model
    with torch.no_grad():
        feats = model_obj.model.encode(data.to(model_obj.device))
        nbest = model_obj.beam_search(feats)
        hyp = nbest[0]
        text = (
            add_results_to_json([hyp.asdict()], model_obj.token_list)
            .replace("▁", " ")
            .replace("<eos>", "")
            .strip()
        )
        # Confidence: mean per-token probability from the attention decoder
        # (more calibrated than the combined beam score, which mixes CTC+LM).
        n = max(len(hyp.yseq) - 1, 1)
        scores = getattr(hyp, "scores", {}) or {}
        token_logprob = float(scores.get("decoder", hyp.score))
        confidence = max(0.0, min(1.0, math.exp(token_logprob / n)))

    return {
        "text": text,
        "confidence": round(confidence, 4),
        "mouth_gif": _mouth_gif_data_url(crop),
        "frame_count": int(len(crop)),
    }
