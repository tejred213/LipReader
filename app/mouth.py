"""Detect and crop the mouth region from an arbitrary video using MediaPipe.

The GRID model expects a 75-frame sequence of 46x140 grayscale mouth crops.
For uploaded videos we don't have GRID's fixed crop, so we:
  1. locate the lips per frame with MediaPipe FaceMesh,
  2. crop a 140:46 box around them (with GRID-like context),
  3. resize to 46x140, grayscale,
  4. uniformly resample to exactly 75 frames,
  5. normalize identically to training (see utils.normalize_frames).
"""

from typing import List, Tuple

import cv2
import numpy as np
import tensorflow as tf

import config
from utils import normalize_frames

# A representative set of MediaPipe FaceMesh lip landmark indices.
LIP_LANDMARKS = [
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,  # outer lower
    308, 324, 318, 402, 317, 14, 87, 178, 88, 95,        # inner
    185, 40, 39, 37, 0, 267, 269, 270, 409,              # outer upper
    78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308,    # inner ring
]

# Target aspect ratio (width / height) of the crop, matching GRID (140/46).
_ASPECT = config.FRAME_WIDTH / config.FRAME_HEIGHT
# Crop width as a multiple of the detected lip width. Calibrated against GRID,
# where the working crop (140px) is ~3.6x the lip-landmark width (~39px).
_HORIZONTAL_PADDING = 3.6
# Horizontal shift of the crop center, as a fraction of lip width (GRID frames
# the mouth slightly left of the lip centroid). Negative = shift left.
_HORIZONTAL_BIAS = -0.26


def _lip_bbox(landmarks, w: int, h: int) -> Tuple[int, int, int, int]:
    """Pixel bbox (x1, y1, x2, y2) tightly around the lip landmarks."""
    xs = [landmarks[i].x * w for i in LIP_LANDMARKS]
    ys = [landmarks[i].y * h for i in LIP_LANDMARKS]
    return int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))


def _fixed_box(measurements, frame_w: int, frame_h: int):
    """Build ONE crop box from per-frame lip measurements (median for stability).

    Like GRID, the same box is applied to every frame so the zoom/position stays
    constant across the sequence. `measurements` is a list of (cx, cy, lip_w).
    """
    cx = float(np.median([m[0] for m in measurements]))
    cy = float(np.median([m[1] for m in measurements]))
    lip_w = float(np.median([m[2] for m in measurements]))

    # GRID frames the crop slightly left of the lip centroid (~0.26*lip_w).
    cx += _HORIZONTAL_BIAS * lip_w

    crop_w = lip_w * _HORIZONTAL_PADDING
    crop_h = crop_w / _ASPECT

    nx1 = int(max(cx - crop_w / 2, 0))
    ny1 = int(max(cy - crop_h / 2, 0))
    nx2 = int(min(cx + crop_w / 2, frame_w))
    ny2 = int(min(cy + crop_h / 2, frame_h))
    return nx1, ny1, nx2, ny2


def extract_mouth_frames(path: str, target_frames: int = config.FRAME_COUNT) -> tf.Tensor:
    """Return a normalized (target_frames, 46, 140, 1) tensor from any video.

    Raises ValueError if no face/mouth is found in any frame.
    """
    import mediapipe as mp

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"Could not open video: {path}")

    raw_frames: List[np.ndarray] = []
    measurements: List[Tuple[float, float, float]] = []

    with mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False, max_num_faces=1, refine_landmarks=True,
        min_detection_confidence=0.5,
    ) as face_mesh:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            raw_frames.append(frame)
            h, w = frame.shape[:2]
            result = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if result.multi_face_landmarks:
                x1, y1, x2, y2 = _lip_bbox(result.multi_face_landmarks[0].landmark, w, h)
                measurements.append(((x1 + x2) / 2, (y1 + y2) / 2, max(x2 - x1, 1)))
    cap.release()

    if not raw_frames or not measurements:
        raise ValueError(
            "No face/mouth detected in the video. Use a clear, front-facing clip."
        )

    h, w = raw_frames[0].shape[:2]
    bx1, by1, bx2, by2 = _fixed_box(measurements, w, h)

    crops: List[np.ndarray] = []
    for frame in raw_frames:
        gray = cv2.cvtColor(frame[by1:by2, bx1:bx2], cv2.COLOR_BGR2GRAY)
        crops.append(cv2.resize(gray, (config.FRAME_WIDTH, config.FRAME_HEIGHT)))

    crops = _resample(crops, target_frames)
    # Shape (T, H, W) -> (T, H, W, 1) as uint8, then normalize like training.
    frames = tf.convert_to_tensor(np.stack(crops)[..., np.newaxis], dtype=tf.uint8)
    return normalize_frames(frames)


def _resample(crops: List[np.ndarray], target: int) -> List[np.ndarray]:
    """Uniformly resample a list of frames to exactly `target` length."""
    n = len(crops)
    if n == target:
        return crops
    if n < target:  # pad by repeating the last frame
        return crops + [crops[-1]] * (target - n)
    idx = np.linspace(0, n - 1, target).round().astype(int)  # uniform subsample
    return [crops[i] for i in idx]
