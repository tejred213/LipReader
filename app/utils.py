"""Video / alignment loading and the character<->integer vocabulary.

Two entry points:
  * `load_video(path)`           -> frames only (use this for inference)
  * `load_sample(name)`          -> frames + ground-truth alignment (for eval)
"""

import os
from typing import List, Tuple

import cv2
import tensorflow as tf

import config

# Character <-> integer mappings (shared by training, inference, decoding).
char_to_num = tf.keras.layers.StringLookup(vocabulary=config.VOCAB, oov_token="")
num_to_char = tf.keras.layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True
)


def load_video(path: str, crop=config.GRID_CROP) -> tf.Tensor:
    """Read a video file into a normalized grayscale frame tensor.

    Returns a float32 tensor of shape (num_frames, HEIGHT, WIDTH, 1),
    mean-centered and divided by the std deviation (as the model expects).

    `crop` is a fixed (top, bottom, left, right) mouth region applied to every
    frame, matching how the model was trained on the GRID corpus. Pass
    crop=None for already-cropped input (e.g. dynamic mouth detection).
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"Could not open video: {path}")

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:  # end of stream / unreadable frame
            break
        frame = tf.image.rgb_to_grayscale(frame)
        if crop is not None:
            top, bottom, left, right = crop
            frame = frame[top:bottom, left:right, :]
        else:
            frame = tf.image.resize(frame, (config.FRAME_HEIGHT, config.FRAME_WIDTH))
        frames.append(frame)
    cap.release()

    if not frames:
        raise ValueError(f"No frames could be read from: {path}")

    return normalize_frames(frames)


def normalize_frames(frames) -> tf.Tensor:
    """Apply the model's training normalization to a stack of uint8 frames.

    NOTE: the subtraction happens on uint8 data, so values below the mean wrap
    around (post-norm mean ≈4.6, not 0). This is intentional — it reproduces the
    exact preprocessing the weights were trained with. Do not "fix" it to float
    normalization or the trained model outputs blanks.
    """
    frames = tf.stack(frames) if isinstance(frames, (list, tuple)) else frames
    mean = tf.math.reduce_mean(frames)
    std = tf.math.reduce_std(tf.cast(frames, tf.float32))
    return tf.cast((frames - mean), tf.float32) / std


def load_alignments(path: str) -> tf.Tensor:
    """Parse a GRID `.align` file into a tensor of character indices."""
    with open(path, "r") as f:
        lines = f.readlines()
    tokens = []
    for line in lines:
        line = line.split()
        if line[2] != "sil":
            tokens = [*tokens, " ", line[2]]
    return char_to_num(
        tf.reshape(tf.strings.unicode_split(tokens, input_encoding="UTF-8"), (-1))
    )[1:]


def load_sample(name: str) -> Tuple[tf.Tensor, tf.Tensor]:
    """Load frames + ground-truth alignment for a GRID sample by name.

    `name` may be a bare id ("bbal6n"), a filename, or a full path; only the
    stem is used to locate the matching .mpg and .align files under the
    configured data directory.
    """
    if isinstance(name, bytes):
        name = name.decode()
    file_name = os.path.splitext(os.path.basename(name))[0]

    video_path = config.VIDEO_DIR / f"{file_name}.mpg"
    alignment_path = config.ALIGN_DIR / f"{file_name}.align"

    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    if not alignment_path.exists():
        raise FileNotFoundError(f"Alignment file not found: {alignment_path}")

    return load_video(str(video_path)), load_alignments(str(alignment_path))
