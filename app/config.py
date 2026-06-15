"""Central configuration for LipReader.

All paths and model dimensions live here so nothing is hardcoded to one
machine. Paths can be overridden with environment variables, which is handy
when the data/weights live outside the repo (e.g. when running from a git
worktree):

    LIPREADER_DATA_DIR    -> directory containing `s1/` and `alignments/s1/`
    LIPREADER_MODEL_PATH  -> path to the `*.weights.h5` checkpoint
"""

import os
from pathlib import Path

# app/ directory and the repo root (its parent).
APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = APP_DIR.parent


def _first_existing(*candidates) -> Path | None:
    """Return the first candidate path that exists, else None."""
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return Path(candidate)
    return None


# --- Data locations -------------------------------------------------------
DATA_DIR = Path(os.environ.get("LIPREADER_DATA_DIR", REPO_ROOT / "data"))
VIDEO_DIR = DATA_DIR / "s1"
ALIGN_DIR = DATA_DIR / "alignments" / "s1"

# --- Model weights --------------------------------------------------------
# Prefer the fully-trained pretrained TF checkpoint (models/checkpoint.*) over
# any partially-trained *.weights.h5. A TF checkpoint is referenced by its
# prefix (no extension); we detect it via the companion .index file.
# Override everything with LIPREADER_MODEL_PATH.
def _resolve_model_path():
    env = os.environ.get("LIPREADER_MODEL_PATH")
    if env:
        return env
    ckpt_prefix = REPO_ROOT / "models" / "checkpoint"
    if (REPO_ROOT / "models" / "checkpoint.index").exists():
        return str(ckpt_prefix)  # TF checkpoint (pretrained, fully trained)
    found = _first_existing(
        REPO_ROOT / "models" / "checkpoint.weights.h5",
        REPO_ROOT / "checkpoints1.weights.h5",
        APP_DIR / "checkpoints1.weights.h5",
    )
    return str(found) if found else None


MODEL_PATH = _resolve_model_path()

# --- Model / preprocessing constants -------------------------------------
# The character set the model was trained to emit (GRID corpus).
VOCAB = [c for c in "abcdefghijklmnopqrstuvwxyz'?!123456789 "]

# Input tensor shape the network expects: (frames, height, width, channels).
FRAME_COUNT = 75
FRAME_HEIGHT = 46
FRAME_WIDTH = 140

# Fixed mouth-region crop for the GRID corpus, matching how the model was
# trained: frame[top:bottom, left:right] == frame[190:236, 80:220] -> 46x140.
# (Arbitrary uploaded videos need dynamic mouth detection instead; see Phase 2.)
GRID_CROP = (190, 236, 80, 220)  # (top, bottom, left, right)

# Number of output classes = vocab + 1 (CTC blank).
NUM_CLASSES = len(VOCAB) + 2  # StringLookup adds an OOV token; matches Dense(41)
