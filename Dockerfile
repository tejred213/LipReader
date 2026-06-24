# syntax=docker/dockerfile:1.6
# =============================================================================
#  LipReader — production container.
#
#  Sized for Hugging Face Spaces (Docker SDK). Two stages:
#    1. Build the React/Vite frontend with Node.
#    2. Install Python deps, copy code + built SPA, bake the ~1.2 GB VSR
#       checkpoints into the image, run FastAPI on the HF default port (7860).
#
#  The same FastAPI process serves both the API (/api/*) and the SPA
#  (everything else, from `frontend/dist`) — single origin, no CORS in prod.
# =============================================================================

# -----------------------------------------------------------------------------
# Stage 1: build the React frontend
# -----------------------------------------------------------------------------
FROM node:20-alpine AS frontend-build

WORKDIR /build/frontend

# Install deps from the lockfile (deterministic + cache-friendly).
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund

# Build the SPA.
COPY frontend/ ./
RUN npm run build

# -----------------------------------------------------------------------------
# Stage 2: Python runtime with the VSR engine + FastAPI
# -----------------------------------------------------------------------------
FROM python:3.10-slim AS runtime

# System dependencies:
#   - ffmpeg          : video transcoding for uploaded clips
#   - libgl1/libglib  : runtime libs OpenCV needs even in headless builds
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Run as a non-root user with UID 1000 (Hugging Face Spaces convention).
RUN useradd -m -u 1000 user
USER user

ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # HF Spaces always sends traffic to port 7860 by default.
    PORT=7860

WORKDIR /home/user/app

# ---- Python dependencies (cached layer) ------------------------------------
# Install the CPU build of torch from the official wheel index so we don't drag
# in CUDA — same machine-independent stack used in `requirements.txt` works
# fine on linux/amd64 (HF runners) as well as Apple Silicon.
COPY --chown=user:user requirements.txt .
RUN pip install --user --upgrade pip \
    && pip install --user --extra-index-url https://download.pytorch.org/whl/cpu \
        -r requirements.txt

# ---- Application code -------------------------------------------------------
COPY --chown=user:user app/  ./app/
COPY --chown=user:user vsr/  ./vsr/

# Built SPA from stage 1 — FastAPI serves this at "/".
COPY --chown=user:user --from=frontend-build /build/frontend/dist ./frontend/dist

# ---- Bake the model weights (1.2 GB) into the image ------------------------
# Pre-downloading at build time makes cold starts fast: container boot just
# loads the cached .pth from disk (~5 s) instead of pulling 1.2 GB over HTTPS
# every restart (~90 s).
RUN python app/download_vsr_weights.py

EXPOSE 7860

# Bind to 0.0.0.0 so HF's reverse proxy can reach us. PORT env var is honored
# in case HF sends a different one.
CMD ["sh", "-c", "uvicorn --app-dir app api:app --host 0.0.0.0 --port ${PORT:-7860}"]
