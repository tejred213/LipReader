---
title: LipReader
emoji: 🎬
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: apache-2.0
short_description: Open-vocabulary lip reading from silent video
---

# LipReader

> **Read lips from any silent video — in your browser.**

LipReader is an open-source web app that transcribes natural English speech
from **silent video**. Upload a clip or record yourself, and a modern
Auto-AVSR lip-reading model returns the transcript along with a per-frame
mouth crop and a confidence score.

No audio. No fixed vocabulary. CPU inference in ~2 seconds per clip.

[![Auto-AVSR](https://img.shields.io/badge/Engine-Auto--AVSR-0ea5e9?style=flat-square)](https://github.com/mpc001/auto_avsr)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.12-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-19-61DAFB?style=flat-square&logo=react&logoColor=black)](https://react.dev/)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue?style=flat-square)](LICENSE.txt)

---

## How it works

```
[ Silent video ]
      │
      ▼
1. Detect — MediaPipe FaceLandmarker finds and tracks the speaker's face,
              live in the browser. A glowing crop overlay shows users
              exactly what the model will see.
      │
      ▼
2. Crop   — A stable 75-frame mouth region is extracted server-side,
              normalised, and aligned to the model's expected geometry.
      │
      ▼
3. Read   — Auto-AVSR (3D-CNN → Conformer → CTC + attention) decodes
              the visual signal into open-vocabulary English text.
      │
      ▼
[ Transcript + confidence + mouth-crop preview ]
```

The model is the [`LRS3_V_WER19.1`](https://huggingface.co/Amanvir/LRS3_V_WER19.1)
checkpoint (≈19% WER on LRS3), with a subword RNN-LM beam search for cleaner
decoding.

---

## Architecture

```
                      │  http://localhost:5173  (dev)
 [ React + Vite SPA ] ┤                         │
                      │  /api/*  ── proxy ──>   │  [ FastAPI + PyTorch ]
                      │                         │      └─ Auto-AVSR (vsr/)
                      │                         │      └─ MediaPipe
```

In production both the SPA and the API are served by the **same FastAPI
process** at the same origin — no CORS needed, deploys as a single container.

---

## Running locally

You need Python 3.10 and Node 20.

```bash
# 1. Python venv + backend deps
python3.10 -m venv .venv-vsr
.venv-vsr/bin/pip install -r requirements.txt

# 2. Download the model weights (~1.2 GB, one-time)
.venv-vsr/bin/python app/download_vsr_weights.py

# 3. Frontend deps
npm --prefix frontend install

# 4. Run — two terminals:
#    Terminal A (backend):
.venv-vsr/bin/uvicorn --app-dir app api:app --port 8000

#    Terminal B (frontend dev server, proxies /api → :8000):
npm --prefix frontend run dev
```

Then open **http://localhost:5173**.

### Headless CLI

```bash
.venv-vsr/bin/python app/predict.py path/to/clip.mp4
```

---

## Deployment

The repo ships a multi-stage `Dockerfile` sized for **Hugging Face Spaces**
(Docker SDK). One container hosts both the SPA and the API; model weights are
baked into the image at build time so cold starts are fast (~5 s).

See [**DEPLOYMENT.md**](DEPLOYMENT.md) for the full step-by-step.

Quick path:

```bash
# After creating an empty Docker Space at https://huggingface.co/new-space
git remote add hf https://huggingface.co/spaces/<your-username>/<space-name>
git push hf main
```

---

## Tech stack

| Layer    | Pieces                                                                       |
|----------|------------------------------------------------------------------------------|
| Frontend | React 19 · Vite · TailwindCSS 4 · framer-motion · Lenis · MediaPipe Tasks    |
| Backend  | FastAPI · PyTorch (CPU) · MediaPipe · vendored Auto-AVSR + espnet            |
| Model    | LRS3_V_WER19.1 (≈19 % WER) + subword RNN-LM, beam-search decoding            |
| Build    | Multi-stage Docker (Node 20-alpine → Python 3.10-slim), targets HF Spaces    |

---

## Project layout

```
app/                  # FastAPI backend
  api.py              # HTTP API + static SPA mount
  vsr_infer.py        # Wrapper around the vendored engine
  download_vsr_weights.py  # One-shot fetch from HF Hub
  predict.py          # Headless CLI

vsr/                  # Vendored Auto-AVSR engine (Apache-2.0)
  pipelines/          # Mouth detection + dataloader + model
  espnet/             # Trimmed espnet (transformer + beam search)
  configs/            # Model configuration
  benchmarks/         # Weights (gitignored; ~1.2 GB)

frontend/             # React + Vite SPA
  src/components/     # Hero, HowItWorks, Demo, Features, Footer, …
  src/lib/            # mouthTracker, useSmoothScroll

Dockerfile            # Production image (see DEPLOYMENT.md)
requirements.txt      # Python deps (pinned)
```

---

## Attribution

This project stands on the shoulders of open research and tooling:

- **[Auto-AVSR](https://github.com/mpc001/auto_avsr)** — Pingchuan Ma et al.,
  Imperial College London. The vendored visual speech recognition engine and
  pretrained checkpoint. (Apache-2.0)
- **[Chaplin](https://github.com/amanvirparhar/chaplin)** — the packaging of
  Auto-AVSR + MediaPipe detector used here is adapted from this real-time
  silent-speech project. (Apache-2.0)
- **[MediaPipe](https://github.com/google-ai-edge/mediapipe)** — face
  landmark detection, in-browser and server-side. (Apache-2.0)

Vendored sources retain their original Apache-2.0 copyright headers; see
[`vsr/NOTICE.md`](vsr/NOTICE.md).

## License

[Apache-2.0](LICENSE.txt).
