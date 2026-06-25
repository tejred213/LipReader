# Deploying LipReader to Hugging Face Spaces

This guide walks you through deploying LipReader as a **Docker Space** on
Hugging Face. The Space serves the React SPA and the FastAPI backend from a
single container; model weights are baked into the image so cold starts are
fast.

> **Time:** ~5 minutes of clicking and pushing, then ~15 minutes of HF build
> time (one-time). Subsequent deploys are much faster thanks to layer caching.

---

## 1. Prerequisites

- A free Hugging Face account: <https://huggingface.co/join>
- `git` and the [`huggingface_hub` CLI] installed locally
  ([install instructions](https://huggingface.co/docs/huggingface_hub/installation)).
- This repo cloned with the latest `main` branch.

---

## 2. Create a new Space

1. Visit **<https://huggingface.co/new-space>**.
2. Pick a name (e.g. `lipreader`).
3. **Select SDK → Docker → "Blank"** (we ship our own Dockerfile).
4. **Hardware:** *CPU basic* (free tier — 2 vCPU / 16 GB RAM). Plenty for
   this model.
5. **Visibility:** Public (or Private — both work).
6. Click **Create Space**. Hugging Face will create an empty git repo at
   `https://huggingface.co/spaces/<your-username>/<space-name>`.

---

## 3. Push your code

Authenticate once (token from <https://huggingface.co/settings/tokens>):

```bash
huggingface-cli login
```

From your clone of this repo:

```bash
# Replace <your-username>/<space-name> with the values from step 2.
git remote add hf https://huggingface.co/spaces/<your-username>/<space-name>

# Push the branch you want deployed (main is fine).
git push hf main
```

That's it. Hugging Face will pick up the `Dockerfile` automatically and start
building.

---

## 4. Watch the first build

Open the **Logs** tab on your Space page. Expect:

| Phase                                  | Approx. duration |
|----------------------------------------|------------------|
| `frontend-build`: `npm ci` + `npm run build` | 1–2 min       |
| `runtime`: system deps + PyTorch + Python deps | 5–8 min       |
| `python app/download_vsr_weights.py` (1.2 GB) | 3–5 min       |
| Image push + container start           | 1–2 min          |
| **Total (first build)**                | **~15 min**      |

When the **App** tab lights up green, you're live. Visit
`https://<your-username>-<space-name>.hf.space` and try the demo.

---

## 5. Custom domain (optional)

Spaces support custom domains on the free tier. From your Space's **Settings**
tab → **Custom domains** → add e.g. `lipreader.yourname.com`. You'll need to
add a CNAME pointing at the Space URL with your DNS provider.

---

## 6. About sleep mode

The free Space sleeps after **~30 minutes of inactivity**. The first visit
after a sleep waits ~20–30 s for the container to wake (model load is fast
because weights are already in the image — just `torch.load` time).

To eliminate this:

- Upgrade hardware to a paid tier (cheapest is CPU upgrade ~$0.03 / hour).
- Or use [HF "Always-on" Spaces](https://huggingface.co/docs/hub/spaces-overview#hardware)
  feature (~$9 / month for permanent uptime).

---

## 7. Re-deploying

Every push to `main` on the `hf` remote triggers a rebuild. Layer caching
means dependency-only changes finish in ~3 minutes; code-only changes (no
`requirements.txt` change) finish in ~1 minute.

```bash
# Iterate locally, then:
git push hf main
```

---

## 8. Configuration (environment variables)

Set these under your Space's **Settings → Variables and secrets** (no rebuild
needed — just restart the Space).

| Variable | Default | Purpose |
|----------|---------|---------|
| `LIPREADER_BEAM_SIZE` | `10` | Decoding beam width. Lower = faster. Output is identical down to ~5 in testing; `1` (greedy) degrades. Raise toward `40` for marginal quality at higher latency. |
| `LIPREADER_ALLOWED_ORIGINS` | local dev origins | Comma-separated CORS origins. Not needed for the default same-origin deploy; set only if you serve the SPA from a different host. |
| `PORT` | `7860` | Port uvicorn binds. HF sets this; leave it alone unless self-hosting. |

---

## 9. Troubleshooting

| Symptom                               | Likely cause / fix                                                                   |
|---------------------------------------|--------------------------------------------------------------------------------------|
| Build fails on `npm ci`               | Check `frontend/package-lock.json` is committed and in sync with `package.json`.     |
| Build fails on `pip install torch …`  | HF runners are x86; the Dockerfile already targets `linux/amd64`. Re-check pin.      |
| Build fails on `download_vsr_weights.py` | Hugging Face Hub rate-limit. Just **Restart Space** — the next attempt usually wins. |
| 500 on `/api/predict` after deploy    | Check Logs — usually a path mismatch. The Dockerfile expects code at `app/` + `vsr/`. |
| Container OOM-killed during inference | You're probably on the Free tier with another big Space taking RAM. Refresh; or upgrade. |
| Health check returns `model_loaded: false` | The weights download in the Dockerfile failed. Inspect build logs.                |

If you get stuck, the Space's `Logs` and `Files` tabs are your friends.

---

## What's actually in the image?

- Built React SPA at `frontend/dist/`
- FastAPI backend at `app/`
- Vendored Auto-AVSR engine at `vsr/`
- Pre-downloaded checkpoints at `vsr/benchmarks/LRS3/`
- `ffmpeg`, OpenCV runtime libs, Python 3.10 + the pinned stack from
  `requirements.txt`

The final image is roughly **3 GB** (PyTorch + weights account for most of
it). The HF free Space has 50 GB of storage, so you're nowhere near the limit.
