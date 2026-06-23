# Vendored VSR engine — attribution

The code under `vsr/pipelines/` and `vsr/espnet/` and the config in
`vsr/configs/` is vendored (lightly trimmed: the RetinaFace detector was
removed so only the MediaPipe path remains) from:

- **Auto-AVSR** — Pingchuan Ma et al., Imperial College London
  https://github.com/mpc001/auto_avsr  (Apache License 2.0)
- Packaging / MediaPipe detector adapted from **Chaplin**
  https://github.com/amanvirparhar/chaplin

The pretrained checkpoints (downloaded by `app/download_vsr_weights.py`, not
committed) are the `LRS3_V_WER19.1` visual model and `lm_en_subword` language
model hosted on the Hugging Face Hub.

All vendored source retains its original Apache-2.0 copyright headers.
