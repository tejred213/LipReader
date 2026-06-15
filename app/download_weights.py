"""Download the fully-trained pretrained LipNet weights.

The repo ships no weights (they're large and git-ignored). This fetches the
canonical pretrained checkpoint and places it under models/ where config.py
expects it. Run once after cloning:

    python app/download_weights.py
"""

import zipfile
from pathlib import Path

import gdown

import config

# Pretrained LipNet weights (TF checkpoint format), same source as the notebook.
WEIGHTS_DRIVE_ID = "1vWscXs4Vt0a_1IH1-ct2TCgXAZT-N3_Y"


def main() -> None:
    models_dir = config.REPO_ROOT / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    if (models_dir / "checkpoint.index").exists():
        print(f"Weights already present in {models_dir}. Nothing to do.")
        return

    zip_path = models_dir / "checkpoints.zip"
    print("Downloading pretrained weights…")
    gdown.download(id=WEIGHTS_DRIVE_ID, output=str(zip_path), quiet=False)

    print("Extracting…")
    with zipfile.ZipFile(zip_path) as z:
        for member in z.namelist():
            # Skip the macOS resource-fork junk in the archive.
            if member.startswith("__MACOSX") or "/._" in member:
                continue
            z.extract(member, models_dir)
    zip_path.unlink()

    if (models_dir / "checkpoint.index").exists():
        print(f"Done. Weights ready in {models_dir}")
    else:
        raise RuntimeError("Extraction finished but checkpoint.index is missing.")


if __name__ == "__main__":
    main()
