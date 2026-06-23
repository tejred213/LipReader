"""Download the open-vocabulary VSR model + language-model weights.

Weights are large (~1.2 GB) and git-ignored. Run once after cloning:

    python app/download_vsr_weights.py
"""

from pathlib import Path
from urllib.request import urlretrieve

REPO_ROOT = Path(__file__).resolve().parent.parent
BENCH = REPO_ROOT / "vsr" / "benchmarks" / "LRS3"

# (url, destination) — same checkpoints the Chaplin project hosts on HF Hub.
FILES = [
    ("https://huggingface.co/Amanvir/LRS3_V_WER19.1/resolve/main/model.json",
     BENCH / "models" / "LRS3_V_WER19.1" / "model.json"),
    ("https://huggingface.co/Amanvir/LRS3_V_WER19.1/resolve/main/model.pth",
     BENCH / "models" / "LRS3_V_WER19.1" / "model.pth"),
    ("https://huggingface.co/Amanvir/lm_en_subword/resolve/main/model.json",
     BENCH / "language_models" / "lm_en_subword" / "model.json"),
    ("https://huggingface.co/Amanvir/lm_en_subword/resolve/main/model.pth",
     BENCH / "language_models" / "lm_en_subword" / "model.pth"),
]


def main() -> None:
    for url, dest in FILES:
        if dest.exists():
            print(f"✓ {dest.relative_to(REPO_ROOT)} (exists)")
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        print(f"↓ {dest.relative_to(REPO_ROOT)} …")
        urlretrieve(url, dest)
    print("Done. VSR weights ready.")


if __name__ == "__main__":
    main()
