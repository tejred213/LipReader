"""LipReader / LipBuddy — Streamlit demo.

Two tabs:
  * GRID samples — run the trained model on the dataset it was trained on.
  * Upload your own — detect the mouth in any video and attempt to read it.

Paths and model dims are resolved in `config.py`.
"""

import subprocess
import tempfile
from pathlib import Path

import imageio
import numpy as np
import streamlit as st
import tensorflow as tf

import config
from modelutil import load_model
from utils import load_sample, num_to_char

st.set_page_config(page_title="LipBuddy", layout="wide")


@st.cache_resource
def get_model():
    """Load the model once and cache it across reruns."""
    return load_model()


def frames_to_gif(video: tf.Tensor, path: str, fps: int = 10) -> None:
    """Render the preprocessed (grayscale, normalized) frames to a GIF."""
    frames = video.numpy()
    frames = frames - frames.min()
    frames = (frames / (frames.max() + 1e-8) * 255).astype(np.uint8)
    if frames.ndim == 4 and frames.shape[-1] == 1:
        frames = np.repeat(frames, 3, axis=-1)
    imageio.mimsave(path, frames, fps=fps)


def transcode_to_mp4(src: str) -> bytes:
    """Re-encode a video to browser-playable H.264 mp4 and return its bytes."""
    out_path = Path(tempfile.gettempdir()) / "lipreader_preview.mp4"
    subprocess.run(
        ["ffmpeg", "-i", src, "-vcodec", "libx264", "-y", str(out_path)],
        check=True,
        capture_output=True,
    )
    return out_path.read_bytes()


def predict(video: tf.Tensor) -> str:
    """Run the model on a single video tensor and decode to text."""
    model = get_model()
    yhat = model.predict(tf.expand_dims(video, axis=0), verbose=0)
    decoded = tf.keras.backend.ctc_decode(
        yhat, input_length=[yhat.shape[1]], greedy=True
    )[0][0].numpy()
    return tf.strings.reduce_join(num_to_char(decoded)).numpy().decode("utf-8")


def show_prediction(video: tf.Tensor, gif_name: str) -> None:
    """Shared right-column view: mouth GIF + decoded text."""
    st.info("What the model sees (cropped, grayscale, normalized)")
    gif_path = Path(tempfile.gettempdir()) / gif_name
    frames_to_gif(video, str(gif_path))
    st.image(str(gif_path), width=400)

    if not config.MODEL_PATH:
        st.warning("Model weights unavailable — run `python app/download_weights.py`.")
        return
    with st.spinner("Reading lips…"):
        prediction = predict(video)
    st.subheader("Prediction")
    st.success(prediction if prediction.strip() else "(no text decoded)")


# --- Sidebar --------------------------------------------------------------
with st.sidebar:
    st.title("LipBuddy")
    st.info("A deep-learning lip-reading demo based on the LipNet model.")
    if config.MODEL_PATH:
        st.success("Model weights loaded.")
    else:
        st.error("No model weights found. Run `python app/download_weights.py`.")

st.title("LipBuddy — reading lips from video")

tab_grid, tab_upload = st.tabs(["GRID samples", "Upload your own"])

# --- Tab 1: GRID sample demo ---------------------------------------------
with tab_grid:
    if not config.VIDEO_DIR.exists():
        st.error(
            f"Data directory not found: {config.VIDEO_DIR}\n\n"
            "Set LIPREADER_DATA_DIR to the folder containing `s1/`."
        )
    else:
        options = sorted(p.name for p in config.VIDEO_DIR.glob("*.mpg"))
        if not options:
            st.error(f"No .mpg videos found in {config.VIDEO_DIR}")
        else:
            selected_video = st.selectbox("Choose a sample video", options)
            file_path = config.VIDEO_DIR / selected_video
            col1, col2 = st.columns(2)
            with col1:
                st.info("Original video")
                try:
                    st.video(transcode_to_mp4(str(file_path)))
                except subprocess.CalledProcessError:
                    st.error("ffmpeg failed to transcode this video.")
            with col2:
                video, _ = load_sample(selected_video)
                show_prediction(video, "grid_frames.gif")

# --- Tab 2: Upload your own ----------------------------------------------
with tab_upload:
    st.warning(
        "⚠️ This model was trained on a **single speaker** (GRID corpus), so "
        "accuracy on arbitrary videos is limited — expect the opening words to "
        "land and the rest to drift. Best results: a clear, front-facing clip of "
        "a few seconds. This demonstrates the full detect→crop→read pipeline."
    )
    uploaded = st.file_uploader(
        "Upload a video", type=["mp4", "mov", "avi", "mpg", "mpeg", "webm"]
    )
    if uploaded is not None:
        suffix = Path(uploaded.name).suffix or ".mp4"
        tmp = Path(tempfile.gettempdir()) / f"lipreader_upload{suffix}"
        tmp.write_bytes(uploaded.getbuffer())

        col1, col2 = st.columns(2)
        with col1:
            st.info("Your video")
            st.video(uploaded)
        with col2:
            try:
                from mouth import extract_mouth_frames

                with st.spinner("Detecting mouth…"):
                    video = extract_mouth_frames(str(tmp))
                show_prediction(video, "upload_frames.gif")
            except ValueError as e:
                st.error(str(e))
            except Exception as e:  # noqa: BLE001 — surface any pipeline failure
                st.error(f"Could not process video: {e}")
