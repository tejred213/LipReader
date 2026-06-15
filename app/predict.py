"""Command-line lip-reading inference.

Examples:
    # A GRID sample by id (looks under the configured data dir):
    python predict.py --sample bbal6n

    # Any video file:
    python predict.py path/to/video.mp4
"""

import argparse

import tensorflow as tf

from modelutil import load_model
from utils import load_video, load_sample, num_to_char


def decode(model, video: tf.Tensor) -> str:
    yhat = model.predict(tf.expand_dims(video, axis=0), verbose=0)
    decoded = tf.keras.backend.ctc_decode(
        yhat, input_length=[yhat.shape[1]], greedy=True
    )[0][0].numpy()
    return tf.strings.reduce_join(num_to_char(decoded)).numpy().decode("utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("video", nargs="?", help="Path to a video file")
    group.add_argument("--sample", help="GRID sample id (e.g. bbal6n)")
    args = parser.parse_args()

    model = load_model()

    if args.sample:
        video, alignment = load_sample(args.sample)
        truth = tf.strings.reduce_join(num_to_char(alignment)).numpy().decode("utf-8")
        print(f"Ground truth : {truth.strip()}")
    else:
        video = load_video(args.video)

    print(f"Prediction   : {decode(model, video).strip()}")


if __name__ == "__main__":
    main()
