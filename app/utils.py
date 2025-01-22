import tensorflow as tf
from typing import List
import cv2
import os

vocab = [x for x in "abcdefghijklmnopqrstuvwxyz'?!123456789 "]
char_to_num = tf.keras.layers.StringLookup(vocabulary=vocab, oov_token="")
# Mapping integers back to original characters
num_to_char = tf.keras.layers.StringLookup(vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True)


def load_video(path: str) -> List[float]:
    cap = cv2.VideoCapture(path)
    frames = []
    for _ in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
        ret, frame = cap.read()
        frame = cv2.resize(frame, (140, 46))  # Resize to match input shape
        frame = tf.image.rgb_to_grayscale(frame)
        frames.append(frame)
    cap.release()

    frames = tf.stack(frames)
    mean = tf.math.reduce_mean(frames)
    std = tf.math.reduce_std(tf.cast(frames, tf.float32))
    return tf.cast((frames - mean), tf.float32) / std


def load_alignments(path: str) -> List[str]:
    with open(path, "r") as f:
        lines = f.readlines()
    tokens = []
    for line in lines:
        line = line.split()
        if line[2] != "sil":
            tokens = [*tokens, " ", line[2]]
    return char_to_num(tf.reshape(tf.strings.unicode_split(tokens, input_encoding="UTF-8"), (-1)))[1:]


def load_data(path: str):
    path = bytes.decode(path.numpy())
    file_name = os.path.splitext(os.path.basename(path))[0]
    video_path = os.path.join(os.getcwd(), "data", "s1", f"{file_name}.mpg")
    alignment_path = os.path.join(os.getcwd(), "data", "alignments", "s1", f"{file_name}.align")

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    if not os.path.exists(alignment_path):
        raise FileNotFoundError(f"Alignment file not found: {alignment_path}")

    frames = load_video(video_path)
    alignments = load_alignments(alignment_path)
    return frames, alignments

# def load_data(path: str):
#     # Handle both string and Tensor inputs
#     if isinstance(path, bytes):  # TensorFlow byte input
#         path = path.decode()
#     elif not isinstance(path, str):  # Invalid input type
#         raise TypeError("Expected a string or bytes, but got: {}".format(type(path)))

#     file_name = os.path.splitext(os.path.basename(path))[0]
#     video_path = os.path.join(os.getcwd(), "data", "s1", f"{file_name}.mpg")
#     alignment_path = os.path.join(os.getcwd(), "data", "alignments", "s1", f"{file_name}.align")

#     if not os.path.exists(video_path):
#         raise FileNotFoundError(f"Video file not found: {video_path}")
#     if not os.path.exists(alignment_path):
#         raise FileNotFoundError(f"Alignment file not found: {alignment_path}")

#     frames = load_video(video_path)
#     alignments = load_alignments(alignment_path)
#     return frames, alignments


# def load_data(path: str):
#     # Handle both string and Tensor inputs
#     if isinstance(path, bytes):  # TensorFlow byte input
#         path = path.decode()  # Convert bytes to string
#     elif not isinstance(path, str):  # Invalid input type
#         raise TypeError(f"Expected a string or bytes, but got: {type(path)}")

#     # Debugging the received path
#     print(f"Received path: {path}")

#     # Extract file name without extension
#     file_name = os.path.splitext(os.path.basename(path))[0]
    
#     # Construct the video and alignment file paths
#     video_path = os.path.join(os.getcwd(), "data", "s1", f"{file_name}.mpg")
#     alignment_path = os.path.join(os.getcwd(), "data", "alignments", "s1", f"{file_name}.align")
#     if not os.path.exists(alignment_path):
#         print(f"Alignment file not found: {alignment_path}")
#         return None, None

#     # Debugging the constructed paths
#     print(f"Video path: {video_path}")
#     print(f"Alignment path: {alignment_path}")

#     # Check if the video file exists
#     if not os.path.exists(video_path):
#         raise FileNotFoundError(f"Video file not found: {video_path}")
    
#     # Check if the alignment file exists
#     if not os.path.exists(alignment_path):
#         raise FileNotFoundError(f"Alignment file not found: {alignment_path}")

#     # Debugging the file existence checks
#     print(f"Video file exists: {os.path.exists(video_path)}")
#     print(f"Alignment file exists: {os.path.exists(alignment_path)}")

#     # Load video and alignment files
#     frames = load_video(video_path)
#     alignments = load_alignments(alignment_path)
#     return frames, alignments



