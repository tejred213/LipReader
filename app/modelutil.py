"""Build the LipNet architecture and load trained weights."""

import os
from typing import Optional

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv3D,
    LSTM,
    Dense,
    Dropout,
    Bidirectional,
    MaxPool3D,
    Activation,
    TimeDistributed,
    Flatten,
)

import config


def build_model() -> Sequential:
    """Construct the (untrained) LipNet model architecture."""
    model = Sequential()

    input_shape = (config.FRAME_COUNT, config.FRAME_HEIGHT, config.FRAME_WIDTH, 1)
    model.add(Conv3D(128, 3, input_shape=input_shape, padding="same"))
    model.add(Activation("relu"))
    model.add(MaxPool3D((1, 2, 2)))

    model.add(Conv3D(256, 3, padding="same"))
    model.add(Activation("relu"))
    model.add(MaxPool3D((1, 2, 2)))

    model.add(Conv3D(75, 3, padding="same"))
    model.add(Activation("relu"))
    model.add(MaxPool3D((1, 2, 2)))

    model.add(TimeDistributed(Flatten()))

    model.add(Bidirectional(LSTM(128, kernel_initializer="orthogonal", return_sequences=True)))
    model.add(Dropout(0.5))

    model.add(Bidirectional(LSTM(128, kernel_initializer="orthogonal", return_sequences=True)))
    model.add(Dropout(0.5))

    model.add(Dense(config.NUM_CLASSES, kernel_initializer="he_normal", activation="softmax"))
    return model


def _weights_exist(model_path: str) -> bool:
    """True for both an .h5 file and a TF-checkpoint prefix (foo -> foo.index)."""
    return os.path.exists(model_path) or os.path.exists(model_path + ".index")


def load_model(model_path: Optional[str] = None) -> Sequential:
    """Build the architecture and load trained weights.

    Accepts either an `.h5` weights file or a TF-checkpoint prefix, and falls
    back to the path resolved in `config.MODEL_PATH` when none is given.
    """
    if model_path is None:
        model_path = config.MODEL_PATH
    if model_path is None or not _weights_exist(str(model_path)):
        raise FileNotFoundError(
            f"Model weights not found (looked at: {model_path}). "
            "Run `python app/download_weights.py` or set LIPREADER_MODEL_PATH."
        )

    model = build_model()
    status = model.load_weights(str(model_path))
    # TF checkpoints return a status object; ignore unrestored optimizer state.
    if status is not None:
        status.expect_partial()
    return model
