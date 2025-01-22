import os
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

def load_model(model_path="/Users/tejasredkar/Developer/LipReader/app/checkpoints1.weights.h5") -> Sequential:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")

    # Initialize the model
    model = Sequential()

    # First Conv3D Block
    model.add(Conv3D(filters=128, kernel_size=(3, 3, 3), input_shape=(75, 46, 140, 1), padding="same"))
    model.add(Activation("relu"))
    model.add(MaxPool3D(pool_size=(1, 2, 2)))

    # Second Conv3D Block
    model.add(Conv3D(filters=256, kernel_size=(3, 3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(MaxPool3D(pool_size=(1, 2, 2)))

    # Third Conv3D Block
    model.add(Conv3D(filters=75, kernel_size=(3, 3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(MaxPool3D(pool_size=(1, 2, 2)))

    # Flatten layer wrapped in TimeDistributed for 3D data
    model.add(TimeDistributed(Flatten()))

    # Bidirectional LSTM layers
    model.add(Bidirectional(LSTM(units=128, kernel_initializer="orthogonal", return_sequences=True)))
    model.add(Dropout(0.5))

    model.add(Bidirectional(LSTM(units=128, kernel_initializer="orthogonal", return_sequences=True)))
    model.add(Dropout(0.5))

    # Dense output layer
    model.add(Dense(units=41, kernel_initializer="he_normal", activation="softmax"))

    # Load weights from the provided model.h5 file
    model.load_weights(model_path)
    return model
