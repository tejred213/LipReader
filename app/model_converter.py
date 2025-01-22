import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv3D, LSTM, Dense, Dropout, Bidirectional, MaxPool3D,
    Activation, TimeDistributed, Flatten
)

def convert_h5_to_keras(h5_weight_path, keras_model_path):
    """
    Converts a weights.h5 file into a .keras model file.
    
    Args:
        h5_weight_path (str): Path to the .weights.h5 file.
        keras_model_path (str): Path to save the .keras model file.
    """
    # Define the model architecture
    model = Sequential()

    model.add(Conv3D(128, 3, input_shape=(75, 46, 140, 1), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPool3D((1, 2, 2)))

    model.add(Conv3D(256, 3, padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPool3D((1, 2, 2)))

    model.add(Conv3D(75, 3, padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPool3D((1, 2, 2)))

    model.add(TimeDistributed(Flatten()))

    model.add(Bidirectional(LSTM(128, kernel_initializer='Orthogonal', return_sequences=True)))
    model.add(Dropout(0.5))

    model.add(Bidirectional(LSTM(128, kernel_initializer='Orthogonal', return_sequences=True)))
    model.add(Dropout(0.5))

    model.add(Dense(41, kernel_initializer='he_normal', activation='softmax'))

    # Check if weight file exists
    if not os.path.exists(h5_weight_path):
        raise FileNotFoundError(f"Weight file not found at: {h5_weight_path}")

    # Load weights from the .h5 file
    try:
        model.load_weights(h5_weight_path)
        print(f"Successfully loaded weights from: {h5_weight_path}")
    except ValueError as e:
        raise ValueError(
            "Error loading weights. Ensure the model architecture matches the weights."
        ) from e

    # Save the model in .keras format
    model.save(keras_model_path, save_format='keras')
    print(f"Model saved in .keras format at: {keras_model_path}")

# Paths for the .weights.h5 file and the new .keras file
h5_weight_path = '/Users/tejasredkar/Developer/LipReader/models/checkpoint.weights.h5'  # Replace with your .weights.h5 file path
keras_model_path = '/Users/tejasredkar/Developer/LipReader/models/checkpoint.keras'  # Replace with the desired .keras file path

# Convert weights to .keras format
convert_h5_to_keras(h5_weight_path, keras_model_path)