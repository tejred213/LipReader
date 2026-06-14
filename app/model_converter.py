"""Convert a `*.weights.h5` checkpoint into a self-contained `.keras` model.

Usage:
    python model_converter.py <weights.h5> <output.keras>
"""

import argparse
import os

from modelutil import build_model


def convert_h5_to_keras(h5_weight_path: str, keras_model_path: str) -> None:
    """Load weights into the LipNet architecture and save a full .keras model."""
    if not os.path.exists(h5_weight_path):
        raise FileNotFoundError(f"Weight file not found at: {h5_weight_path}")

    model = build_model()
    try:
        model.load_weights(h5_weight_path)
    except ValueError as e:
        raise ValueError(
            "Error loading weights. Ensure the model architecture matches the weights."
        ) from e
    print(f"Loaded weights from: {h5_weight_path}")

    model.save(keras_model_path)
    print(f"Saved full model to: {keras_model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("weights", help="Path to the input *.weights.h5 file")
    parser.add_argument("output", help="Path to the output *.keras file")
    args = parser.parse_args()
    convert_h5_to_keras(args.weights, args.output)
