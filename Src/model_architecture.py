from pathlib import Path

import h5py
import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization, Conv2D, Dense, Dropout, Flatten, MaxPooling2D


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WEIGHTS_PATH = PROJECT_ROOT / "Model" / "traffic.weights.h5"


def create_model(input_shape=(50, 50, 3), num_classes=43) -> tf.keras.Model:
    """Create the CNN architecture used during training."""
    tf.keras.backend.clear_session()

    model = tf.keras.Sequential(
        [
            Conv2D(
                filters=64,
                kernel_size=(3, 3),
                input_shape=input_shape,
                activation="relu",
                padding="same",
            ),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),
            Conv2D(filters=64, kernel_size=(3, 3), activation="relu"),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),
            Flatten(),
            Dense(128, activation="relu"),
            Dropout(0.5),
            Dense(num_classes, activation="softmax"),
        ]
    )
    model.build((None, *input_shape))

    return model


def load_traffic_sign_model(weights_path=DEFAULT_WEIGHTS_PATH) -> tf.keras.Model:
    """Create the architecture, load weights, and warm the graph for Grad-CAM."""
    weights_path = Path(weights_path)
    if not weights_path.exists():
        raise FileNotFoundError(f"Model weights were not found at {weights_path}")

    model = create_model()
    try:
        model.load_weights(weights_path)
    except ValueError:
        _load_keras3_weights_by_layer_name(model, weights_path)

    dummy_input = tf.zeros((1, 50, 50, 3), dtype=tf.float32)
    _ = model(dummy_input, training=False)

    return model


def _load_keras3_weights_by_layer_name(model: tf.keras.Model, weights_path: Path) -> None:
    """Load Keras 3 `.weights.h5` files in older tf.keras runtimes."""
    with h5py.File(weights_path, "r") as file:
        if "layers" not in file:
            raise ValueError(f"Unsupported weights file format: {weights_path}")

        layer_groups = file["layers"]
        for layer in model.layers:
            if layer.name not in layer_groups:
                continue

            variable_group = layer_groups[layer.name]["vars"]
            weights = [
                variable_group[key][()]
                for key in sorted(variable_group.keys(), key=lambda value: int(value))
            ]
            if weights:
                layer.set_weights(weights)
