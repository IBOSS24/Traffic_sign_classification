import numpy as np
import tensorflow as tf
from PIL import Image


def find_last_conv_layer(model: tf.keras.Model) -> str:
    """Return the name of the final Conv2D layer in a Keras model."""
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name

    raise ValueError("No Conv2D layer found in the model")


def make_gradcam_heatmap(
    img_array: np.ndarray,
    model: tf.keras.Model,
    last_conv_layer_name: str | None = None,
    pred_index: int | None = None,
) -> np.ndarray:
    """Generate a normalized Grad-CAM heatmap for one preprocessed image."""
    if last_conv_layer_name is None:
        last_conv_layer_name = find_last_conv_layer(model)

    grad_model = _build_gradcam_model(model, img_array.shape[1:], last_conv_layer_name)

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array, training=False)
        if pred_index is None:
            pred_index = int(tf.argmax(predictions[0]))
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    if grads is None:
        height, width = conv_outputs.shape[1], conv_outputs.shape[2]
        return np.zeros((height, width), dtype=np.float32)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
    heatmap = tf.maximum(heatmap, 0)

    max_heat = tf.reduce_max(heatmap)
    if float(max_heat) > 0:
        heatmap = heatmap / max_heat

    return heatmap.numpy()


def _build_gradcam_model(
    model: tf.keras.Model,
    input_shape: tuple[int, ...],
    last_conv_layer_name: str,
) -> tf.keras.Model:
    """Create a functional model for Grad-CAM without relying on Sequential.output."""
    inputs = tf.keras.Input(shape=input_shape)
    x = inputs
    conv_outputs = None

    for layer in model.layers:
        try:
            x = layer(x, training=False)
        except TypeError:
            x = layer(x)

        if layer.name == last_conv_layer_name:
            conv_outputs = x

    if conv_outputs is None:
        raise ValueError(f"Layer {last_conv_layer_name!r} was not found in the model")

    return tf.keras.Model(inputs=inputs, outputs=[conv_outputs, x])


def _jet_colormap(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, 0.0, 1.0)
    red = np.clip(1.5 - np.abs(4.0 * values - 3.0), 0.0, 1.0)
    green = np.clip(1.5 - np.abs(4.0 * values - 2.0), 0.0, 1.0)
    blue = np.clip(1.5 - np.abs(4.0 * values - 1.0), 0.0, 1.0)

    return np.stack([red, green, blue], axis=-1)


def overlay_heatmap(image, heatmap: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """Overlay a Grad-CAM heatmap on a PIL image and return an RGB array."""
    img = np.asarray(image.convert("RGB"), dtype=np.float32)
    heatmap_image = Image.fromarray(np.uint8(np.clip(heatmap, 0.0, 1.0) * 255))
    heatmap_image = heatmap_image.resize((img.shape[1], img.shape[0]), Image.Resampling.BILINEAR)
    heatmap_array = np.asarray(heatmap_image, dtype=np.float32) / 255.0
    colored_heatmap = _jet_colormap(heatmap_array) * 255.0

    blended = (1.0 - alpha) * img + alpha * colored_heatmap

    return np.clip(blended, 0, 255).astype(np.uint8)
