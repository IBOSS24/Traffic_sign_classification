import numpy as np
from PIL import Image


IMAGE_SIZE = (50, 50)


def preprocess_image(image: Image.Image, image_size=IMAGE_SIZE) -> np.ndarray:
    """Prepare a PIL image for model inference."""
    if not isinstance(image, Image.Image):
        raise TypeError("preprocess_image expects a PIL.Image.Image instance")

    image = image.convert("RGB").resize(image_size)
    array = np.asarray(image, dtype=np.float32) / 255.0

    return np.expand_dims(array, axis=0)
