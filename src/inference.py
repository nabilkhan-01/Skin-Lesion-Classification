from pathlib import Path

import numpy as np
import tensorflow as tf


MODEL_PATH = Path(
    "models/efficientnet_b0_finetuned_best.keras"
)

IMAGE_SIZE = (224, 224)

CLASS_NAMES = [
    "akiec",
    "bcc",
    "bkl",
    "df",
    "mel",
    "nv",
    "scc",
    "vasc",
]


def load_model():
    """Load the final trained model."""
    return tf.keras.models.load_model(MODEL_PATH)


def load_image(image_source):
    """
    Load an image from a file path or bytes.

    Returns:
        image: Tensor with shape (224, 224, 3)
        original: Tensor with shape (224, 224, 3)
    """

    if isinstance(image_source, (str, Path)):
        image_bytes = tf.io.read_file(str(image_source))
    else:
        image_bytes = tf.convert_to_tensor(
            image_source,
            dtype=tf.string
        )

    image = tf.image.decode_jpeg(
        image_bytes,
        channels=3
    )

    original = tf.image.resize(
        image,
        IMAGE_SIZE
    )

    image = tf.cast(
        original,
        tf.float32
    )

    return image, original


def predict(model, image):
    """
    Generate prediction probabilities.

    Returns:
        predicted_index
        predicted_class
        confidence
        probabilities
    """

    image_batch = tf.expand_dims(
        image,
        axis=0
    )

    predictions = model.predict(
        image_batch,
        verbose=0
    )[0]

    predicted_index = int(
        np.argmax(predictions)
    )

    predicted_class = CLASS_NAMES[
        predicted_index
    ]

    confidence = float(
        predictions[predicted_index]
    )

    return (
        predicted_index,
        predicted_class,
        confidence,
        predictions
    )