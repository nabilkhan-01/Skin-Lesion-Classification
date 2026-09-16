import pandas as pd
import tensorflow as tf
from pathlib import Path


IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16

DATA_DIR = Path("data/HAM10000")
IMAGE_DIR = DATA_DIR / "images"

CLASS_NAMES = [
    "akiec",
    "bcc",
    "bkl",
    "df",
    "mel",
    "nv",
    "scc",
    "vasc"
]

CLASS_TO_INDEX = {
    name: index
    for index, name in enumerate(CLASS_NAMES)
}


def load_dataframe(csv_file):
    df = pd.read_csv(csv_file)

    df["image_path"] = df["isic_id"].apply(
        lambda x: str(IMAGE_DIR / f"{x}.jpg")
    )

    df["label"] = df["diagnosis"].map(CLASS_TO_INDEX)

    return df


def load_image(image_path, label):

    image = tf.io.read_file(image_path)

    image = tf.image.decode_jpeg(
        image,
        channels=3
    )

    image = tf.image.resize(
        image,
        IMAGE_SIZE
    )

    # Keep pixel values in [0, 255].
    # EfficientNetB0 contains its own input preprocessing.
    image = tf.cast(
        image,
        tf.float32
    )

    return image, label


data_augmentation = tf.keras.Sequential([

    tf.keras.layers.RandomFlip(
        "horizontal_and_vertical"
    ),

    tf.keras.layers.RandomRotation(
        0.05
    ),

    tf.keras.layers.RandomZoom(
        0.10
    ),

    tf.keras.layers.RandomTranslation(
        height_factor=0.05,
        width_factor=0.05
    )

])


def create_dataset(csv_file, training=False):

    df = load_dataframe(csv_file)

    image_paths = df["image_path"].values
    labels = df["label"].values

    dataset = tf.data.Dataset.from_tensor_slices(
        (image_paths, labels)
    )

    if training:
        dataset = dataset.shuffle(
            buffer_size=len(df),
            seed=42
        )

    dataset = dataset.map(
        load_image,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    if training:
        dataset = dataset.map(
            lambda image, label: (
                data_augmentation(
                    image,
                    training=True
                ),
                label
            ),
            num_parallel_calls=tf.data.AUTOTUNE
        )

    dataset = dataset.batch(BATCH_SIZE)

    dataset = dataset.prefetch(
        tf.data.AUTOTUNE
    )

    return dataset