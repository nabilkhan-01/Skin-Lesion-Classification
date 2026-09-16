import tensorflow as tf


IMAGE_SIZE = (224, 224, 3)
NUM_CLASSES = 8


def build_efficientnet():

    base_model = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=IMAGE_SIZE
    )

    # Freeze pretrained backbone
    base_model.trainable = False

    inputs = tf.keras.Input(
        shape=IMAGE_SIZE
    )

    x = base_model(
        inputs,
        training=False
    )

    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    x = tf.keras.layers.Dropout(
        0.4
    )(x)

    x = tf.keras.layers.Dense(
        128,
        activation="relu"
    )(x)

    x = tf.keras.layers.Dropout(
        0.3
    )(x)

    outputs = tf.keras.layers.Dense(
        NUM_CLASSES,
        activation="softmax"
    )(x)

    model = tf.keras.Model(
        inputs,
        outputs
    )

    return model