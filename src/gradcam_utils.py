import numpy as np
import tensorflow as tf

from .inference import CLASS_NAMES


def get_target_layer(model):
    """
    Find the final convolutional layer (top_conv)
    inside the EfficientNetB0 backbone.
    """

    backbone = model.get_layer(
        "efficientnetb0"
    )

    try:
        target_layer = backbone.get_layer("top_conv")
    except (ValueError, KeyError):
        conv_layers = [
            layer
            for layer in backbone.layers
            if isinstance(
                layer,
                tf.keras.layers.Conv2D
            )
        ]
        if not conv_layers:
            raise RuntimeError(
                "No convolutional layer found in backbone."
            )
        target_layer = conv_layers[-1]

    return backbone, target_layer


def generate_gradcam(
    model,
    image,
    class_index=None
):
    """
    Generate a Grad-CAM heatmap.

    Args:
        model: trained classification model
        image: tensor with shape (224, 224, 3)
        class_index: target class. If None,
                     uses predicted class.

    Returns:
        heatmap
        class_index
        probabilities
    """

    backbone, target_layer = get_target_layer(
        model
    )

    feature_model = tf.keras.Model(
        inputs=backbone.input,
        outputs=[
            target_layer.output,
            backbone.output
        ]
    )

    # Everything after EfficientNetB0 is the
    # classifier head.
    backbone_position = model.layers.index(
        backbone
    )

    classifier_layers = model.layers[
        backbone_position + 1:
    ]

    image_batch = tf.expand_dims(
        image,
        axis=0
    )

    with tf.GradientTape() as tape:

        conv_outputs, backbone_output = (
            feature_model(
                image_batch,
                training=False
            )
        )

        x = backbone_output

        for layer in classifier_layers:
            x = layer(
                x,
                training=False
            )

        predictions = x

        if class_index is None:
            target_class_idx = int(
                tf.argmax(predictions[0]).numpy()
            )
        elif isinstance(class_index, tf.Tensor):
            target_class_idx = int(class_index.numpy())
        else:
            target_class_idx = int(class_index)

        class_score = predictions[
            0,
            target_class_idx
        ]

    gradients = tape.gradient(
        class_score,
        conv_outputs
    )

    pooled_gradients = tf.reduce_mean(
        gradients,
        axis=(0, 1, 2)
    )

    conv_outputs = conv_outputs[0]

    heatmap = tf.reduce_sum(
        conv_outputs * pooled_gradients,
        axis=-1
    )

    heatmap = tf.maximum(
        heatmap,
        0
    )

    max_value = tf.reduce_max(
        heatmap
    )

    heatmap = tf.where(
        max_value > 0,
        heatmap / max_value,
        tf.zeros_like(heatmap)
    )

    return (
        heatmap.numpy(),
        target_class_idx,
        predictions[0].numpy()
    )


def resize_heatmap(
    heatmap,
    image_size=(224, 224)
):
    """Resize Grad-CAM heatmap to image size."""

    resized = tf.image.resize(
        heatmap[..., np.newaxis],
        image_size
    )

    return resized.numpy().squeeze()


def get_top_predictions(
    probabilities,
    top_k=3
):
    """
    Return top-k predictions as:

    [
        {
            "class": "...",
            "confidence": 0.00
        },
        ...
    ]
    """

    top_indices = np.argsort(
        probabilities
    )[::-1][:top_k]

    results = []

    for index in top_indices:

        results.append({
            "class": CLASS_NAMES[index],
            "confidence": float(
                probabilities[index]
            )
        })

    return results