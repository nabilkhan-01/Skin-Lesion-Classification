from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt


# --------------------------------------------------
# Configuration
# --------------------------------------------------

MODEL_PATH = "models/efficientnet_b0_finetuned_best.keras"

TEST_CSV = "data/HAM10000/test.csv"

IMAGE_DIR = Path("data/HAM10000/images")

OUTPUT_DIR = Path("results/gradcam")

IMAGE_SIZE = (224, 224)

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


# --------------------------------------------------
# Load model
# --------------------------------------------------

print("Loading fine-tuned EfficientNetB0...")

model = tf.keras.models.load_model(
    MODEL_PATH
)

print("Model loaded successfully.")


# --------------------------------------------------
# Get EfficientNet backbone
# --------------------------------------------------

backbone = model.get_layer(
    "efficientnetb0"
)

print(
    f"EfficientNetB0 layers: "
    f"{len(backbone.layers)}"
)


# --------------------------------------------------
# Find final convolutional layer
# --------------------------------------------------

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
        "No Conv2D layers found."
    )


target_layer = conv_layers[-1]

print(
    f"Grad-CAM target layer: "
    f"{target_layer.name}"
)


# --------------------------------------------------
# Feature model
#
# This model extracts the feature maps from the
# selected convolutional layer.
# --------------------------------------------------

feature_model = tf.keras.Model(
    inputs=backbone.input,
    outputs=[
        target_layer.output,
        backbone.output
    ]
)


# --------------------------------------------------
# Classifier head
#
# The outer model contains:
#
# EfficientNetB0
#      ↓
# GlobalAveragePooling2D
#      ↓
# Dropout
#      ↓
# Dense
#      ↓
# Dropout
#      ↓
# Dense
#
# We manually apply those layers so that the
# gradient remains connected to the target layer.
# --------------------------------------------------

backbone_index = model.layers.index(
    backbone
)

classifier_layers = model.layers[
    backbone_index + 1:
]


# --------------------------------------------------
# Load image
# --------------------------------------------------

def load_image(image_path):

    image_bytes = tf.io.read_file(
        image_path
    )

    image = tf.image.decode_jpeg(
        image_bytes,
        channels=3
    )

    image = tf.image.resize(
        image,
        IMAGE_SIZE
    )

    image = tf.cast(
        image,
        tf.float32
    )

    return image


# --------------------------------------------------
# Generate Grad-CAM
# --------------------------------------------------

def generate_gradcam(image):

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

        # Pass through classifier head
        for layer in classifier_layers:
            x = layer(
                x,
                training=False
            )

        predictions = x

        predicted_class = tf.argmax(
            predictions[0]
        )

        class_score = predictions[
            0,
            predicted_class
        ]

    # Gradient of class score with respect to
    # target convolutional feature maps
    gradients = tape.gradient(
        class_score,
        conv_outputs
    )

    # Global average pooling over spatial dimensions
    pooled_gradients = tf.reduce_mean(
        gradients,
        axis=(0, 1, 2)
    )

    conv_outputs = conv_outputs[0]

    # Weighted combination of feature maps
    heatmap = tf.reduce_sum(
        conv_outputs * pooled_gradients,
        axis=-1
    )

    # Keep positive influence
    heatmap = tf.maximum(
        heatmap,
        0
    )

    # Normalize
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
        int(predicted_class.numpy()),
        predictions[0].numpy()
    )


# --------------------------------------------------
# Save visualization
# --------------------------------------------------

def save_gradcam_visualization(
    original_image,
    heatmap,
    true_class,
    predicted_class,
    confidence,
    image_id
):

    heatmap_resized = tf.image.resize(
        heatmap[..., np.newaxis],
        IMAGE_SIZE
    ).numpy().squeeze()


    plt.figure(
        figsize=(12, 4)
    )


    # --------------------------------------------------
    # Original image
    # --------------------------------------------------

    plt.subplot(
        1,
        3,
        1
    )

    plt.imshow(
        original_image.numpy().astype(
            np.uint8
        )
    )

    plt.title(
        f"Original\nTrue: {true_class}"
    )

    plt.axis("off")


    # --------------------------------------------------
    # Heatmap
    # --------------------------------------------------

    plt.subplot(
        1,
        3,
        2
    )

    plt.imshow(
        heatmap_resized,
        cmap="jet"
    )

    plt.title(
        "Grad-CAM Heatmap"
    )

    plt.axis("off")


    # --------------------------------------------------
    # Overlay
    # --------------------------------------------------

    plt.subplot(
        1,
        3,
        3
    )

    plt.imshow(
        original_image.numpy().astype(
            np.uint8
        )
    )

    plt.imshow(
        heatmap_resized,
        cmap="jet",
        alpha=0.45
    )

    plt.title(
        f"Prediction: {predicted_class}\n"
        f"Confidence: {confidence:.2%}"
    )

    plt.axis("off")


    plt.tight_layout()


    output_path = (
        OUTPUT_DIR /
        f"gradcam_{image_id}.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


    print(
        f"Saved: {output_path}"
    )


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    test_df = pd.read_csv(
        TEST_CSV
    )

    print(
        f"\nTest images available: "
        f"{len(test_df)}"
    )

    print(
        "\nSelecting one test image "
        "from each class..."
    )


    # One representative test image
    # from every class
    samples = (
        test_df
        .groupby(
            "diagnosis",
            group_keys=False
        )
        .head(1)
        .reset_index(drop=True)
    )


    for _, row in samples.iterrows():

        image_id = row["isic_id"]

        true_class = row["diagnosis"]

        image_path = (
            IMAGE_DIR /
            f"{image_id}.jpg"
        )


        if not image_path.exists():

            print(
                f"Skipping missing image: "
                f"{image_path}"
            )

            continue


        print(
            f"\nProcessing {image_id}..."
        )


        original_image = load_image(
            str(image_path)
        )


        heatmap, predicted_index, predictions = (
            generate_gradcam(
                original_image
            )
        )


        predicted_class = (
            CLASS_NAMES[predicted_index]
        )

        confidence = float(
            predictions[predicted_index]
        )


        print(
            f"True class:      "
            f"{true_class}"
        )

        print(
            f"Predicted class: "
            f"{predicted_class}"
        )

        print(
            f"Confidence:      "
            f"{confidence:.4f}"
        )


        save_gradcam_visualization(
            original_image=original_image,
            heatmap=heatmap,
            true_class=true_class,
            predicted_class=predicted_class,
            confidence=confidence,
            image_id=image_id
        )


    print(
        "\n========================================"
    )

    print(
        "Grad-CAM generation completed."
    )

    print(
        f"Results saved to: "
        f"{OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()