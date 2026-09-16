import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.utils.class_weight import compute_class_weight

from transfer_data_pipeline import (
    create_dataset,
    CLASS_NAMES
)


# --------------------------------------------------
# Configuration
# --------------------------------------------------

TRAIN_CSV = "data/HAM10000/train.csv"
VAL_CSV = "data/HAM10000/validation.csv"

BASE_MODEL_PATH = "models/efficientnet_b0_best.keras"

MODEL_DIR = Path("models")
RESULTS_DIR = Path("results")

MODEL_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

EPOCHS = 10
LEARNING_RATE = 1e-5


# --------------------------------------------------
# Load datasets
# --------------------------------------------------

print("Loading training dataset...")

train_dataset = create_dataset(
    TRAIN_CSV,
    training=True
)

print("Loading validation dataset...")

val_dataset = create_dataset(
    VAL_CSV,
    training=False
)


# --------------------------------------------------
# Class weights
# --------------------------------------------------

train_df = pd.read_csv(TRAIN_CSV)

class_indices = train_df["diagnosis"].map(
    {
        name: index
        for index, name in enumerate(CLASS_NAMES)
    }
).values

raw_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.arange(len(CLASS_NAMES)),
    y=class_indices
)

class_weights = {
    index: float(np.sqrt(weight))
    for index, weight in enumerate(raw_weights)
}

print("\nModerated class weights:")

for index, class_name in enumerate(CLASS_NAMES):
    print(
        f"{class_name}: "
        f"{class_weights[index]:.4f}"
    )


# --------------------------------------------------
# Load pretrained EfficientNet model
# --------------------------------------------------

print("\nLoading best EfficientNetB0 model...")

model = tf.keras.models.load_model(
    BASE_MODEL_PATH
)


# --------------------------------------------------
# Locate EfficientNet backbone
# --------------------------------------------------

base_model = model.get_layer(
    "efficientnetb0"
)

print(
    "\nEfficientNetB0 layers:",
    len(base_model.layers)
)


# --------------------------------------------------
# Freeze everything first
# --------------------------------------------------

base_model.trainable = True

for layer in base_model.layers:
    layer.trainable = False


# --------------------------------------------------
# Unfreeze upper portion
# --------------------------------------------------

# EfficientNetB0 has roughly 230 layers.
# We fine-tune only the final ~30 layers.

for layer in base_model.layers[-30:]:

    # Keep BatchNormalization frozen.
    if not isinstance(
        layer,
        tf.keras.layers.BatchNormalization
    ):
        layer.trainable = True


# --------------------------------------------------
# Print trainable layers
# --------------------------------------------------

print("\nTrainable EfficientNet layers:")

trainable_count = 0

for layer in base_model.layers:

    if layer.trainable:

        print(
            f"  {layer.name}"
        )

        trainable_count += 1


print(
    f"\nTrainable EfficientNet layers: "
    f"{trainable_count}"
)


# --------------------------------------------------
# Compile
# --------------------------------------------------

model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE
    ),

    loss="sparse_categorical_crossentropy",

    metrics=["accuracy"]
)


print("\n========================================")
print("EXPERIMENT 4: EfficientNetB0 Fine-Tuning")
print("========================================")

print(
    f"Learning rate: {LEARNING_RATE}"
)

print(
    f"Epochs: {EPOCHS}"
)

print(
    "Fine-tuning: final ~30 layers"
)

print(
    "BatchNormalization layers: frozen"
)


# --------------------------------------------------
# Callbacks
# --------------------------------------------------

callbacks = [

    tf.keras.callbacks.EarlyStopping(

        monitor="val_loss",

        patience=3,

        restore_best_weights=True,

        verbose=1
    ),

    tf.keras.callbacks.ReduceLROnPlateau(

        monitor="val_loss",

        factor=0.5,

        patience=1,

        min_lr=1e-7,

        verbose=1
    ),

    tf.keras.callbacks.ModelCheckpoint(

        filepath=(
            MODEL_DIR /
            "efficientnet_b0_finetuned_best.keras"
        ),

        monitor="val_loss",

        save_best_only=True,

        verbose=1
    )
]


# --------------------------------------------------
# Train
# --------------------------------------------------

history = model.fit(

    train_dataset,

    validation_data=val_dataset,

    epochs=EPOCHS,

    class_weight=class_weights,

    callbacks=callbacks
)


# --------------------------------------------------
# Save history
# --------------------------------------------------

history_path = (
    RESULTS_DIR /
    "efficientnet_b0_finetuned_history.json"
)

with open(
    history_path,
    "w"
) as file:

    json.dump(
        history.history,
        file
    )

print(
    f"\nTraining history saved to: "
    f"{history_path}"
)


# --------------------------------------------------
# Save final model
# --------------------------------------------------

final_model_path = (
    MODEL_DIR /
    "efficientnet_b0_finetuned_final.keras"
)

model.save(
    final_model_path
)

print(
    f"Final model saved to: "
    f"{final_model_path}"
)


print(
    "\nExperiment 4 completed successfully."
)5678