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

from efficientnet_model import build_efficientnet


# --------------------------------------------------
# Configuration
# --------------------------------------------------

TRAIN_CSV = "data/HAM10000/train.csv"
VAL_CSV = "data/HAM10000/validation.csv"

MODEL_DIR = Path("models")
RESULTS_DIR = Path("results")

MODEL_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

EPOCHS = 15
LEARNING_RATE = 0.0003


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
# Calculate moderated class weights
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


# Same moderated weighting strategy
# used in Experiment 2

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
# Build EfficientNetB0
# --------------------------------------------------

print("\nBuilding EfficientNetB0...")

model = build_efficientnet()

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()


# --------------------------------------------------
# Callbacks
# --------------------------------------------------

callbacks = [

    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=4,
        restore_best_weights=True,
        verbose=1
    ),

    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
        verbose=1
    ),

    tf.keras.callbacks.ModelCheckpoint(
        filepath=(
            MODEL_DIR /
            "efficientnet_b0_best.keras"
        ),
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    )
]


# --------------------------------------------------
# Train
# --------------------------------------------------

print("\n========================================")
print("EXPERIMENT 3: EfficientNetB0")
print("========================================")

print(
    f"Learning rate: {LEARNING_RATE}"
)

print(
    f"Epochs: {EPOCHS}"
)

print(
    "Backbone: ImageNet pretrained + frozen"
)


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
    "efficientnet_b0_history.json"
)

with open(history_path, "w") as file:

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
    "efficientnet_b0_final.keras"
)

model.save(
    final_model_path
)

print(
    f"Final model saved to: "
    f"{final_model_path}"
)

print(
    "\nExperiment 3 completed successfully."
)