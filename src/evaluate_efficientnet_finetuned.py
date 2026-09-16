import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from transfer_data_pipeline import create_dataset, CLASS_NAMES


MODEL_PATH = "models/efficientnet_b0_finetuned_best.keras"
TEST_CSV = "data/HAM10000/test.csv"

CM_OUTPUT = "results/efficientnet_b0_finetuned_confusion_matrix.csv"
METRICS_OUTPUT = "results/efficientnet_b0_finetuned_test_metrics.csv"


print("Loading fine-tuned EfficientNetB0...")
model = tf.keras.models.load_model(MODEL_PATH)

print("Loading test dataset...")
test_dataset = create_dataset(
    TEST_CSV,
    training=False
)

print("\nRunning predictions...")

y_true = []
y_pred = []

for images, labels in test_dataset:

    predictions = model.predict(
        images,
        verbose=0
    )

    predicted_classes = np.argmax(
        predictions,
        axis=1
    )

    y_true.extend(labels.numpy())
    y_pred.extend(predicted_classes)


y_true = np.array(y_true)
y_pred = np.array(y_pred)


# --------------------------------------------------
# Metrics
# --------------------------------------------------

accuracy = accuracy_score(
    y_true,
    y_pred
)

weighted_precision = precision_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0
)

weighted_recall = recall_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0
)

weighted_f1 = f1_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0
)

macro_f1 = f1_score(
    y_true,
    y_pred,
    average="macro",
    zero_division=0
)


print("\n========================================")
print("FINE-TUNED EFFICIENTNETB0 - TEST RESULTS")
print("========================================")

print(f"Accuracy:           {accuracy:.4f}")
print(f"Weighted Precision: {weighted_precision:.4f}")
print(f"Weighted Recall:    {weighted_recall:.4f}")
print(f"Weighted F1:        {weighted_f1:.4f}")
print(f"Macro F1:           {macro_f1:.4f}")


# --------------------------------------------------
# Classification report
# --------------------------------------------------

print("\nClassification Report:\n")

report = classification_report(
    y_true,
    y_pred,
    target_names=CLASS_NAMES,
    zero_division=0
)

print(report)


# --------------------------------------------------
# Confusion matrix
# --------------------------------------------------

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=range(len(CLASS_NAMES))
)

cm_df = pd.DataFrame(
    cm,
    index=CLASS_NAMES,
    columns=CLASS_NAMES
)

cm_df.to_csv(CM_OUTPUT)

print(
    f"Confusion matrix saved to: "
    f"{CM_OUTPUT}"
)


# --------------------------------------------------
# Save metrics
# --------------------------------------------------

metrics_df = pd.DataFrame({
    "metric": [
        "accuracy",
        "weighted_precision",
        "weighted_recall",
        "weighted_f1",
        "macro_f1"
    ],
    "value": [
        accuracy,
        weighted_precision,
        weighted_recall,
        weighted_f1,
        macro_f1
    ]
})

metrics_df.to_csv(
    METRICS_OUTPUT,
    index=False
)

print(
    f"Metrics saved to: "
    f"{METRICS_OUTPUT}"
)