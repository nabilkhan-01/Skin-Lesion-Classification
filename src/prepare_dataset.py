import pandas as pd
from pathlib import Path

# --------------------------------------------------
# Paths
# --------------------------------------------------

DATA_DIR = Path("data/HAM10000")

METADATA_FILE = DATA_DIR / "metadata.csv"
OUTPUT_FILE = DATA_DIR / "labels.csv"

# --------------------------------------------------
# Load metadata
# --------------------------------------------------

df = pd.read_csv(METADATA_FILE)

# --------------------------------------------------
# Create diagnosis labels
# --------------------------------------------------

def create_label(row):

    diagnosis = row["diagnosis_3"]

    # Standard HAM10000-style classes
    if diagnosis == "Nevus":
        return "nv"

    elif diagnosis == "Pigmented benign keratosis":
        return "bkl"

    elif diagnosis == "Melanoma, NOS":
        return "mel"

    elif diagnosis == "Basal cell carcinoma":
        return "bcc"

    elif diagnosis == "Solar or actinic keratosis":
        return "akiec"

    elif diagnosis == "Dermatofibroma":
        return "df"

    # Vascular lesions do not have diagnosis_3
    # in the current ISIC metadata.
    elif row["diagnosis_2"] == "Benign soft tissue proliferations - Vascular":
        return "vasc"

    # Additional class in the current collection
    elif diagnosis == "Squamous cell carcinoma, NOS":
        return "scc"

    return None


df["diagnosis"] = df.apply(create_label, axis=1)

# --------------------------------------------------
# Remove records without a usable diagnosis
# --------------------------------------------------

df = df.dropna(subset=["diagnosis"])

# --------------------------------------------------
# Keep only the columns we need
# --------------------------------------------------

labels = df[["isic_id", "diagnosis"]].copy()

# --------------------------------------------------
# Save labels
# --------------------------------------------------

labels.to_csv(OUTPUT_FILE, index=False)

# --------------------------------------------------
# Print results
# --------------------------------------------------

print("=" * 60)
print("DATASET PREPARATION")
print("=" * 60)

print(f"\nTotal labelled images: {len(labels)}")

print("\nClass distribution:")
print(labels["diagnosis"].value_counts())

print(f"\nSaved labels to:")
print(OUTPUT_FILE)

print("\n" + "=" * 60)