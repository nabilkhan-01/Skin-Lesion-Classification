import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedGroupKFold

# --------------------------------------------------
# Paths
# --------------------------------------------------

DATA_DIR = Path("data/HAM10000")
METADATA_FILE = DATA_DIR / "metadata.csv"

# --------------------------------------------------
# Load metadata
# --------------------------------------------------

df = pd.read_csv(METADATA_FILE)


# --------------------------------------------------
# Create our 8-class label
# --------------------------------------------------

def create_label(row):

    diagnosis = row["diagnosis_3"]

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

    elif row["diagnosis_2"] == "Benign soft tissue proliferations - Vascular":
        return "vasc"

    elif diagnosis == "Squamous cell carcinoma, NOS":
        return "scc"

    return None


df["diagnosis"] = df.apply(create_label, axis=1)

# Remove records without a label
df = df.dropna(subset=["diagnosis"]).copy()

# --------------------------------------------------
# Keep only required information
# --------------------------------------------------

df = df[
    [
        "isic_id",
        "lesion_id",
        "diagnosis"
    ]
].copy()

# --------------------------------------------------
# Stratified group splitting
# --------------------------------------------------

X = df["isic_id"]
y = df["diagnosis"]
groups = df["lesion_id"]

splitter = StratifiedGroupKFold(
    n_splits=7,
    shuffle=True,
    random_state=42
)

# Generate the 7 folds
folds = list(splitter.split(X, y, groups))

# --------------------------------------------------
# Fold 0 → TEST
# Fold 1 → VALIDATION
# Remaining folds → TRAIN
# --------------------------------------------------

train_indices = []
validation_indices = []
test_indices = []

for fold_number, (_, test_idx) in enumerate(folds):

    if fold_number == 0:
        test_indices.extend(test_idx)

    elif fold_number == 1:
        validation_indices.extend(test_idx)

    else:
        train_indices.extend(test_idx)

train_df = df.iloc[train_indices].copy()
validation_df = df.iloc[validation_indices].copy()
test_df = df.iloc[test_indices].copy()

# --------------------------------------------------
# Save splits
# --------------------------------------------------

train_df.to_csv(DATA_DIR / "train.csv", index=False)
validation_df.to_csv(DATA_DIR / "validation.csv", index=False)
test_df.to_csv(DATA_DIR / "test.csv", index=False)

# --------------------------------------------------
# Check for lesion leakage
# --------------------------------------------------

train_lesions = set(train_df["lesion_id"])
validation_lesions = set(validation_df["lesion_id"])
test_lesions = set(test_df["lesion_id"])

train_val_overlap = train_lesions & validation_lesions
train_test_overlap = train_lesions & test_lesions
val_test_overlap = validation_lesions & test_lesions

# --------------------------------------------------
# Print results
# --------------------------------------------------

print("=" * 65)
print("DATASET SPLITTING")
print("=" * 65)

print(f"\nTotal images:       {len(df)}")

print("\nSplit sizes:")
print(f"Training:           {len(train_df)}")
print(f"Validation:         {len(validation_df)}")
print(f"Testing:            {len(test_df)}")

print("\nPercentages:")

total = len(df)

print(f"Training:           {len(train_df) / total * 100:.2f}%")
print(f"Validation:         {len(validation_df) / total * 100:.2f}%")
print(f"Testing:            {len(test_df) / total * 100:.2f}%")

print("\nClass distribution:")
print("\nTRAINING")
print(train_df["diagnosis"].value_counts())

print("\nVALIDATION")
print(validation_df["diagnosis"].value_counts())

print("\nTESTING")
print(test_df["diagnosis"].value_counts())

print("\nLesion leakage check:")
print(f"Train ∩ Validation: {len(train_val_overlap)}")
print(f"Train ∩ Test:       {len(train_test_overlap)}")
print(f"Validation ∩ Test:  {len(val_test_overlap)}")

print("\n" + "=" * 65)

if (
    len(train_val_overlap) == 0
    and len(train_test_overlap) == 0
    and len(val_test_overlap) == 0
):
    print("SUCCESS: No lesion leakage detected.")
else:
    print("WARNING: Lesion leakage detected!")

print("=" * 65)