import os
import numpy as np

from sklearn.model_selection import train_test_split


# ============================================================
# SETTINGS
# ============================================================

SCRIPT_FOLDER = os.path.dirname(
    os.path.abspath(__file__)
)

X_FILE = os.path.join(
    SCRIPT_FOLDER,
    "X_all_seizures.npy"
)

Y_FILE = os.path.join(
    SCRIPT_FOLDER,
    "y_all_seizures.npy"
)


# ============================================================
# Load full dataset
# ============================================================

print("========================================")
print("LOADING FULL DATASET")
print("========================================")

X = np.load(X_FILE)
y = np.load(Y_FILE)

print(
    "X shape:",
    X.shape
)

print(
    "y shape:",
    y.shape
)


# ============================================================
# Original label distribution
# ============================================================

print("\n========================================")
print("ORIGINAL LABEL DISTRIBUTION")
print("========================================")

unique, counts = np.unique(
    y,
    return_counts=True
)

print(
    dict(
        zip(unique, counts)
    )
)


# ============================================================
# First split
# Train = 70%
# Temporary = 30%
# ============================================================

print("\n========================================")
print("CREATING TRAIN / TEMPORARY SPLIT")
print("========================================")

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)


# ============================================================
# Second split
# Validation = 15%
# Test = 15%
#
# Temporary = 30%
# Therefore:
# Validation = 50% of temporary
# Test       = 50% of temporary
# ============================================================

print("\n========================================")
print("CREATING VALIDATION / TEST SPLIT")
print("========================================")

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)


# ============================================================
# Function for displaying distribution
# ============================================================

def print_distribution(
    name,
    X_data,
    y_data
):

    unique, counts = np.unique(
        y_data,
        return_counts=True
    )

    distribution = dict(
        zip(unique, counts)
    )

    print("\n" + name)

    print(
        "X shape:",
        X_data.shape
    )

    print(
        "y shape:",
        y_data.shape
    )

    print(
        "Distribution:",
        distribution
    )

    print(
        "Seizure:",
        distribution.get(1, 0)
    )

    print(
        "Normal:",
        distribution.get(0, 0)
    )


# ============================================================
# Display final split
# ============================================================

print("\n========================================")
print("FINAL DATASET SPLIT")
print("========================================")

print_distribution(
    "TRAIN",
    X_train,
    y_train
)

print_distribution(
    "VALIDATION",
    X_val,
    y_val
)

print_distribution(
    "TEST",
    X_test,
    y_test
)


# ============================================================
# Save datasets
# ============================================================

print("\n========================================")
print("SAVING DATASETS")
print("========================================")


np.save(
    os.path.join(
        SCRIPT_FOLDER,
        "X_train_full.npy"
    ),
    X_train
)

np.save(
    os.path.join(
        SCRIPT_FOLDER,
        "y_train_full.npy"
    ),
    y_train
)


np.save(
    os.path.join(
        SCRIPT_FOLDER,
        "X_val_full.npy"
    ),
    X_val
)

np.save(
    os.path.join(
        SCRIPT_FOLDER,
        "y_val_full.npy"
    ),
    y_val
)


np.save(
    os.path.join(
        SCRIPT_FOLDER,
        "X_test_full.npy"
    ),
    X_test
)

np.save(
    os.path.join(
        SCRIPT_FOLDER,
        "y_test_full.npy"
    ),
    y_test
)


# ============================================================
# Final verification
# ============================================================

print("\n========================================")
print("DATASETS SAVED SUCCESSFULLY")
print("========================================")

print(
    "X_train_full.npy"
)

print(
    "y_train_full.npy"
)

print(
    "X_val_full.npy"
)

print(
    "y_val_full.npy"
)

print(
    "X_test_full.npy"
)

print(
    "y_test_full.npy"
)

print("\nDONE")