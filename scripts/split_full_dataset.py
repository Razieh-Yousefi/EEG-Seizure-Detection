import numpy as np
import os

# ============================================================
# Load full dataset
# ============================================================

print("Loading full dataset...")

X = np.load("X_full.npy")
y = np.load("y_full.npy")

print("\nFull dataset:")
print("X shape:", X.shape)
print("y shape:", y.shape)

print("\nLabel distribution:")
print(np.unique(y, return_counts=True))

# ============================================================
# Separate seizure and normal samples
# ============================================================

seizure_indices = np.where(y == 1)[0]
normal_indices = np.where(y == 0)[0]

print("\nSeizure samples:", len(seizure_indices))
print("Normal samples:", len(normal_indices))

# ============================================================
# Shuffle indices
# ============================================================

np.random.seed(42)

np.random.shuffle(seizure_indices)
np.random.shuffle(normal_indices)

# ============================================================
# Split ratios
# ============================================================

# 70% Train
# 15% Validation
# 15% Test

def split_indices(indices):

    n = len(indices)

    train_end = int(0.70 * n)
    val_end = int(0.85 * n)

    train = indices[:train_end]
    val = indices[train_end:val_end]
    test = indices[val_end:]

    return train, val, test


seizure_train, seizure_val, seizure_test = split_indices(
    seizure_indices
)

normal_train, normal_val, normal_test = split_indices(
    normal_indices
)

# ============================================================
# Combine indices
# ============================================================

train_indices = np.concatenate(
    [seizure_train, normal_train]
)

val_indices = np.concatenate(
    [seizure_val, normal_val]
)

test_indices = np.concatenate(
    [seizure_test, normal_test]
)

# Shuffle each set

np.random.shuffle(train_indices)
np.random.shuffle(val_indices)
np.random.shuffle(test_indices)

# ============================================================
# Create datasets
# ============================================================

print("\nCreating Train dataset...")

X_train = X[train_indices]
y_train = y[train_indices]

print("Creating Validation dataset...")

X_val = X[val_indices]
y_val = y[val_indices]

print("Creating Test dataset...")

X_test = X[test_indices]
y_test = y[test_indices]

# ============================================================
# Print results
# ============================================================

print("\n==============================")
print("FINAL DATASET SPLIT")
print("==============================")

print("\nTRAIN")
print("X_train:", X_train.shape)
print("Labels:", np.unique(y_train, return_counts=True))

print("\nVALIDATION")
print("X_val:", X_val.shape)
print("Labels:", np.unique(y_val, return_counts=True))

print("\nTEST")
print("X_test:", X_test.shape)
print("Labels:", np.unique(y_test, return_counts=True))

# ============================================================
# Save datasets
# ============================================================

np.save("X_train_full.npy", X_train)
np.save("y_train_full.npy", y_train)

np.save("X_val_full.npy", X_val)
np.save("y_val_full.npy", y_val)

np.save("X_test_full.npy", X_test)
np.save("y_test_full.npy", y_test)

print("\n==============================")
print("DATASETS SAVED SUCCESSFULLY")
print("==============================")