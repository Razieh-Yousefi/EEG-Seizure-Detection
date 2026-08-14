import numpy as np
from sklearn.model_selection import train_test_split


# ============================================================
# Load full dataset
# ============================================================

X = np.load("X_full.npy")
y = np.load("y_full.npy")

print("Original dataset:")
print("X shape:", X.shape)
print("y shape:", y.shape)

print("\nOriginal label distribution:")
print(np.unique(y, return_counts=True))


# ============================================================
# First split:
# 70% Training
# 30% Temporary
# ============================================================

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)


# ============================================================
# Second split:
# Temporary -> 50% Validation + 50% Test
#
# Final:
# Training   = 70%
# Validation = 15%
# Test       = 15%
# ============================================================

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)


# ============================================================
# Print dataset sizes
# ============================================================

print("\nTraining data shape:")
print(X_train.shape)

print("Validation data shape:")
print(X_val.shape)

print("Testing data shape:")
print(X_test.shape)


# ============================================================
# Print label distributions
# ============================================================

print("\nTraining labels:")
print(np.unique(y_train, return_counts=True))

print("\nValidation labels:")
print(np.unique(y_val, return_counts=True))

print("\nTesting labels:")
print(np.unique(y_test, return_counts=True))


# ============================================================
# Save datasets
# ============================================================

np.save("X_train.npy", X_train)
np.save("y_train.npy", y_train)

np.save("X_val.npy", X_val)
np.save("y_val.npy", y_val)

np.save("X_test.npy", X_test)
np.save("y_test.npy", y_test)


print("\nTraining, validation and testing datasets saved successfully!")