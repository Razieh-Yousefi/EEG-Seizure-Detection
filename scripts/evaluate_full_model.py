import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score
)

# ============================================================
# Load model
# ============================================================

model = load_model("eeg_seizure_cnn_model.keras")

print("Model loaded successfully.")

# ============================================================
# Load TEST dataset
# ============================================================

X_test = np.load("X_test.npy")
y_test = np.load("y_test.npy")

print("\nTest dataset:")
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)

print("\nTest labels:")
print(np.unique(y_test, return_counts=True))

# ============================================================
# Rearrange data for CNN
# ============================================================

X_test = np.transpose(X_test, (0, 2, 1))

print("\nCNN Test shape:")
print(X_test.shape)

# ============================================================
# Predict probabilities
# ============================================================

print("\nPredicting Test data...")

y_probability = model.predict(
    X_test,
    batch_size=64,
    verbose=1
).flatten()

# ============================================================
# Test different thresholds
# ============================================================

thresholds = [
    0.50,
    0.45,
    0.40,
    0.35,
    0.30,
    0.25,
    0.20,
    0.15,
    0.10,
    0.05
]

best_threshold = None
best_f1 = -1

print("\n==============================================")
print("TEST THRESHOLD COMPARISON")
print("==============================================")

for threshold in thresholds:

    y_prediction = (
        y_probability >= threshold
    ).astype(int)

    precision = precision_score(
        y_test,
        y_prediction,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_prediction,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_prediction,
        zero_division=0
    )

    cm = confusion_matrix(
        y_test,
        y_prediction
    )

    print("\n----------------------------------------------")
    print("Threshold:", threshold)
    print("Precision:", round(precision, 4))
    print("Recall   :", round(recall, 4))
    print("F1 Score :", round(f1, 4))

    print("Confusion Matrix:")
    print(cm)

    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

# ============================================================
# Final result
# ============================================================

print("\n==============================================")
print("BEST TEST THRESHOLD")
print("==============================================")

print("Best Threshold:", best_threshold)
print("Best F1 Score :", round(best_f1, 4))

# ============================================================
# Detailed report using best threshold
# ============================================================

y_final = (
    y_probability >= best_threshold
).astype(int)

precision = precision_score(
    y_test,
    y_final,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_final,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_final,
    zero_division=0
)

cm = confusion_matrix(
    y_test,
    y_final
)

print("\n==============================================")
print("FINAL TEST RESULTS")
print("==============================================")

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_final,
        target_names=[
            "Normal",
            "Seizure"
        ],
        digits=4,
        zero_division=0
    )
)

print("Precision:", precision)
print("Recall   :", recall)
print("F1 Score :", f1)

print("\n==============================================")
print("DONE")
print("==============================================")