import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix


# ==========================
# Load test dataset
# ==========================

X_test = np.load("X_test.npy")
y_test = np.load("y_test.npy")


print("X_test shape:")
print(X_test.shape)

print()

print("y_test shape:")
print(y_test.shape)


# ==========================
# Prepare data for CNN
# ==========================

X_test = np.transpose(X_test, (0, 2, 1))


print()

print("New X_test shape:")
print(X_test.shape)



# ==========================
# Load trained model
# ==========================

model = load_model(
    "eeg_seizure_cnn_model.keras"
)



# ==========================
# Prediction
# ==========================

y_pred_prob = model.predict(X_test)



# ==========================
# Show seizure probabilities
# ==========================

seizure_probs = y_pred_prob[y_test == 1]


print("\nSeizure probabilities:")
print(seizure_probs)


print("\nFirst 20 prediction probabilities:")
print(y_pred_prob[:20])



# ==========================
# Threshold
# ==========================

threshold = 0.5


y_pred = (
    y_pred_prob > threshold
).astype(int).flatten()


# ==========================
# Evaluation
# ==========================

print("\nUsing threshold:")
print(threshold)


print("\nConfusion Matrix:")
print(
    confusion_matrix(
        y_test,
        y_pred
    )
)


print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred
    )
)



# ==========================
# Find missed seizures
# ==========================

missed = np.where(
    (y_test == 1) &
    (y_pred == 0)
)[0]


print("\nMissed seizure indexes:")
print(missed)


print("\nNumber of missed seizures:")
print(len(missed))