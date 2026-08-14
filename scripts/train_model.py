import os
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input,
    Conv1D,
    MaxPooling1D,
    BatchNormalization,
    GlobalAveragePooling1D,
    Dense,
    Dropout
)

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau
)

from sklearn.utils.class_weight import compute_class_weight



# ============================================================
# DATA PATHS
# ============================================================

PROJECT_DIR = r"C:\Users\rezay\Desktop\EEG_Seizure_Project"

MODEL_FILE = os.path.join(
    PROJECT_DIR,
    "eeg_seizure_cnn_full_model.keras"
)

HISTORY_FILE = os.path.join(
    PROJECT_DIR,
    "training_history_full.npy"
)

X_TRAIN_FILE = os.path.join(PROJECT_DIR, "X_train_full.npy")
Y_TRAIN_FILE = os.path.join(PROJECT_DIR, "y_train_full.npy")

X_VAL_FILE = os.path.join(PROJECT_DIR, "X_val_full.npy")
Y_VAL_FILE = os.path.join(PROJECT_DIR, "y_val_full.npy")

X_TEST_FILE = os.path.join(PROJECT_DIR, "X_test_full.npy")
Y_TEST_FILE = os.path.join(PROJECT_DIR, "y_test_full.npy")

# ============================================================
# CHECK FILES
# ============================================================

print("\n==============================")
print("CHECKING DATA FILES")
print("==============================")

files = [
    X_TRAIN_FILE,
    Y_TRAIN_FILE,
    X_VAL_FILE,
    Y_VAL_FILE,
    X_TEST_FILE,
    Y_TEST_FILE
]

for file in files:
    print(file)

    if not os.path.exists(file):
        raise FileNotFoundError(
            "\nFILE NOT FOUND:\n" + file
        )

print("\nAll dataset files found successfully!")

# ============================================================
# RANDOM SEED
# ============================================================

np.random.seed(42)
tf.random.set_seed(42)


# ============================================================
# CHECK FILES
# ============================================================

print("=" * 70)
print("CHECKING DATA FILES")
print("=" * 70)

files = [
    X_TRAIN_FILE,
    Y_TRAIN_FILE,
    X_VAL_FILE,
    Y_VAL_FILE,
    X_TEST_FILE,
    Y_TEST_FILE
]

for file in files:

    print(file)

    if not os.path.exists(file):
        raise FileNotFoundError(
            "\nFILE NOT FOUND:\n" + file
        )

print("\nAll dataset files found.")


# ============================================================
# LOAD DATA
# ============================================================

print("\n" + "=" * 70)
print("LOADING DATA")
print("=" * 70)

X_train = np.load(X_TRAIN_FILE)
y_train = np.load(Y_TRAIN_FILE)

X_val = np.load(X_VAL_FILE)
y_val = np.load(Y_VAL_FILE)

X_test = np.load(X_TEST_FILE)
y_test = np.load(Y_TEST_FILE)


print("\nOriginal shapes:")

print("X_train:", X_train.shape)
print("y_train:", y_train.shape)

print("X_val:", X_val.shape)
print("y_val:", y_val.shape)

print("X_test:", X_test.shape)
print("y_test:", y_test.shape)


# ============================================================
# LABEL DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("ORIGINAL LABEL DISTRIBUTION")
print("=" * 70)

print("\nTRAIN:")

train_unique, train_counts = np.unique(
    y_train,
    return_counts=True
)

print(
    dict(
        zip(
            train_unique.tolist(),
            train_counts.tolist()
        )
    )
)

print("\nVALIDATION:")

val_unique, val_counts = np.unique(
    y_val,
    return_counts=True
)

print(
    dict(
        zip(
            val_unique.tolist(),
            val_counts.tolist()
        )
    )
)

print("\nTEST:")

test_unique, test_counts = np.unique(
    y_test,
    return_counts=True
)

print(
    dict(
        zip(
            test_unique.tolist(),
            test_counts.tolist()
        )
    )
)


# ============================================================
# OVERSAMPLING TRAIN ONLY
# ============================================================

print("\n" + "=" * 70)
print("OVERSAMPLING TRAINING SEIZURES")
print("=" * 70)

seizure_indices = np.where(
    y_train == 1
)[0]

normal_indices = np.where(
    y_train == 0
)[0]

print(
    "Original seizure samples:",
    len(seizure_indices)
)

print(
    "Original normal samples:",
    len(normal_indices)
)


# Target number of seizure samples
TARGET_SEIZURES = 500

if len(seizure_indices) < TARGET_SEIZURES:

    additional = (
        TARGET_SEIZURES -
        len(seizure_indices)
    )

    print(
        "Additional seizure samples:",
        additional
    )

    extra_indices = np.random.choice(
        seizure_indices,
        size=additional,
        replace=True
    )

    X_train = np.concatenate(
        [
            X_train,
            X_train[extra_indices]
        ],
        axis=0
    )

    y_train = np.concatenate(
        [
            y_train,
            y_train[extra_indices]
        ],
        axis=0
    )


print("\nAfter oversampling:")

print("X_train:", X_train.shape)
print("y_train:", y_train.shape)

unique_after, counts_after = np.unique(
    y_train,
    return_counts=True
)

print(
    dict(
        zip(
            unique_after.tolist(),
            counts_after.tolist()
        )
    )
)


# ============================================================
# SHUFFLE TRAIN DATA
# ============================================================

print("\n" + "=" * 70)
print("SHUFFLING TRAINING DATA")
print("=" * 70)

shuffle_indices = np.random.permutation(
    len(X_train)
)

X_train = X_train[
    shuffle_indices
]

y_train = y_train[
    shuffle_indices
]

print("Training data shuffled.")


# ============================================================
# CONVERT TO FLOAT32
# ============================================================

print("\n" + "=" * 70)
print("CONVERTING DATA TO FLOAT32")
print("=" * 70)

X_train = X_train.astype(
    np.float32
)

X_val = X_val.astype(
    np.float32
)

X_test = X_test.astype(
    np.float32
)

print("X_train dtype:", X_train.dtype)
print("X_val dtype:", X_val.dtype)
print("X_test dtype:", X_test.dtype)


# ============================================================
# TRANSPOSE FOR CNN
# ============================================================

print("\n" + "=" * 70)
print("PREPARING CNN DATA")
print("=" * 70)

X_train = np.transpose(
    X_train,
    (0, 2, 1)
)

X_val = np.transpose(
    X_val,
    (0, 2, 1)
)

X_test = np.transpose(
    X_test,
    (0, 2, 1)
)

print("\nCNN shapes:")

print("X_train:", X_train.shape)
print("X_val:", X_val.shape)
print("X_test:", X_test.shape)


# ============================================================
# CLASS WEIGHTS
# ============================================================

print("\n" + "=" * 70)
print("CALCULATING CLASS WEIGHTS")
print("=" * 70)

classes = np.array(
    [0, 1]
)

weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train
)

class_weights = {
    0: float(weights[0]),
    1: float(weights[1])
}

print(
    "Class weights:",
    class_weights
)


# ============================================================
# CREATE CNN MODEL
# ============================================================

print("\n" + "=" * 70)
print("CREATING CNN MODEL")
print("=" * 70)

model = Sequential([

    Input(
        shape=(1280, 23)
    ),

    Conv1D(
        filters=32,
        kernel_size=7,
        activation="relu",
        padding="same"
    ),

    BatchNormalization(),

    MaxPooling1D(
        pool_size=2
    ),

    Conv1D(
        filters=64,
        kernel_size=5,
        activation="relu",
        padding="same"
    ),

    BatchNormalization(),

    MaxPooling1D(
        pool_size=2
    ),

    Conv1D(
        filters=128,
        kernel_size=5,
        activation="relu",
        padding="same"
    ),

    BatchNormalization(),

    MaxPooling1D(
        pool_size=2
    ),

    Conv1D(
        filters=256,
        kernel_size=3,
        activation="relu",
        padding="same"
    ),

    BatchNormalization(),

    GlobalAveragePooling1D(),

    Dense(
        128,
        activation="relu"
    ),

    Dropout(
        0.4
    ),

    Dense(
        1,
        activation="sigmoid"
    )
])


model.summary()


# ============================================================
# COMPILE
# ============================================================

print("\n" + "=" * 70)
print("COMPILING MODEL")
print("=" * 70)

model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.0005
    ),

    loss="binary_crossentropy",

    metrics=[
        "accuracy",
        tf.keras.metrics.Precision(
            name="precision"
        ),
        tf.keras.metrics.Recall(
            name="recall"
        )
    ]
)


# ============================================================
# CALLBACKS
# ============================================================

early_stop = EarlyStopping(

    monitor="val_loss",

    patience=6,

    restore_best_weights=True,

    verbose=1
)


checkpoint = ModelCheckpoint(

    MODEL_FILE,

    monitor="val_loss",

    save_best_only=True,

    verbose=1
)


reduce_lr = ReduceLROnPlateau(

    monitor="val_loss",

    factor=0.5,

    patience=2,

    min_lr=1e-6,

    verbose=1
)


# ============================================================
# TRAIN
# ============================================================

print("\n" + "=" * 70)
print("STARTING TRAINING")
print("=" * 70)

history = model.fit(

    X_train,

    y_train,

    epochs=30,

    batch_size=64,

    validation_data=(
        X_val,
        y_val
    ),

    class_weight=class_weights,

    callbacks=[
        early_stop,
        checkpoint,
        reduce_lr
    ],

    verbose=1
)


# ============================================================
# LOAD BEST MODEL
# ============================================================

print("\n" + "=" * 70)
print("LOADING BEST MODEL")
print("=" * 70)

model = tf.keras.models.load_model(
    MODEL_FILE
)

print(
    "Best model loaded successfully."
)


# ============================================================
# VALIDATION EVALUATION
# ============================================================

print("\n" + "=" * 70)
print("VALIDATION EVALUATION")
print("=" * 70)

val_results = model.evaluate(
    X_val,
    y_val,
    batch_size=64,
    verbose=1
)

print(
    "\nValidation results:"
)

for name, value in zip(
    model.metrics_names,
    val_results
):

    print(
        name,
        ":",
        value
    )


# ============================================================
# TEST EVALUATION
# ============================================================

print("\n" + "=" * 70)
print("TEST EVALUATION")
print("=" * 70)

test_results = model.evaluate(
    X_test,
    y_test,
    batch_size=64,
    verbose=1
)

print(
    "\nTest results:"
)

for name, value in zip(
    model.metrics_names,
    test_results
):

    print(
        name,
        ":",
        value
    )


# ============================================================
# TEST PREDICTIONS
# ============================================================

print("\n" + "=" * 70)
print("TEST PREDICTIONS")
print("=" * 70)

probabilities = model.predict(
    X_test,
    batch_size=64,
    verbose=1
).ravel()


# ============================================================
# THRESHOLD ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("THRESHOLD ANALYSIS")
print("=" * 70)

thresholds = [
    0.50,
    0.30,
    0.20,
    0.15,
    0.10,
    0.05,
    0.02,
    0.01,
    0.005,
    0.001
]

for threshold in thresholds:

    predictions = (
        probabilities >= threshold
    ).astype(np.int8)

    actual_seizures = np.sum(
        y_test == 1
    )

    detected = np.sum(
        (predictions == 1) &
        (y_test == 1)
    )

    missed = (
        actual_seizures -
        detected
    )

    false_positive = np.sum(
        (predictions == 1) &
        (y_test == 0)
    )

    print(
        f"Threshold {threshold:>7.3f} | "
        f"Detected: {detected:>3} | "
        f"Missed: {missed:>3} | "
        f"False Positive: {false_positive:>4}"
    )


# ============================================================
# HIGHEST SEIZURE PROBABILITIES
# ============================================================

print("\n" + "=" * 70)
print("HIGHEST SEIZURE PROBABILITIES")
print("=" * 70)

top_indices = np.argsort(
    probabilities
)[::-1][:20]

for rank, index in enumerate(
    top_indices,
    start=1
):

    print(
        f"{rank:>2}. "
        f"Index={index:<5} "
        f"True={y_test[index]} "
        f"Probability={probabilities[index]:.8f}"
    )


# ============================================================
# SAVE HISTORY
# ============================================================

np.save(
    HISTORY_FILE,
    history.history
)


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)

print(
    "\nModel saved to:"
)

print(
    MODEL_FILE
)

print(
    "\nHistory saved to:"
)

print(
    HISTORY_FILE
)

print("\nDONE.")