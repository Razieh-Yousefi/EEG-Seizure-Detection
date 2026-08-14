import os
import json
import numpy as np


# ================================================================
# 1. CONFIGURATION
# ================================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

X_FILE = os.path.join(
    BASE_DIR,
    "X_chbmit_full.npy"
)

Y_FILE = os.path.join(
    BASE_DIR,
    "y_chbmit_full.npy"
)

PATIENTS_FILE = os.path.join(
    BASE_DIR,
    "patients_chbmit_full.npy"
)

TEST_INDICES_FILE = os.path.join(
    BASE_DIR,
    "test_indices.npy"
)

THRESHOLD_FILE = os.path.join(
    BASE_DIR,
    "validation_threshold_results.json"
)

PROB_FILE = os.path.join(
    BASE_DIR,
    "test_window_probabilities.npz"
)

OUTPUT_JSON = os.path.join(
    BASE_DIR,
    "fp_tp_examples.json"
)

OUTPUT_NPZ = os.path.join(
    BASE_DIR,
    "fp_tp_examples.npz"
)

# Number of examples from each category
N_EXAMPLES = 20


# ================================================================
# 2. HEADER
# ================================================================

print()
print("=" * 70)
print("FP / TP EEG EXAMPLE EXTRACTION")
print("=" * 70)

print()
print("Base directory:")
print(BASE_DIR)


# ================================================================
# 3. CHECK FILES
# ================================================================

print()
print("=" * 70)
print("1. CHECKING INPUT FILES")
print("=" * 70)

required_files = [
    X_FILE,
    Y_FILE,
    PATIENTS_FILE,
    TEST_INDICES_FILE,
    THRESHOLD_FILE,
    PROB_FILE,
]

for path in required_files:

    if os.path.exists(path):
        print("[OK]", path)
    else:
        raise FileNotFoundError(path)


# ================================================================
# 4. LOAD DATA
# ================================================================

print()
print("=" * 70)
print("2. LOADING DATA")
print("=" * 70)

X = np.load(
    X_FILE,
    mmap_mode="r"
)

y = np.load(
    Y_FILE
)

patients = np.load(
    PATIENTS_FILE,
    allow_pickle=True
)

test_indices = np.load(
    TEST_INDICES_FILE
)

prob_data = np.load(
    PROB_FILE
)

probabilities = np.asarray(
    prob_data["probabilities"],
    dtype=np.float32
)

print()
print("X shape:", X.shape)
print("y shape:", y.shape)
print("patients shape:", patients.shape)
print("test indices:", test_indices.shape)
print("probabilities:", probabilities.shape)


# ================================================================
# 5. LOAD THRESHOLD
# ================================================================

print()
print("=" * 70)
print("3. LOADING VALIDATION THRESHOLD")
print("=" * 70)

with open(
    THRESHOLD_FILE,
    "r",
    encoding="utf-8"
) as f:
    threshold_data = json.load(f)


def find_threshold(obj):

    if isinstance(obj, dict):

        keys = [
            "selected_threshold",
            "validation_threshold",
            "best_threshold",
            "threshold",
        ]

        for key in keys:

            if key in obj:

                value = obj[key]

                if isinstance(
                    value,
                    (int, float)
                ):
                    return float(value)

        for value in obj.values():

            result = find_threshold(value)

            if result is not None:
                return result

    elif isinstance(obj, list):

        for item in obj:

            result = find_threshold(item)

            if result is not None:
                return result

    return None


THRESHOLD = find_threshold(
    threshold_data
)

if THRESHOLD is None:
    raise RuntimeError(
        "Validation threshold not found."
    )

print()
print(
    f"Validation threshold: {THRESHOLD:.4f}"
)


# ================================================================
# 6. VERIFY ALIGNMENT
# ================================================================

print()
print("=" * 70)
print("4. VERIFYING ALIGNMENT")
print("=" * 70)

saved_indices = np.asarray(
    prob_data["test_indices"]
)

saved_labels = np.asarray(
    prob_data["labels"]
)

saved_patients = np.asarray(
    prob_data["patients"]
)

y_test = y[test_indices]

patients_test = patients[test_indices]

if not np.array_equal(
    saved_indices,
    test_indices
):
    raise ValueError(
        "Test index mismatch."
    )

if not np.array_equal(
    saved_labels,
    y_test
):
    raise ValueError(
        "Label mismatch."
    )

if not np.array_equal(
    saved_patients,
    patients_test
):
    raise ValueError(
        "Patient mismatch."
    )

print()
print("[OK] Alignment verified.")


# ================================================================
# 7. CREATE GROUPS
# ================================================================

print()
print("=" * 70)
print("5. CREATING TP / FP / TN GROUPS")
print("=" * 70)

predictions = (
    probabilities >= THRESHOLD
).astype(np.int64)

tp_mask = (
    (predictions == 1)
    & (y_test == 1)
)

fp_mask = (
    (predictions == 1)
    & (y_test == 0)
)

tn_mask = (
    (predictions == 0)
    & (y_test == 0)
)

fn_mask = (
    (predictions == 0)
    & (y_test == 1)
)

print()
print("TP:", int(tp_mask.sum()))
print("FP:", int(fp_mask.sum()))
print("TN:", int(tn_mask.sum()))
print("FN:", int(fn_mask.sum()))


# ================================================================
# 8. SELECT EXAMPLES
# ================================================================

print()
print("=" * 70)
print("6. SELECTING REPRESENTATIVE EXAMPLES")
print("=" * 70)


fp_positions = np.where(
    fp_mask
)[0]

tp_positions = np.where(
    tp_mask
)[0]


# ------------------------------------------------
# FP HIGH CONFIDENCE
# ------------------------------------------------

fp_high = fp_positions[
    np.argsort(
        probabilities[fp_positions]
    )[::-1]
][:N_EXAMPLES]


# ------------------------------------------------
# FP MEDIUM CONFIDENCE
# ------------------------------------------------

fp_sorted = fp_positions[
    np.argsort(
        probabilities[fp_positions]
    )
]

if len(fp_sorted) > N_EXAMPLES:

    middle_start = (
        len(fp_sorted) // 2
        - N_EXAMPLES // 2
    )

    fp_medium = fp_sorted[
        middle_start:
        middle_start + N_EXAMPLES
    ]

else:

    fp_medium = fp_sorted


# ------------------------------------------------
# TP
# ------------------------------------------------

tp_examples = tp_positions[
    np.argsort(
        probabilities[tp_positions]
    )[::-1]
][:N_EXAMPLES]


print()
print(
    "High-confidence FP examples:",
    len(fp_high)
)

print(
    "Medium-confidence FP examples:",
    len(fp_medium)
)

print(
    "TP examples:",
    len(tp_examples)
)


# ================================================================
# 9. FEATURE EXTRACTION FOR EXAMPLES
# ================================================================

def extract_features(
    position
):

    signal = np.asarray(
        X[
            test_indices[position]
        ],
        dtype=np.float32
    )

    # signal shape:
    # channels x time

    channel_rms = np.sqrt(
        np.mean(
            signal ** 2,
            axis=1
        )
    )

    channel_ptp = (
        np.max(
            signal,
            axis=1
        )
        -
        np.min(
            signal,
            axis=1
        )
    )

    channel_max_abs = np.max(
        np.abs(signal),
        axis=1
    )

    dominant_rms_channel = int(
        np.argmax(channel_rms)
    )

    dominant_ptp_channel = int(
        np.argmax(channel_ptp)
    )

    dominant_max_channel = int(
        np.argmax(channel_max_abs)
    )

    return {

        "test_position": int(
            position
        ),

        "original_index": int(
            test_indices[position]
        ),

        "patient": str(
            patients_test[position]
        ),

        "label": int(
            y_test[position]
        ),

        "prediction": int(
            predictions[position]
        ),

        "probability": float(
            probabilities[position]
        ),

        "global_rms": float(
            np.sqrt(
                np.mean(
                    signal ** 2
                )
            )
        ),

        "global_std": float(
            np.std(signal)
        ),

        "global_ptp": float(
            np.max(signal)
            -
            np.min(signal)
        ),

        "global_max_abs": float(
            np.max(
                np.abs(signal)
            )
        ),

        "dominant_rms_channel": (
            dominant_rms_channel
        ),

        "dominant_ptp_channel": (
            dominant_ptp_channel
        ),

        "dominant_max_abs_channel": (
            dominant_max_channel
        ),

        "channel_rms": [
            float(v)
            for v in channel_rms
        ],

        "channel_ptp": [
            float(v)
            for v in channel_ptp
        ],

        "channel_max_abs": [
            float(v)
            for v in channel_max_abs
        ],
    }


# ================================================================
# 10. BUILD EXAMPLE METADATA
# ================================================================

print()
print("=" * 70)
print("7. EXTRACTING EXAMPLE FEATURES")
print("=" * 70)


high_fp_metadata = []

for position in fp_high:

    high_fp_metadata.append(
        extract_features(position)
    )


medium_fp_metadata = []

for position in fp_medium:

    medium_fp_metadata.append(
        extract_features(position)
    )


tp_metadata = []

for position in tp_examples:

    tp_metadata.append(
        extract_features(position)
    )


print()
print("[OK] Example features extracted.")


# ================================================================
# 11. LOAD SIGNALS
# ================================================================

print()
print("=" * 70)
print("8. EXTRACTING RAW EEG SIGNALS")
print("=" * 70)


def load_signals(
    positions
):

    signals = []

    for position in positions:

        signal = np.asarray(
            X[
                test_indices[position]
            ],
            dtype=np.float32
        )

        signals.append(signal)

    if len(signals) == 0:

        return np.empty(
            (0, X.shape[1], X.shape[2]),
            dtype=np.float32
        )

    return np.stack(
        signals,
        axis=0
    )


high_fp_signals = load_signals(
    fp_high
)

medium_fp_signals = load_signals(
    fp_medium
)

tp_signals = load_signals(
    tp_examples
)


print()
print(
    "High FP signals:",
    high_fp_signals.shape
)

print(
    "Medium FP signals:",
    medium_fp_signals.shape
)

print(
    "TP signals:",
    tp_signals.shape
)


# ================================================================
# 12. SAVE RAW SIGNALS
# ================================================================

print()
print("=" * 70)
print("9. SAVING EEG EXAMPLES")
print("=" * 70)

np.savez_compressed(
    OUTPUT_NPZ,

    high_confidence_fp_signals=(
        high_fp_signals
    ),

    medium_confidence_fp_signals=(
        medium_fp_signals
    ),

    tp_signals=(
        tp_signals
    ),

    high_confidence_fp_positions=(
        fp_high
    ),

    medium_confidence_fp_positions=(
        fp_medium
    ),

    tp_positions=(
        tp_examples
    )
)


print()
print(
    "[OK] EEG examples saved:"
)

print(
    OUTPUT_NPZ
)


# ================================================================
# 13. SAVE METADATA
# ================================================================

print()
print("=" * 70)
print("10. SAVING EXAMPLE METADATA")
print("=" * 70)


results = {

    "validation_threshold": float(
        THRESHOLD
    ),

    "window_shape": [
        int(X.shape[1]),
        int(X.shape[2])
    ],

    "window_duration_seconds": 5.0,

    "high_confidence_fp": (
        high_fp_metadata
    ),

    "medium_confidence_fp": (
        medium_fp_metadata
    ),

    "true_positive_examples": (
        tp_metadata
    ),

    "note": (
        "Representative EEG windows extracted "
        "for diagnostic analysis only. "
        "No model, dataset, or threshold was modified."
    ),
}


with open(
    OUTPUT_JSON,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        results,
        f,
        indent=2,
        ensure_ascii=False
    )


print()
print(
    "[OK] Metadata saved:"
)

print(
    OUTPUT_JSON
)


# ================================================================
# 14. PRINT SUMMARY
# ================================================================

print()
print("=" * 70)
print("11. EXAMPLE SUMMARY")
print("=" * 70)


def print_summary(
    name,
    metadata
):

    print()
    print(
        f"{name}:"
    )

    for i, item in enumerate(
        metadata[:10],
        start=1
    ):

        print(
            f"{i:02d}. "
            f"{item['patient']} | "
            f"prob="
            f"{item['probability']:.6f} | "
            f"RMS="
            f"{item['global_rms']:.6f} | "
            f"PTP="
            f"{item['global_ptp']:.6f} | "
            f"dominant_channel="
            f"{item['dominant_rms_channel']}"
        )


print_summary(
    "HIGH-CONFIDENCE FP",
    high_fp_metadata
)

print_summary(
    "MEDIUM-CONFIDENCE FP",
    medium_fp_metadata
)

print_summary(
    "TRUE POSITIVE",
    tp_metadata
)


# ================================================================
# 15. FINAL
# ================================================================

print()
print("=" * 70)
print("FP / TP EEG EXAMPLE EXTRACTION COMPLETED")
print("=" * 70)

print()
print("Model was NOT modified.")
print("Dataset was NOT modified.")
print("Threshold was NOT modified.")
print("No threshold optimization was performed.")

print()
print("=" * 70)