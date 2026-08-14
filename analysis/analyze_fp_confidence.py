# ================================================================
# analyze_fp_confidence.py
#
# Confidence analysis of false-positive predictions.
#
# PURPOSE:
# - Compare confidence of TP vs FP vs FN.
# - Identify high-confidence false positives.
# - Analyze FP probability distribution.
# - Analyze FP confidence separately for each patient.
#
# IMPORTANT:
# - Does NOT modify the model.
# - Does NOT modify X/y datasets.
# - Does NOT modify validation threshold.
# - Does NOT optimize or select a new threshold.
# ================================================================

import os
import json
import numpy as np


# ================================================================
# 1. CONFIGURATION
# ================================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
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

PROBABILITIES_FILE = os.path.join(
    BASE_DIR,
    "test_window_probabilities.npz"
)

OUTPUT_FILE = os.path.join(
    BASE_DIR,
    "fp_confidence_analysis_results.json"
)


# ================================================================
# 2. HEADER
# ================================================================

print()
print("=" * 70)
print("FALSE-POSITIVE CONFIDENCE ANALYSIS")
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
    Y_FILE,
    PATIENTS_FILE,
    TEST_INDICES_FILE,
    THRESHOLD_FILE,
    PROBABILITIES_FILE,
]

for path in required_files:

    if os.path.exists(path):

        print("[OK]", path)

    else:

        raise FileNotFoundError(
            f"Required file not found:\n{path}"
        )


# ================================================================
# 4. LOAD DATA
# ================================================================

print()
print("=" * 70)
print("2. LOADING DATA")
print("=" * 70)

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

print()
print("Labels shape:")
print(y.shape)

print()
print("Patients shape:")
print(patients.shape)

print()
print("Test indices shape:")
print(test_indices.shape)


# ================================================================
# 5. LOAD VALIDATION THRESHOLD
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

        preferred_keys = [
            "selected_threshold",
            "validation_threshold",
            "best_threshold",
            "threshold",
        ]

        for key in preferred_keys:

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


VALIDATION_THRESHOLD = find_threshold(
    threshold_data
)

if VALIDATION_THRESHOLD is None:

    raise RuntimeError(
        "Could not find validation threshold."
    )

print()
print(
    f"Validation threshold: "
    f"{VALIDATION_THRESHOLD:.4f}"
)


# ================================================================
# 6. LOAD INDIVIDUAL PROBABILITIES
# ================================================================

print()
print("=" * 70)
print("4. LOADING TEST PROBABILITIES")
print("=" * 70)

probability_data = np.load(
    PROBABILITIES_FILE
)

required_keys = [
    "test_indices",
    "patients",
    "labels",
    "probabilities",
]

for key in required_keys:

    if key not in probability_data:

        raise RuntimeError(
            f"Missing '{key}' in "
            f"test_window_probabilities.npz"
        )


saved_indices = np.asarray(
    probability_data["test_indices"]
)

saved_patients = np.asarray(
    probability_data["patients"]
)

saved_labels = np.asarray(
    probability_data["labels"]
)

probabilities = np.asarray(
    probability_data["probabilities"],
    dtype=np.float32
)


# ================================================================
# 7. VERIFY ALIGNMENT
# ================================================================

print()
print("=" * 70)
print("5. VERIFYING DATA ALIGNMENT")
print("=" * 70)

if len(probabilities) != len(test_indices):

    raise ValueError(
        "Probability count does not match "
        "test sample count."
    )

if not np.array_equal(
    saved_indices,
    test_indices
):

    raise ValueError(
        "Saved test indices do not match "
        "test_indices.npy."
    )

expected_labels = y[
    test_indices
]

expected_patients = patients[
    test_indices
]

if not np.array_equal(
    saved_labels,
    expected_labels
):

    raise ValueError(
        "Saved labels do not match "
        "y[test_indices]."
    )

if not np.array_equal(
    saved_patients,
    expected_patients
):

    raise ValueError(
        "Saved patients do not match "
        "patients[test_indices]."
    )

print()
print("[OK] Probability alignment verified.")

print(
    "Probability count:",
    len(probabilities)
)


# ================================================================
# 8. CREATE TEST PREDICTIONS
# ================================================================

print()
print("=" * 70)
print("6. CREATING TEST PREDICTIONS")
print("=" * 70)

labels = expected_labels
patients_test = expected_patients

predictions = (
    probabilities >= VALIDATION_THRESHOLD
).astype(np.int64)


# ------------------------------------------------
# Classification masks
# ------------------------------------------------

tp_mask = (
    (predictions == 1)
    & (labels == 1)
)

fp_mask = (
    (predictions == 1)
    & (labels == 0)
)

fn_mask = (
    (predictions == 0)
    & (labels == 1)
)

tn_mask = (
    (predictions == 0)
    & (labels == 0)
)


print()
print(
    "TP:",
    int(tp_mask.sum())
)

print(
    "FP:",
    int(fp_mask.sum())
)

print(
    "FN:",
    int(fn_mask.sum())
)

print(
    "TN:",
    int(tn_mask.sum())
)


# ================================================================
# 9. BASIC CONFIDENCE STATISTICS
# ================================================================

print()
print("=" * 70)
print("7. CONFIDENCE STATISTICS")
print("=" * 70)


def probability_statistics(
    name,
    values
):

    print()
    print(name)

    print(
        "Count  :",
        len(values)
    )

    if len(values) == 0:

        print("No samples.")

        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
            "std": None,
        }

    print(
        "Min    :",
        f"{values.min():.6f}"
    )

    print(
        "Max    :",
        f"{values.max():.6f}"
    )

    print(
        "Mean   :",
        f"{values.mean():.6f}"
    )

    print(
        "Median :",
        f"{np.median(values):.6f}"
    )

    print(
        "Std    :",
        f"{values.std():.6f}"
    )

    return {
        "count": int(len(values)),
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "std": float(values.std()),
    }


tp_probabilities = probabilities[
    tp_mask
]

fp_probabilities = probabilities[
    fp_mask
]

fn_probabilities = probabilities[
    fn_mask
]

tn_probabilities = probabilities[
    tn_mask
]


tp_stats = probability_statistics(
    "TRUE POSITIVES",
    tp_probabilities
)

fp_stats = probability_statistics(
    "FALSE POSITIVES",
    fp_probabilities
)

fn_stats = probability_statistics(
    "FALSE NEGATIVES",
    fn_probabilities
)

tn_stats = probability_statistics(
    "TRUE NEGATIVES",
    tn_probabilities
)


# ================================================================
# 10. FP CONFIDENCE BINS
# ================================================================

print()
print("=" * 70)
print("8. FALSE-POSITIVE CONFIDENCE BINS")
print("=" * 70)

confidence_bins = [
    (0.56, 0.60),
    (0.60, 0.70),
    (0.70, 0.80),
    (0.80, 0.90),
    (0.90, 0.95),
    (0.95, 1.01),
]

fp_bin_results = []

print()

for low, high in confidence_bins:

    mask = (
        fp_probabilities >= low
    ) & (
        fp_probabilities < high
    )

    count = int(
        mask.sum()
    )

    percentage = (
        100.0 * count / len(fp_probabilities)
        if len(fp_probabilities) > 0
        else 0.0
    )

    print(
        f"{low:.2f} - "
        f"{min(high, 1.00):.2f}"
        f" | FP={count:4d}"
        f" | {percentage:6.2f}%"
    )

    fp_bin_results.append({

        "lower_bound":
            float(low),

        "upper_bound":
            float(min(high, 1.00)),

        "count":
            count,

        "percentage_of_fp":
            float(percentage),
    })


# ================================================================
# 11. HIGH-CONFIDENCE FALSE POSITIVES
# ================================================================

print()
print("=" * 70)
print("9. HIGH-CONFIDENCE FALSE POSITIVES")
print("=" * 70)


confidence_levels = [
    0.70,
    0.80,
    0.90,
    0.95,
    0.99,
]


high_confidence_results = {}


for level in confidence_levels:

    count = int(
        np.sum(
            fp_probabilities >= level
        )
    )

    percentage = (
        100.0
        * count
        / len(fp_probabilities)
        if len(fp_probabilities) > 0
        else 0.0
    )

    total_test_percentage = (
        100.0
        * count
        / len(probabilities)
    )

    print(
        f"FP probability >= {level:.2f}"
        f" : {count:4d}"
        f" ({percentage:.2f}% of FP)"
        f" | {total_test_percentage:.2f}%"
        f" of all test windows"
    )

    high_confidence_results[
        f">={level:.2f}"
    ] = {

        "count":
            count,

        "percentage_of_fp":
            float(percentage),

        "percentage_of_all_test_windows":
            float(total_test_percentage),
    }


# ================================================================
# 12. TOP FALSE POSITIVES
# ================================================================

print()
print("=" * 70)
print("10. HIGHEST-CONFIDENCE FALSE POSITIVES")
print("=" * 70)


fp_indices = np.where(
    fp_mask
)[0]

sorted_fp_positions = fp_indices[
    np.argsort(
        probabilities[fp_indices]
    )[::-1]
]


top_fp = []

print()

for rank, position in enumerate(
    sorted_fp_positions[:20],
    start=1
):

    probability = float(
        probabilities[position]
    )

    original_index = int(
        test_indices[position]
    )

    patient = str(
        patients_test[position]
    )

    item = {

        "rank":
            rank,

        "test_position":
            int(position),

        "original_index":
            original_index,

        "patient":
            patient,

        "probability":
            probability,
    }

    top_fp.append(item)

    print(
        f"{rank:02d}. "
        f"{patient}"
        f" | probability="
        f"{probability:.6f}"
        f" | original_index="
        f"{original_index}"
    )


# ================================================================
# 13. PATIENT-LEVEL CONFIDENCE ANALYSIS
# ================================================================

print()
print("=" * 70)
print("11. PATIENT-LEVEL FP CONFIDENCE")
print("=" * 70)

patient_results = {}

unique_patients = np.unique(
    patients_test
)

for patient in unique_patients:

    patient_mask = (
        patients_test == patient
    )

    patient_fp_mask = (
        patient_mask
        & fp_mask
    )

    patient_tp_mask = (
        patient_mask
        & tp_mask
    )

    patient_fp_probs = probabilities[
        patient_fp_mask
    ]

    patient_tp_probs = probabilities[
        patient_tp_mask
    ]

    patient_name = str(
        patient
    )

    print()
    print("-" * 70)
    print(patient_name)
    print("-" * 70)

    print(
        "FP count:",
        len(patient_fp_probs)
    )

    if len(patient_fp_probs) > 0:

        print(
            "FP mean:",
            f"{patient_fp_probs.mean():.6f}"
        )

        print(
            "FP median:",
            f"{np.median(patient_fp_probs):.6f}"
        )

        print(
            "FP max:",
            f"{patient_fp_probs.max():.6f}"
        )

        print(
            "FP >= 0.90:",
            int(
                np.sum(
                    patient_fp_probs >= 0.90
                )
            )
        )

        print(
            "FP >= 0.95:",
            int(
                np.sum(
                    patient_fp_probs >= 0.95
                )
            )
        )

        print(
            "FP >= 0.99:",
            int(
                np.sum(
                    patient_fp_probs >= 0.99
                )
            )
        )

    else:

        print(
            "No false positives."
        )

    patient_results[
        patient_name
    ] = {

        "fp_count":
            int(len(patient_fp_probs)),

        "fp_min":
            (
                float(patient_fp_probs.min())
                if len(patient_fp_probs) > 0
                else None
            ),

        "fp_max":
            (
                float(patient_fp_probs.max())
                if len(patient_fp_probs) > 0
                else None
            ),

        "fp_mean":
            (
                float(patient_fp_probs.mean())
                if len(patient_fp_probs) > 0
                else None
            ),

        "fp_median":
            (
                float(np.median(patient_fp_probs))
                if len(patient_fp_probs) > 0
                else None
            ),

        "fp_ge_0_90":
            int(
                np.sum(
                    patient_fp_probs >= 0.90
                )
            ),

        "fp_ge_0_95":
            int(
                np.sum(
                    patient_fp_probs >= 0.95
                )
            ),

        "fp_ge_0_99":
            int(
                np.sum(
                    patient_fp_probs >= 0.99
                )
            ),

        "tp_count":
            int(len(patient_tp_probs)),

        "tp_mean":
            (
                float(patient_tp_probs.mean())
                if len(patient_tp_probs) > 0
                else None
            ),
    }


# ================================================================
# 14. FP VS TP CONFIDENCE GAP
# ================================================================

print()
print("=" * 70)
print("12. FP VS TP CONFIDENCE COMPARISON")
print("=" * 70)

if (
    len(tp_probabilities) > 0
    and len(fp_probabilities) > 0
):

    mean_gap = (
        tp_probabilities.mean()
        - fp_probabilities.mean()
    )

    median_gap = (
        np.median(tp_probabilities)
        - np.median(fp_probabilities)
    )

    print()
    print(
        "TP mean probability:",
        f"{tp_probabilities.mean():.6f}"
    )

    print(
        "FP mean probability:",
        f"{fp_probabilities.mean():.6f}"
    )

    print(
        "TP - FP mean gap:",
        f"{mean_gap:.6f}"
    )

    print()
    print(
        "TP median probability:",
        f"{np.median(tp_probabilities):.6f}"
    )

    print(
        "FP median probability:",
        f"{np.median(fp_probabilities):.6f}"
    )

    print(
        "TP - FP median gap:",
        f"{median_gap:.6f}"
    )

else:

    mean_gap = None
    median_gap = None

    print(
        "Not enough data for comparison."
    )


# ================================================================
# 15. SAVE RESULTS
# ================================================================

print()
print("=" * 70)
print("13. SAVING CONFIDENCE ANALYSIS")
print("=" * 70)

results = {

    "validation_threshold":
        float(VALIDATION_THRESHOLD),

    "test_samples":
        int(len(probabilities)),

    "classification_counts": {

        "tp":
            int(tp_mask.sum()),

        "fp":
            int(fp_mask.sum()),

        "fn":
            int(fn_mask.sum()),

        "tn":
            int(tn_mask.sum()),
    },

    "probability_statistics": {

        "true_positive":
            tp_stats,

        "false_positive":
            fp_stats,

        "false_negative":
            fn_stats,

        "true_negative":
            tn_stats,
    },

    "false_positive_confidence_bins":
        fp_bin_results,

    "high_confidence_false_positives":
        high_confidence_results,

    "top_20_highest_confidence_false_positives":
        top_fp,

    "patient_results":
        patient_results,

    "tp_fp_confidence_gap": {

        "mean_probability_gap":
            (
                float(mean_gap)
                if mean_gap is not None
                else None
            ),

        "median_probability_gap":
            (
                float(median_gap)
                if median_gap is not None
                else None
            ),
    },

    "note": (
        "Diagnostic confidence analysis only. "
        "No model, dataset, or validation "
        "threshold was modified. "
        "No threshold was selected using "
        "test data."
    ),
}


with open(
    OUTPUT_FILE,
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
    "[OK] Confidence analysis saved:"
)

print(
    OUTPUT_FILE
)


# ================================================================
# 16. FINAL
# ================================================================

print()
print("=" * 70)
print("FALSE-POSITIVE CONFIDENCE ANALYSIS COMPLETED")
print("=" * 70)

print()
print("Model was NOT modified.")
print("Dataset was NOT modified.")
print("Validation threshold was NOT modified.")
print("No threshold was selected using test data.")

print()
print("=" * 70)