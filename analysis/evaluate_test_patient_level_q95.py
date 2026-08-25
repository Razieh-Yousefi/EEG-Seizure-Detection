import json
from pathlib import Path

import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_DIR / "results"

PROB_FILE = RESULTS_DIR / "test_window_probabilities.npz"
THRESHOLD_FILE = RESULTS_DIR / "validation_threshold_results.json"

OUTPUT_FILE = RESULTS_DIR / "test_patient_q95_evaluation.json"

Q = 95
REQUIRED_SENSITIVITY = 0.90


# ============================================================
# HELPERS
# ============================================================

def calculate_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = (
        2 * precision * sensitivity / (precision + sensitivity)
        if (precision + sensitivity) > 0
        else 0.0
    )

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "f1": f1,
    }


def patient_truth_from_windows(labels):
    """
    Patient is considered positive if at least one window
    has a positive seizure label.
    """
    return int(np.any(labels == 1))


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("TEST PATIENT-LEVEL Q95 EVALUATION")
print("=" * 70)

print("\nProject directory:")
print(PROJECT_DIR)

print("\nResults directory:")
print(RESULTS_DIR)


# ============================================================
# 1. CHECK INPUT FILES
# ============================================================

print("\n" + "=" * 70)
print("1. CHECKING INPUT FILES")
print("=" * 70)

if not PROB_FILE.exists():
    raise FileNotFoundError(
        f"Test probability file not found:\n{PROB_FILE}"
    )

if not THRESHOLD_FILE.exists():
    raise FileNotFoundError(
        f"Validation threshold file not found:\n{THRESHOLD_FILE}"
    )

print(f"[OK] {PROB_FILE}")
print(f"[OK] {THRESHOLD_FILE}")


# ============================================================
# 2. LOAD TEST PROBABILITIES
# ============================================================

print("\n" + "=" * 70)
print("2. LOADING TEST PROBABILITIES")
print("=" * 70)

data = np.load(PROB_FILE)

required_arrays = [
    "test_indices",
    "patients",
    "labels",
    "probabilities",
]

for name in required_arrays:
    if name not in data:
        raise KeyError(
            f"Required array '{name}' not found in:\n{PROB_FILE}"
        )

test_indices = np.asarray(data["test_indices"])
patients = np.asarray(data["patients"])
labels = np.asarray(data["labels"])
probabilities = np.asarray(data["probabilities"], dtype=float)

print(f"Test samples: {len(probabilities)}")
print(f"Indices shape: {test_indices.shape}")
print(f"Patients shape: {patients.shape}")
print(f"Labels shape: {labels.shape}")
print(f"Probability shape: {probabilities.shape}")


# ============================================================
# 3. VERIFY ALIGNMENT
# ============================================================

print("\n" + "=" * 70)
print("3. VERIFYING ARRAY ALIGNMENT")
print("=" * 70)

n = len(probabilities)

if not (
    len(test_indices) == n
    and len(patients) == n
    and len(labels) == n
):
    raise RuntimeError("Test arrays are not aligned.")

if not np.all(np.isfinite(probabilities)):
    raise RuntimeError("Probabilities contain NaN or infinite values.")

print("[OK] Arrays are aligned.")
print("[OK] Probabilities are finite.")


# ============================================================
# 4. LOAD VALIDATION THRESHOLD
# ============================================================

print("\n" + "=" * 70)
print("4. LOADING VALIDATION THRESHOLD")
print("=" * 70)

with open(THRESHOLD_FILE, "r", encoding="utf-8") as f:
    threshold_data = json.load(f)


def find_threshold(obj):
    """
    Recursively search common threshold field names.
    """

    if isinstance(obj, dict):

        preferred_keys = [
            "best_threshold",
            "validation_threshold",
            "threshold",
            "optimal_threshold",
        ]

        for key in preferred_keys:
            if key in obj:
                value = obj[key]

                if isinstance(value, (int, float)):
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


threshold = find_threshold(threshold_data)

if threshold is None:
    raise RuntimeError(
        "Could not find validation threshold in:\n"
        f"{THRESHOLD_FILE}"
    )

print(f"Validation threshold: {threshold:.6f}")


# ============================================================
# 5. WINDOW-LEVEL BASELINE
# ============================================================

print("\n" + "=" * 70)
print("5. WINDOW-LEVEL BASELINE")
print("=" * 70)

window_predictions = (
    probabilities >= threshold
).astype(int)

window_metrics = calculate_metrics(
    labels,
    window_predictions,
)

for key, value in window_metrics.items():

    if isinstance(value, float):
        print(f"{key}: {value:.6f}")
    else:
        print(f"{key}: {value}")


# ============================================================
# 6. IDENTIFY TEST PATIENTS
# ============================================================

print("\n" + "=" * 70)
print("6. IDENTIFYING TEST PATIENTS")
print("=" * 70)

unique_patients = np.unique(patients)

print(f"Number of test patients: {len(unique_patients)}")

for patient in unique_patients:

    mask = patients == patient

    print(
        f"{patient}: "
        f"{np.sum(mask)} windows"
    )


# ============================================================
# 7. PATIENT-LEVEL Q95 CALCULATION
# ============================================================

print("\n" + "=" * 70)
print("7. CALCULATING PATIENT-LEVEL Q95")
print("=" * 70)

patient_results = []

patient_true = []
patient_pred = []

for patient in unique_patients:

    mask = patients == patient

    patient_probs = probabilities[mask]
    patient_labels = labels[mask]

    q95 = float(np.percentile(patient_probs, Q))

    max_probability = float(np.max(patient_probs))
    mean_probability = float(np.mean(patient_probs))
    median_probability = float(np.median(patient_probs))

    positive_windows = int(
        np.sum(patient_probs >= threshold)
    )

    true_seizure_windows = int(
        np.sum(patient_labels == 1)
    )

    true_patient_label = patient_truth_from_windows(
        patient_labels
    )

    patient_prediction = int(
        q95 >= threshold
    )

    patient_true.append(true_patient_label)
    patient_pred.append(patient_prediction)

    patient_results.append(
        {
            "patient": str(patient),
            "total_windows": int(len(patient_probs)),
            "positive_windows_at_threshold": positive_windows,
            "true_seizure_windows": true_seizure_windows,
            "true_patient_label": true_patient_label,
            "q95_probability": q95,
            "max_probability": max_probability,
            "mean_probability": mean_probability,
            "median_probability": median_probability,
            "patient_prediction": patient_prediction,
            "patient_prediction_label": (
                "positive"
                if patient_prediction == 1
                else "negative"
            ),
        }
    )

    print("-" * 70)
    print(f"Patient: {patient}")
    print(f"Total windows       : {len(patient_probs)}")
    print(f"Positive windows    : {positive_windows}")
    print(f"True seizure labels : {true_seizure_windows}")
    print(f"Q95 probability     : {q95:.6f}")
    print(f"Max probability     : {max_probability:.6f}")
    print(f"Mean probability    : {mean_probability:.6f}")
    print(f"Median probability  : {median_probability:.6f}")
    print(f"True patient label  : {true_patient_label}")
    print(f"Patient prediction   : {patient_prediction}")


# ============================================================
# 8. PATIENT-LEVEL METRICS
# ============================================================

print("\n" + "=" * 70)
print("8. PATIENT-LEVEL Q95 RESULTS")
print("=" * 70)

patient_metrics = calculate_metrics(
    patient_true,
    patient_pred,
)

for key, value in patient_metrics.items():

    if isinstance(value, float):
        print(f"{key}: {value:.6f}")
    else:
        print(f"{key}: {value}")


# ============================================================
# 9. COMPARE WITH WINDOW LEVEL
# ============================================================

print("\n" + "=" * 70)
print("9. WINDOW-LEVEL vs PATIENT-LEVEL")
print("=" * 70)

print(
    f"Window sensitivity    : "
    f"{window_metrics['sensitivity']:.6f}"
)

print(
    f"Patient Q95 sensitivity: "
    f"{patient_metrics['sensitivity']:.6f}"
)

print(
    f"Window precision      : "
    f"{window_metrics['precision']:.6f}"
)

print(
    f"Patient Q95 precision : "
    f"{patient_metrics['precision']:.6f}"
)

print(
    f"Window F1             : "
    f"{window_metrics['f1']:.6f}"
)

print(
    f"Patient Q95 F1        : "
    f"{patient_metrics['f1']:.6f}"
)


# ============================================================
# 10. SAFETY CHECK
# ============================================================

print("\n" + "=" * 70)
print("10. VALIDATION OF EVALUATION PROCEDURE")
print("=" * 70)

print(
    "Required sensitivity for reference: "
    f"{REQUIRED_SENSITIVITY:.2f}"
)

if patient_metrics["sensitivity"] >= REQUIRED_SENSITIVITY:
    print(
        "[INFO] Patient-level Q95 reached the reference "
        "sensitivity requirement."
    )
else:
    print(
        "[INFO] Patient-level Q95 did NOT reach the "
        "reference sensitivity requirement."
    )

print(
    "\nIMPORTANT: This is TEST evaluation only."
)

print(
    "No threshold optimization was performed on Test."
)

print(
    "The threshold came from Validation."
)


# ============================================================
# 11. SAVE RESULTS
# ============================================================

print("\n" + "=" * 70)
print("11. SAVING RESULTS")
print("=" * 70)

output = {
    "analysis": "test_patient_level_q95_evaluation",

    "threshold_source": (
        "validation_threshold_results.json"
    ),

    "threshold": threshold,

    "aggregation_method": "q95",

    "q": Q,

    "window_level": window_metrics,

    "patient_level_q95": patient_metrics,

    "patient_results": patient_results,

    "num_test_windows": int(len(probabilities)),

    "num_test_patients": int(len(unique_patients)),

    "test_patients": [
        str(x) for x in unique_patients
    ],

    "test_was_used_for_optimization": False,

    "threshold_was_modified": False,

    "model_was_modified": False,

    "dataset_was_modified": False,
}

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        output,
        f,
        indent=2,
        ensure_ascii=False,
    )

print(f"[OK] Results saved:")
print(OUTPUT_FILE)


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 70)
print("TEST PATIENT-LEVEL Q95 EVALUATION COMPLETED")
print("=" * 70)

print("\nNo model was modified.")
print("No dataset was modified.")
print("Validation threshold was NOT modified.")
print("No optimization was performed on Test.")

print("\nOutput:")
print(OUTPUT_FILE)

print("=" * 70)