import json
from pathlib import Path

import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_DIR / "results"

TEST_NPZ = RESULTS_DIR / "test_window_probabilities.npz"
THRESHOLD_JSON = RESULTS_DIR / "validation_threshold_results.json"

OUTPUT_JSON = RESULTS_DIR / "test_patient_temporal_rule_evaluation.json"

# Frozen from Validation analysis
WINDOW_THRESHOLD = None
Q95_THRESHOLD = 0.50
MIN_POSITIVE_FRACTION = 0.005


# ============================================================
# HELPERS
# ============================================================

def print_header(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def calculate_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))

    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = (
        2 * precision * sensitivity / (precision + sensitivity)
        if (precision + sensitivity)
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


def count_runs(binary_array):
    binary_array = np.asarray(binary_array, dtype=int)

    if len(binary_array) == 0:
        return 0, 0, 0.0

    runs = []
    current = 0

    for value in binary_array:
        if value == 1:
            current += 1
        else:
            if current > 0:
                runs.append(current)
                current = 0

    if current > 0:
        runs.append(current)

    if not runs:
        return 0, 0, 0.0

    return (
        len(runs),
        max(runs),
        float(np.mean(runs)),
    )


# ============================================================
# MAIN
# ============================================================

print_header("TEST PATIENT TEMPORAL RULE EVALUATION")

print()
print("Project directory:")
print(PROJECT_DIR)

print()
print("Results directory:")
print(RESULTS_DIR)


# ============================================================
# 1. CHECK INPUT FILES
# ============================================================

print_header("1. CHECKING INPUT FILES")

if not TEST_NPZ.exists():
    raise FileNotFoundError(f"Missing file: {TEST_NPZ}")

if not THRESHOLD_JSON.exists():
    raise FileNotFoundError(f"Missing file: {THRESHOLD_JSON}")

print(f"[OK] {TEST_NPZ}")
print(f"[OK] {THRESHOLD_JSON}")


# ============================================================
# 2. LOAD VALIDATION THRESHOLD
# ============================================================

print_header("2. LOADING FROZEN VALIDATION THRESHOLD")

with open(THRESHOLD_JSON, "r", encoding="utf-8") as f:
    threshold_data = json.load(f)


def find_threshold(obj):
    if isinstance(obj, dict):
        preferred_keys = [
            "threshold",
            "best_threshold",
            "validation_threshold",
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


WINDOW_THRESHOLD = find_threshold(threshold_data)

if WINDOW_THRESHOLD is None:
    raise RuntimeError(
        "Could not determine validation threshold "
        "from validation_threshold_results.json"
    )

print(f"Frozen window threshold: {WINDOW_THRESHOLD:.6f}")
print()
print("IMPORTANT:")
print("This threshold comes from Validation.")
print("No threshold optimization is performed on Test.")


# ============================================================
# 3. LOAD TEST DATA
# ============================================================

print_header("3. LOADING TEST DATA")

data = np.load(TEST_NPZ, allow_pickle=True)

print()
print("Available NPZ arrays:")

for key in data.files:
    print(f"  {key:30s} shape={data[key].shape}")


# ------------------------------------------------------------
# Resolve arrays robustly
# ------------------------------------------------------------

probabilities = np.asarray(data["probabilities"], dtype=float)
patients = np.asarray(data["patients"])
labels = np.asarray(data["labels"], dtype=int)

if "indices" in data.files:
    indices = np.asarray(data["indices"], dtype=int)
elif "test_indices" in data.files:
    indices = np.asarray(data["test_indices"], dtype=int)
elif "validation_indices" in data.files:
    indices = np.asarray(data["validation_indices"], dtype=int)
else:
    print()
    print("[INFO] No explicit indices array found.")
    print("[INFO] Using sequential sample indices.")
    indices = np.arange(len(probabilities), dtype=int)


print()
print(f"Test samples: {len(probabilities)}")
print(f"Probabilities shape: {probabilities.shape}")
print(f"Labels shape       : {labels.shape}")
print(f"Patients shape     : {patients.shape}")
print(f"Indices shape      : {indices.shape}")


# ============================================================
# 4. VERIFY DATA
# ============================================================

print_header("4. VERIFYING TEST DATA")

if not (
    len(probabilities)
    == len(labels)
    == len(patients)
    == len(indices)
):
    raise RuntimeError("Array lengths are not aligned.")

print("[OK] Arrays aligned.")

if not np.all(np.isfinite(probabilities)):
    raise RuntimeError("Probabilities contain NaN or infinite values.")

print("[OK] Probabilities finite.")


# ============================================================
# 5. BUILD PATIENT STATISTICS
# ============================================================

print_header("5. BUILDING TEST PATIENT STATISTICS")

unique_patients = np.unique(patients)

patient_stats = []

for patient in unique_patients:

    mask = patients == patient

    p = probabilities[mask]
    y = labels[mask]

    window_predictions = (p >= WINDOW_THRESHOLD).astype(int)

    total_windows = len(p)
    positive_windows = int(np.sum(window_predictions))

    positive_fraction = (
        positive_windows / total_windows
        if total_windows
        else 0.0
    )

    q95 = float(np.quantile(p, 0.95))

    positive_runs, max_positive_run, mean_positive_run = count_runs(
        window_predictions
    )

    true_label = int(np.any(y == 1))

    patient_prediction = int(
        (q95 >= Q95_THRESHOLD)
        and (positive_fraction >= MIN_POSITIVE_FRACTION)
    )

    stat = {
        "patient": str(patient),
        "total_windows": total_windows,
        "positive_windows": positive_windows,
        "positive_fraction": positive_fraction,
        "q95": q95,
        "max_probability": float(np.max(p)),
        "mean_probability": float(np.mean(p)),
        "median_probability": float(np.median(p)),
        "positive_runs": positive_runs,
        "max_positive_run": max_positive_run,
        "mean_positive_run": mean_positive_run,
        "true_label": true_label,
        "prediction": patient_prediction,
    }

    patient_stats.append(stat)

    print(
        f"{str(patient):8s} "
        f"windows={total_windows:4d} "
        f"positive={positive_windows:4d} "
        f"fraction={positive_fraction:.4f} "
        f"runs={positive_runs:3d} "
        f"max_run={max_positive_run:2d} "
        f"Q95={q95:.6f} "
        f"true={true_label} "
        f"pred={patient_prediction}"
    )


# ============================================================
# 6. FROZEN TEMPORAL RULE
# ============================================================

print_header("6. APPLYING FROZEN VALIDATION RULE")

print()
print("RULE:")
print(
    f"Q95 >= {Q95_THRESHOLD:.2f} "
    f"AND positive_fraction >= {MIN_POSITIVE_FRACTION:.3f}"
)

print()
print("No rule search is performed.")
print("No Test optimization is performed.")


# ============================================================
# 7. PATIENT-LEVEL METRICS
# ============================================================

y_true = [item["true_label"] for item in patient_stats]
y_pred = [item["prediction"] for item in patient_stats]

metrics = calculate_metrics(y_true, y_pred)

print_header("7. TEST PATIENT-LEVEL RESULTS")

print(json.dumps(metrics, indent=2))

print()
print("Patient-by-patient predictions:")

for item in patient_stats:
    print(
        f"{item['patient']:8s} "
        f"true={item['true_label']} "
        f"pred={item['prediction']} "
        f"Q95={item['q95']:.6f} "
        f"fraction={item['positive_fraction']:.6f} "
        f"positive_windows={item['positive_windows']}"
    )


# ============================================================
# 8. WINDOW-LEVEL BASELINE
# ============================================================

print_header("8. WINDOW-LEVEL BASELINE")

window_predictions = (
    probabilities >= WINDOW_THRESHOLD
).astype(int)

window_metrics = calculate_metrics(
    labels,
    window_predictions,
)

print(json.dumps(window_metrics, indent=2))


# ============================================================
# 9. COMPARISON
# ============================================================

print_header("9. WINDOW-LEVEL VS PATIENT-LEVEL")

print(
    f"Window sensitivity        : "
    f"{window_metrics['sensitivity']:.6f}"
)

print(
    f"Patient temporal sensitivity: "
    f"{metrics['sensitivity']:.6f}"
)

print(
    f"Window specificity        : "
    f"{window_metrics['specificity']:.6f}"
)

print(
    f"Patient temporal specificity: "
    f"{metrics['specificity']:.6f}"
)

print(
    f"Window precision          : "
    f"{window_metrics['precision']:.6f}"
)

print(
    f"Patient temporal precision : "
    f"{metrics['precision']:.6f}"
)

print(
    f"Window F1                 : "
    f"{window_metrics['f1']:.6f}"
)

print(
    f"Patient temporal F1        : "
    f"{metrics['f1']:.6f}"
)


# ============================================================
# 10. VALIDATION RULE REFERENCE
# ============================================================

print_header("10. VALIDATION RULE REFERENCE")

print()
print("The rule was frozen using Validation data.")

print()
print(f"Validation Q95 threshold      : {Q95_THRESHOLD:.3f}")
print(
    f"Validation minimum fraction   : "
    f"{MIN_POSITIVE_FRACTION:.3f}"
)

print()
print("Validation reference result:")
print("Sensitivity = 1.000000")
print("Specificity = 0.500000")
print("Precision   = 0.750000")
print("F1          = 0.857143")

print()
print("Test evaluation is independent.")
print("No Test-based optimization was performed.")


# ============================================================
# 11. SAVE RESULTS
# ============================================================

print_header("11. SAVING RESULTS")

output = {
    "analysis": "test_patient_temporal_rule_evaluation",

    "data_split": "test",

    "inputs": {
        "test_probability_file": str(TEST_NPZ),
        "validation_threshold_file": str(THRESHOLD_JSON),
    },

    "frozen_parameters": {
        "window_threshold": float(WINDOW_THRESHOLD),
        "q95_threshold": float(Q95_THRESHOLD),
        "minimum_positive_fraction": float(
            MIN_POSITIVE_FRACTION
        ),
    },

    "rule": (
        f"Q95 >= {Q95_THRESHOLD:.2f} "
        f"AND positive_fraction >= "
        f"{MIN_POSITIVE_FRACTION:.3f}"
    ),

    "patient_level_metrics": metrics,

    "window_level_metrics": window_metrics,

    "patients": patient_stats,

    "safety": {
        "model_modified": False,
        "dataset_modified": False,
        "validation_threshold_modified": False,
        "test_optimization_performed": False,
        "test_rule_search_performed": False,
    },
}

with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2)

print(f"[OK] Results saved:")
print(OUTPUT_JSON)


# ============================================================
# 12. FINAL SUMMARY
# ============================================================

print_header("TEST PATIENT TEMPORAL RULE EVALUATION COMPLETED")

print()
print("Frozen Validation rule:")
print(
    f"Q95 >= {Q95_THRESHOLD:.2f} "
    f"AND positive_fraction >= {MIN_POSITIVE_FRACTION:.3f}"
)

print()
print("TEST RESULTS:")
print(f"TP: {metrics['tp']}")
print(f"FP: {metrics['fp']}")
print(f"FN: {metrics['fn']}")
print(f"TN: {metrics['tn']}")
print(f"Sensitivity: {metrics['sensitivity']:.6f}")
print(f"Specificity: {metrics['specificity']:.6f}")
print(f"Precision: {metrics['precision']:.6f}")
print(f"F1: {metrics['f1']:.6f}")

print()
print("No model was modified.")
print("No dataset was modified.")
print("Validation threshold was NOT modified.")
print("No Test optimization was performed.")
print("No Test rule search was performed.")

print()
print("Output:")
print(OUTPUT_JSON)

print("=" * 70)