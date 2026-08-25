
import json
from pathlib import Path

import numpy as np


# ============================================================
# PATIENT-LEVEL AGGREGATION ANALYSIS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_DIR / "results"

PROB_FILE = RESULTS_DIR / "test_window_probabilities.npz"
THRESHOLD_FILE = RESULTS_DIR / "validation_threshold_results.json"

OUTPUT_FILE = RESULTS_DIR / "patient_level_aggregation_analysis.json"


print("=" * 70)
print("PATIENT-LEVEL AGGREGATION ANALYSIS")
print("=" * 70)

print()
print("Project directory:")
print(PROJECT_DIR)

print()
print("Results directory:")
print(RESULTS_DIR)


# ============================================================
# 1. CHECK INPUT FILES
# ============================================================

print()
print("=" * 70)
print("1. CHECKING INPUT FILES")
print("=" * 70)

if not PROB_FILE.exists():
    raise FileNotFoundError(f"Probability file not found:\n{PROB_FILE}")

if not THRESHOLD_FILE.exists():
    raise FileNotFoundError(f"Threshold file not found:\n{THRESHOLD_FILE}")

print(f"[OK] {PROB_FILE}")
print(f"[OK] {THRESHOLD_FILE}")


# ============================================================
# 2. LOAD TEST PROBABILITIES
# ============================================================

print()
print("=" * 70)
print("2. LOADING TEST PROBABILITIES")
print("=" * 70)

data = np.load(PROB_FILE, allow_pickle=True)

required_keys = [
    "test_indices",
    "patients",
    "labels",
    "probabilities",
]

for key in required_keys:
    if key not in data:
        raise KeyError(
            f"Required array '{key}' not found in:\n{PROB_FILE}"
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

print()
print("=" * 70)
print("3. VERIFYING ARRAY ALIGNMENT")
print("=" * 70)

n = len(probabilities)

if not (
    len(test_indices) == n
    and len(patients) == n
    and len(labels) == n
):
    raise RuntimeError(
        "Test indices, patients, labels and probabilities "
        "are not aligned."
    )

if not np.all(np.isfinite(probabilities)):
    raise RuntimeError("Probabilities contain non-finite values.")

print("[OK] Arrays are aligned.")
print("[OK] Probabilities are finite.")


# ============================================================
# 4. LOAD VALIDATION THRESHOLD
# ============================================================

print()
print("=" * 70)
print("4. LOADING VALIDATION THRESHOLD")
print("=" * 70)

with open(THRESHOLD_FILE, "r", encoding="utf-8") as f:
    threshold_data = json.load(f)

threshold = None

# Support several possible JSON structures.
candidate_paths = [
    ("best_threshold",),
    ("threshold",),
    ("validation_threshold",),
    ("best_result", "threshold"),
    ("best", "threshold"),
]

for path in candidate_paths:
    obj = threshold_data

    try:
        for key in path:
            obj = obj[key]

        if isinstance(obj, (int, float)):
            threshold = float(obj)
            break
    except (KeyError, TypeError):
        pass

if threshold is None:
    raise RuntimeError(
        "Could not find validation threshold in:\n"
        f"{THRESHOLD_FILE}"
    )

print(f"Validation threshold: {threshold}")


# ============================================================
# 5. WINDOW-LEVEL BASELINE
# ============================================================

print()
print("=" * 70)
print("5. WINDOW-LEVEL BASELINE")
print("=" * 70)

window_pred = probabilities >= threshold

tp = int(np.sum((window_pred == 1) & (labels == 1)))
fp = int(np.sum((window_pred == 1) & (labels == 0)))
fn = int(np.sum((window_pred == 0) & (labels == 1)))
tn = int(np.sum((window_pred == 0) & (labels == 0)))

sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
specificity = tn / (tn + fp) if (tn + fp) else 0.0
precision = tp / (tp + fp) if (tp + fp) else 0.0
f1 = (
    2 * precision * sensitivity / (precision + sensitivity)
    if (precision + sensitivity)
    else 0.0
)

print(f"TP: {tp}")
print(f"FP: {fp}")
print(f"FN: {fn}")
print(f"TN: {tn}")
print(f"Sensitivity: {sensitivity:.6f}")
print(f"Specificity: {specificity:.6f}")
print(f"Precision: {precision:.6f}")
print(f"F1: {f1:.6f}")


# ============================================================
# 6. PATIENT IDENTIFICATION
# ============================================================

print()
print("=" * 70)
print("6. IDENTIFYING TEST PATIENTS")
print("=" * 70)

unique_patients = np.unique(patients)

print(f"Number of test patients: {len(unique_patients)}")
print()

for patient in unique_patients:
    count = int(np.sum(patients == patient))
    print(f"{patient}: {count} windows")


# ============================================================
# 7. PATIENT-LEVEL AGGREGATION
# ============================================================

print()
print("=" * 70)
print("7. PATIENT-LEVEL AGGREGATION")
print("=" * 70)

patient_results = {}

for patient in unique_patients:

    mask = patients == patient

    patient_probs = probabilities[mask]
    patient_labels = labels[mask]

    positive_windows = patient_probs >= threshold

    true_positive_windows = int(
        np.sum((positive_windows == 1) & (patient_labels == 1))
    )

    false_positive_windows = int(
        np.sum((positive_windows == 1) & (patient_labels == 0))
    )

    total_windows = int(np.sum(mask))
    positive_predictions = int(np.sum(positive_windows))
    true_positive_labels = int(np.sum(patient_labels == 1))

    patient_results[str(patient)] = {
        "total_windows": total_windows,
        "positive_windows": positive_predictions,
        "true_positive_labels": true_positive_labels,
        "true_positive_windows": true_positive_windows,
        "false_positive_windows": false_positive_windows,
        "max_probability": float(np.max(patient_probs)),
        "mean_probability": float(np.mean(patient_probs)),
        "median_probability": float(np.median(patient_probs)),
        "q90_probability": float(np.quantile(patient_probs, 0.90)),
        "q95_probability": float(np.quantile(patient_probs, 0.95)),
        "q99_probability": float(np.quantile(patient_probs, 0.99)),
    }


# ============================================================
# 8. PRINT PATIENT RESULTS
# ============================================================

print()

for patient in unique_patients:

    result = patient_results[str(patient)]

    print("-" * 70)
    print(f"Patient: {patient}")
    print(f"Total windows       : {result['total_windows']}")
    print(f"Positive windows    : {result['positive_windows']}")
    print(f"True seizure labels : {result['true_positive_labels']}")
    print(f"TP windows          : {result['true_positive_windows']}")
    print(f"FP windows          : {result['false_positive_windows']}")
    print(f"Max probability     : {result['max_probability']:.6f}")
    print(f"Mean probability    : {result['mean_probability']:.6f}")
    print(f"Median probability : {result['median_probability']:.6f}")
    print(f"Q90                 : {result['q90_probability']:.6f}")
    print(f"Q95                 : {result['q95_probability']:.6f}")
    print(f"Q99                 : {result['q99_probability']:.6f}")


# ============================================================
# 9. PATIENT-LEVEL RULE SEARCH
# ============================================================

print()
print("=" * 70)
print("9. PATIENT-LEVEL RULE SEARCH")
print("=" * 70)

print()
print(
    "Testing whether requiring multiple positive windows "
    "per patient can reduce false alarms."
)

run_lengths = [1, 2, 3, 4, 5, 10]

patient_rule_results = []

for minimum_positive_windows in run_lengths:

    predicted_positive_patients = set()

    for patient in unique_patients:

        result = patient_results[str(patient)]

        if result["positive_windows"] >= minimum_positive_windows:
            predicted_positive_patients.add(str(patient))

    true_positive_patients = 0
    false_positive_patients = 0
    actual_positive_patients = 0

    for patient in unique_patients:

        patient_string = str(patient)
        result = patient_results[patient_string]

        actual_positive = result["true_positive_labels"] > 0
        predicted_positive = (
            patient_string in predicted_positive_patients
        )

        if actual_positive:
            actual_positive_patients += 1

        if predicted_positive and actual_positive:
            true_positive_patients += 1

        elif predicted_positive and not actual_positive:
            false_positive_patients += 1

    false_negative_patients = (
        actual_positive_patients - true_positive_patients
    )

    patient_sensitivity = (
        true_positive_patients / actual_positive_patients
        if actual_positive_patients
        else 0.0
    )

    patient_precision = (
        true_positive_patients
        / (true_positive_patients + false_positive_patients)
        if (true_positive_patients + false_positive_patients)
        else 0.0
    )

    patient_f1 = (
        2
        * patient_precision
        * patient_sensitivity
        / (patient_precision + patient_sensitivity)
        if (patient_precision + patient_sensitivity)
        else 0.0
    )

    patient_rule_results.append(
        {
            "minimum_positive_windows": minimum_positive_windows,
            "tp_patients": true_positive_patients,
            "fp_patients": false_positive_patients,
            "fn_patients": false_negative_patients,
            "sensitivity": patient_sensitivity,
            "precision": patient_precision,
            "f1": patient_f1,
        }
    )

    print(
        f"minimum_windows={minimum_positive_windows} "
        f"TP={true_positive_patients} "
        f"FP={false_positive_patients} "
        f"FN={false_negative_patients} "
        f"Sens={patient_sensitivity:.4f} "
        f"Precision={patient_precision:.4f} "
        f"F1={patient_f1:.4f}"
    )


# ============================================================
# 10. WINDOW-LEVEL PATIENT AGGREGATION SEARCH
# ============================================================

print()
print("=" * 70)
print("10. WINDOW-LEVEL AGGREGATION SEARCH")
print("=" * 70)

aggregation_rules = [
    ("max", None),
    ("mean", None),
    ("median", None),
    ("q90", None),
    ("q95", None),
    ("q99", None),
]

aggregation_results = []

for method, _ in aggregation_rules:

    patient_scores = {}
    patient_actual = {}

    for patient in unique_patients:

        mask = patients == patient

        p = probabilities[mask]
        y = labels[mask]

        if method == "max":
            score = float(np.max(p))
        elif method == "mean":
            score = float(np.mean(p))
        elif method == "median":
            score = float(np.median(p))
        elif method == "q90":
            score = float(np.quantile(p, 0.90))
        elif method == "q95":
            score = float(np.quantile(p, 0.95))
        elif method == "q99":
            score = float(np.quantile(p, 0.99))
        else:
            continue

        patient_scores[str(patient)] = score
        patient_actual[str(patient)] = bool(np.any(y == 1))

    predicted = {
        patient
        for patient, score in patient_scores.items()
        if score >= threshold
    }

    tp_pat = sum(
        1
        for patient in predicted
        if patient_actual[patient]
    )

    fp_pat = sum(
        1
        for patient in predicted
        if not patient_actual[patient]
    )

    actual_pat = sum(patient_actual.values())

    fn_pat = actual_pat - tp_pat

    sens_pat = tp_pat / actual_pat if actual_pat else 0.0
    prec_pat = (
        tp_pat / (tp_pat + fp_pat)
        if (tp_pat + fp_pat)
        else 0.0
    )

    f1_pat = (
        2 * sens_pat * prec_pat / (sens_pat + prec_pat)
        if (sens_pat + prec_pat)
        else 0.0
    )

    aggregation_results.append(
        {
            "method": method,
            "tp_patients": int(tp_pat),
            "fp_patients": int(fp_pat),
            "fn_patients": int(fn_pat),
            "sensitivity": float(sens_pat),
            "precision": float(prec_pat),
            "f1": float(f1_pat),
            "patient_scores": patient_scores,
        }
    )

    print(
        f"{method:>6} | "
        f"TP={tp_pat} "
        f"FP={fp_pat} "
        f"FN={fn_pat} "
        f"Sens={sens_pat:.4f} "
        f"Precision={prec_pat:.4f} "
        f"F1={f1_pat:.4f}"
    )


# ============================================================
# 11. SAVE RESULTS
# ============================================================

print()
print("=" * 70)
print("11. SAVING RESULTS")
print("=" * 70)

output = {
    "analysis": "patient_level_aggregation",
    "threshold": threshold,
    "test_samples": int(n),
    "number_of_patients": int(len(unique_patients)),
    "window_level_baseline": {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "f1": f1,
    },
    "patients": patient_results,
    "minimum_positive_window_rules": patient_rule_results,
    "aggregation_rules": aggregation_results,
}

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2)

print(f"[OK] Results saved:")
print(OUTPUT_FILE)


# ============================================================
# 12. FINAL MESSAGE
# ============================================================

print()
print("=" * 70)
print("PATIENT-LEVEL AGGREGATION ANALYSIS COMPLETED")
print("=" * 70)

print()
print("No model was modified.")
print("No dataset was modified.")
print("Validation threshold was NOT modified.")
print("This analysis uses the TEST predictions only for evaluation.")
