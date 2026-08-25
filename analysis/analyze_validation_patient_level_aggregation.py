
import json
from pathlib import Path

import numpy as np


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_DIR / "results"

PROB_FILE = RESULTS_DIR / "validation_window_probabilities.npz"
THRESHOLD_FILE = RESULTS_DIR / "validation_threshold_results.json"

OUTPUT_FILE = RESULTS_DIR / "validation_patient_level_aggregation_analysis.json"


# ============================================================
# HELPERS
# ============================================================

def print_section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def calculate_metrics(tp, fp, fn, tn):
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

    if precision + sensitivity > 0:
        f1 = 2 * precision * sensitivity / (precision + sensitivity)
    else:
        f1 = 0.0

    return {
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "precision": float(precision),
        "f1": float(f1),
    }


def safe_float(value):
    if isinstance(value, (np.floating, np.integer)):
        return float(value)
    return value


def patient_statistics(probabilities):
    return {
        "count": int(len(probabilities)),
        "max": float(np.max(probabilities)),
        "mean": float(np.mean(probabilities)),
        "median": float(np.median(probabilities)),
        "q90": float(np.quantile(probabilities, 0.90)),
        "q95": float(np.quantile(probabilities, 0.95)),
        "q99": float(np.quantile(probabilities, 0.99)),
    }


# ============================================================
# MAIN
# ============================================================

print_section("VALIDATION PATIENT-LEVEL AGGREGATION ANALYSIS")

print("\nProject directory:")
print(PROJECT_DIR)

print("\nResults directory:")
print(RESULTS_DIR)


# ============================================================
# 1. CHECK INPUT FILES
# ============================================================

print_section("1. CHECKING INPUT FILES")

if not PROB_FILE.exists():
    raise FileNotFoundError(
        f"Validation probability file not found:\n{PROB_FILE}"
    )

if not THRESHOLD_FILE.exists():
    raise FileNotFoundError(
        f"Validation threshold file not found:\n{THRESHOLD_FILE}"
    )

print(f"[OK] {PROB_FILE}")
print(f"[OK] {THRESHOLD_FILE}")


# ============================================================
# 2. LOAD VALIDATION PROBABILITIES
# ============================================================

print_section("2. LOADING VALIDATION PROBABILITIES")

data = np.load(PROB_FILE)

required_arrays = [
    "validation_indices",
    "patients",
    "labels",
    "probabilities",
]

for name in required_arrays:
    if name not in data:
        raise KeyError(
            f"Required array '{name}' not found in:\n{PROB_FILE}"
        )

validation_indices = data["validation_indices"]
patients = data["patients"]
labels = data["labels"]
probabilities = data["probabilities"]

print(f"Validation samples: {len(probabilities)}")
print(f"Indices shape: {validation_indices.shape}")
print(f"Patients shape: {patients.shape}")
print(f"Labels shape: {labels.shape}")
print(f"Probability shape: {probabilities.shape}")


# ============================================================
# 3. VERIFY ALIGNMENT
# ============================================================

print_section("3. VERIFYING ARRAY ALIGNMENT")

n = len(probabilities)

if not (
    len(validation_indices) == n
    and len(patients) == n
    and len(labels) == n
):
    raise RuntimeError(
        "Validation arrays are not aligned."
    )

if not np.all(np.isfinite(probabilities)):
    raise RuntimeError(
        "Validation probabilities contain non-finite values."
    )

print("[OK] Arrays are aligned.")
print("[OK] Probabilities are finite.")


# ============================================================
# 4. LOAD VALIDATION THRESHOLD
# ============================================================

print_section("4. LOADING VALIDATION THRESHOLD")

with open(THRESHOLD_FILE, "r", encoding="utf-8") as f:
    threshold_results = json.load(f)


def find_threshold(obj):
    if isinstance(obj, dict):
        preferred_keys = [
            "best_threshold",
            "validation_threshold",
            "threshold",
            "optimal_threshold",
        ]

        for key in preferred_keys:
            if key in obj and isinstance(
                obj[key], (int, float)
            ):
                return float(obj[key])

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


threshold = find_threshold(threshold_results)

if threshold is None:
    raise RuntimeError(
        "Could not identify validation threshold "
        "inside validation_threshold_results.json"
    )

print(f"Validation threshold: {threshold}")


# ============================================================
# 5. WINDOW-LEVEL BASELINE
# ============================================================

print_section("5. WINDOW-LEVEL BASELINE")

window_predictions = probabilities >= threshold

tp = int(np.sum((window_predictions == 1) & (labels == 1)))
fp = int(np.sum((window_predictions == 1) & (labels == 0)))
fn = int(np.sum((window_predictions == 0) & (labels == 1)))
tn = int(np.sum((window_predictions == 0) & (labels == 0)))

window_metrics = calculate_metrics(tp, fp, fn, tn)

for key, value in window_metrics.items():
    print(f"{key}: {value:.6f}" if isinstance(value, float)
          else f"{key}: {value}")


# ============================================================
# 6. IDENTIFY VALIDATION PATIENTS
# ============================================================

print_section("6. IDENTIFYING VALIDATION PATIENTS")

unique_patients = np.unique(patients)

print(f"Number of validation patients: {len(unique_patients)}")

patient_data = {}

for patient in unique_patients:
    mask = patients == patient

    patient_probabilities = probabilities[mask]
    patient_labels = labels[mask]

    positive_windows = int(
        np.sum(patient_probabilities >= threshold)
    )

    true_positive_windows = int(
        np.sum(
            (patient_probabilities >= threshold)
            & (patient_labels == 1)
        )
    )

    true_seizure_windows = int(
        np.sum(patient_labels == 1)
    )

    patient_data[str(patient)] = {
        "total_windows": int(np.sum(mask)),
        "positive_windows": positive_windows,
        "true_seizure_windows": true_seizure_windows,
        "true_positive_windows": true_positive_windows,
        "has_true_seizure": bool(true_seizure_windows > 0),
        "probabilities": patient_probabilities,
        "labels": patient_labels,
    }

    print(
        f"{patient}: "
        f"{int(np.sum(mask))} windows"
    )


# ============================================================
# 7. PATIENT STATISTICS
# ============================================================

print_section("7. PATIENT-LEVEL PROBABILITY STATISTICS")

patient_statistics_output = {}

for patient in unique_patients:
    key = str(patient)

    probs = patient_data[key]["probabilities"]

    stats = patient_statistics(probs)

    patient_statistics_output[key] = {
        **stats,
        "total_windows": patient_data[key]["total_windows"],
        "positive_windows": patient_data[key]["positive_windows"],
        "true_seizure_windows": patient_data[key][
            "true_seizure_windows"
        ],
        "has_true_seizure": patient_data[key][
            "has_true_seizure"
        ],
    }

    print("\n" + "-" * 70)
    print(f"Patient: {patient}")
    print(f"Total windows       : {stats['count']}")
    print(
        f"Positive windows    : "
        f"{patient_data[key]['positive_windows']}"
    )
    print(
        f"True seizure labels : "
        f"{patient_data[key]['true_seizure_windows']}"
    )
    print(f"Max probability     : {stats['max']:.6f}")
    print(f"Mean probability    : {stats['mean']:.6f}")
    print(f"Median probability : {stats['median']:.6f}")
    print(f"Q90                 : {stats['q90']:.6f}")
    print(f"Q95                 : {stats['q95']:.6f}")
    print(f"Q99                 : {stats['q99']:.6f}")


# ============================================================
# 8. PATIENT-LEVEL EVALUATION FUNCTION
# ============================================================

def evaluate_patient_rule(patient_predictions):
    patient_true = {}

    for patient in unique_patients:
        key = str(patient)
        patient_true[key] = patient_data[key]["has_true_seizure"]

    tp = 0
    fp = 0
    fn = 0
    tn = 0

    for patient in unique_patients:
        key = str(patient)

        predicted = bool(patient_predictions[key])
        actual = bool(patient_true[key])

        if predicted and actual:
            tp += 1
        elif predicted and not actual:
            fp += 1
        elif not predicted and actual:
            fn += 1
        else:
            tn += 1

    return calculate_metrics(tp, fp, fn, tn)


# ============================================================
# 9. MINIMUM POSITIVE WINDOW SEARCH
# ============================================================

print_section("9. MINIMUM POSITIVE WINDOW SEARCH")

minimum_window_results = []

max_positive_windows = max(
    patient_data[str(p)]["positive_windows"]
    for p in unique_patients
)

for minimum_windows in [
    1,
    2,
    3,
    4,
    5,
    10,
    15,
    20,
    30,
]:
    if minimum_windows > max_positive_windows:
        continue

    predictions = {}

    for patient in unique_patients:
        key = str(patient)

        predictions[key] = (
            patient_data[key]["positive_windows"]
            >= minimum_windows
        )

    metrics = evaluate_patient_rule(predictions)

    result = {
        "minimum_positive_windows": int(minimum_windows),
        **metrics,
    }

    minimum_window_results.append(result)

    print(
        f"minimum_windows={minimum_windows} "
        f"TP={metrics['tp']} "
        f"FP={metrics['fp']} "
        f"FN={metrics['fn']} "
        f"Sens={metrics['sensitivity']:.4f} "
        f"Precision={metrics['precision']:.4f} "
        f"F1={metrics['f1']:.4f}"
    )


# ============================================================
# 10. AGGREGATION SEARCH
# ============================================================

print_section("10. WINDOW-LEVEL AGGREGATION SEARCH")

aggregation_results = []

aggregation_methods = {
    "max": lambda x: np.max(x),
    "mean": lambda x: np.mean(x),
    "median": lambda x: np.median(x),
    "q80": lambda x: np.quantile(x, 0.80),
    "q85": lambda x: np.quantile(x, 0.85),
    "q90": lambda x: np.quantile(x, 0.90),
    "q95": lambda x: np.quantile(x, 0.95),
    "q99": lambda x: np.quantile(x, 0.99),
}

for method_name, aggregator in aggregation_methods.items():

    patient_scores = {}

    for patient in unique_patients:
        key = str(patient)

        probs = patient_data[key]["probabilities"]

        patient_scores[key] = float(
            aggregator(probs)
        )

    predictions = {
        key: score >= threshold
        for key, score in patient_scores.items()
    }

    metrics = evaluate_patient_rule(predictions)

    result = {
        "method": method_name,
        "threshold": float(threshold),
        "patient_scores": patient_scores,
        **metrics,
    }

    aggregation_results.append(result)

    print(
        f"{method_name:>6} | "
        f"TP={metrics['tp']} "
        f"FP={metrics['fp']} "
        f"FN={metrics['fn']} "
        f"Sens={metrics['sensitivity']:.4f} "
        f"Precision={metrics['precision']:.4f} "
        f"F1={metrics['f1']:.4f}"
    )


# ============================================================
# 11. AGGREGATION + MULTIPLE WINDOW RULE
# ============================================================

print_section("11. AGGREGATION + MINIMUM POSITIVE WINDOW SEARCH")

combined_results = []

for method_name, aggregator in aggregation_methods.items():

    patient_scores = {}

    for patient in unique_patients:
        key = str(patient)

        probs = patient_data[key]["probabilities"]

        patient_scores[key] = float(
            aggregator(probs)
        )

    for minimum_windows in [1, 2, 3, 5, 10]:

        predictions = {}

        for patient in unique_patients:
            key = str(patient)

            score_condition = (
                patient_scores[key] >= threshold
            )

            count_condition = (
                patient_data[key]["positive_windows"]
                >= minimum_windows
            )

            predictions[key] = (
                score_condition and count_condition
            )

        metrics = evaluate_patient_rule(predictions)

        result = {
            "aggregation": method_name,
            "minimum_positive_windows": int(
                minimum_windows
            ),
            **metrics,
        }

        combined_results.append(result)

        print(
            f"{method_name:>6} + "
            f"min_windows={minimum_windows:2d} | "
            f"TP={metrics['tp']} "
            f"FP={metrics['fp']} "
            f"FN={metrics['fn']} "
            f"Sens={metrics['sensitivity']:.4f} "
            f"Precision={metrics['precision']:.4f} "
            f"F1={metrics['f1']:.4f}"
        )


# ============================================================
# 12. SAFE CANDIDATE SELECTION
# ============================================================

print_section("12. SAFE VALIDATION CANDIDATES")

REQUIRED_SENSITIVITY = 0.90

safe_candidates = []

for result in aggregation_results:
    if result["sensitivity"] >= REQUIRED_SENSITIVITY:
        safe_candidates.append(result)

for result in minimum_window_results:
    if result["sensitivity"] >= REQUIRED_SENSITIVITY:
        safe_candidates.append(result)

for result in combined_results:
    if result["sensitivity"] >= REQUIRED_SENSITIVITY:
        safe_candidates.append(result)

print(
    f"Required sensitivity: "
    f"{REQUIRED_SENSITIVITY}"
)

print(
    f"Safe candidates: "
    f"{len(safe_candidates)}"
)

if safe_candidates:

    safe_candidates = sorted(
        safe_candidates,
        key=lambda x: (
            x["f1"],
            x["precision"],
            x["sensitivity"],
        ),
        reverse=True,
    )

    print("\nTOP SAFE CANDIDATES")
    print(
        "-" * 100
    )

    for candidate in safe_candidates[:15]:

        if "method" in candidate:
            description = candidate["method"]
        elif "minimum_positive_windows" in candidate:
            description = (
                f"minimum_windows="
                f"{candidate['minimum_positive_windows']}"
            )
        else:
            description = "combined"

        if "aggregation" in candidate:
            description += (
                f" + min_windows="
                f"{candidate['minimum_positive_windows']}"
            )

        print(
            f"{description:35s} | "
            f"TP={candidate['tp']} "
            f"FP={candidate['fp']} "
            f"FN={candidate['fn']} "
            f"Sens={candidate['sensitivity']:.4f} "
            f"Precision={candidate['precision']:.4f} "
            f"F1={candidate['f1']:.4f}"
        )

else:
    print(
        "\nNo validation patient-level aggregation "
        "candidate reached sensitivity >= 0.90."
    )


# ============================================================
# 13. BEST CANDIDATE
# ============================================================

print_section("13. BEST VALIDATION PATIENT-LEVEL CANDIDATE")

best_candidate = None

if safe_candidates:
    best_candidate = safe_candidates[0]

    print("Best candidate:")

    for key, value in best_candidate.items():
        if key != "patient_scores":
            print(f"{key}: {value}")

else:
    print(
        "No candidate reached the required "
        "sensitivity."
    )

    all_candidates = (
        aggregation_results
        + minimum_window_results
        + combined_results
    )

    all_candidates = sorted(
        all_candidates,
        key=lambda x: (
            x["sensitivity"],
            x["f1"],
            x["precision"],
        ),
        reverse=True,
    )

    print("\nTOP OVERALL CANDIDATES:")
    print("-" * 100)

    for candidate in all_candidates[:10]:

        if "method" in candidate:
            description = candidate["method"]
        elif "minimum_positive_windows" in candidate:
            description = (
                f"minimum_windows="
                f"{candidate['minimum_positive_windows']}"
            )
        else:
            description = "combined"

        if "aggregation" in candidate:
            description = (
                f"{candidate['aggregation']} + "
                f"min_windows="
                f"{candidate['minimum_positive_windows']}"
            )

        print(
            f"{description:35s} | "
            f"Sens={candidate['sensitivity']:.4f} "
            f"FP={candidate['fp']} "
            f"Precision={candidate['precision']:.4f} "
            f"F1={candidate['f1']:.4f}"
        )


# ============================================================
# 14. SAVE RESULTS
# ============================================================

print_section("14. SAVING RESULTS")

output = {
    "analysis": "validation_patient_level_aggregation",
    "validation_threshold": float(threshold),
    "required_sensitivity": float(
        REQUIRED_SENSITIVITY
    ),
    "validation_samples": int(n),
    "number_of_validation_patients": int(
        len(unique_patients)
    ),
    "window_level_baseline": window_metrics,
    "patient_statistics": patient_statistics_output,
    "minimum_positive_window_search": minimum_window_results,
    "aggregation_search": aggregation_results,
    "aggregation_plus_minimum_window_search": combined_results,
    "safe_candidates": safe_candidates,
    "best_candidate": best_candidate,
    "test_data_used": False,
    "model_modified": False,
    "dataset_modified": False,
    "threshold_modified": False,
}

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        output,
        f,
        indent=2,
        ensure_ascii=False,
    )

print(f"[OK] Results saved:")
print(OUTPUT_FILE)

print("\nNo model was modified.")
print("No dataset was modified.")
print("Validation threshold was NOT modified.")
print("Test data was NOT used.")

print_section(
    "VALIDATION PATIENT-LEVEL AGGREGATION ANALYSIS COMPLETED"
)

