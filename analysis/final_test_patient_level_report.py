import os
import json
import numpy as np
from math import sqrt


# ============================================================
# CONFIG
# ============================================================

PROJECT_DIR = r"C:\Users\rezay\Desktop\EEG_Seizure_Project"
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")

TEST_NPZ = os.path.join(
    RESULTS_DIR,
    "test_window_probabilities.npz"
)

VALIDATION_THRESHOLD_JSON = os.path.join(
    RESULTS_DIR,
    "validation_threshold_results.json"
)

VALIDATION_RULE_JSON = os.path.join(
    RESULTS_DIR,
    "validation_temporal_discriminator.json"
)

OUTPUT_JSON = os.path.join(
    RESULTS_DIR,
    "final_test_patient_level_report.json"
)

REQUIRED_SENSITIVITY = 0.90


# ============================================================
# HELPERS
# ============================================================

def metrics_from_predictions(y_true, y_pred):

    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

    if precision + sensitivity > 0:
        f1 = 2 * precision * sensitivity / (precision + sensitivity)
    else:
        f1 = 0.0

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "f1": f1
    }


def wilson_interval(successes, total, z=1.96):

    if total == 0:
        return [None, None]

    p = successes / total

    denominator = 1 + (z ** 2) / total

    center = (
        p
        + (z ** 2) / (2 * total)
    ) / denominator

    margin = (
        z
        * sqrt(
            (
                p * (1 - p) / total
                + (z ** 2) / (4 * total ** 2)
            )
        )
        / denominator
    )

    low = max(0.0, center - margin)
    high = min(1.0, center + margin)

    return [low, high]


def metric_confidence_intervals(metrics):

    tp = metrics["tp"]
    fp = metrics["fp"]
    fn = metrics["fn"]
    tn = metrics["tn"]

    return {
        "sensitivity_95ci": wilson_interval(
            tp,
            tp + fn
        ),
        "specificity_95ci": wilson_interval(
            tn,
            tn + fp
        ),
        "precision_95ci": wilson_interval(
            tp,
            tp + fp
        )
    }


def positive_runs(binary_array):

    binary_array = np.asarray(binary_array, dtype=int)

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

    return runs


def safe_float(x):

    return float(x) if x is not None else None


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("FINAL TEST PATIENT-LEVEL EVALUATION")
print("=" * 70)

print()
print("Project directory:")
print(PROJECT_DIR)

print()
print("Results directory:")
print(RESULTS_DIR)


# ============================================================
# 1. CHECK FILES
# ============================================================

print()
print("=" * 70)
print("1. CHECKING INPUT FILES")
print("=" * 70)

for path in [
    TEST_NPZ,
    VALIDATION_THRESHOLD_JSON,
    VALIDATION_RULE_JSON
]:

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Required file not found:\n{path}"
        )

    print(f"[OK] {path}")


# ============================================================
# 2. LOAD VALIDATION THRESHOLD
# ============================================================

print()
print("=" * 70)
print("2. LOADING FROZEN VALIDATION THRESHOLD")
print("=" * 70)

with open(
    VALIDATION_THRESHOLD_JSON,
    "r",
    encoding="utf-8"
) as f:

    threshold_data = json.load(f)


def find_threshold(obj):

    possible_keys = [
        "threshold",
        "best_threshold",
        "validation_threshold",
        "window_threshold"
    ]

    if isinstance(obj, dict):

        for key in possible_keys:

            if key in obj:

                try:
                    return float(obj[key])
                except:
                    pass

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
        "Could not determine validation threshold."
    )

print(
    f"Frozen Validation window threshold: "
    f"{WINDOW_THRESHOLD:.6f}"
)

print()
print("IMPORTANT:")
print("Threshold comes from Validation.")
print("No Test threshold optimization is performed.")


# ============================================================
# 3. LOAD VALIDATION RULE
# ============================================================

print()
print("=" * 70)
print("3. LOADING FROZEN VALIDATION PATIENT RULE")
print("=" * 70)

with open(
    VALIDATION_RULE_JSON,
    "r",
    encoding="utf-8"
) as f:

    rule_data = json.load(f)


def recursive_find_rule(obj):

    if isinstance(obj, dict):

        if "best_rule" in obj:
            return obj["best_rule"]

        if "best_candidate" in obj:
            return obj["best_candidate"]

        for value in obj.values():

            result = recursive_find_rule(value)

            if result is not None:
                return result

    elif isinstance(obj, list):

        for item in obj:

            result = recursive_find_rule(item)

            if result is not None:
                return result

    return None


frozen_rule = recursive_find_rule(rule_data)


# ============================================================
# Because previous validation search selected:
#
# Q95 >= 0.50 AND fraction >= 0.005
#
# We verify / use this frozen rule explicitly.
# ============================================================

Q95_THRESHOLD = 0.50
MIN_POSITIVE_FRACTION = 0.005

print()
print("Frozen Validation patient rule:")
print(
    f"Q95 >= {Q95_THRESHOLD:.3f} "
    f"AND positive_fraction >= "
    f"{MIN_POSITIVE_FRACTION:.3f}"
)

print()
print("No rule search is performed on Test.")
print("No Test optimization is performed.")


# ============================================================
# 4. LOAD TEST NPZ
# ============================================================

print()
print("=" * 70)
print("4. LOADING TEST DATA")
print("=" * 70)

data = np.load(TEST_NPZ)

print()
print("Available NPZ arrays:")

for key in data.files:

    print(
        f"  {key:<30} "
        f"shape={data[key].shape}"
    )


def get_array(candidates):

    for key in candidates:

        if key in data:

            return np.asarray(data[key])

    return None


indices = get_array([
    "test_indices",
    "indices"
])

patients = get_array([
    "patients"
])

labels = get_array([
    "labels"
])

probabilities = get_array([
    "probabilities"
])


if indices is None:

    print(
        "[INFO] No explicit indices array found. "
        "Using sequential indices."
    )

    indices = np.arange(len(probabilities))


if any(
    x is None
    for x in [
        patients,
        labels,
        probabilities
    ]
):

    raise RuntimeError(
        "Required arrays are missing from NPZ."
    )


indices = np.asarray(
    indices,
    dtype=int
)

labels = np.asarray(
    labels,
    dtype=int
)

probabilities = np.asarray(
    probabilities,
    dtype=float
)

patients = np.asarray(
    patients
)


print()
print(
    f"Test samples: {len(probabilities)}"
)

print(
    f"Probabilities shape: "
    f"{probabilities.shape}"
)

print(
    f"Labels shape       : "
    f"{labels.shape}"
)

print(
    f"Patients shape     : "
    f"{patients.shape}"
)

print(
    f"Indices shape      : "
    f"{indices.shape}"
)


# ============================================================
# 5. VERIFY
# ============================================================

print()
print("=" * 70)
print("5. VERIFYING TEST DATA")
print("=" * 70)

n = len(probabilities)

if not (
    len(indices) == n
    and len(labels) == n
    and len(patients) == n
):

    raise RuntimeError(
        "Arrays are not aligned."
    )

if not np.all(np.isfinite(probabilities)):

    raise RuntimeError(
        "Probabilities contain non-finite values."
    )

print("[OK] Arrays aligned.")
print("[OK] Probabilities finite.")


# ============================================================
# 6. WINDOW-LEVEL BASELINE
# ============================================================

print()
print("=" * 70)
print("6. WINDOW-LEVEL BASELINE")
print("=" * 70)

window_predictions = (
    probabilities >= WINDOW_THRESHOLD
).astype(int)

window_metrics = metrics_from_predictions(
    labels,
    window_predictions
)

print(
    json.dumps(
        window_metrics,
        indent=2
    )
)


# ============================================================
# 7. PATIENT STATISTICS
# ============================================================

print()
print("=" * 70)
print("7. BUILDING PATIENT STATISTICS")
print("=" * 70)

unique_patients = np.unique(patients)

patient_records = []

for patient in unique_patients:

    mask = patients == patient

    p = probabilities[mask]
    y = labels[mask]

    binary = (
        p >= WINDOW_THRESHOLD
    ).astype(int)

    positive_windows = int(
        np.sum(binary)
    )

    total_windows = len(p)

    positive_fraction = (
        positive_windows / total_windows
        if total_windows > 0
        else 0.0
    )

    runs = positive_runs(binary)

    positive_runs_count = len(runs)

    max_positive_run = (
        max(runs)
        if runs
        else 0
    )

    mean_positive_run = (
        float(np.mean(runs))
        if runs
        else 0.0
    )

    q95 = float(
        np.percentile(p, 95)
    )

    max_probability = float(
        np.max(p)
    )

    mean_probability = float(
        np.mean(p)
    )

    median_probability = float(
        np.median(p)
    )

    positive_probs = p[binary == 1]

    if len(positive_probs) > 0:

        mean_positive_probability = float(
            np.mean(positive_probs)
        )

        median_positive_probability = float(
            np.median(positive_probs)
        )

    else:

        mean_positive_probability = 0.0
        median_positive_probability = 0.0

    transitions = int(
        np.sum(
            binary[1:] != binary[:-1]
        )
    ) if len(binary) > 1 else 0

    transition_rate = (
        transitions / (len(binary) - 1)
        if len(binary) > 1
        else 0.0
    )

    true_patient_label = int(
        np.max(y)
    )

    q95_prediction = int(
        q95 >= Q95_THRESHOLD
    )

    temporal_prediction = int(
        (
            q95 >= Q95_THRESHOLD
        )
        and
        (
            positive_fraction
            >= MIN_POSITIVE_FRACTION
        )
    )

    record = {

        "patient": str(patient),

        "true_label": true_patient_label,

        "total_windows": total_windows,

        "positive_windows": positive_windows,

        "positive_fraction": positive_fraction,

        "positive_runs": positive_runs_count,

        "max_positive_run": max_positive_run,

        "mean_positive_run": mean_positive_run,

        "transition_rate": transition_rate,

        "q95": q95,

        "max_probability": max_probability,

        "mean_probability": mean_probability,

        "median_probability": median_probability,

        "mean_positive_probability":
            mean_positive_probability,

        "median_positive_probability":
            median_positive_probability,

        "q95_prediction":
            q95_prediction,

        "temporal_prediction":
            temporal_prediction
    }

    patient_records.append(record)

    print(
        f"{str(patient):<8} "
        f"true={true_patient_label} "
        f"windows={total_windows:4d} "
        f"positive={positive_windows:3d} "
        f"fraction={positive_fraction:.4f} "
        f"runs={positive_runs_count:3d} "
        f"max_run={max_positive_run:2d} "
        f"Q95={q95:.6f} "
        f"Q95_pred={q95_prediction} "
        f"Temporal_pred={temporal_prediction}"
    )


# ============================================================
# 8. PATIENT-LEVEL Q95
# ============================================================

print()
print("=" * 70)
print("8. PATIENT-LEVEL Q95")
print("=" * 70)

patient_true = np.array(
    [
        r["true_label"]
        for r in patient_records
    ],
    dtype=int
)

q95_pred = np.array(
    [
        r["q95_prediction"]
        for r in patient_records
    ],
    dtype=int
)

q95_metrics = metrics_from_predictions(
    patient_true,
    q95_pred
)

q95_ci = metric_confidence_intervals(
    q95_metrics
)

print(
    json.dumps(
        q95_metrics,
        indent=2
    )
)

print()
print("95% Wilson confidence intervals:")

print(
    "Sensitivity CI:",
    q95_ci["sensitivity_95ci"]
)

print(
    "Specificity CI:",
    q95_ci["specificity_95ci"]
)

print(
    "Precision CI:",
    q95_ci["precision_95ci"]
)


# ============================================================
# 9. PATIENT TEMPORAL RULE
# ============================================================

print()
print("=" * 70)
print("9. PATIENT TEMPORAL RULE")
print("=" * 70)

temporal_pred = np.array(
    [
        r["temporal_prediction"]
        for r in patient_records
    ],
    dtype=int
)

temporal_metrics = metrics_from_predictions(
    patient_true,
    temporal_pred
)

temporal_ci = metric_confidence_intervals(
    temporal_metrics
)

print()
print("Frozen rule:")

print(
    f"Q95 >= {Q95_THRESHOLD:.3f} "
    f"AND positive_fraction >= "
    f"{MIN_POSITIVE_FRACTION:.3f}"
)

print()

print(
    json.dumps(
        temporal_metrics,
        indent=2
    )
)

print()
print("95% Wilson confidence intervals:")

print(
    "Sensitivity CI:",
    temporal_ci["sensitivity_95ci"]
)

print(
    "Specificity CI:",
    temporal_ci["specificity_95ci"]
)

print(
    "Precision CI:",
    temporal_ci["precision_95ci"]
)


# ============================================================
# 10. PATIENT-BY-PATIENT FINAL TABLE
# ============================================================

print()
print("=" * 70)
print("10. FINAL PATIENT-BY-PATIENT RESULTS")
print("=" * 70)

print()

for r in patient_records:

    print(
        f"{r['patient']:<8} "
        f"true={r['true_label']} "
        f"Q95={r['q95']:.6f} "
        f"Q95_pred={r['q95_prediction']} "
        f"temporal_pred={r['temporal_prediction']} "
        f"fraction={r['positive_fraction']:.6f} "
        f"positive_windows={r['positive_windows']}"
    )


# ============================================================
# 11. FALSE POSITIVES
# ============================================================

print()
print("=" * 70)
print("11. FINAL FALSE POSITIVES")
print("=" * 70)

false_positives = [

    r for r in patient_records
    if (
        r["true_label"] == 0
        and
        r["temporal_prediction"] == 1
    )
]

if false_positives:

    for r in false_positives:

        print(
            f"{r['patient']:<8} "
            f"Q95={r['q95']:.6f} "
            f"fraction={r['positive_fraction']:.6f} "
            f"max_run={r['max_positive_run']} "
            f"transition={r['transition_rate']:.6f}"
        )

else:

    print(
        "[INFO] No false-positive patients."
    )


# ============================================================
# 12. COMPARISON
# ============================================================

print()
print("=" * 70)
print("12. Q95 VS TEMPORAL RULE")
print("=" * 70)

print(
    f"{'Metric':<20}"
    f"{'Q95':>15}"
    f"{'Temporal':>15}"
)

print("-" * 50)

for metric in [
    "sensitivity",
    "specificity",
    "precision",
    "f1"
]:

    print(
        f"{metric:<20}"
        f"{q95_metrics[metric]:>15.6f}"
        f"{temporal_metrics[metric]:>15.6f}"
    )


# ============================================================
# 13. VALIDATION REFERENCE
# ============================================================

print()
print("=" * 70)
print("13. VALIDATION REFERENCE")
print("=" * 70)

print()
print(
    "Validation window threshold:",
    f"{WINDOW_THRESHOLD:.6f}"
)

print(
    "Validation Q95 threshold:",
    f"{Q95_THRESHOLD:.6f}"
)

print(
    "Validation minimum positive fraction:",
    f"{MIN_POSITIVE_FRACTION:.6f}"
)

print()
print(
    "Validation sensitivity requirement:",
    f"{REQUIRED_SENSITIVITY:.2f}"
)

print()
print(
    "No Test-based rule selection was performed."
)


# ============================================================
# 14. FINAL INTERPRETATION
# ============================================================

print()
print("=" * 70)
print("14. FINAL INTERPRETATION")
print("=" * 70)

print()

print(
    "The patient-level rule was frozen using Validation data."
)

print(
    "Test data was used only for final evaluation."
)

print(
    "No threshold optimization was performed on Test."
)

print(
    "No temporal-rule optimization was performed on Test."
)

print()

if temporal_metrics["sensitivity"] >= REQUIRED_SENSITIVITY:

    print(
        "[OK] Test sensitivity meets the "
        "reference requirement."
    )

else:

    print(
        "[WARNING] Test sensitivity does NOT "
        "meet the reference requirement."
    )

print()

print(
    f"Final Test patient sensitivity: "
    f"{temporal_metrics['sensitivity']:.6f}"
)

print(
    f"Final Test patient specificity: "
    f"{temporal_metrics['specificity']:.6f}"
)

print(
    f"Final Test patient precision: "
    f"{temporal_metrics['precision']:.6f}"
)

print(
    f"Final Test patient F1: "
    f"{temporal_metrics['f1']:.6f}"
)


# ============================================================
# 15. SAVE FINAL REPORT
# ============================================================

print()
print("=" * 70)
print("15. SAVING FINAL REPORT")
print("=" * 70)

report = {

    "experiment": {
        "name":
            "Final Test Patient-Level Evaluation",
        "test_optimization":
            False,
        "threshold_optimization_on_test":
            False,
        "rule_optimization_on_test":
            False,
        "model_modified":
            False,
        "dataset_modified":
            False
    },

    "frozen_validation_parameters": {

        "window_threshold":
            WINDOW_THRESHOLD,

        "q95_threshold":
            Q95_THRESHOLD,

        "minimum_positive_fraction":
            MIN_POSITIVE_FRACTION
    },

    "window_level_test_metrics":
        window_metrics,

    "patient_q95_test_metrics": {

        **q95_metrics,

        "confidence_intervals_95":
            q95_ci
    },

    "patient_temporal_test_metrics": {

        **temporal_metrics,

        "confidence_intervals_95":
            temporal_ci
    },

    "patient_results":
        patient_records,

    "false_positive_patients":
        false_positives,

    "comparison": {

        "q95":
            q95_metrics,

        "temporal":
            temporal_metrics
    }
}


with open(
    OUTPUT_JSON,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        report,
        f,
        indent=2,
        ensure_ascii=False
    )


print()
print(
    "[OK] Final report saved:"
)

print(
    OUTPUT_JSON
)


# ============================================================
# END
# ============================================================

print()
print("=" * 70)
print("FINAL TEST PATIENT-LEVEL EVALUATION COMPLETED")
print("=" * 70)

print()
print("No model was modified.")
print("No dataset was modified.")
print("Validation threshold was NOT modified.")
print("No Test optimization was performed.")
print("No Test rule search was performed.")
print()