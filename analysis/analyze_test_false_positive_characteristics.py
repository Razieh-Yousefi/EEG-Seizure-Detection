import json
from pathlib import Path

import numpy as np


# ============================================================
# CONFIG
# ============================================================

PROJECT_DIR = Path(r"C:\Users\rezay\Desktop\EEG_Seizure_Project")
RESULTS_DIR = PROJECT_DIR / "results"

TEST_NPZ = RESULTS_DIR / "test_window_probabilities.npz"
THRESHOLD_JSON = RESULTS_DIR / "validation_threshold_results.json"

OUTPUT_JSON = RESULTS_DIR / "test_false_positive_characteristics.json"

REQUIRED_SENSITIVITY = 0.90


# ============================================================
# HELPERS
# ============================================================

def calculate_runs(binary_array):
    """
    Calculate lengths of consecutive positive runs.
    """
    x = np.asarray(binary_array, dtype=int)

    if len(x) == 0:
        return []

    runs = []
    current = 0

    for value in x:
        if value == 1:
            current += 1
        else:
            if current > 0:
                runs.append(current)
                current = 0

    if current > 0:
        runs.append(current)

    return runs


def transition_rate(binary_array):
    """
    Fraction of adjacent windows whose binary state changes.
    """
    x = np.asarray(binary_array, dtype=int)

    if len(x) <= 1:
        return 0.0

    return float(np.mean(x[1:] != x[:-1]))


def safe_float(x):
    return float(x) if np.isfinite(x) else None


def percentile_dict(probabilities):
    p = np.asarray(probabilities, dtype=float)

    return {
        "p01": safe_float(np.percentile(p, 1)),
        "p05": safe_float(np.percentile(p, 5)),
        "p10": safe_float(np.percentile(p, 10)),
        "p25": safe_float(np.percentile(p, 25)),
        "p50": safe_float(np.percentile(p, 50)),
        "p75": safe_float(np.percentile(p, 75)),
        "p90": safe_float(np.percentile(p, 90)),
        "p95": safe_float(np.percentile(p, 95)),
        "p97": safe_float(np.percentile(p, 97)),
        "p98": safe_float(np.percentile(p, 98)),
        "p99": safe_float(np.percentile(p, 99)),
        "max": safe_float(np.max(p)),
        "mean": safe_float(np.mean(p)),
        "std": safe_float(np.std(p)),
    }


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("TEST FALSE-POSITIVE CHARACTERISTICS ANALYSIS")
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

if not TEST_NPZ.exists():
    raise FileNotFoundError(f"Missing: {TEST_NPZ}")

if not THRESHOLD_JSON.exists():
    raise FileNotFoundError(f"Missing: {THRESHOLD_JSON}")

print(f"[OK] {TEST_NPZ}")
print(f"[OK] {THRESHOLD_JSON}")


# ============================================================
# 2. LOAD TEST DATA
# ============================================================

print()
print("=" * 70)
print("2. LOADING TEST DATA")
print("=" * 70)

data = np.load(TEST_NPZ, allow_pickle=True)

print()
print("Available NPZ arrays:")

for key in data.files:
    print(f"  {key:30s} shape={data[key].shape}")

probabilities = np.asarray(data["probabilities"], dtype=float)
labels = np.asarray(data["labels"], dtype=int)
patients = np.asarray(data["patients"]).astype(str)

if "test_indices" in data.files:
    indices = np.asarray(data["test_indices"], dtype=int)
elif "indices" in data.files:
    indices = np.asarray(data["indices"], dtype=int)
else:
    print()
    print("[INFO] No explicit indices array found.")
    print("[INFO] Using sequential indices.")
    indices = np.arange(len(probabilities), dtype=int)

print()
print(f"Test samples: {len(probabilities)}")
print(f"Probabilities shape: {probabilities.shape}")
print(f"Labels shape       : {labels.shape}")
print(f"Patients shape     : {patients.shape}")
print(f"Indices shape      : {indices.shape}")


# ============================================================
# 3. VERIFY DATA
# ============================================================

print()
print("=" * 70)
print("3. VERIFYING DATA")
print("=" * 70)

if not (
    len(probabilities)
    == len(labels)
    == len(patients)
    == len(indices)
):
    raise ValueError("Array lengths are not aligned.")

print("[OK] Arrays aligned.")

if not np.all(np.isfinite(probabilities)):
    raise ValueError("Probabilities contain NaN or Inf.")

print("[OK] Probabilities finite.")


# ============================================================
# 4. LOAD FROZEN VALIDATION THRESHOLD
# ============================================================

print()
print("=" * 70)
print("4. LOADING FROZEN VALIDATION THRESHOLD")
print("=" * 70)

with open(THRESHOLD_JSON, "r", encoding="utf-8") as f:
    threshold_data = json.load(f)


def find_threshold(obj):
    """
    Recursively search JSON for a likely validation threshold.
    """

    if isinstance(obj, dict):
        preferred_keys = [
            "threshold",
            "best_threshold",
            "validation_threshold",
            "selected_threshold",
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
        for value in obj:
            result = find_threshold(value)

            if result is not None:
                return result

    return None


window_threshold = find_threshold(threshold_data)

if window_threshold is None:
    raise ValueError(
        "Could not find validation threshold in JSON."
    )

print(f"Frozen validation threshold: {window_threshold:.6f}")


# ============================================================
# 5. WINDOW-LEVEL PREDICTIONS
# ============================================================

window_predictions = (
    probabilities >= window_threshold
).astype(int)


# ============================================================
# 6. BUILD PATIENT STATISTICS
# ============================================================

print()
print("=" * 70)
print("6. BUILDING TEST PATIENT STATISTICS")
print("=" * 70)

patient_results = {}

unique_patients = sorted(np.unique(patients))

for patient in unique_patients:

    mask = patients == patient

    p = probabilities[mask]
    y = labels[mask]
    pred = window_predictions[mask]
    idx = indices[mask]

    # Sort by sample/index order
    order = np.argsort(idx)

    p = p[order]
    y = y[order]
    pred = pred[order]
    idx = idx[order]

    positive_windows = int(np.sum(pred))
    total_windows = int(len(pred))

    positive_fraction = (
        positive_windows / total_windows
        if total_windows > 0
        else 0.0
    )

    runs = calculate_runs(pred)

    q95 = float(np.percentile(p, 95))

    true_label = int(np.max(y)) if len(y) else 0

    mean_positive_probability = (
        float(np.mean(p[pred == 1]))
        if positive_windows > 0
        else 0.0
    )

    median_positive_probability = (
        float(np.median(p[pred == 1]))
        if positive_windows > 0
        else 0.0
    )

    max_probability = float(np.max(p))

    mean_probability = float(np.mean(p))

    median_probability = float(np.median(p))

    transition = transition_rate(pred)

    stats = {
        "patient": patient,
        "true_label": true_label,
        "total_windows": total_windows,
        "positive_windows": positive_windows,
        "positive_fraction": positive_fraction,
        "positive_runs": len(runs),
        "max_positive_run": max(runs) if runs else 0,
        "mean_positive_run": (
            float(np.mean(runs)) if runs else 0.0
        ),
        "median_positive_run": (
            float(np.median(runs)) if runs else 0.0
        ),
        "transition_rate": transition,
        "q95": q95,
        "max_probability": max_probability,
        "mean_probability": mean_probability,
        "median_probability": median_probability,
        "mean_positive_probability": mean_positive_probability,
        "median_positive_probability": median_positive_probability,
        "percentiles": percentile_dict(p),
    }

    patient_results[patient] = stats

    print()
    print(
        f"{patient:6s} "
        f"true={true_label} "
        f"windows={total_windows:4d} "
        f"positive={positive_windows:3d} "
        f"fraction={positive_fraction:.4f} "
        f"runs={len(runs):3d} "
        f"max_run={max(runs) if runs else 0:2d} "
        f"mean_run={np.mean(runs) if runs else 0:.2f} "
        f"transition={transition:.4f} "
        f"Q95={q95:.6f}"
    )


# ============================================================
# 7. BASELINE PATIENT Q95
# ============================================================

print()
print("=" * 70)
print("7. BASELINE PATIENT-LEVEL Q95")
print("=" * 70)

baseline_predictions = {}

for patient, stats in patient_results.items():
    baseline_predictions[patient] = int(
        stats["q95"] >= 0.50
    )

tp = fp = fn = tn = 0

for patient, stats in patient_results.items():

    true_label = stats["true_label"]
    pred = baseline_predictions[patient]

    if true_label == 1 and pred == 1:
        tp += 1
    elif true_label == 0 and pred == 1:
        fp += 1
    elif true_label == 1 and pred == 0:
        fn += 1
    elif true_label == 0 and pred == 0:
        tn += 1


def metrics(tp, fp, fn, tn):

    sensitivity = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0.0
    )

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0.0
    )

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0.0
    )

    f1 = (
        2 * precision * sensitivity
        / (precision + sensitivity)
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


baseline_metrics = metrics(tp, fp, fn, tn)

print(json.dumps(
    baseline_metrics,
    indent=2
))


# ============================================================
# 8. IDENTIFY FALSE POSITIVES
# ============================================================

print()
print("=" * 70)
print("8. TEST FALSE POSITIVES")
print("=" * 70)

false_positives = []

for patient, stats in patient_results.items():

    if (
        stats["true_label"] == 0
        and baseline_predictions[patient] == 1
    ):
        false_positives.append(patient)

        print()
        print("[FALSE POSITIVE]")
        print(f"Patient                    : {patient}")
        print(f"Q95                        : {stats['q95']:.6f}")
        print(f"Max probability            : {stats['max_probability']:.6f}")
        print(f"Mean probability           : {stats['mean_probability']:.6f}")
        print(f"Median probability         : {stats['median_probability']:.6f}")
        print(f"Positive fraction          : {stats['positive_fraction']:.6f}")
        print(f"Positive windows           : {stats['positive_windows']}")
        print(f"Positive runs              : {stats['positive_runs']}")
        print(f"Maximum positive run       : {stats['max_positive_run']}")
        print(f"Mean positive run          : {stats['mean_positive_run']:.4f}")
        print(f"Transition rate            : {stats['transition_rate']:.6f}")
        print(
            f"Mean positive probability  : "
            f"{stats['mean_positive_probability']:.6f}"
        )
        print(
            f"Median positive probability: "
            f"{stats['median_positive_probability']:.6f}"
        )

print()
print(f"Total false positives: {len(false_positives)}")


# ============================================================
# 9. COMPARE TRUE POSITIVE PATIENTS VS FALSE POSITIVES
# ============================================================

print()
print("=" * 70)
print("9. TRUE POSITIVE VS FALSE POSITIVE COMPARISON")
print("=" * 70)

true_positives = []

for patient, stats in patient_results.items():

    if (
        stats["true_label"] == 1
        and baseline_predictions[patient] == 1
    ):
        true_positives.append(patient)


comparison_features = [
    "q95",
    "max_probability",
    "mean_probability",
    "median_probability",
    "positive_fraction",
    "positive_windows",
    "positive_runs",
    "max_positive_run",
    "mean_positive_run",
    "median_positive_run",
    "transition_rate",
    "mean_positive_probability",
    "median_positive_probability",
]


print()
print(
    f"{'FEATURE':30s} "
    f"{'TRUE POS MEAN':>15s} "
    f"{'FALSE POS MEAN':>15s}"
)

print("-" * 70)

for feature in comparison_features:

    tp_values = [
        patient_results[p][feature]
        for p in true_positives
    ]

    fp_values = [
        patient_results[p][feature]
        for p in false_positives
    ]

    tp_mean = (
        float(np.mean(tp_values))
        if tp_values
        else float("nan")
    )

    fp_mean = (
        float(np.mean(fp_values))
        if fp_values
        else float("nan")
    )

    print(
        f"{feature:30s} "
        f"{tp_mean:15.6f} "
        f"{fp_mean:15.6f}"
    )


# ============================================================
# 10. MANUAL INTERPRETATION
# ============================================================

print()
print("=" * 70)
print("10. INTERPRETATION")
print("=" * 70)

print()
print(
    "The purpose of this analysis is NOT to optimize a rule on Test."
)

print(
    "We are only characterizing how Test false positives differ "
    "from true-positive patients."
)

print()

if false_positives:

    for patient in false_positives:

        stats = patient_results[patient]

        print(
            f"{patient}: "
            f"Q95={stats['q95']:.6f}, "
            f"fraction={stats['positive_fraction']:.6f}, "
            f"max_run={stats['max_positive_run']}, "
            f"mean_run={stats['mean_positive_run']:.2f}, "
            f"transition={stats['transition_rate']:.4f}"
        )

print()
print(
    "IMPORTANT:"
)
print(
    "No threshold or classification rule is being changed here."
)
print(
    "This step is diagnostic only."
)


# ============================================================
# 11. SAVE RESULTS
# ============================================================

print()
print("=" * 70)
print("11. SAVING RESULTS")
print("=" * 70)

output = {
    "analysis": "test_false_positive_characteristics",
    "validation_threshold": window_threshold,
    "baseline_rule": "Q95 >= 0.50",
    "baseline_metrics": baseline_metrics,
    "true_positive_patients": true_positives,
    "false_positive_patients": false_positives,
    "patient_statistics": patient_results,
    "comparison_features": comparison_features,
    "notes": [
        "Diagnostic analysis only.",
        "No Test rule optimization performed.",
        "No model modified.",
        "No dataset modified.",
        "Validation threshold remained frozen.",
    ],
}

with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print(f"[OK] Results saved:")
print(OUTPUT_JSON)

print()
print("=" * 70)
print("TEST FALSE-POSITIVE CHARACTERISTICS ANALYSIS COMPLETED")
print("=" * 70)

print()
print("No model was modified.")
print("No dataset was modified.")
print("Validation threshold was NOT modified.")
print("No Test optimization was performed.")
print()