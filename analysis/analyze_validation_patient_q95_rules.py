import json
from pathlib import Path

import numpy as np


# ============================================================
# CONFIG
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_DIR / "results"

PROB_FILE = RESULTS_DIR / "validation_window_probabilities.npz"
THRESHOLD_FILE = RESULTS_DIR / "validation_threshold_results.json"

OUTPUT_FILE = RESULTS_DIR / "validation_patient_q95_rules.json"

REQUIRED_SENSITIVITY = 0.90

# Q95 probability thresholds to investigate.
# 0.56 is the original validation threshold.
Q95_THRESHOLDS = [
    0.50,
    0.52,
    0.54,
    0.56,
    0.58,
    0.60,
    0.62,
    0.64,
    0.66,
    0.68,
    0.70,
    0.72,
    0.75,
    0.80,
    0.85,
    0.90,
    0.95,
]

# Minimum number of positive windows at the original
# window-level threshold.
MIN_POSITIVE_WINDOWS = [
    1,
    2,
    3,
    5,
    10,
    15,
    20,
    30,
    40,
    50,
    75,
    100,
]

# Additional rule:
# Q95 >= q95_threshold AND positive_windows >= min_windows
#
# We also test OR variants only as diagnostics.
# The preferred rule is AND because it is more conservative.


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
        2 * precision * sensitivity /
        (precision + sensitivity)
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


def find_threshold(obj):

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


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("VALIDATION PATIENT-LEVEL Q95 RULE SEARCH")
print("=" * 70)

print("\nProject directory:")
print(PROJECT_DIR)

print("\nResults directory:")
print(RESULTS_DIR)


# ============================================================
# 1. CHECK FILES
# ============================================================

print("\n" + "=" * 70)
print("1. CHECKING INPUT FILES")
print("=" * 70)

if not PROB_FILE.exists():
    raise FileNotFoundError(PROB_FILE)

if not THRESHOLD_FILE.exists():
    raise FileNotFoundError(THRESHOLD_FILE)

print(f"[OK] {PROB_FILE}")
print(f"[OK] {THRESHOLD_FILE}")


# ============================================================
# 2. LOAD VALIDATION DATA
# ============================================================

print("\n" + "=" * 70)
print("2. LOADING VALIDATION PROBABILITIES")
print("=" * 70)

data = np.load(PROB_FILE)

required = [
    "validation_indices",
    "patients",
    "labels",
    "probabilities",
]

for name in required:

    if name not in data:

        raise KeyError(
            f"Missing array '{name}' in {PROB_FILE}"
        )

indices = np.asarray(data["validation_indices"])
patients = np.asarray(data["patients"])
labels = np.asarray(data["labels"])
probabilities = np.asarray(
    data["probabilities"],
    dtype=float,
)

print(f"Validation samples: {len(probabilities)}")
print(f"Indices shape: {indices.shape}")
print(f"Patients shape: {patients.shape}")
print(f"Labels shape: {labels.shape}")
print(f"Probability shape: {probabilities.shape}")


# ============================================================
# 3. VERIFY
# ============================================================

print("\n" + "=" * 70)
print("3. VERIFYING VALIDATION DATA")
print("=" * 70)

n = len(probabilities)

if not (
    len(indices) == n
    and len(patients) == n
    and len(labels) == n
):
    raise RuntimeError(
        "Validation arrays are not aligned."
    )

if not np.all(np.isfinite(probabilities)):
    raise RuntimeError(
        "Probabilities contain NaN or infinite values."
    )

print("[OK] Arrays aligned.")
print("[OK] Probabilities finite.")


# ============================================================
# 4. LOAD ORIGINAL VALIDATION THRESHOLD
# ============================================================

print("\n" + "=" * 70)
print("4. LOADING VALIDATION THRESHOLD")
print("=" * 70)

with open(
    THRESHOLD_FILE,
    "r",
    encoding="utf-8",
) as f:

    threshold_data = json.load(f)

window_threshold = find_threshold(
    threshold_data
)

if window_threshold is None:

    raise RuntimeError(
        "Could not find validation threshold."
    )

print(
    f"Window-level validation threshold: "
    f"{window_threshold:.6f}"
)


# ============================================================
# 5. BUILD PATIENT STATISTICS
# ============================================================

print("\n" + "=" * 70)
print("5. BUILDING PATIENT-LEVEL STATISTICS")
print("=" * 70)

unique_patients = np.unique(patients)

patient_stats = []

for patient in unique_patients:

    mask = patients == patient

    p = probabilities[mask]
    y = labels[mask]

    positive_windows = int(
        np.sum(p >= window_threshold)
    )

    q95 = float(np.percentile(p, 95))

    q90 = float(np.percentile(p, 90))
    q99 = float(np.percentile(p, 99))

    true_label = int(
        np.any(y == 1)
    )

    patient_stats.append(
        {
            "patient": str(patient),
            "total_windows": int(len(p)),
            "positive_windows": positive_windows,
            "q90": q90,
            "q95": q95,
            "q99": q99,
            "max_probability": float(np.max(p)),
            "mean_probability": float(np.mean(p)),
            "median_probability": float(np.median(p)),
            "true_label": true_label,
        }
    )

    print(
        f"{str(patient):8s} "
        f"windows={len(p):4d} "
        f"positive={positive_windows:3d} "
        f"Q95={q95:.6f} "
        f"true={true_label}"
    )


# ============================================================
# 6. BASELINE PATIENT Q95
# ============================================================

print("\n" + "=" * 70)
print("6. BASELINE PATIENT Q95")
print("=" * 70)

baseline_true = [
    x["true_label"]
    for x in patient_stats
]

baseline_pred = [
    int(x["q95"] >= window_threshold)
    for x in patient_stats
]

baseline_metrics = calculate_metrics(
    baseline_true,
    baseline_pred,
)

for key, value in baseline_metrics.items():

    if isinstance(value, float):

        print(
            f"{key}: {value:.6f}"
        )

    else:

        print(
            f"{key}: {value}"
        )


# ============================================================
# 7. SEARCH Q95 THRESHOLD ONLY
# ============================================================

print("\n" + "=" * 70)
print("7. Q95 THRESHOLD SEARCH")
print("=" * 70)

q95_candidates = []

for q_threshold in Q95_THRESHOLDS:

    y_true = [
        x["true_label"]
        for x in patient_stats
    ]

    y_pred = [
        int(
            x["q95"] >= q_threshold
        )
        for x in patient_stats
    ]

    metrics = calculate_metrics(
        y_true,
        y_pred,
    )

    candidate = {
        "type": "q95_threshold",
        "q95_threshold": q_threshold,
        **metrics,
    }

    q95_candidates.append(candidate)

    print(
        f"Q95 >= {q_threshold:.2f} | "
        f"TP={metrics['tp']} "
        f"FP={metrics['fp']} "
        f"FN={metrics['fn']} "
        f"Sens={metrics['sensitivity']:.4f} "
        f"Prec={metrics['precision']:.4f} "
        f"F1={metrics['f1']:.4f}"
    )


# ============================================================
# 8. Q95 + MINIMUM WINDOWS (AND)
# ============================================================

print("\n" + "=" * 70)
print("8. Q95 + MINIMUM POSITIVE WINDOWS SEARCH")
print("=" * 70)

and_candidates = []

for q_threshold in Q95_THRESHOLDS:

    for min_windows in MIN_POSITIVE_WINDOWS:

        y_true = [
            x["true_label"]
            for x in patient_stats
        ]

        y_pred = []

        for x in patient_stats:

            rule = (
                x["q95"] >= q_threshold
                and
                x["positive_windows"] >= min_windows
            )

            y_pred.append(
                int(rule)
            )

        metrics = calculate_metrics(
            y_true,
            y_pred,
        )

        candidate = {
            "type": "q95_and_min_windows",
            "q95_threshold": q_threshold,
            "min_positive_windows": min_windows,
            **metrics,
        }

        and_candidates.append(candidate)


# ============================================================
# 9. SAFE CANDIDATES
# ============================================================

print("\n" + "=" * 70)
print("9. SAFE CANDIDATES")
print("=" * 70)

all_candidates = (
    q95_candidates
    + and_candidates
)

safe_candidates = [
    c
    for c in all_candidates
    if c["sensitivity"]
    >= REQUIRED_SENSITIVITY
]

print(
    f"Required sensitivity: "
    f"{REQUIRED_SENSITIVITY:.2f}"
)

print(
    f"Total candidates: "
    f"{len(all_candidates)}"
)

print(
    f"Safe candidates: "
    f"{len(safe_candidates)}"
)


# ============================================================
# 10. RANK CANDIDATES
# ============================================================

print("\n" + "=" * 70)
print("10. TOP SAFE CANDIDATES")
print("=" * 70)

if len(safe_candidates) == 0:

    print(
        "No candidate reached the required "
        "sensitivity."
    )

else:

    # Priority:
    # 1. lowest FP
    # 2. highest sensitivity
    # 3. highest F1
    # 4. highest precision
    safe_sorted = sorted(
        safe_candidates,
        key=lambda x: (
            x["fp"],
            -x["sensitivity"],
            -x["f1"],
            -x["precision"],
        ),
    )

    print(
        "TYPE | PARAMETERS | TP | FP | FN | "
        "SENS | PRECISION | F1"
    )

    print("-" * 110)

    for c in safe_sorted[:20]:

        if c["type"] == "q95_threshold":

            params = (
                f"Q95 >= "
                f"{c['q95_threshold']:.2f}"
            )

        else:

            params = (
                f"Q95 >= "
                f"{c['q95_threshold']:.2f} "
                f"AND positive_windows >= "
                f"{c['min_positive_windows']}"
            )

        print(
            f"{c['type']:22s} | "
            f"{params:55s} | "
            f"{c['tp']:2d} | "
            f"{c['fp']:2d} | "
            f"{c['fn']:2d} | "
            f"{c['sensitivity']:.4f} | "
            f"{c['precision']:.4f} | "
            f"{c['f1']:.4f}"
        )


# ============================================================
# 11. BEST CANDIDATE
# ============================================================

print("\n" + "=" * 70)
print("11. BEST VALIDATION PATIENT-LEVEL RULE")
print("=" * 70)

best_candidate = None

if safe_candidates:

    safe_sorted = sorted(
        safe_candidates,
        key=lambda x: (
            x["fp"],
            -x["sensitivity"],
            -x["f1"],
            -x["precision"],
        ),
    )

    best_candidate = safe_sorted[0]

    print("Best candidate:")

    for key, value in best_candidate.items():

        print(
            f"{key}: {value}"
        )

else:

    print(
        "No safe candidate found."
    )


# ============================================================
# 12. PATIENT-BY-PATIENT BEST RULE
# ============================================================

if best_candidate is not None:

    print("\n" + "=" * 70)
    print("12. PATIENT-BY-PATIENT BEST RULE")
    print("=" * 70)

    if best_candidate["type"] == "q95_threshold":

        q_thr = best_candidate["q95_threshold"]

        for x in patient_stats:

            pred = int(
                x["q95"] >= q_thr
            )

            print(
                f"{x['patient']:8s} "
                f"Q95={x['q95']:.6f} "
                f"true={x['true_label']} "
                f"pred={pred}"
            )

    else:

        q_thr = best_candidate[
            "q95_threshold"
        ]

        min_w = best_candidate[
            "min_positive_windows"
        ]

        for x in patient_stats:

            pred = int(
                x["q95"] >= q_thr
                and
                x["positive_windows"] >= min_w
            )

            print(
                f"{x['patient']:8s} "
                f"Q95={x['q95']:.6f} "
                f"positive={x['positive_windows']:3d} "
                f"true={x['true_label']} "
                f"pred={pred}"
            )


# ============================================================
# 13. SAVE RESULTS
# ============================================================

print("\n" + "=" * 70)
print("13. SAVING RESULTS")
print("=" * 70)

output = {
    "analysis":
        "validation_patient_q95_rules",

    "window_threshold":
        window_threshold,

    "required_sensitivity":
        REQUIRED_SENSITIVITY,

    "q95_thresholds_tested":
        Q95_THRESHOLDS,

    "minimum_positive_windows_tested":
        MIN_POSITIVE_WINDOWS,

    "patient_statistics":
        patient_stats,

    "baseline_q95":
        baseline_metrics,

    "q95_threshold_candidates":
        q95_candidates,

    "q95_and_min_windows_candidates":
        and_candidates,

    "safe_candidates":
        safe_candidates,

    "best_candidate":
        best_candidate,

    "test_data_used":
        False,

    "model_modified":
        False,

    "dataset_modified":
        False,

    "threshold_modified":
        False,
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

print(
    f"[OK] Results saved:"
)

print(OUTPUT_FILE)


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 70)
print(
    "VALIDATION PATIENT-LEVEL Q95 RULE SEARCH COMPLETED"
)
print("=" * 70)

print("\nNo model was modified.")
print("No dataset was modified.")
print("Test data was NOT used.")
print("No Test optimization was performed.")

print("\nOutput:")
print(OUTPUT_FILE)

print("=" * 70)