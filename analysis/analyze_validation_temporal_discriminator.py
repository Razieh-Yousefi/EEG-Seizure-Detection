import json
from pathlib import Path

import numpy as np


# ============================================================
# CONFIG
# ============================================================

PROJECT_DIR = Path(r"C:\Users\rezay\Desktop\EEG_Seizure_Project")
RESULTS_DIR = PROJECT_DIR / "results"

VALIDATION_NPZ = RESULTS_DIR / "validation_window_probabilities.npz"
THRESHOLD_JSON = RESULTS_DIR / "validation_threshold_results.json"

OUTPUT_JSON = RESULTS_DIR / "validation_temporal_discriminator.json"

REQUIRED_SENSITIVITY = 0.90


# ============================================================
# HELPERS
# ============================================================

def calculate_runs(binary_array):
    x = np.asarray(binary_array, dtype=int)

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
    x = np.asarray(binary_array, dtype=int)

    if len(x) <= 1:
        return 0.0

    return float(np.mean(x[1:] != x[:-1]))


def find_threshold(obj):

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


def calculate_metrics(predictions, labels):

    predictions = np.asarray(predictions, dtype=int)
    labels = np.asarray(labels, dtype=int)

    tp = int(np.sum((labels == 1) & (predictions == 1)))
    fp = int(np.sum((labels == 0) & (predictions == 1)))
    fn = int(np.sum((labels == 1) & (predictions == 0)))
    tn = int(np.sum((labels == 0) & (predictions == 0)))

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


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("VALIDATION TEMPORAL DISCRIMINATOR SEARCH")
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

if not VALIDATION_NPZ.exists():
    raise FileNotFoundError(VALIDATION_NPZ)

if not THRESHOLD_JSON.exists():
    raise FileNotFoundError(THRESHOLD_JSON)

print(f"[OK] {VALIDATION_NPZ}")
print(f"[OK] {THRESHOLD_JSON}")


# ============================================================
# 2. LOAD VALIDATION DATA
# ============================================================

print()
print("=" * 70)
print("2. LOADING VALIDATION DATA")
print("=" * 70)

data = np.load(VALIDATION_NPZ, allow_pickle=True)

print()
print("Available NPZ arrays:")

for key in data.files:
    print(f"  {key:30s} shape={data[key].shape}")


probabilities = np.asarray(
    data["probabilities"],
    dtype=float
)

labels = np.asarray(
    data["labels"],
    dtype=int
)

patients = np.asarray(
    data["patients"]
).astype(str)


if "validation_indices" in data.files:

    indices = np.asarray(
        data["validation_indices"],
        dtype=int
    )

elif "indices" in data.files:

    indices = np.asarray(
        data["indices"],
        dtype=int
    )

else:

    print()
    print("[INFO] No explicit indices found.")
    print("[INFO] Using sequential indices.")

    indices = np.arange(
        len(probabilities),
        dtype=int
    )


print()
print(f"Validation samples: {len(probabilities)}")
print(f"Probabilities shape: {probabilities.shape}")
print(f"Labels shape       : {labels.shape}")
print(f"Patients shape     : {patients.shape}")
print(f"Indices shape      : {indices.shape}")


# ============================================================
# 3. VERIFY
# ============================================================

print()
print("=" * 70)
print("3. VERIFYING VALIDATION DATA")
print("=" * 70)

if not (
    len(probabilities)
    == len(labels)
    == len(patients)
    == len(indices)
):
    raise ValueError("Validation arrays are not aligned.")

print("[OK] Arrays aligned.")

if not np.all(np.isfinite(probabilities)):
    raise ValueError("Probabilities contain NaN/Inf.")

print("[OK] Probabilities finite.")


# ============================================================
# 4. LOAD FROZEN WINDOW THRESHOLD
# ============================================================

print()
print("=" * 70)
print("4. LOADING VALIDATION WINDOW THRESHOLD")
print("=" * 70)

with open(
    THRESHOLD_JSON,
    "r",
    encoding="utf-8"
) as f:

    threshold_data = json.load(f)


window_threshold = find_threshold(
    threshold_data
)

if window_threshold is None:
    raise ValueError(
        "Could not locate validation threshold."
    )

print(
    f"Window threshold: "
    f"{window_threshold:.6f}"
)


# ============================================================
# 5. WINDOW PREDICTIONS
# ============================================================

window_predictions = (
    probabilities >= window_threshold
).astype(int)


# ============================================================
# 6. BUILD PATIENT FEATURES
# ============================================================

print()
print("=" * 70)
print("6. BUILDING PATIENT FEATURES")
print("=" * 70)

patient_stats = {}

for patient in sorted(np.unique(patients)):

    mask = patients == patient

    p = probabilities[mask]
    y = labels[mask]
    pred = window_predictions[mask]
    idx = indices[mask]

    order = np.argsort(idx)

    p = p[order]
    y = y[order]
    pred = pred[order]

    total_windows = len(pred)

    positive_windows = int(
        np.sum(pred)
    )

    positive_fraction = (
        positive_windows / total_windows
        if total_windows > 0
        else 0.0
    )

    runs = calculate_runs(pred)

    q95 = float(
        np.percentile(p, 95)
    )

    mean_probability = float(
        np.mean(p)
    )

    median_probability = float(
        np.median(p)
    )

    max_probability = float(
        np.max(p)
    )

    positive_values = p[pred == 1]

    mean_positive_probability = (
        float(np.mean(positive_values))
        if len(positive_values) > 0
        else 0.0
    )

    median_positive_probability = (
        float(np.median(positive_values))
        if len(positive_values) > 0
        else 0.0
    )

    stats = {
        "patient": patient,
        "true_label": int(np.max(y)),
        "total_windows": total_windows,
        "positive_windows": positive_windows,
        "positive_fraction": positive_fraction,
        "positive_runs": len(runs),
        "max_positive_run": (
            max(runs)
            if runs
            else 0
        ),
        "mean_positive_run": (
            float(np.mean(runs))
            if runs
            else 0.0
        ),
        "transition_rate": transition_rate(pred),
        "q95": q95,
        "max_probability": max_probability,
        "mean_probability": mean_probability,
        "median_probability": median_probability,
        "mean_positive_probability":
            mean_positive_probability,
        "median_positive_probability":
            median_positive_probability,
    }

    patient_stats[patient] = stats

    print(
        f"{patient:6s} "
        f"true={stats['true_label']} "
        f"Q95={q95:.6f} "
        f"fraction={positive_fraction:.4f} "
        f"max_run={stats['max_positive_run']} "
        f"transition={stats['transition_rate']:.4f} "
        f"mean_pos={mean_positive_probability:.4f}"
    )


# ============================================================
# 7. BASELINE Q95
# ============================================================

print()
print("=" * 70)
print("7. BASELINE Q95 RULE")
print("=" * 70)

baseline_predictions = {}

for patient, stats in patient_stats.items():

    baseline_predictions[patient] = int(
        stats["q95"] >= 0.50
    )

baseline_labels = np.array([
    patient_stats[p]["true_label"]
    for p in sorted(patient_stats)
])

baseline_preds = np.array([
    baseline_predictions[p]
    for p in sorted(patient_stats)
])

baseline_metrics = calculate_metrics(
    baseline_preds,
    baseline_labels
)

print(
    json.dumps(
        baseline_metrics,
        indent=2
    )
)


# ============================================================
# 8. SEARCH TEMPORAL DISCRIMINATORS
# ============================================================

print()
print("=" * 70)
print("8. TEMPORAL DISCRIMINATOR SEARCH")
print("=" * 70)

print()
print(
    "IMPORTANT:"
)

print(
    "This search is performed ONLY on Validation."
)

print(
    "Test data is not used."
)


# ------------------------------------------------------------
# Threshold grids
# ------------------------------------------------------------

q95_thresholds = [
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
    0.95,
]

fraction_thresholds = [
    0.005,
    0.010,
    0.015,
    0.020,
    0.030,
    0.050,
    0.075,
    0.100,
    0.125,
    0.150,
    0.175,
    0.200,
]

transition_thresholds = [
    0.02,
    0.05,
    0.08,
    0.10,
    0.12,
    0.15,
    0.18,
    0.20,
    0.25,
    0.30,
]

max_run_thresholds = [
    1,
    2,
    3,
    4,
    5,
]

mean_run_thresholds = [
    1.00,
    1.05,
    1.10,
    1.15,
    1.20,
    1.25,
    1.30,
]

mean_positive_probability_thresholds = [
    0.75,
    0.80,
    0.82,
    0.84,
    0.86,
    0.88,
    0.90,
    0.92,
    0.94,
    0.96,
]


# ============================================================
# 9. GENERATE CANDIDATES
# ============================================================

patients_sorted = sorted(patient_stats)

labels_vector = np.array([
    patient_stats[p]["true_label"]
    for p in patients_sorted
])


candidates = []


def evaluate_rule(rule_name, prediction_dict):

    predictions = np.array([
        prediction_dict[p]
        for p in patients_sorted
    ])

    m = calculate_metrics(
        predictions,
        labels_vector
    )

    candidate = {
        "rule": rule_name,
        "metrics": m,
        "predictions": {
            p: int(prediction_dict[p])
            for p in patients_sorted
        },
    }

    candidates.append(candidate)


# ============================================================
# A. Q95 + FRACTION
# ============================================================

for q95_t in q95_thresholds:

    for frac_t in fraction_thresholds:

        predictions = {}

        for patient in patients_sorted:

            s = patient_stats[patient]

            predictions[patient] = int(
                s["q95"] >= q95_t
                and s["positive_fraction"] >= frac_t
            )

        evaluate_rule(
            f"Q95 >= {q95_t:.2f} AND "
            f"fraction >= {frac_t:.3f}",
            predictions
        )


# ============================================================
# B. Q95 + TRANSITION RATE
# ============================================================

for q95_t in q95_thresholds:

    for tr_t in transition_thresholds:

        predictions = {}

        for patient in patients_sorted:

            s = patient_stats[patient]

            predictions[patient] = int(
                s["q95"] >= q95_t
                and s["transition_rate"] >= tr_t
            )

        evaluate_rule(
            f"Q95 >= {q95_t:.2f} AND "
            f"transition >= {tr_t:.3f}",
            predictions
        )


# ============================================================
# C. Q95 + MAX RUN
# ============================================================

for q95_t in q95_thresholds:

    for run_t in max_run_thresholds:

        predictions = {}

        for patient in patients_sorted:

            s = patient_stats[patient]

            predictions[patient] = int(
                s["q95"] >= q95_t
                and s["max_positive_run"] >= run_t
            )

        evaluate_rule(
            f"Q95 >= {q95_t:.2f} AND "
            f"max_run >= {run_t}",
            predictions
        )


# ============================================================
# D. Q95 + MEAN RUN
# ============================================================

for q95_t in q95_thresholds:

    for run_t in mean_run_thresholds:

        predictions = {}

        for patient in patients_sorted:

            s = patient_stats[patient]

            predictions[patient] = int(
                s["q95"] >= q95_t
                and s["mean_positive_run"] >= run_t
            )

        evaluate_rule(
            f"Q95 >= {q95_t:.2f} AND "
            f"mean_run >= {run_t:.2f}",
            predictions
        )


# ============================================================
# E. Q95 + MEAN POSITIVE PROBABILITY
# ============================================================

for q95_t in q95_thresholds:

    for prob_t in mean_positive_probability_thresholds:

        predictions = {}

        for patient in patients_sorted:

            s = patient_stats[patient]

            predictions[patient] = int(
                s["q95"] >= q95_t
                and
                s["mean_positive_probability"] >= prob_t
            )

        evaluate_rule(
            f"Q95 >= {q95_t:.2f} AND "
            f"mean_positive_probability >= {prob_t:.2f}",
            predictions
        )


# ============================================================
# 10. THREE-FEATURE RULES
# ============================================================

print()
print("=" * 70)
print("10. THREE-FEATURE RULE SEARCH")
print("=" * 70)


# ------------------------------------------------------------
# Q95 + fraction + transition
# ------------------------------------------------------------

for q95_t in q95_thresholds:

    for frac_t in fraction_thresholds:

        for tr_t in transition_thresholds:

            predictions = {}

            for patient in patients_sorted:

                s = patient_stats[patient]

                predictions[patient] = int(
                    s["q95"] >= q95_t
                    and
                    s["positive_fraction"] >= frac_t
                    and
                    s["transition_rate"] >= tr_t
                )

            evaluate_rule(
                f"Q95 >= {q95_t:.2f} AND "
                f"fraction >= {frac_t:.3f} AND "
                f"transition >= {tr_t:.3f}",
                predictions
            )


# ------------------------------------------------------------
# Q95 + fraction + max run
# ------------------------------------------------------------

for q95_t in q95_thresholds:

    for frac_t in fraction_thresholds:

        for run_t in max_run_thresholds:

            predictions = {}

            for patient in patients_sorted:

                s = patient_stats[patient]

                predictions[patient] = int(
                    s["q95"] >= q95_t
                    and
                    s["positive_fraction"] >= frac_t
                    and
                    s["max_positive_run"] >= run_t
                )

            evaluate_rule(
                f"Q95 >= {q95_t:.2f} AND "
                f"fraction >= {frac_t:.3f} AND "
                f"max_run >= {run_t}",
                predictions
            )


# ------------------------------------------------------------
# Q95 + fraction + mean positive probability
# ------------------------------------------------------------

for q95_t in q95_thresholds:

    for frac_t in fraction_thresholds:

        for prob_t in mean_positive_probability_thresholds:

            predictions = {}

            for patient in patients_sorted:

                s = patient_stats[patient]

                predictions[patient] = int(
                    s["q95"] >= q95_t
                    and
                    s["positive_fraction"] >= frac_t
                    and
                    s["mean_positive_probability"] >= prob_t
                )

            evaluate_rule(
                f"Q95 >= {q95_t:.2f} AND "
                f"fraction >= {frac_t:.3f} AND "
                f"mean_positive_probability >= {prob_t:.2f}",
                predictions
            )


# ============================================================
# 11. SAFE CANDIDATES
# ============================================================

print()
print("=" * 70)
print("11. SAFE VALIDATION CANDIDATES")
print("=" * 70)

safe_candidates = [
    c
    for c in candidates
    if c["metrics"]["sensitivity"]
    >= REQUIRED_SENSITIVITY
]


print()
print(
    f"Total candidates: {len(candidates)}"
)

print(
    f"Safe candidates: {len(safe_candidates)}"
)


# ============================================================
# 12. SORT
# ============================================================

safe_candidates.sort(
    key=lambda c: (
        c["metrics"]["f1"],
        c["metrics"]["specificity"],
        c["metrics"]["precision"],
    ),
    reverse=True,
)


print()
print(
    f"{'RULE':75s} "
    f"{'TP':>3s} "
    f"{'FP':>3s} "
    f"{'FN':>3s} "
    f"{'SENS':>7s} "
    f"{'SPEC':>7s} "
    f"{'PREC':>7s} "
    f"{'F1':>7s}"
)

print("-" * 150)


for candidate in safe_candidates[:40]:

    m = candidate["metrics"]

    print(
        f"{candidate['rule'][:75]:75s} "
        f"{m['tp']:3d} "
        f"{m['fp']:3d} "
        f"{m['fn']:3d} "
        f"{m['sensitivity']:.4f} "
        f"{m['specificity']:.4f} "
        f"{m['precision']:.4f} "
        f"{m['f1']:.4f}"
    )


# ============================================================
# 13. ZERO-FP CHECK
# ============================================================

print()
print("=" * 70)
print("13. ZERO-FP CHECK")
print("=" * 70)

zero_fp_candidates = [
    c
    for c in safe_candidates
    if c["metrics"]["fp"] == 0
]


if zero_fp_candidates:

    print()
    print(
        f"[SUCCESS] "
        f"{len(zero_fp_candidates)} "
        f"zero-FP candidates found."
    )

    zero_fp_candidates.sort(
        key=lambda c: (
            c["metrics"]["sensitivity"],
            c["metrics"]["f1"],
            c["metrics"]["specificity"],
        ),
        reverse=True,
    )

    print()

    for candidate in zero_fp_candidates[:20]:

        print(
            candidate["rule"],
            "=>",
            candidate["metrics"]
        )

else:

    print()
    print(
        "[INFO] No zero-FP rule found while "
        "maintaining sensitivity >= 0.90."
    )


# ============================================================
# 14. BEST RULE
# ============================================================

print()
print("=" * 70)
print("14. BEST VALIDATION TEMPORAL DISCRIMINATOR")
print("=" * 70)

if not safe_candidates:

    print()
    print(
        "[WARNING] No safe candidate found."
    )

    best_candidate = None

else:

    best_candidate = safe_candidates[0]

    print()
    print(
        "Best rule:"
    )

    print(
        best_candidate["rule"]
    )

    print()
    print(
        json.dumps(
            best_candidate["metrics"],
            indent=2
        )
    )

    print()
    print("Patient predictions:")

    for patient in patients_sorted:

        stats = patient_stats[patient]

        pred = best_candidate[
            "predictions"
        ][patient]

        print(
            f"{patient:6s} "
            f"true={stats['true_label']} "
            f"pred={pred} "
            f"Q95={stats['q95']:.6f} "
            f"fraction={stats['positive_fraction']:.6f} "
            f"transition={stats['transition_rate']:.6f} "
            f"max_run={stats['max_positive_run']} "
            f"mean_pos={stats['mean_positive_probability']:.6f}"
        )


# ============================================================
# 15. SAVE
# ============================================================

print()
print("=" * 70)
print("15. SAVING RESULTS")
print("=" * 70)

output = {
    "analysis":
        "validation_temporal_discriminator",

    "validation_window_threshold":
        window_threshold,

    "required_sensitivity":
        REQUIRED_SENSITIVITY,

    "baseline_rule":
        "Q95 >= 0.50",

    "baseline_metrics":
        baseline_metrics,

    "best_candidate":
        best_candidate,

    "zero_fp_candidates":
        zero_fp_candidates[:50],

    "patient_statistics":
        patient_stats,

    "notes": [
        "Validation-only discriminator search.",
        "Test data was not used.",
        "No model was modified.",
        "No dataset was modified.",
        "The original validation threshold was not modified.",
        "Any selected rule must be frozen before Test evaluation.",
    ],
}


with open(
    OUTPUT_JSON,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        output,
        f,
        indent=2,
        ensure_ascii=False
    )


print()
print("[OK] Results saved:")
print(OUTPUT_JSON)

print()
print("=" * 70)
print("VALIDATION TEMPORAL DISCRIMINATOR SEARCH COMPLETED")
print("=" * 70)

print()
print("No model was modified.")
print("No dataset was modified.")
print("Test data was NOT used.")
print("No Test optimization was performed.")