import json
from pathlib import Path

import numpy as np


# ======================================================================
# CONFIG
# ======================================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_DIR / "results"

INPUT_NPZ = RESULTS_DIR / "validation_window_probabilities.npz"
THRESHOLD_JSON = RESULTS_DIR / "validation_threshold_results.json"

OUTPUT_JSON = RESULTS_DIR / "validation_fp_refinement.json"

REQUIRED_SENSITIVITY = 0.90


# ======================================================================
# HELPERS
# ======================================================================

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


def positive_runs(binary_array):
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


def patient_statistics(probabilities, labels, threshold):
    positive = probabilities >= threshold

    runs = positive_runs(positive.astype(int))

    positive_windows = int(np.sum(positive))
    total_windows = len(probabilities)

    fraction = (
        positive_windows / total_windows
        if total_windows > 0
        else 0.0
    )

    transitions = (
        int(np.sum(positive[1:] != positive[:-1]))
        if len(positive) > 1
        else 0
    )

    transition_rate = (
        transitions / (len(positive) - 1)
        if len(positive) > 1
        else 0.0
    )

    return {
        "total_windows": int(total_windows),
        "positive_windows": positive_windows,
        "positive_fraction": float(fraction),
        "positive_runs": int(len(runs)),
        "max_positive_run": int(max(runs)) if runs else 0,
        "mean_positive_run": float(np.mean(runs)) if runs else 0.0,
        "median_positive_run": float(np.median(runs)) if runs else 0.0,
        "q95": float(np.percentile(probabilities, 95)),
        "q90": float(np.percentile(probabilities, 90)),
        "q99": float(np.percentile(probabilities, 99)),
        "max_probability": float(np.max(probabilities)),
        "mean_probability": float(np.mean(probabilities)),
        "median_probability": float(np.median(probabilities)),
        "transition_rate": float(transition_rate),
    }


# ======================================================================
# MAIN
# ======================================================================

print_header("VALIDATION FALSE-POSITIVE REFINEMENT ANALYSIS")

print()
print("Project directory:")
print(PROJECT_DIR)

print()
print("Results directory:")
print(RESULTS_DIR)


# ======================================================================
# 1. CHECK INPUT FILES
# ======================================================================

print_header("1. CHECKING INPUT FILES")

if not INPUT_NPZ.exists():
    raise FileNotFoundError(f"Missing input file: {INPUT_NPZ}")

if not THRESHOLD_JSON.exists():
    raise FileNotFoundError(f"Missing threshold file: {THRESHOLD_JSON}")

print(f"[OK] {INPUT_NPZ}")
print(f"[OK] {THRESHOLD_JSON}")


# ======================================================================
# 2. LOAD DATA
# ======================================================================

print_header("2. LOADING VALIDATION DATA")

data = np.load(INPUT_NPZ, allow_pickle=True)

print()
print("Available NPZ arrays:")

for key in data.files:
    print(f"  {key:30s} shape={data[key].shape}")

probabilities = np.asarray(data["probabilities"], dtype=float)
labels = np.asarray(data["labels"], dtype=int)
patients = np.asarray(data["patients"])

if "validation_indices" in data.files:
    indices = np.asarray(data["validation_indices"], dtype=int)
elif "indices" in data.files:
    indices = np.asarray(data["indices"], dtype=int)
else:
    print()
    print("[INFO] No index array found.")
    print("[INFO] Using sequential sample indices.")
    indices = np.arange(len(probabilities), dtype=int)

print()
print(f"Validation samples: {len(probabilities)}")
print(f"Probabilities shape: {probabilities.shape}")
print(f"Labels shape       : {labels.shape}")
print(f"Patients shape     : {patients.shape}")
print(f"Indices shape      : {indices.shape}")


# ======================================================================
# 3. VERIFY DATA
# ======================================================================

print_header("3. VERIFYING DATA")

if not (
    len(probabilities)
    == len(labels)
    == len(patients)
    == len(indices)
):
    raise ValueError("Arrays are not aligned.")

print("[OK] Arrays aligned.")

if not np.all(np.isfinite(probabilities)):
    raise ValueError("Probabilities contain NaN or Inf.")

print("[OK] Probabilities finite.")


# ======================================================================
# 4. LOAD THRESHOLD
# ======================================================================

print_header("4. LOADING VALIDATION THRESHOLD")

with open(THRESHOLD_JSON, "r", encoding="utf-8") as f:
    threshold_data = json.load(f)

threshold = None

possible_keys = [
    "threshold",
    "validation_threshold",
    "best_threshold",
]

for key in possible_keys:
    if key in threshold_data:
        threshold = float(threshold_data[key])
        break

if threshold is None:
    raise KeyError(
        "Could not find validation threshold in "
        "validation_threshold_results.json"
    )

print(f"Window threshold: {threshold:.6f}")


# ======================================================================
# 5. BUILD PATIENT STATISTICS
# ======================================================================

print_header("5. BUILDING PATIENT STATISTICS")

patient_stats = {}

unique_patients = np.unique(patients)

for patient in unique_patients:

    mask = patients == patient

    p = probabilities[mask]
    y = labels[mask]

    patient_label = int(np.max(y)) if len(y) else 0

    stats = patient_statistics(
        probabilities=p,
        labels=y,
        threshold=threshold,
    )

    stats["true_label"] = patient_label

    patient_stats[str(patient)] = stats

    print(
        f"{str(patient):8s} "
        f"windows={stats['total_windows']:4d} "
        f"positive={stats['positive_windows']:3d} "
        f"fraction={stats['positive_fraction']:.4f} "
        f"runs={stats['positive_runs']:3d} "
        f"max_run={stats['max_positive_run']:2d} "
        f"mean_run={stats['mean_positive_run']:.2f} "
        f"transition={stats['transition_rate']:.4f} "
        f"Q95={stats['q95']:.6f} "
        f"true={patient_label}"
    )


# ======================================================================
# 6. IDENTIFY BASELINE FP
# ======================================================================

print_header("6. IDENTIFYING BASELINE FALSE POSITIVES")

baseline_predictions = []

for patient, stats in patient_stats.items():

    pred = int(stats["q95"] >= 0.50)

    baseline_predictions.append(
        {
            "patient": patient,
            "true_label": stats["true_label"],
            "prediction": pred,
        }
    )

    if stats["true_label"] == 0 and pred == 1:
        print()
        print("[FALSE POSITIVE]")
        print(f"Patient: {patient}")
        print(f"Q95: {stats['q95']:.6f}")
        print(
            f"Positive fraction: "
            f"{stats['positive_fraction']:.6f}"
        )
        print(
            f"Positive windows: "
            f"{stats['positive_windows']}"
        )
        print(
            f"Positive runs: "
            f"{stats['positive_runs']}"
        )
        print(
            f"Maximum positive run: "
            f"{stats['max_positive_run']}"
        )
        print(
            f"Mean positive run: "
            f"{stats['mean_positive_run']:.4f}"
        )
        print(
            f"Transition rate: "
            f"{stats['transition_rate']:.6f}"
        )


# ======================================================================
# 7. BASELINE METRICS
# ======================================================================

print_header("7. BASELINE PATIENT-LEVEL Q95")

y_true = [
    item["true_label"]
    for item in baseline_predictions
]

y_pred = [
    item["prediction"]
    for item in baseline_predictions
]

baseline_metrics = calculate_metrics(
    y_true,
    y_pred,
)

print(json.dumps(baseline_metrics, indent=2))


# ======================================================================
# 8. RULE SEARCH
# ======================================================================

print_header("8. FALSE-POSITIVE REFINEMENT RULE SEARCH")

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
    0.250,
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
    1.20,
    1.30,
    1.40,
    1.50,
]

transition_thresholds = [
    0.05,
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
]

candidates = []


def evaluate_rule(rule_name, rule_function):

    predictions = []

    for patient, stats in patient_stats.items():

        prediction = int(rule_function(stats))

        predictions.append(prediction)

    metrics = calculate_metrics(
        y_true,
        predictions,
    )

    candidates.append(
        {
            "rule": rule_name,
            "metrics": metrics,
            "predictions": {
                patient: int(pred)
                for patient, pred in zip(
                    patient_stats.keys(),
                    predictions,
                )
            },
        }
    )


# ----------------------------------------------------------------------
# Q95 + FRACTION
# ----------------------------------------------------------------------

for q95_t in q95_thresholds:

    for fraction_t in fraction_thresholds:

        rule_name = (
            f"Q95 >= {q95_t:.2f} AND "
            f"fraction >= {fraction_t:.3f}"
        )

        evaluate_rule(
            rule_name,
            lambda s,
                   q=q95_t,
                   f=fraction_t:
                s["q95"] >= q
                and s["positive_fraction"] >= f,
        )


# ----------------------------------------------------------------------
# Q95 + MAX RUN
# ----------------------------------------------------------------------

for q95_t in q95_thresholds:

    for run_t in max_run_thresholds:

        rule_name = (
            f"Q95 >= {q95_t:.2f} AND "
            f"max_run >= {run_t}"
        )

        evaluate_rule(
            rule_name,
            lambda s,
                   q=q95_t,
                   r=run_t:
                s["q95"] >= q
                and s["max_positive_run"] >= r,
        )


# ----------------------------------------------------------------------
# Q95 + MEAN RUN
# ----------------------------------------------------------------------

for q95_t in q95_thresholds:

    for mean_run_t in mean_run_thresholds:

        rule_name = (
            f"Q95 >= {q95_t:.2f} AND "
            f"mean_run >= {mean_run_t:.2f}"
        )

        evaluate_rule(
            rule_name,
            lambda s,
                   q=q95_t,
                   r=mean_run_t:
                s["q95"] >= q
                and s["mean_positive_run"] >= r,
        )


# ----------------------------------------------------------------------
# Q95 + TRANSITION RATE
# ----------------------------------------------------------------------

for q95_t in q95_thresholds:

    for transition_t in transition_thresholds:

        rule_name = (
            f"Q95 >= {q95_t:.2f} AND "
            f"transition_rate >= {transition_t:.2f}"
        )

        evaluate_rule(
            rule_name,
            lambda s,
                   q=q95_t,
                   t=transition_t:
                s["q95"] >= q
                and s["transition_rate"] >= t,
        )


# ----------------------------------------------------------------------
# Q95 + FRACTION + MAX RUN
# ----------------------------------------------------------------------

for q95_t in q95_thresholds:

    for fraction_t in fraction_thresholds:

        for run_t in max_run_thresholds:

            rule_name = (
                f"Q95 >= {q95_t:.2f} AND "
                f"fraction >= {fraction_t:.3f} AND "
                f"max_run >= {run_t}"
            )

            evaluate_rule(
                rule_name,
                lambda s,
                       q=q95_t,
                       f=fraction_t,
                       r=run_t:
                    s["q95"] >= q
                    and s["positive_fraction"] >= f
                    and s["max_positive_run"] >= r,
            )


# ----------------------------------------------------------------------
# Q95 + FRACTION + MEAN RUN
# ----------------------------------------------------------------------

for q95_t in q95_thresholds:

    for fraction_t in fraction_thresholds:

        for mean_run_t in mean_run_thresholds:

            rule_name = (
                f"Q95 >= {q95_t:.2f} AND "
                f"fraction >= {fraction_t:.3f} AND "
                f"mean_run >= {mean_run_t:.2f}"
            )

            evaluate_rule(
                rule_name,
                lambda s,
                       q=q95_t,
                       f=fraction_t,
                       r=mean_run_t:
                    s["q95"] >= q
                    and s["positive_fraction"] >= f
                    and s["mean_positive_run"] >= r,
            )


# ----------------------------------------------------------------------
# Q95 + FRACTION + TRANSITION
# ----------------------------------------------------------------------

for q95_t in q95_thresholds:

    for fraction_t in fraction_thresholds:

        for transition_t in transition_thresholds:

            rule_name = (
                f"Q95 >= {q95_t:.2f} AND "
                f"fraction >= {fraction_t:.3f} AND "
                f"transition_rate >= {transition_t:.2f}"
            )

            evaluate_rule(
                rule_name,
                lambda s,
                       q=q95_t,
                       f=fraction_t,
                       t=transition_t:
                    s["q95"] >= q
                    and s["positive_fraction"] >= f
                    and s["transition_rate"] >= t,
            )


# ======================================================================
# 9. SAFE CANDIDATES
# ======================================================================

print_header("9. SAFE CANDIDATES")

safe_candidates = [
    c
    for c in candidates
    if c["metrics"]["sensitivity"] >= REQUIRED_SENSITIVITY
]

print(f"Total candidates: {len(candidates)}")
print(f"Safe candidates: {len(safe_candidates)}")


# Sort:
# 1. lowest FP
# 2. highest F1
# 3. highest precision
# 4. highest specificity

safe_candidates.sort(
    key=lambda c: (
        c["metrics"]["fp"],
        -c["metrics"]["f1"],
        -c["metrics"]["precision"],
        -c["metrics"]["specificity"],
    )
)


print()
print(
    "RULE | TP | FP | FN | "
    "SENS | SPEC | PREC | F1"
)
print("-" * 110)

for candidate in safe_candidates[:40]:

    m = candidate["metrics"]

    print(
        f"{candidate['rule'][:70]:70s} | "
        f"{m['tp']:2d} | "
        f"{m['fp']:2d} | "
        f"{m['fn']:2d} | "
        f"{m['sensitivity']:.4f} | "
        f"{m['specificity']:.4f} | "
        f"{m['precision']:.4f} | "
        f"{m['f1']:.4f}"
    )


# ======================================================================
# 10. CHECK WHETHER FP CAN BE REMOVED
# ======================================================================

print_header("10. CAN THE FALSE POSITIVE BE REMOVED?")

zero_fp_candidates = [
    c
    for c in safe_candidates
    if c["metrics"]["fp"] == 0
]

if zero_fp_candidates:

    print(
        f"[SUCCESS] Found "
        f"{len(zero_fp_candidates)} safe candidates "
        f"with FP = 0."
    )

    best_zero_fp = zero_fp_candidates[0]

    print()
    print("BEST ZERO-FP RULE:")
    print(best_zero_fp["rule"])
    print()
    print(json.dumps(
        best_zero_fp["metrics"],
        indent=2,
    ))

else:

    print(
        "[INFO] No safe rule removed the false positive "
        "while maintaining required sensitivity."
    )

    print()
    print(
        "This means the current validation sample does not "
        "support a temporal rule that simultaneously gives "
        "sensitivity >= 0.90 and FP = 0."
    )


# ======================================================================
# 11. BEST OVERALL SAFE RULE
# ======================================================================

print_header("11. BEST VALIDATION REFINEMENT RULE")

if safe_candidates:

    best_rule = safe_candidates[0]

    print("Best rule:")
    print(best_rule["rule"])

    print()
    print("Metrics:")

    print(
        json.dumps(
            best_rule["metrics"],
            indent=2,
        )
    )

    print()
    print("Patient predictions:")

    for patient, prediction in (
        best_rule["predictions"].items()
    ):

        stats = patient_stats[patient]

        print(
            f"{patient:8s} "
            f"true={stats['true_label']} "
            f"pred={prediction} "
            f"Q95={stats['q95']:.6f} "
            f"fraction={stats['positive_fraction']:.6f} "
            f"max_run={stats['max_positive_run']} "
            f"mean_run={stats['mean_positive_run']:.2f}"
        )

else:

    best_rule = None

    print(
        "[WARNING] No candidate satisfied "
        "the required sensitivity."
    )


# ======================================================================
# 12. SAVE RESULTS
# ======================================================================

print_header("12. SAVING RESULTS")

output = {
    "analysis": "validation_false_positive_refinement",
    "window_threshold": threshold,
    "required_sensitivity": REQUIRED_SENSITIVITY,
    "baseline_metrics": baseline_metrics,
    "patient_statistics": patient_stats,
    "total_candidates": len(candidates),
    "safe_candidates": len(safe_candidates),
    "zero_fp_candidates": len(zero_fp_candidates),
    "best_rule": best_rule,
    "top_safe_candidates": safe_candidates[:50],
}

with open(
    OUTPUT_JSON,
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
print(OUTPUT_JSON)


# ======================================================================
# FINAL SAFETY STATEMENT
# ======================================================================

print()
print("=" * 70)
print("VALIDATION FALSE-POSITIVE REFINEMENT COMPLETED")
print("=" * 70)

print()
print("No model was modified.")
print("No dataset was modified.")
print("Validation threshold was NOT modified.")
print("Test data was NOT used.")
print("No Test optimization was performed.")

if zero_fp_candidates:
    print()
    print("IMPORTANT:")
    print(
        "A zero-FP rule exists on Validation, "
        "but it must NOT be applied to Test yet."
    )
    print(
        "The next step is to freeze the selected rule "
        "and perform ONE final evaluation on Test."
    )
else:
    print()
    print("IMPORTANT:")
    print(
        "No zero-FP rule was found under the required "
        "sensitivity constraint."
    )
    print(
        "Do NOT tune the rule using Test data."
    )