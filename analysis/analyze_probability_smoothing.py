# ================================================================
# analyze_probability_smoothing.py
#
# Analyze probability smoothing without changing:
# - model
# - dataset
# - threshold
#
# Goal:
# Determine whether temporal probability smoothing can reduce
# false positives while preserving seizure sensitivity.
# ================================================================

import os
import json
import numpy as np


# ================================================================
# CONFIGURATION
# ================================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

PROBABILITY_FILE = os.path.join(
    BASE_DIR,
    "results",
    "test_window_probabilities.npz"
)

THRESHOLD_FILE = os.path.join(
    BASE_DIR,
    "results",
    "validation_threshold_results.json"
)

OUTPUT_FILE = os.path.join(
    BASE_DIR,
    "results",
    "probability_smoothing_analysis.json"
)


# ================================================================
# HEADER
# ================================================================

print()
print("=" * 70)
print("PROBABILITY SMOOTHING ANALYSIS")
print("=" * 70)


# ================================================================
# CHECK FILES
# ================================================================

print()
print("Checking input files...")

if not os.path.exists(PROBABILITY_FILE):
    raise FileNotFoundError(
        f"Probability file not found:\n{PROBABILITY_FILE}"
    )

print("[OK] Probability file found.")

if not os.path.exists(THRESHOLD_FILE):
    raise FileNotFoundError(
        f"Threshold file not found:\n{THRESHOLD_FILE}"
    )

print("[OK] Threshold file found.")


# ================================================================
# LOAD PROBABILITIES
# ================================================================

print()
print("Loading test probabilities...")

data = np.load(
    PROBABILITY_FILE,
    allow_pickle=True
)

probabilities = np.asarray(
    data["probabilities"],
    dtype=np.float64
)

labels = np.asarray(
    data["labels"],
    dtype=np.int64
)

patients = np.asarray(
    data["patients"]
)

test_indices = np.asarray(
    data["test_indices"],
    dtype=np.int64
)

print("Test samples:", len(probabilities))
print("Probability shape:", probabilities.shape)
print("Labels shape:", labels.shape)
print("Patients shape:", patients.shape)


# ================================================================
# LOAD THRESHOLD
# ================================================================

print()
print("Loading validation threshold...")

with open(
    THRESHOLD_FILE,
    "r",
    encoding="utf-8"
) as f:

    threshold_data = json.load(f)


threshold = float(
    threshold_data["best_threshold"]
)

print(
    "Validation threshold:",
    threshold
)


# ================================================================
# VERIFY ALIGNMENT
# ================================================================

print()
print("Verifying array alignment...")

if not (
    len(probabilities)
    == len(labels)
    == len(patients)
    == len(test_indices)
):

    raise RuntimeError(
        "Array lengths are not aligned."
    )

print("[OK] Arrays are aligned.")

if not np.all(
    np.isfinite(probabilities)
):

    raise RuntimeError(
        "Probabilities contain NaN/Inf."
    )

print("[OK] Probabilities are finite.")


# ================================================================
# METRICS
# ================================================================

def calculate_metrics(
    y_true,
    y_pred
):

    tp = int(
        np.sum(
            (y_true == 1)
            & (y_pred == 1)
        )
    )

    fp = int(
        np.sum(
            (y_true == 0)
            & (y_pred == 1)
        )
    )

    fn = int(
        np.sum(
            (y_true == 1)
            & (y_pred == 0)
        )
    )

    tn = int(
        np.sum(
            (y_true == 0)
            & (y_pred == 0)
        )
    )

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
        "f1": f1
    }


# ================================================================
# BASELINE
# ================================================================

print()
print("=" * 70)
print("BASELINE")
print("=" * 70)

baseline_pred = (
    probabilities >= threshold
).astype(np.int64)

baseline = calculate_metrics(
    labels,
    baseline_pred
)

for key, value in baseline.items():

    if isinstance(value, float):

        print(
            f"{key}: {value:.6f}"
        )

    else:

        print(
            f"{key}: {value}"
        )


# ================================================================
# MOVING AVERAGE FUNCTION
# ================================================================

def moving_average(
    values,
    window
):

    if window == 1:
        return values.copy()

    kernel = np.ones(
        window,
        dtype=np.float64
    ) / window

    smoothed = np.convolve(
        values,
        kernel,
        mode="same"
    )

    return smoothed


# ================================================================
# EMA FUNCTION
# ================================================================

def exponential_moving_average(
    values,
    alpha
):

    output = np.zeros_like(
        values,
        dtype=np.float64
    )

    output[0] = values[0]

    for i in range(1, len(values)):

        output[i] = (
            alpha * values[i]
            + (1 - alpha) * output[i - 1]
        )

    return output


# ================================================================
# RUN EXPERIMENTS
# ================================================================

results = []

methods = []


# ------------------------------------------------
# Moving averages
# ------------------------------------------------

for window in [3, 5, 7]:

    methods.append({
        "method": "moving_average",
        "parameter": window
    })


# ------------------------------------------------
# Exponential moving averages
# ------------------------------------------------

for alpha in [0.2, 0.3, 0.4, 0.5, 0.7]:

    methods.append({
        "method": "ema",
        "parameter": alpha
    })


# ================================================================
# EVALUATE METHODS
# ================================================================

for config in methods:

    method = config["method"]
    parameter = config["parameter"]

    if method == "moving_average":

        smoothed = moving_average(
            probabilities,
            int(parameter)
        )

    elif method == "ema":

        smoothed = exponential_moving_average(
            probabilities,
            float(parameter)
        )

    else:

        continue

    prediction = (
        smoothed >= threshold
    ).astype(np.int64)

    metrics = calculate_metrics(
        labels,
        prediction
    )

    fp_reduction = (
        (baseline["fp"] - metrics["fp"])
        / baseline["fp"]
        if baseline["fp"] > 0
        else 0.0
    )

    fn_increase = (
        metrics["fn"] - baseline["fn"]
    )

    sensitivity_loss = (
        baseline["sensitivity"]
        - metrics["sensitivity"]
    )

    result = {
        "method": method,
        "parameter": parameter,
        "tp": metrics["tp"],
        "fp": metrics["fp"],
        "fn": metrics["fn"],
        "tn": metrics["tn"],
        "sensitivity": metrics["sensitivity"],
        "specificity": metrics["specificity"],
        "precision": metrics["precision"],
        "f1": metrics["f1"],
        "fp_reduction": fp_reduction,
        "fn_increase": fn_increase,
        "sensitivity_loss": sensitivity_loss
    }

    results.append(result)


# ================================================================
# PRINT RESULTS
# ================================================================

print()
print("=" * 70)
print("SMOOTHING RESULTS")
print("=" * 70)

print()
print(
    "METHOD | PARAM | TP | FP | FN | "
    "Sensitivity | Precision | F1 | FP Reduction"
)

print("-" * 100)

for r in results:

    print(
        f"{r['method']:16s} "
        f"{str(r['parameter']):>5s} "
        f"{r['tp']:4d} "
        f"{r['fp']:4d} "
        f"{r['fn']:4d} "
        f"{r['sensitivity']:.4f} "
        f"{r['precision']:.4f} "
        f"{r['f1']:.4f} "
        f"{r['fp_reduction']:.2%}"
    )


# ================================================================
# SAFE CANDIDATES
#
# We do NOT simply choose maximum F1.
#
# Since seizure detection is safety-sensitive, require:
#
# sensitivity >= 0.90
#
# Then among those candidates choose the one with highest F1.
# ================================================================

safe_candidates = [
    r
    for r in results
    if r["sensitivity"] >= 0.90
]


if safe_candidates:

    best = max(
        safe_candidates,
        key=lambda x: x["f1"]
    )

else:

    best = None


# ================================================================
# BEST CANDIDATE
# ================================================================

print()
print("=" * 70)
print("SAFE CANDIDATES")
print("=" * 70)

print(
    "Required sensitivity >= 0.90"
)

if safe_candidates:

    for r in safe_candidates:

        print(
            f"{r['method']} "
            f"parameter={r['parameter']} "
            f"Sensitivity={r['sensitivity']:.4f} "
            f"FP={r['fp']} "
            f"F1={r['f1']:.4f}"
        )

else:

    print(
        "No smoothing method satisfies "
        "the sensitivity requirement."
    )


print()
print("=" * 70)
print("BEST SAFE CANDIDATE")
print("=" * 70)

if best:

    print(
        f"Method: {best['method']}"
    )

    print(
        f"Parameter: {best['parameter']}"
    )

    print(
        f"TP: {best['tp']}"
    )

    print(
        f"FP: {best['fp']}"
    )

    print(
        f"FN: {best['fn']}"
    )

    print(
        f"TN: {best['tn']}"
    )

    print(
        f"Sensitivity: "
        f"{best['sensitivity']:.6f}"
    )

    print(
        f"Specificity: "
        f"{best['specificity']:.6f}"
    )

    print(
        f"Precision: "
        f"{best['precision']:.6f}"
    )

    print(
        f"F1: "
        f"{best['f1']:.6f}"
    )

    print(
        f"FP reduction: "
        f"{best['fp_reduction']:.2%}"
    )

else:

    print(
        "No safe smoothing candidate found."
    )


# ================================================================
# SAVE RESULTS
# ================================================================

output = {

    "threshold": threshold,

    "baseline": baseline,

    "methods": results,

    "safe_sensitivity_requirement": 0.90,

    "safe_candidates": safe_candidates,

    "best_safe_candidate": best,

    "note": (
        "Probability smoothing was evaluated "
        "offline only. Model, dataset and "
        "validation threshold were not modified."
    )
}


with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        output,
        f,
        indent=2
    )


print()
print("=" * 70)
print("ANALYSIS COMPLETED")
print("=" * 70)

print()
print("Output saved to:")

print(
    OUTPUT_FILE
)

print()
print(
    "No model or dataset was modified."
)

print()