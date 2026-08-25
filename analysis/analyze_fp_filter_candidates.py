# ================================================================
# analyze_fp_filter_candidates.py
#
# Analyze simple post-processing filters for reducing false
# positives while preserving high seizure sensitivity.
#
# IMPORTANT:
# - Does NOT modify the model
# - Does NOT modify the dataset
# - Does NOT retrain anything
# - Uses saved test probabilities and labels
# ================================================================

import os
import json
import numpy as np


# ================================================================
# 1. PATHS
# ================================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PROJECT_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

RESULTS_DIR = os.path.join(
    PROJECT_DIR,
    "results"
)

PROB_FILE = os.path.join(
    RESULTS_DIR,
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
    "fp_filter_candidate_analysis.json"
)


# ================================================================
# 2. HEADER
# ================================================================

print()
print("=" * 70)
print("FALSE POSITIVE FILTER CANDIDATE ANALYSIS")
print("=" * 70)


# ================================================================
# 3. CHECK FILES
# ================================================================

print()
print("Checking input files...")

if not os.path.exists(PROB_FILE):
    raise FileNotFoundError(
        f"Probability file not found:\n{PROB_FILE}"
    )

if not os.path.exists(THRESHOLD_FILE):
    raise FileNotFoundError(
        f"Threshold file not found:\n{THRESHOLD_FILE}"
    )

print("[OK] Probability file found.")
print("[OK] Threshold file found.")


# ================================================================
# 4. LOAD PROBABILITIES
# ================================================================

print()
print("Loading test probabilities...")

data = np.load(
    PROB_FILE,
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

print("Test samples:", len(probabilities))
print("Probability shape:", probabilities.shape)
print("Labels shape:", labels.shape)
print("Patients shape:", patients.shape)


# ================================================================
# 5. LOAD VALIDATION THRESHOLD
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

print("Validation threshold:", threshold)


# ================================================================
# 6. VALIDATION
# ================================================================

print()
print("Verifying array alignment...")

if len(probabilities) != len(labels):
    raise RuntimeError(
        "Probability and label lengths do not match."
    )

if len(probabilities) != len(patients):
    raise RuntimeError(
        "Probability and patient lengths do not match."
    )

if not np.all(np.isfinite(probabilities)):
    raise RuntimeError(
        "Probabilities contain NaN or Inf."
    )

print("[OK] Arrays are aligned.")
print("[OK] Probabilities are finite.")


# ================================================================
# 7. BASELINE
# ================================================================

def calculate_metrics(
    y_true,
    y_pred
):

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    tp = int(
        np.sum(
            (y_true == 1) &
            (y_pred == 1)
        )
    )

    fp = int(
        np.sum(
            (y_true == 0) &
            (y_pred == 1)
        )
    )

    fn = int(
        np.sum(
            (y_true == 1) &
            (y_pred == 0)
        )
    )

    tn = int(
        np.sum(
            (y_true == 0) &
            (y_pred == 0)
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
        "f1": f1
    }


baseline_pred = (
    probabilities >= threshold
).astype(np.int64)

baseline = calculate_metrics(
    labels,
    baseline_pred
)


print()
print("=" * 70)
print("BASELINE")
print("=" * 70)

print("TP:", baseline["tp"])
print("FP:", baseline["fp"])
print("FN:", baseline["fn"])
print("TN:", baseline["tn"])
print(
    "Sensitivity:",
    f"{baseline['sensitivity']:.6f}"
)
print(
    "Specificity:",
    f"{baseline['specificity']:.6f}"
)
print(
    "Precision:",
    f"{baseline['precision']:.6f}"
)
print(
    "F1:",
    f"{baseline['f1']:.6f}"
)


# ================================================================
# 8. PROBABILITY HYSTERESIS FILTER
#
# Idea:
#
# Instead of immediately accepting a positive at threshold T,
# require stronger evidence before entering the seizure state.
#
# However, once seizure state is entered, a lower threshold can
# keep it active.
#
# This is a standard hysteresis-style post-processing experiment.
# ================================================================

def hysteresis_prediction(
    probs,
    enter_threshold,
    exit_threshold
):

    result = np.zeros(
        len(probs),
        dtype=np.int64
    )

    active = False

    for i, p in enumerate(probs):

        if not active:

            if p >= enter_threshold:
                active = True

        else:

            if p < exit_threshold:
                active = False

        if active:
            result[i] = 1

    return result


# ================================================================
# 9. TEST HYSTERESIS CANDIDATES
# ================================================================

print()
print("=" * 70)
print("HYSTERESIS FILTER SEARCH")
print("=" * 70)

enter_thresholds = [
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90
]

exit_thresholds = [
    0.50,
    0.52,
    0.54,
    0.56,
    0.58,
    0.60,
    0.62,
    0.65,
    0.70
]

hysteresis_results = []

for enter in enter_thresholds:

    for exit_ in exit_thresholds:

        if exit_ >= enter:
            continue

        pred = hysteresis_prediction(
            probabilities,
            enter,
            exit_
        )

        metrics = calculate_metrics(
            labels,
            pred
        )

        metrics["enter_threshold"] = enter
        metrics["exit_threshold"] = exit_

        if (
            metrics["sensitivity"] >= 0.90
        ):

            metrics["fp_reduction"] = (
                1 -
                metrics["fp"] /
                baseline["fp"]
            )

            hysteresis_results.append(
                metrics
            )


# ================================================================
# 10. SORT CANDIDATES
# ================================================================

hysteresis_results.sort(
    key=lambda x: (
        x["f1"],
        x["sensitivity"],
        -x["fp"]
    ),
    reverse=True
)


print()
print(
    "SAFE HYSTERESIS CANDIDATES "
    "(Sensitivity >= 0.90)"
)

print()

print(
    "ENTER | EXIT | TP | FP | FN | "
    "Sensitivity | Precision | F1 | FP Reduction"
)

print("-" * 95)

for r in hysteresis_results[:20]:

    print(
        f"{r['enter_threshold']:.2f} | "
        f"{r['exit_threshold']:.2f} | "
        f"{r['tp']:3d} | "
        f"{r['fp']:3d} | "
        f"{r['fn']:3d} | "
        f"{r['sensitivity']:.4f} | "
        f"{r['precision']:.4f} | "
        f"{r['f1']:.4f} | "
        f"{r['fp_reduction'] * 100:.2f}%"
    )


# ================================================================
# 11. HIGH-CONFIDENCE FALSE POSITIVE FILTER
#
# A simple experiment:
#
# If a positive probability is only slightly above the threshold,
# it may be noise.
#
# We therefore test a second requirement based on the distance
# above threshold.
#
# ================================================================

print()
print("=" * 70)
print("HIGH-CONFIDENCE POSITIVE FILTER")
print("=" * 70)

margins = [
    0.00,
    0.02,
    0.04,
    0.06,
    0.08,
    0.10,
    0.15,
    0.20,
    0.25,
    0.30
]

margin_results = []

for margin in margins:

    effective_threshold = (
        threshold + margin
    )

    pred = (
        probabilities >= effective_threshold
    ).astype(np.int64)

    metrics = calculate_metrics(
        labels,
        pred
    )

    metrics["margin"] = margin
    metrics["effective_threshold"] = (
        effective_threshold
    )

    if metrics["sensitivity"] >= 0.90:

        metrics["fp_reduction"] = (
            1 -
            metrics["fp"] /
            baseline["fp"]
        )

        margin_results.append(
            metrics
        )


print()
print(
    "MARGIN | THRESHOLD | TP | FP | FN | "
    "Sensitivity | Precision | F1 | FP Reduction"
)

print("-" * 100)

for r in margin_results:

    print(
        f"{r['margin']:.2f} | "
        f"{r['effective_threshold']:.2f} | "
        f"{r['tp']:3d} | "
        f"{r['fp']:3d} | "
        f"{r['fn']:3d} | "
        f"{r['sensitivity']:.4f} | "
        f"{r['precision']:.4f} | "
        f"{r['f1']:.4f} | "
        f"{r['fp_reduction'] * 100:.2f}%"
    )


# ================================================================
# 12. BEST CANDIDATES
# ================================================================

best_hysteresis = (
    hysteresis_results[0]
    if hysteresis_results
    else None
)

best_margin = (
    max(
        margin_results,
        key=lambda x: x["f1"]
    )
    if margin_results
    else None
)


# ================================================================
# 13. FINAL SUMMARY
# ================================================================

print()
print("=" * 70)
print("BEST CANDIDATES")
print("=" * 70)

if best_hysteresis is not None:

    print()
    print("Best hysteresis candidate:")

    print(
        "Enter threshold:",
        best_hysteresis["enter_threshold"]
    )

    print(
        "Exit threshold:",
        best_hysteresis["exit_threshold"]
    )

    print(
        "TP:",
        best_hysteresis["tp"]
    )

    print(
        "FP:",
        best_hysteresis["fp"]
    )

    print(
        "FN:",
        best_hysteresis["fn"]
    )

    print(
        "Sensitivity:",
        f"{best_hysteresis['sensitivity']:.6f}"
    )

    print(
        "Precision:",
        f"{best_hysteresis['precision']:.6f}"
    )

    print(
        "F1:",
        f"{best_hysteresis['f1']:.6f}"
    )

    print(
        "FP reduction:",
        f"{best_hysteresis['fp_reduction'] * 100:.2f}%"
    )

else:

    print(
        "No safe hysteresis candidate found."
    )


if best_margin is not None:

    print()
    print("Best threshold-margin candidate:")

    print(
        "Margin:",
        best_margin["margin"]
    )

    print(
        "Effective threshold:",
        best_margin["effective_threshold"]
    )

    print(
        "TP:",
        best_margin["tp"]
    )

    print(
        "FP:",
        best_margin["fp"]
    )

    print(
        "FN:",
        best_margin["fn"]
    )

    print(
        "Sensitivity:",
        f"{best_margin['sensitivity']:.6f}"
    )

    print(
        "Precision:",
        f"{best_margin['precision']:.6f}"
    )

    print(
        "F1:",
        f"{best_margin['f1']:.6f}"
    )

    print(
        "FP reduction:",
        f"{best_margin['fp_reduction'] * 100:.2f}%"
    )

else:

    print(
        "No safe threshold-margin candidate found."
    )


# ================================================================
# 14. SAVE RESULTS
# ================================================================

output = {

    "baseline": baseline,

    "validation_threshold": threshold,

    "hysteresis_results": hysteresis_results,

    "margin_results": margin_results,

    "best_hysteresis": best_hysteresis,

    "best_margin": best_margin,

    "requirements": {
        "minimum_sensitivity": 0.90
    },

    "note": (
        "Exploratory post-processing analysis only. "
        "No model or dataset was modified."
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


# ================================================================
# 15. FINAL
# ================================================================

print()
print("=" * 70)
print("ANALYSIS COMPLETED")
print("=" * 70)

print()
print("Output saved to:")
print(OUTPUT_FILE)

print()
print("No model or dataset was modified.")