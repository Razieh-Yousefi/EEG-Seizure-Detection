# ================================================================
# validate_fp_filter_on_validation.py
#
# PURPOSE:
# Find the best hysteresis threshold pair using VALIDATION data.
#
# IMPORTANT:
# - Does NOT modify the model.
# - Does NOT modify the dataset.
# - Does NOT use the TEST set for parameter selection.
# - Does NOT change the final threshold.
#
# The selected parameters will later be evaluated ONCE on TEST.
# ================================================================

import os
import json
import numpy as np


# ================================================================
# 1. PROJECT PATHS
# ================================================================

PROJECT_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATA_DIR = os.path.join(
    PROJECT_DIR,
    "data"
)

RESULTS_DIR = os.path.join(
    PROJECT_DIR,
    "results"
)

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)


# ================================================================
# 2. INPUT FILE
# ================================================================

# The probability file generated previously contains TEST
# probabilities only.
#
# Therefore we need a validation probability file separately.

VALIDATION_PROB_FILE = os.path.join(
    RESULTS_DIR,
    "validation_window_probabilities.npz"
)


# ================================================================
# 3. OUTPUT FILE
# ================================================================

OUTPUT_FILE = os.path.join(
    RESULTS_DIR,
    "validation_fp_filter_analysis.json"
)


# ================================================================
# 4. VALIDATION BASE THRESHOLD
# ================================================================

THRESHOLD_FILE = os.path.join(
    RESULTS_DIR,
    "validation_threshold_results.json"
)


# ================================================================
# HEADER
# ================================================================

print()
print("=" * 70)
print("VALIDATION FALSE POSITIVE FILTER ANALYSIS")
print("=" * 70)

print()
print("Project directory:")
print(PROJECT_DIR)

print()
print("Validation probability file:")
print(VALIDATION_PROB_FILE)

print()
print("Output file:")
print(OUTPUT_FILE)


# ================================================================
# 5. CHECK INPUT FILES
# ================================================================

print()
print("=" * 70)
print("1. CHECKING INPUT FILES")
print("=" * 70)

if not os.path.exists(VALIDATION_PROB_FILE):

    raise FileNotFoundError(
        "\nValidation probability file was not found:\n"
        f"{VALIDATION_PROB_FILE}\n\n"
        "We must generate validation probabilities first."
    )

if not os.path.exists(THRESHOLD_FILE):

    raise FileNotFoundError(
        "\nValidation threshold file was not found:\n"
        f"{THRESHOLD_FILE}"
    )

print("[OK] Validation probability file found.")
print("[OK] Validation threshold file found.")


# ================================================================
# 6. LOAD VALIDATION PROBABILITIES
# ================================================================

print()
print("=" * 70)
print("2. LOADING VALIDATION PROBABILITIES")
print("=" * 70)

data = np.load(
    VALIDATION_PROB_FILE,
    allow_pickle=True
)

required_keys = [
    "labels",
    "probabilities"
]

for key in required_keys:

    if key not in data.files:

        raise KeyError(
            f"Required key missing from validation file: {key}"
        )


labels = np.asarray(
    data["labels"],
    dtype=np.int64
)

probabilities = np.asarray(
    data["probabilities"],
    dtype=np.float32
)

print()
print("Validation samples:", len(probabilities))
print("Probability shape:", probabilities.shape)
print("Labels shape:", labels.shape)


# ================================================================
# 7. LOAD VALIDATION THRESHOLD
# ================================================================

print()
print("=" * 70)
print("3. LOADING VALIDATION THRESHOLD")
print("=" * 70)

with open(
    THRESHOLD_FILE,
    "r",
    encoding="utf-8"
) as f:

    threshold_data = json.load(f)


base_threshold = float(
    threshold_data["best_threshold"]
)

print()
print("Validation threshold:", base_threshold)


# ================================================================
# 8. VERIFY ALIGNMENT
# ================================================================

print()
print("=" * 70)
print("4. VERIFYING DATA")
print("=" * 70)

if len(labels) != len(probabilities):

    raise RuntimeError(
        "Labels and probabilities have different lengths."
    )

if not np.all(np.isfinite(probabilities)):

    raise RuntimeError(
        "Validation probabilities contain NaN or Inf."
    )

unique_labels = np.unique(labels)

if not np.all(
    np.isin(
        unique_labels,
        [0, 1]
    )
):

    raise RuntimeError(
        f"Unexpected labels: {unique_labels}"
    )

print("[OK] Arrays are aligned.")
print("[OK] Probabilities are finite.")
print("[OK] Labels are binary.")


# ================================================================
# 9. METRIC FUNCTION
# ================================================================

def calculate_metrics(
    labels,
    predictions
):

    labels = np.asarray(
        labels
    )

    predictions = np.asarray(
        predictions
    )

    tp = int(
        np.sum(
            (labels == 1) &
            (predictions == 1)
        )
    )

    fp = int(
        np.sum(
            (labels == 0) &
            (predictions == 1)
        )
    )

    fn = int(
        np.sum(
            (labels == 1) &
            (predictions == 0)
        )
    )

    tn = int(
        np.sum(
            (labels == 0) &
            (predictions == 0)
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
# 10. HYSTERESIS FILTER
# ================================================================

def apply_hysteresis(
    probabilities,
    enter_threshold,
    exit_threshold
):
    """
    State-based hysteresis detector.

    OFF:
        probability >= enter_threshold
        -> ON

    ON:
        probability < exit_threshold
        -> OFF

    While ON, prediction remains positive.

    This reduces isolated borderline positives.
    """

    predictions = np.zeros(
        len(probabilities),
        dtype=np.int64
    )

    state = 0

    for i, probability in enumerate(
        probabilities
    ):

        if state == 0:

            if probability >= enter_threshold:

                state = 1

        else:

            if probability < exit_threshold:

                state = 0

        predictions[i] = state

    return predictions


# ================================================================
# 11. BASELINE
# ================================================================

print()
print("=" * 70)
print("5. VALIDATION BASELINE")
print("=" * 70)

baseline_predictions = (
    probabilities >= base_threshold
).astype(
    np.int64
)

baseline = calculate_metrics(
    labels,
    baseline_predictions
)

print()
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
# 12. SEARCH SPACE
# ================================================================

enter_thresholds = [
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
    0.95
]

exit_thresholds = [
    0.40,
    0.42,
    0.44,
    0.46,
    0.48,
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


# ================================================================
# 13. HYSTERESIS SEARCH
# ================================================================

print()
print("=" * 70)
print("6. HYSTERESIS SEARCH ON VALIDATION")
print("=" * 70)

print()
print(
    "Safety requirement:"
)

print(
    "Sensitivity >= 0.90"
)

results = []

for enter_threshold in enter_thresholds:

    for exit_threshold in exit_thresholds:

        if exit_threshold >= enter_threshold:

            continue

        predictions = apply_hysteresis(
            probabilities,
            enter_threshold,
            exit_threshold
        )

        metrics = calculate_metrics(
            labels,
            predictions
        )

        fp_reduction = (
            100.0
            * (
                baseline["fp"]
                - metrics["fp"]
            )
            / baseline["fp"]
            if baseline["fp"] > 0
            else 0.0
        )

        result = {
            "enter_threshold": float(
                enter_threshold
            ),

            "exit_threshold": float(
                exit_threshold
            ),

            **metrics,

            "fp_reduction_percent": fp_reduction,

            "sensitivity_drop": (
                baseline["sensitivity"]
                - metrics["sensitivity"]
            )
        }

        results.append(
            result
        )


# ================================================================
# 14. SAFE CANDIDATES
# ================================================================

safe_candidates = [
    r
    for r in results
    if r["sensitivity"] >= 0.90
]


# Sort primarily by F1.
# If F1 is equal, prefer lower FP.

safe_candidates = sorted(
    safe_candidates,
    key=lambda r: (
        r["f1"],
        -r["fp"]
    ),
    reverse=True
)


print()
print("=" * 70)
print("SAFE VALIDATION CANDIDATES")
print("=" * 70)

if len(safe_candidates) == 0:

    print()
    print(
        "NO SAFE CANDIDATE FOUND."
    )

else:

    print()

    print(
        "ENTER | EXIT | TP | FP | FN | "
        "Sensitivity | Precision | F1 | FP Reduction"
    )

    print(
        "-" * 105
    )

    for result in safe_candidates[:20]:

        print(
            f"{result['enter_threshold']:.2f} | "
            f"{result['exit_threshold']:.2f} | "
            f"{result['tp']:3d} | "
            f"{result['fp']:3d} | "
            f"{result['fn']:3d} | "
            f"{result['sensitivity']:.4f} | "
            f"{result['precision']:.4f} | "
            f"{result['f1']:.4f} | "
            f"{result['fp_reduction_percent']:.2f}%"
        )


# ================================================================
# 15. SELECT BEST VALIDATION CANDIDATE
# ================================================================

best_candidate = None

if safe_candidates:

    best_candidate = safe_candidates[0]


# ================================================================
# 16. REPORT BEST CANDIDATE
# ================================================================

print()
print("=" * 70)
print("7. BEST VALIDATION CANDIDATE")
print("=" * 70)

if best_candidate is None:

    print()
    print(
        "No hysteresis candidate satisfies "
        "Sensitivity >= 0.90."
    )

else:

    print()
    print(
        "Enter threshold:",
        best_candidate["enter_threshold"]
    )

    print(
        "Exit threshold:",
        best_candidate["exit_threshold"]
    )

    print(
        "TP:",
        best_candidate["tp"]
    )

    print(
        "FP:",
        best_candidate["fp"]
    )

    print(
        "FN:",
        best_candidate["fn"]
    )

    print(
        "TN:",
        best_candidate["tn"]
    )

    print(
        "Sensitivity:",
        f"{best_candidate['sensitivity']:.6f}"
    )

    print(
        "Specificity:",
        f"{best_candidate['specificity']:.6f}"
    )

    print(
        "Precision:",
        f"{best_candidate['precision']:.6f}"
    )

    print(
        "F1:",
        f"{best_candidate['f1']:.6f}"
    )

    print(
        "FP reduction:",
        f"{best_candidate['fp_reduction_percent']:.2f}%"
    )


# ================================================================
# 17. SAVE RESULTS
# ================================================================

output = {

    "method": "validation_hysteresis_search",

    "selection_dataset": "validation",

    "test_set_used_for_selection": False,

    "base_threshold": base_threshold,

    "safety_requirement": {
        "minimum_sensitivity": 0.90
    },

    "baseline": baseline,

    "search_space": {
        "enter_thresholds": enter_thresholds,
        "exit_thresholds": exit_thresholds
    },

    "safe_candidate_count": len(
        safe_candidates
    ),

    "safe_candidates": safe_candidates,

    "best_candidate": best_candidate,

    "methodology": (
        "Hysteresis parameters were selected "
        "using validation data only. "
        "The test set was not used for parameter "
        "selection."
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
# 18. FINAL
# ================================================================

print()
print("=" * 70)
print("VALIDATION ANALYSIS COMPLETED")
print("=" * 70)

print()
print("Output saved to:")
print(OUTPUT_FILE)

print()
print("IMPORTANT:")
print(
    "The selected hysteresis parameters are "
    "NOT yet evaluated on the TEST set."
)

print()
print(
    "No model was modified."
)

print(
    "No dataset was modified."
)

print(
    "No test-set parameter selection was performed."
)

print()
print("=" * 70)