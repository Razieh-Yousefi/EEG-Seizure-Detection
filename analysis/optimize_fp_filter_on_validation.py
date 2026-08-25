# ================================================================
# optimize_fp_filter_on_validation.py
#
# Optimize FP-reduction filters using VALIDATION data only.
#
# IMPORTANT:
# - Test probabilities are NOT used.
# - Model is NOT modified.
# - Dataset is NOT modified.
# - This script only selects candidate filter parameters.
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

RESULTS_DIR = os.path.join(
    PROJECT_DIR,
    "results"
)

PROB_FILE = os.path.join(
    RESULTS_DIR,
    "validation_window_probabilities.npz"
)

THRESHOLD_FILE = os.path.join(
    RESULTS_DIR,
    "validation_threshold_results.json"
)

OUTPUT_FILE = os.path.join(
    RESULTS_DIR,
    "validation_fp_filter_optimization.json"
)


# ================================================================
# 2. CONFIGURATION
# ================================================================

SENSITIVITY_REQUIREMENT = 0.90

BASE_THRESHOLD = 0.56

# Hysteresis search
ENTER_THRESHOLDS = [
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
    0.95
]

EXIT_THRESHOLDS = [
    0.30,
    0.35,
    0.40,
    0.45,
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

# Threshold-margin search
MARGINS = [
    0.00,
    0.02,
    0.04,
    0.06,
    0.08,
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.35,
    0.40
]

# Minimum-run search
MIN_RUNS = [
    1,
    2,
    3,
    4
]


# ================================================================
# 3. HEADER
# ================================================================

print()
print("=" * 70)
print("VALIDATION-ONLY FP FILTER OPTIMIZATION")
print("=" * 70)

print()
print("Project directory:")
print(PROJECT_DIR)

print()
print("Validation probability file:")
print(PROB_FILE)

print()
print("Validation threshold file:")
print(THRESHOLD_FILE)


# ================================================================
# 4. CHECK FILES
# ================================================================

print()
print("=" * 70)
print("1. CHECKING INPUT FILES")
print("=" * 70)

if not os.path.exists(PROB_FILE):

    raise FileNotFoundError(
        f"Validation probability file not found:\n{PROB_FILE}"
    )

print(
    "[OK] Validation probability file found."
)


if not os.path.exists(THRESHOLD_FILE):

    raise FileNotFoundError(
        f"Validation threshold file not found:\n{THRESHOLD_FILE}"
    )

print(
    "[OK] Validation threshold file found."
)


# ================================================================
# 5. LOAD VALIDATION PROBABILITIES
# ================================================================

print()
print("=" * 70)
print("2. LOADING VALIDATION PROBABILITIES")
print("=" * 70)

data = np.load(
    PROB_FILE,
    allow_pickle=True
)

validation_indices = data[
    "validation_indices"
]

patients = data[
    "patients"
]

labels = data[
    "labels"
]

probabilities = data[
    "probabilities"
]


print()
print(
    "Validation samples:",
    len(probabilities)
)

print(
    "Probability shape:",
    probabilities.shape
)

print(
    "Labels shape:",
    labels.shape
)

print(
    "Patients shape:",
    patients.shape
)


# ================================================================
# 6. VERIFY ALIGNMENT
# ================================================================

print()
print("=" * 70)
print("3. VERIFYING ARRAY ALIGNMENT")
print("=" * 70)

if not (
    len(validation_indices)
    == len(patients)
    == len(labels)
    == len(probabilities)
):

    raise RuntimeError(
        "Validation arrays are not aligned."
    )


if not np.all(
    np.isfinite(probabilities)
):

    raise RuntimeError(
        "Validation probabilities contain NaN/Inf."
    )


if not np.all(
    (probabilities >= 0)
    &
    (probabilities <= 1)
):

    raise RuntimeError(
        "Probabilities outside [0,1]."
    )


print(
    "[OK] Arrays are aligned."
)

print(
    "[OK] Probabilities are finite."
)


# ================================================================
# 7. LOAD VALIDATION THRESHOLD
# ================================================================

print()
print("=" * 70)
print("4. LOADING VALIDATION THRESHOLD")
print("=" * 70)

with open(
    THRESHOLD_FILE,
    "r",
    encoding="utf-8"
) as f:

    threshold_data = json.load(f)


validation_threshold = float(
    threshold_data[
        "best_threshold"
    ]
)


print()
print(
    "Stored validation threshold:",
    validation_threshold
)


# Use the actual validation threshold as baseline.
BASE_THRESHOLD = validation_threshold


# ================================================================
# 8. METRICS
# ================================================================

def calculate_metrics(
    y_true,
    y_pred
):

    y_true = np.asarray(
        y_true
    )

    y_pred = np.asarray(
        y_pred
    )


    tp = int(
        np.sum(
            (y_true == 1)
            &
            (y_pred == 1)
        )
    )


    fp = int(
        np.sum(
            (y_true == 0)
            &
            (y_pred == 1)
        )
    )


    fn = int(
        np.sum(
            (y_true == 1)
            &
            (y_pred == 0)
        )
    )


    tn = int(
        np.sum(
            (y_true == 0)
            &
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
        2.0
        * precision
        * sensitivity
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
# 9. BASELINE
# ================================================================

print()
print("=" * 70)
print("5. VALIDATION BASELINE")
print("=" * 70)

baseline_pred = (
    probabilities
    >= BASE_THRESHOLD
).astype(
    np.int32
)

baseline = calculate_metrics(
    labels,
    baseline_pred
)

print()

for key, value in baseline.items():

    if isinstance(
        value,
        float
    ):

        print(
            f"{key}: {value:.6f}"
        )

    else:

        print(
            f"{key}: {value}"
        )


# ================================================================
# 10. RUN DETECTION
# ================================================================

def minimum_run_filter(
    probabilities,
    threshold,
    minimum_run
):

    raw = (
        probabilities
        >= threshold
    )

    result = np.zeros(
        len(raw),
        dtype=bool
    )


    i = 0


    while i < len(raw):

        if raw[i]:

            start = i

            while (
                i < len(raw)
                and raw[i]
            ):

                i += 1


            run_length = (
                i - start
            )


            if run_length >= minimum_run:

                result[
                    start:i
                ] = True

        else:

            i += 1


    return result


# ================================================================
# 11. HYSTERESIS FILTER
# ================================================================

def hysteresis_filter(
    probabilities,
    enter_threshold,
    exit_threshold
):

    result = np.zeros(
        len(probabilities),
        dtype=bool
    )

    active = False


    for i, p in enumerate(
        probabilities
    ):

        if not active:

            if p >= enter_threshold:

                active = True

                result[i] = True

        else:

            result[i] = True

            if p < exit_threshold:

                active = False

                result[i] = False


    return result


# ================================================================
# 12. THRESHOLD-MARGIN FILTER
# ================================================================

def threshold_filter(
    probabilities,
    threshold
):

    return (
        probabilities
        >= threshold
    )


# ================================================================
# 13. SEARCH MINIMUM-RUN FILTER
# ================================================================

print()
print("=" * 70)
print("6. MINIMUM-RUN SEARCH")
print("=" * 70)

minimum_run_results = []


for run_length in MIN_RUNS:

    prediction = minimum_run_filter(
        probabilities,
        BASE_THRESHOLD,
        run_length
    )


    metrics = calculate_metrics(
        labels,
        prediction.astype(
            np.int32
        )
    )


    fp_reduction = (

        (
            baseline["fp"]
            - metrics["fp"]
        )
        / baseline["fp"]
        * 100.0

        if baseline["fp"] > 0
        else 0.0
    )


    result = {

        "method":
            "minimum_run",

        "parameter":
            run_length,

        **metrics,

        "fp_reduction_percent":
            fp_reduction
    }


    minimum_run_results.append(
        result
    )


    print(
        f"run={run_length} "
        f"TP={metrics['tp']} "
        f"FP={metrics['fp']} "
        f"FN={metrics['fn']} "
        f"Sens={metrics['sensitivity']:.4f} "
        f"Precision={metrics['precision']:.4f} "
        f"F1={metrics['f1']:.4f} "
        f"FP reduction={fp_reduction:.2f}%"
    )


# ================================================================
# 14. SEARCH HYSTERESIS
# ================================================================

print()
print("=" * 70)
print("7. HYSTERESIS SEARCH")
print("=" * 70)


hysteresis_results = []


for enter in ENTER_THRESHOLDS:

    for exit_threshold in EXIT_THRESHOLDS:

        if exit_threshold >= enter:

            continue


        prediction = hysteresis_filter(
            probabilities,
            enter,
            exit_threshold
        )


        metrics = calculate_metrics(
            labels,
            prediction.astype(
                np.int32
            )
        )


        fp_reduction = (

            (
                baseline["fp"]
                - metrics["fp"]
            )
            / baseline["fp"]
            * 100.0

            if baseline["fp"] > 0
            else 0.0
        )


        result = {

            "method":
                "hysteresis",

            "enter_threshold":
                float(enter),

            "exit_threshold":
                float(exit_threshold),

            **metrics,

            "fp_reduction_percent":
                fp_reduction
        }


        hysteresis_results.append(
            result
        )


# ================================================================
# 15. SEARCH THRESHOLD MARGIN
# ================================================================

print()
print("=" * 70)
print("8. THRESHOLD-MARGIN SEARCH")
print("=" * 70)


threshold_margin_results = []


for margin in MARGINS:

    effective_threshold = (
        BASE_THRESHOLD
        + margin
    )


    prediction = threshold_filter(
        probabilities,
        effective_threshold
    )


    metrics = calculate_metrics(
        labels,
        prediction.astype(
            np.int32
        )
    )


    fp_reduction = (

        (
            baseline["fp"]
            - metrics["fp"]
        )
        / baseline["fp"]
        * 100.0

        if baseline["fp"] > 0
        else 0.0
    )


    result = {

        "method":
            "threshold_margin",

        "margin":
            float(margin),

        "effective_threshold":
            float(effective_threshold),

        **metrics,

        "fp_reduction_percent":
            fp_reduction
    }


    threshold_margin_results.append(
        result
    )


    print(
        f"margin={margin:.2f} "
        f"threshold={effective_threshold:.2f} "
        f"TP={metrics['tp']} "
        f"FP={metrics['fp']} "
        f"FN={metrics['fn']} "
        f"Sens={metrics['sensitivity']:.4f} "
        f"Precision={metrics['precision']:.4f} "
        f"F1={metrics['f1']:.4f} "
        f"FP reduction={fp_reduction:.2f}%"
    )


# ================================================================
# 16. SELECT SAFE CANDIDATES
# ================================================================

print()
print("=" * 70)
print("9. SELECTING SAFE VALIDATION CANDIDATES")
print("=" * 70)


all_results = (
    minimum_run_results
    + hysteresis_results
    + threshold_margin_results
)


safe_candidates = [

    r

    for r in all_results

    if r["sensitivity"]
    >= SENSITIVITY_REQUIREMENT
]


# Sort primarily by F1.
# If F1 is equal, prefer higher sensitivity.
# If still equal, prefer fewer FP.

safe_candidates_sorted = sorted(
    safe_candidates,
    key=lambda r: (
        r["f1"],
        r["sensitivity"],
        -r["fp"]
    ),
    reverse=True
)


print()

print(
    "Required sensitivity:",
    SENSITIVITY_REQUIREMENT
)

print(
    "Safe candidates:",
    len(
        safe_candidates_sorted
    )
)


print()
print(
    "TOP VALIDATION CANDIDATES"
)

print(
    "-" * 110
)

print(
    "METHOD | PARAMETER | TP | FP | FN | "
    "Sensitivity | Precision | F1 | FP Reduction"
)

print(
    "-" * 110
)


for r in safe_candidates_sorted[:20]:

    if r["method"] == "hysteresis":

        parameter = (
            f"enter={r['enter_threshold']:.2f},"
            f"exit={r['exit_threshold']:.2f}"
        )

    elif r["method"] == "threshold_margin":

        parameter = (
            f"margin={r['margin']:.2f},"
            f"thr={r['effective_threshold']:.2f}"
        )

    else:

        parameter = str(
            r["parameter"]
        )


    print(
        f"{r['method']:16s} | "
        f"{parameter:28s} | "
        f"{r['tp']:3d} | "
        f"{r['fp']:3d} | "
        f"{r['fn']:3d} | "
        f"{r['sensitivity']:.4f} | "
        f"{r['precision']:.4f} | "
        f"{r['f1']:.4f} | "
        f"{r['fp_reduction_percent']:.2f}%"
    )


# ================================================================
# 17. BEST CANDIDATE
# ================================================================

best_candidate = None


if len(
    safe_candidates_sorted
) > 0:

    best_candidate = (
        safe_candidates_sorted[0]
    )


print()
print("=" * 70)
print("10. BEST VALIDATION CANDIDATE")
print("=" * 70)


if best_candidate is None:

    print()
    print(
        "No candidate reached the required sensitivity."
    )

else:

    print()

    for key, value in (
        best_candidate.items()
    ):

        if isinstance(
            value,
            float
        ):

            print(
                f"{key}: {value:.6f}"
            )

        else:

            print(
                f"{key}: {value}"
            )


# ================================================================
# 18. SAVE RESULTS
# ================================================================

output = {

    "methodology": {

        "dataset":
            "validation",

        "test_data_used":
            False,

        "model_modified":
            False,

        "dataset_modified":
            False,

        "base_threshold":
            float(BASE_THRESHOLD),

        "sensitivity_requirement":
            float(
                SENSITIVITY_REQUIREMENT
            )
    },

    "baseline":
        baseline,

    "minimum_run_results":
        minimum_run_results,

    "hysteresis_results":
        hysteresis_results,

    "threshold_margin_results":
        threshold_margin_results,

    "safe_candidates":
        safe_candidates_sorted,

    "best_candidate":
        best_candidate
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
print("11. RESULTS SAVED")
print("=" * 70)

print()
print(
    OUTPUT_FILE
)

print()
print(
    "No model or dataset was modified."
)

print(
    "Test probabilities were NOT used."
)

print(
    "This result is VALIDATION-ONLY."
)

print()
print("=" * 70)
print("VALIDATION OPTIMIZATION COMPLETED")
print("=" * 70)