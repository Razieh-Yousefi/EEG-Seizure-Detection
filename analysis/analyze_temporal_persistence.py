import json
import numpy as np
from pathlib import Path


# ============================================================
# TEMPORAL PERSISTENCE ANALYSIS
#
# Purpose:
#   Analyze whether requiring consecutive positive windows
#   can reduce false positives.
#
# IMPORTANT:
#   - No model retraining
#   - No dataset modification
#   - No threshold modification
#   - No labels modification
#   - Diagnostic analysis only
# ============================================================


# ============================================================
# 1. PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"

PROBABILITY_FILE = (
    RESULTS_DIR / "test_window_probabilities.npz"
)

THRESHOLD_FILE = (
    RESULTS_DIR / "validation_threshold_results.json"
)

OUTPUT_FILE = (
    RESULTS_DIR / "temporal_persistence_analysis.json"
)


print()
print("=" * 70)
print("TEMPORAL PERSISTENCE ANALYSIS")
print("=" * 70)


# ============================================================
# 2. CHECK INPUT FILES
# ============================================================

print()
print("Checking input files...")

if not PROBABILITY_FILE.exists():
    raise FileNotFoundError(
        f"\nProbability file not found:\n"
        f"{PROBABILITY_FILE}\n\n"
        f"First run:\n"
        f"python archive\\save_test_probabilities.py"
    )

if not THRESHOLD_FILE.exists():
    raise FileNotFoundError(
        f"\nThreshold file not found:\n"
        f"{THRESHOLD_FILE}"
    )

print("[OK] Probability file found.")
print("[OK] Threshold file found.")


# ============================================================
# 3. LOAD TEST PROBABILITIES
# ============================================================

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

test_indices = np.asarray(
    data["test_indices"],
    dtype=np.int64
)

patients = np.asarray(
    data["patients"]
)


print("Test samples:", len(probabilities))
print("Probability shape:", probabilities.shape)
print("Labels shape:", labels.shape)
print("Patients shape:", patients.shape)


# ============================================================
# 4. LOAD VALIDATION THRESHOLD
# ============================================================

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


# ============================================================
# 5. VERIFY ALIGNMENT
# ============================================================

print()
print("Verifying array alignment...")

if not (
    len(probabilities)
    == len(labels)
    == len(test_indices)
    == len(patients)
):
    raise RuntimeError(
        "Probability / label / index / patient "
        "arrays are not aligned."
    )


if not np.all(
    np.isfinite(probabilities)
):
    raise RuntimeError(
        "Probabilities contain NaN or Inf."
    )


print("[OK] Arrays are aligned.")
print("[OK] Probabilities are finite.")


# ============================================================
# 6. RAW PREDICTIONS
# ============================================================

raw_predictions = (
    probabilities >= threshold
).astype(np.int64)


# ============================================================
# 7. METRIC FUNCTION
# ============================================================

def calculate_metrics(
    y_true,
    y_pred
):

    y_true = np.asarray(
        y_true,
        dtype=np.int64
    )

    y_pred = np.asarray(
        y_pred,
        dtype=np.int64
    )

    tn = int(
        np.sum(
            (y_true == 0)
            & (y_pred == 0)
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

    tp = int(
        np.sum(
            (y_true == 1)
            & (y_pred == 1)
        )
    )

    accuracy = (
        (tp + tn) / len(y_true)
        if len(y_true) > 0
        else 0.0
    )

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0.0
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

    f1 = (
        2.0
        * precision
        * sensitivity
        / (precision + sensitivity)
        if (precision + sensitivity) > 0
        else 0.0
    )

    return {
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "accuracy": accuracy,
        "precision": precision,
        "recall": sensitivity,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "f1": f1
    }


# ============================================================
# 8. TEMPORAL PERSISTENCE FILTER
# ============================================================

def apply_minimum_run(
    predictions,
    minimum_run
):

    predictions = np.asarray(
        predictions,
        dtype=np.int64
    )

    filtered = np.zeros_like(
        predictions
    )

    start = None

    for i in range(
        len(predictions) + 1
    ):

        is_positive = (
            i < len(predictions)
            and predictions[i] == 1
        )

        if (
            is_positive
            and start is None
        ):

            start = i

        elif (
            not is_positive
            and start is not None
        ):

            run_length = i - start

            if run_length >= minimum_run:

                filtered[start:i] = 1

            start = None

    return filtered


# ============================================================
# 9. BASELINE
# ============================================================

print()
print("=" * 70)
print("BASELINE")
print("=" * 70)

baseline_metrics = calculate_metrics(
    labels,
    raw_predictions
)

print(
    "TP:",
    baseline_metrics["tp"]
)

print(
    "FP:",
    baseline_metrics["fp"]
)

print(
    "FN:",
    baseline_metrics["fn"]
)

print(
    "TN:",
    baseline_metrics["tn"]
)

print(
    "Sensitivity:",
    f"{baseline_metrics['sensitivity']:.6f}"
)

print(
    "Specificity:",
    f"{baseline_metrics['specificity']:.6f}"
)

print(
    "Precision:",
    f"{baseline_metrics['precision']:.6f}"
)

print(
    "F1:",
    f"{baseline_metrics['f1']:.6f}"
)


# ============================================================
# 10. TEST DIFFERENT PERSISTENCE LEVELS
# ============================================================

minimum_runs = [
    2,
    3,
    4
]

methods = {}

comparison = []


for minimum_run in minimum_runs:

    print()
    print("=" * 70)
    print(
        f"MINIMUM RUN = {minimum_run} WINDOWS"
    )
    print("=" * 70)

    filtered_predictions = (
        apply_minimum_run(
            raw_predictions,
            minimum_run
        )
    )

    metrics = calculate_metrics(
        labels,
        filtered_predictions
    )

    fp_reduction = (
        (
            baseline_metrics["fp"]
            - metrics["fp"]
        )
        / baseline_metrics["fp"]
        * 100.0
        if baseline_metrics["fp"] > 0
        else 0.0
    )

    fn_increase = (
        metrics["fn"]
        - baseline_metrics["fn"]
    )

    sensitivity_change = (
        metrics["sensitivity"]
        - baseline_metrics["sensitivity"]
    )

    methods[
        f"minimum_run_{minimum_run}"
    ] = {
        "minimum_positive_run": minimum_run,
        "metrics": metrics,
        "fp_reduction_percent": fp_reduction,
        "fn_increase": fn_increase,
        "sensitivity_change": sensitivity_change
    }

    comparison.append(
        {
            "method": (
                f"minimum_run_{minimum_run}"
            ),
            "minimum_positive_run": minimum_run,
            "tp": metrics["tp"],
            "fp": metrics["fp"],
            "fn": metrics["fn"],
            "tn": metrics["tn"],
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "sensitivity": metrics["sensitivity"],
            "specificity": metrics["specificity"],
            "f1": metrics["f1"],
            "fp_reduction_percent": fp_reduction,
            "fn_increase": fn_increase,
            "sensitivity_change": sensitivity_change
        }
    )

    print(
        "TP:",
        metrics["tp"]
    )

    print(
        "FP:",
        metrics["fp"]
    )

    print(
        "FN:",
        metrics["fn"]
    )

    print(
        "TN:",
        metrics["tn"]
    )

    print(
        "Sensitivity:",
        f"{metrics['sensitivity']:.6f}"
    )

    print(
        "Specificity:",
        f"{metrics['specificity']:.6f}"
    )

    print(
        "Precision:",
        f"{metrics['precision']:.6f}"
    )

    print(
        "F1:",
        f"{metrics['f1']:.6f}"
    )

    print(
        "FP reduction:",
        f"{fp_reduction:.2f}%"
    )

    print(
        "FN increase:",
        fn_increase
    )


# ============================================================
# 11. EXTRACT POSITIVE RUNS
# ============================================================

def extract_positive_runs(
    predictions
):

    runs = []

    start = None

    for i in range(
        len(predictions) + 1
    ):

        is_positive = (
            i < len(predictions)
            and predictions[i] == 1
        )

        if (
            is_positive
            and start is None
        ):

            start = i

        elif (
            not is_positive
            and start is not None
        ):

            end = i - 1

            run_probabilities = (
                probabilities[
                    start:end + 1
                ]
            )

            run_labels = (
                labels[
                    start:end + 1
                ]
            )

            run_patients = (
                patients[
                    start:end + 1
                ]
            )

            runs.append(
                {
                    "start_position": int(
                        start
                    ),
                    "end_position": int(
                        end
                    ),
                    "length_windows": int(
                        end - start + 1
                    ),
                    "duration_seconds": int(
                        (end - start + 1)
                        * 5
                    ),
                    "probability_min": float(
                        np.min(
                            run_probabilities
                        )
                    ),
                    "probability_mean": float(
                        np.mean(
                            run_probabilities
                        )
                    ),
                    "probability_max": float(
                        np.max(
                            run_probabilities
                        )
                    ),
                    "labels": [
                        int(x)
                        for x in sorted(
                            set(
                                run_labels.tolist()
                            )
                        )
                    ],
                    "patients": [
                        str(x)
                        for x in sorted(
                            set(
                                str(y)
                                for y in run_patients
                            )
                        )
                    ]
                }
            )

            start = None

    return runs


positive_runs = extract_positive_runs(
    raw_predictions
)


# ============================================================
# 12. SEPARATE FP AND TP RUNS
# ============================================================

fp_runs = [
    run
    for run in positive_runs
    if run["labels"] == [0]
]

tp_runs = [
    run
    for run in positive_runs
    if run["labels"] == [1]
]

mixed_runs = [
    run
    for run in positive_runs
    if len(run["labels"]) > 1
]


# ============================================================
# 13. RUN-LENGTH DISTRIBUTION
# ============================================================

def run_length_distribution(
    runs
):

    distribution = {}

    for run in runs:

        length = (
            run["length_windows"]
        )

        key = str(length)

        distribution[key] = (
            distribution.get(
                key,
                0
            )
            + 1
        )

    return distribution


fp_run_distribution = (
    run_length_distribution(
        fp_runs
    )
)

tp_run_distribution = (
    run_length_distribution(
        tp_runs
    )
)


# ============================================================
# 14. RUN SUMMARY
# ============================================================

def summarize_runs(
    runs
):

    if len(runs) == 0:

        return {
            "count": 0,
            "mean_length_windows": 0.0,
            "median_length_windows": 0.0,
            "max_length_windows": 0,
            "mean_duration_seconds": 0.0,
            "median_duration_seconds": 0.0,
            "max_duration_seconds": 0.0
        }

    lengths = np.asarray(
        [
            r["length_windows"]
            for r in runs
        ],
        dtype=np.float64
    )

    durations = (
        lengths * 5.0
    )

    return {
        "count": int(len(runs)),
        "mean_length_windows": float(
            np.mean(lengths)
        ),
        "median_length_windows": float(
            np.median(lengths)
        ),
        "max_length_windows": int(
            np.max(lengths)
        ),
        "mean_duration_seconds": float(
            np.mean(durations)
        ),
        "median_duration_seconds": float(
            np.median(durations)
        ),
        "max_duration_seconds": float(
            np.max(durations)
        )
    }


fp_run_summary = summarize_runs(
    fp_runs
)

tp_run_summary = summarize_runs(
    tp_runs
)


# ============================================================
# 15. FIND BEST F1
# ============================================================

best_method = None
best_f1 = baseline_metrics["f1"]

for item in comparison:

    if item["f1"] > best_f1:

        best_f1 = item["f1"]

        best_method = item[
            "method"
        ]


# ============================================================
# 16. PATIENT-LEVEL BASELINE
# ============================================================

patient_results = {}

unique_patients = sorted(
    set(
        str(x)
        for x in patients
    )
)


for patient in unique_patients:

    mask = np.array(
        [
            str(x) == patient
            for x in patients
        ]
    )

    patient_labels = (
        labels[mask]
    )

    patient_predictions = (
        raw_predictions[mask]
    )

    patient_metrics = (
        calculate_metrics(
            patient_labels,
            patient_predictions
        )
    )

    patient_results[
        patient
    ] = patient_metrics


# ============================================================
# 17. SAVE RESULTS
# ============================================================

output = {

    "settings": {
        "validation_threshold": threshold,
        "window_duration_seconds": 5,
        "tested_minimum_runs": minimum_runs,
        "note": (
            "Diagnostic temporal persistence analysis. "
            "No model, dataset, labels, or original "
            "validation threshold was modified."
        )
    },

    "dataset": {
        "test_samples": int(
            len(labels)
        ),
        "seizure_samples": int(
            np.sum(labels == 1)
        ),
        "nonseizure_samples": int(
            np.sum(labels == 0)
        )
    },

    "baseline": baseline_metrics,

    "comparison": comparison,

    "methods": methods,

    "temporal_runs": {

        "total_positive_runs": int(
            len(positive_runs)
        ),

        "fp_runs": int(
            len(fp_runs)
        ),

        "tp_runs": int(
            len(tp_runs)
        ),

        "mixed_runs": int(
            len(mixed_runs)
        ),

        "fp_run_distribution": (
            fp_run_distribution
        ),

        "tp_run_distribution": (
            tp_run_distribution
        ),

        "fp_summary": fp_run_summary,

        "tp_summary": tp_run_summary
    },

    "fp_runs": fp_runs,

    "tp_runs": tp_runs,

    "mixed_runs": mixed_runs,

    "patient_results": patient_results,

    "best_f1_candidate": best_method,

    "best_f1": float(
        best_f1
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


# ============================================================
# 18. FINAL REPORT
# ============================================================

print()
print("=" * 70)
print("FINAL SUMMARY")
print("=" * 70)

print()
print(
    "Baseline:"
)

print(
    f"  TP={baseline_metrics['tp']} "
    f"FP={baseline_metrics['fp']} "
    f"FN={baseline_metrics['fn']} "
    f"TN={baseline_metrics['tn']}"
)

for item in comparison:

    print()

    print(
        f"{item['method']}:"
    )

    print(
        f"  TP={item['tp']} "
        f"FP={item['fp']} "
        f"FN={item['fn']} "
        f"TN={item['tn']}"
    )

    print(
        f"  Sensitivity="
        f"{item['sensitivity']:.4f}"
    )

    print(
        f"  Specificity="
        f"{item['specificity']:.4f}"
    )

    print(
        f"  Precision="
        f"{item['precision']:.4f}"
    )

    print(
        f"  F1="
        f"{item['f1']:.4f}"
    )

    print(
        f"  FP reduction="
        f"{item['fp_reduction_percent']:.2f}%"
    )

    print(
        f"  FN increase="
        f"{item['fn_increase']}"
    )


print()
print("=" * 70)
print("TEMPORAL RUN INFORMATION")
print("=" * 70)

print()
print(
    "Total positive runs:",
    len(positive_runs)
)

print(
    "Pure FP runs:",
    len(fp_runs)
)

print(
    "Pure TP runs:",
    len(tp_runs)
)

print(
    "Mixed runs:",
    len(mixed_runs)
)

print()
print(
    "FP run distribution:"
)

print(
    fp_run_distribution
)

print()
print(
    "TP run distribution:"
)

print(
    tp_run_distribution
)

print()
print(
    "Best F1 candidate:",
    best_method
)

print(
    "Best F1:",
    f"{best_f1:.6f}"
)

print()
print(
    "Output saved to:"
)

print(
    OUTPUT_FILE
)

print()
print("=" * 70)
print("ANALYSIS COMPLETED")
print("=" * 70)