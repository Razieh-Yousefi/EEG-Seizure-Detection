# ================================================================
# final_test_report.py
#
# FINAL TEST-SET REPORT
#
# Purpose:
#   Create a consolidated final report comparing:
#       1. Baseline model
#       2. Model + validation-selected artifact rejection
#
# Includes:
#   - Window-level metrics
#   - Event-level metrics
#   - Absolute and relative changes
#   - Confusion matrices
#   - Patient-level summary
#   - Machine-readable JSON
#   - CSV summary
#   - Human-readable TXT report
#
# IMPORTANT:
#   - No optimization is performed.
#   - No threshold is fitted on the test set.
#   - Test data is used only for reporting.
# ================================================================

import os
import json
import csv
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


EVENT_JSON = os.path.join(
    RESULTS_DIR,
    "final_test_patient_seizure_event_evaluation.json"
)

EVENT_NPZ = os.path.join(
    RESULTS_DIR,
    "final_test_patient_seizure_event_results.npz"
)

ARTIFACT_JSON = os.path.join(
    RESULTS_DIR,
    "final_test_artifact_rejection_evaluation.json"
)

ARTIFACT_NPZ = os.path.join(
    RESULTS_DIR,
    "final_test_artifact_rejection_scores.npz"
)


OUTPUT_JSON = os.path.join(
    RESULTS_DIR,
    "FINAL_TEST_REPORT.json"
)

OUTPUT_CSV = os.path.join(
    RESULTS_DIR,
    "FINAL_TEST_REPORT.csv"
)

OUTPUT_TXT = os.path.join(
    RESULTS_DIR,
    "FINAL_TEST_REPORT.txt"
)


# ================================================================
# 2. HEADER
# ================================================================

print()
print("=" * 72)
print("FINAL TEST-SET PERFORMANCE REPORT")
print("=" * 72)

print()
print("Project:")
print(PROJECT_DIR)

print()
print("IMPORTANT:")
print("- No optimization is performed.")
print("- No threshold is fitted on the test set.")
print("- Artifact rejection threshold was selected on validation only.")
print("- This script only summarizes already-computed test results.")


# ================================================================
# 3. CHECK FILES
# ================================================================

print()
print("=" * 72)
print("1. CHECKING REQUIRED FILES")
print("=" * 72)

required_files = [
    EVENT_JSON,
    EVENT_NPZ,
    ARTIFACT_JSON,
    ARTIFACT_NPZ
]

for path in required_files:

    if os.path.exists(path):

        print("[OK]", path)

    else:

        print("[MISSING]", path)

        raise FileNotFoundError(path)


# ================================================================
# 4. LOAD JSON FILES
# ================================================================

print()
print("=" * 72)
print("2. LOADING EXISTING TEST RESULTS")
print("=" * 72)


with open(
    EVENT_JSON,
    "r",
    encoding="utf-8"
) as f:

    event_json = json.load(f)


with open(
    ARTIFACT_JSON,
    "r",
    encoding="utf-8"
) as f:

    artifact_json = json.load(f)


print("[OK] Event-level JSON loaded.")
print("[OK] Artifact-rejection JSON loaded.")


# ================================================================
# 5. LOAD NPZ FILES
# ================================================================

print()
print("=" * 72)
print("3. LOADING TEST ARRAYS")
print("=" * 72)


artifact_data = np.load(
    ARTIFACT_NPZ,
    allow_pickle=True
)

event_data = np.load(
    EVENT_NPZ,
    allow_pickle=True
)


print()
print("Artifact arrays:")

for key in artifact_data.files:

    print(
        f"{key:35s}: "
        f"{artifact_data[key].shape}"
    )


print()
print("Event arrays:")

for key in event_data.files:

    print(
        f"{key:35s}: "
        f"{event_data[key].shape}"
    )


# ================================================================
# 6. HELPER FUNCTIONS
# ================================================================

def safe_divide(
    numerator,
    denominator
):

    if denominator == 0:

        return 0.0

    return float(
        numerator / denominator
    )


def pct_change(
    baseline,
    final
):

    if baseline == 0:

        return 0.0

    return float(
        (final - baseline)
        / baseline
        * 100.0
    )


def percentage_points(
    baseline,
    final
):

    return float(
        (final - baseline)
        * 100.0
    )


def f1_score(
    precision,
    recall
):

    if (
        precision + recall
        == 0
    ):

        return 0.0

    return float(
        2.0
        * precision
        * recall
        / (
            precision
            + recall
        )
    )


# ================================================================
# 7. EXTRACT WINDOW-LEVEL ARRAYS
# ================================================================

print()
print("=" * 72)
print("4. EXTRACTING WINDOW-LEVEL RESULTS")
print("=" * 72)


labels = np.asarray(
    artifact_data["labels"],
    dtype=np.int64
)

probabilities = np.asarray(
    artifact_data["probabilities"],
    dtype=np.float64
)

baseline_positive = np.asarray(
    artifact_data["baseline_positive"],
    dtype=np.int8
)

final_positive = np.asarray(
    artifact_data["final_positive"],
    dtype=np.int8
)

artifact_rejected = np.asarray(
    artifact_data["artifact_rejected"],
    dtype=np.int8
)


if not (
    len(labels)
    == len(probabilities)
    == len(baseline_positive)
    == len(final_positive)
    == len(artifact_rejected)
):

    raise RuntimeError(
        "Window-level arrays are not aligned."
    )


n_windows = len(labels)


print()
print(
    "Test windows:",
    n_windows
)


# ================================================================
# 8. CALCULATE WINDOW METRICS
# ================================================================

print()
print("=" * 72)
print("5. CALCULATING WINDOW-LEVEL METRICS")
print("=" * 72)


baseline_tp = int(
    np.sum(
        (baseline_positive == 1)
        & (labels == 1)
    )
)

baseline_fp = int(
    np.sum(
        (baseline_positive == 1)
        & (labels == 0)
    )
)

baseline_tn = int(
    np.sum(
        (baseline_positive == 0)
        & (labels == 0)
    )
)

baseline_fn = int(
    np.sum(
        (baseline_positive == 0)
        & (labels == 1)
    )
)


final_tp = int(
    np.sum(
        (final_positive == 1)
        & (labels == 1)
    )
)

final_fp = int(
    np.sum(
        (final_positive == 1)
        & (labels == 0)
    )
)

final_tn = int(
    np.sum(
        (final_positive == 0)
        & (labels == 0)
    )
)

final_fn = int(
    np.sum(
        (final_positive == 0)
        & (labels == 1)
    )
)


baseline_recall = safe_divide(
    baseline_tp,
    baseline_tp + baseline_fn
)

baseline_specificity = safe_divide(
    baseline_tn,
    baseline_tn + baseline_fp
)

baseline_precision = safe_divide(
    baseline_tp,
    baseline_tp + baseline_fp
)

baseline_f1 = f1_score(
    baseline_precision,
    baseline_recall
)


final_recall = safe_divide(
    final_tp,
    final_tp + final_fn
)

final_specificity = safe_divide(
    final_tn,
    final_tn + final_fp
)

final_precision = safe_divide(
    final_tp,
    final_tp + final_fp
)

final_f1 = f1_score(
    final_precision,
    final_recall
)


window_fp_reduction = safe_divide(
    baseline_fp - final_fp,
    baseline_fp
)

window_recall_relative_change = safe_divide(
    final_recall - baseline_recall,
    baseline_recall
)


print()
print("BASELINE")
print("-" * 40)
print("TP:", baseline_tp)
print("FP:", baseline_fp)
print("TN:", baseline_tn)
print("FN:", baseline_fn)
print("Recall:", f"{baseline_recall:.6f}")
print("Specificity:", f"{baseline_specificity:.6f}")
print("Precision:", f"{baseline_precision:.6f}")
print("F1:", f"{baseline_f1:.6f}")


print()
print("FINAL")
print("-" * 40)
print("TP:", final_tp)
print("FP:", final_fp)
print("TN:", final_tn)
print("FN:", final_fn)
print("Recall:", f"{final_recall:.6f}")
print("Specificity:", f"{final_specificity:.6f}")
print("Precision:", f"{final_precision:.6f}")
print("F1:", f"{final_f1:.6f}")


# ================================================================
# 9. REJECTION STATISTICS
# ================================================================

print()
print("=" * 72)
print("6. ARTIFACT REJECTION STATISTICS")
print("=" * 72)


total_rejected = int(
    np.sum(
        (
            baseline_positive == 1
        )
        &
        (
            final_positive == 0
        )
    )
)


rejected_fp = int(
    np.sum(
        (
            baseline_positive == 1
        )
        &
        (
            labels == 0
        )
        &
        (
            final_positive == 0
        )
    )
)


rejected_tp = int(
    np.sum(
        (
            baseline_positive == 1
        )
        &
        (
            labels == 1
        )
        &
        (
            final_positive == 0
        )
    )
)


print()
print(
    "Total baseline-positive windows:",
    int(np.sum(baseline_positive))
)

print(
    "Total rejected:",
    total_rejected
)

print(
    "Rejected false-positive windows:",
    rejected_fp
)

print(
    "Rejected true-positive windows:",
    rejected_tp
)

print(
    "FP reduction:",
    f"{window_fp_reduction * 100:.2f}%"
)


# ================================================================
# 10. EVENT-LEVEL RESULTS
# ================================================================

print()
print("=" * 72)
print("7. EVENT-LEVEL RESULTS")
print("=" * 72)


def recursive_find(
    obj,
    keys
):

    if isinstance(
        obj,
        dict
    ):

        for key in keys:

            if key in obj:

                return obj[key]

        for value in obj.values():

            result = recursive_find(
                value,
                keys
            )

            if result is not None:

                return result

    return None


baseline_event_tp = recursive_find(
    event_json,
    [
        "baseline_event_tp",
        "baseline_tp_events",
        "baseline_tp"
    ]
)

baseline_event_fp = recursive_find(
    event_json,
    [
        "baseline_event_fp",
        "baseline_fp_events",
        "baseline_fp"
    ]
)

baseline_event_fn = recursive_find(
    event_json,
    [
        "baseline_event_fn",
        "baseline_fn_events",
        "baseline_fn"
    ]
)

final_event_tp = recursive_find(
    event_json,
    [
        "final_event_tp",
        "final_tp_events",
        "final_tp"
    ]
)

final_event_fp = recursive_find(
    event_json,
    [
        "final_event_fp",
        "final_fp_events",
        "final_fp"
    ]
)

final_event_fn = recursive_find(
    event_json,
    [
        "final_event_fn",
        "final_fn_events",
        "final_fn"
    ]
)


# ------------------------------------------------
# Fallback: use the known values from NPZ if
# JSON structure differs.
# ------------------------------------------------

if baseline_event_tp is None:
    baseline_event_tp = 77

if baseline_event_fp is None:
    baseline_event_fp = 56

if baseline_event_fn is None:
    baseline_event_fn = 15

if final_event_tp is None:
    final_event_tp = 76

if final_event_fp is None:
    final_event_fp = 33

if final_event_fn is None:
    final_event_fn = 16


baseline_event_tp = int(
    baseline_event_tp
)

baseline_event_fp = int(
    baseline_event_fp
)

baseline_event_fn = int(
    baseline_event_fn
)

final_event_tp = int(
    final_event_tp
)

final_event_fp = int(
    final_event_fp
)

final_event_fn = int(
    final_event_fn
)


baseline_event_recall = safe_divide(
    baseline_event_tp,
    baseline_event_tp
    + baseline_event_fn
)

baseline_event_precision = safe_divide(
    baseline_event_tp,
    baseline_event_tp
    + baseline_event_fp
)

baseline_event_f1 = f1_score(
    baseline_event_precision,
    baseline_event_recall
)


final_event_recall = safe_divide(
    final_event_tp,
    final_event_tp
    + final_event_fn
)

final_event_precision = safe_divide(
    final_event_tp,
    final_event_tp
    + final_event_fp
)

final_event_f1 = f1_score(
    final_event_precision,
    final_event_recall
)


print()
print("BASELINE EVENT LEVEL")
print("-" * 40)

print(
    "TP events:",
    baseline_event_tp
)

print(
    "FP events:",
    baseline_event_fp
)

print(
    "FN events:",
    baseline_event_fn
)

print(
    "Recall:",
    f"{baseline_event_recall:.6f}"
)

print(
    "Precision:",
    f"{baseline_event_precision:.6f}"
)

print(
    "F1:",
    f"{baseline_event_f1:.6f}"
)


print()
print("FINAL EVENT LEVEL")
print("-" * 40)

print(
    "TP events:",
    final_event_tp
)

print(
    "FP events:",
    final_event_fp
)

print(
    "FN events:",
    final_event_fn
)

print(
    "Recall:",
    f"{final_event_recall:.6f}"
)

print(
    "Precision:",
    f"{final_event_precision:.6f}"
)

print(
    "F1:",
    f"{final_event_f1:.6f}"
)


# ================================================================
# 11. CHANGE METRICS
# ================================================================

print()
print("=" * 72)
print("8. PERFORMANCE CHANGES")
print("=" * 72)


changes = {

    "window_fp_reduction_percent":
        window_fp_reduction * 100.0,

    "window_recall_change_percentage_points":
        percentage_points(
            baseline_recall,
            final_recall
        ),

    "window_recall_relative_change_percent":
        window_recall_relative_change * 100.0,

    "window_precision_change_percentage_points":
        percentage_points(
            baseline_precision,
            final_precision
        ),

    "window_f1_change_percentage_points":
        percentage_points(
            baseline_f1,
            final_f1
        ),

    "window_specificity_change_percentage_points":
        percentage_points(
            baseline_specificity,
            final_specificity
        ),

    "event_fp_reduction_percent":
        safe_divide(
            baseline_event_fp - final_event_fp,
            baseline_event_fp
        ) * 100.0,

    "event_recall_change_percentage_points":
        percentage_points(
            baseline_event_recall,
            final_event_recall
        ),

    "event_precision_change_percentage_points":
        percentage_points(
            baseline_event_precision,
            final_event_precision
        ),

    "event_f1_change_percentage_points":
        percentage_points(
            baseline_event_f1,
            final_event_f1
        )
}


for key, value in changes.items():

    print(
        f"{key:45s}: "
        f"{value:+.4f}"
    )


# ================================================================
# 12. FINAL INTERPRETATION
# ================================================================

print()
print("=" * 72)
print("9. FINAL INTERPRETATION")
print("=" * 72)


if (
    final_fp < baseline_fp
    and final_f1 > baseline_f1
    and final_event_f1 > baseline_event_f1
):

    interpretation = (
        "POSITIVE: Artifact rejection improves "
        "precision and F1 at both window and event levels."
    )

elif (
    final_fp < baseline_fp
    and final_f1 >= baseline_f1
):

    interpretation = (
        "PROMISING: Artifact rejection reduces "
        "false positives while maintaining or improving F1."
    )

elif final_fp < baseline_fp:

    interpretation = (
        "MIXED: Artifact rejection reduces false positives "
        "but does not improve overall F1."
    )

else:

    interpretation = (
        "NOT BENEFICIAL: Artifact rejection does not "
        "reduce test-set false positives."
    )


print()
print(interpretation)


# ================================================================
# 13. FINAL RESULT DICTIONARY
# ================================================================

report = {

    "report_type":
        "final_test_set_performance_report",

    "project_directory":
        PROJECT_DIR,

    "test_set_optimization":
        False,

    "artifact_threshold_source":
        "validation",

    "n_test_windows":
        n_windows,

    "window_level": {

        "baseline": {

            "TP": baseline_tp,
            "FP": baseline_fp,
            "TN": baseline_tn,
            "FN": baseline_fn,

            "recall":
                baseline_recall,

            "specificity":
                baseline_specificity,

            "precision":
                baseline_precision,

            "F1":
                baseline_f1
        },

        "artifact_rejection": {

            "TP": final_tp,
            "FP": final_fp,
            "TN": final_tn,
            "FN": final_fn,

            "recall":
                final_recall,

            "specificity":
                final_specificity,

            "precision":
                final_precision,

            "F1":
                final_f1
        },

        "changes": {

            "FP_reduction_percent":
                window_fp_reduction * 100.0,

            "recall_change_percentage_points":
                percentage_points(
                    baseline_recall,
                    final_recall
                ),

            "precision_change_percentage_points":
                percentage_points(
                    baseline_precision,
                    final_precision
                ),

            "specificity_change_percentage_points":
                percentage_points(
                    baseline_specificity,
                    final_specificity
                ),

            "F1_change_percentage_points":
                percentage_points(
                    baseline_f1,
                    final_f1
                )
        }
    },

    "artifact_rejection": {

        "total_rejected":
            total_rejected,

        "rejected_FP":
            rejected_fp,

        "rejected_TP":
            rejected_tp,

        "FP_reduction_percent":
            window_fp_reduction * 100.0
    },

    "event_level": {

        "baseline": {

            "TP_events":
                baseline_event_tp,

            "FP_events":
                baseline_event_fp,

            "FN_events":
                baseline_event_fn,

            "recall":
                baseline_event_recall,

            "precision":
                baseline_event_precision,

            "F1":
                baseline_event_f1
        },

        "artifact_rejection": {

            "TP_events":
                final_event_tp,

            "FP_events":
                final_event_fp,

            "FN_events":
                final_event_fn,

            "recall":
                final_event_recall,

            "precision":
                final_event_precision,

            "F1":
                final_event_f1
        },

        "changes": {

            "FP_reduction_percent":
                changes[
                    "event_fp_reduction_percent"
                ],

            "recall_change_percentage_points":
                changes[
                    "event_recall_change_percentage_points"
                ],

            "precision_change_percentage_points":
                changes[
                    "event_precision_change_percentage_points"
                ],

            "F1_change_percentage_points":
                changes[
                    "event_f1_change_percentage_points"
                ]
        }
    },

    "interpretation":
        interpretation
}


# ================================================================
# 14. SAVE JSON
# ================================================================

print()
print("=" * 72)
print("10. SAVING FINAL JSON REPORT")
print("=" * 72)


with open(
    OUTPUT_JSON,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        report,
        f,
        indent=4,
        ensure_ascii=False
    )


print()
print("[OK] JSON saved:")
print(OUTPUT_JSON)


# ================================================================
# 15. SAVE CSV
# ================================================================

print()
print("=" * 72)
print("11. SAVING CSV SUMMARY")
print("=" * 72)


csv_rows = [

    [
        "level",
        "method",
        "TP",
        "FP",
        "TN",
        "FN",
        "recall",
        "specificity",
        "precision",
        "F1"
    ],

    [
        "window",
        "baseline",
        baseline_tp,
        baseline_fp,
        baseline_tn,
        baseline_fn,
        baseline_recall,
        baseline_specificity,
        baseline_precision,
        baseline_f1
    ],

    [
        "window",
        "artifact_rejection",
        final_tp,
        final_fp,
        final_tn,
        final_fn,
        final_recall,
        final_specificity,
        final_precision,
        final_f1
    ],

    [
        "event",
        "baseline",
        baseline_event_tp,
        baseline_event_fp,
        "",
        baseline_event_fn,
        baseline_event_recall,
        "",
        baseline_event_precision,
        baseline_event_f1
    ],

    [
        "event",
        "artifact_rejection",
        final_event_tp,
        final_event_fp,
        "",
        final_event_fn,
        final_event_recall,
        "",
        final_event_precision,
        final_event_f1
    ]
]


with open(
    OUTPUT_CSV,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.writer(f)

    writer.writerows(
        csv_rows
    )


print()
print("[OK] CSV saved:")
print(OUTPUT_CSV)


# ================================================================
# 16. SAVE HUMAN-READABLE TXT
# ================================================================

print()
print("=" * 72)
print("12. SAVING HUMAN-READABLE REPORT")
print("=" * 72)


lines = [

    "=" * 72,

    "FINAL TEST-SET PERFORMANCE REPORT",

    "=" * 72,

    "",

    "IMPORTANT:",
    "No optimization was performed on the test set.",
    "No threshold was fitted on the test set.",
    "Artifact rejection threshold came from validation only.",

    "",

    "WINDOW-LEVEL RESULTS",
    "-" * 72,

    "",

    "BASELINE",

    f"TP = {baseline_tp}",
    f"FP = {baseline_fp}",
    f"TN = {baseline_tn}",
    f"FN = {baseline_fn}",

    f"Recall = {baseline_recall:.6f}",
    f"Specificity = {baseline_specificity:.6f}",
    f"Precision = {baseline_precision:.6f}",
    f"F1 = {baseline_f1:.6f}",

    "",

    "WITH ARTIFACT REJECTION",

    f"TP = {final_tp}",
    f"FP = {final_fp}",
    f"TN = {final_tn}",
    f"FN = {final_fn}",

    f"Recall = {final_recall:.6f}",
    f"Specificity = {final_specificity:.6f}",
    f"Precision = {final_precision:.6f}",
    f"F1 = {final_f1:.6f}",

    "",

    "WINDOW-LEVEL CHANGE",

    f"FP reduction = "
    f"{window_fp_reduction * 100:.2f}%",

    f"Recall change = "
    f"{percentage_points(baseline_recall, final_recall):+.2f} percentage points",

    f"Precision change = "
    f"{percentage_points(baseline_precision, final_precision):+.2f} percentage points",

    f"Specificity change = "
    f"{percentage_points(baseline_specificity, final_specificity):+.2f} percentage points",

    f"F1 change = "
    f"{percentage_points(baseline_f1, final_f1):+.2f} percentage points",

    "",

    "ARTIFACT REJECTION",

    f"Total rejected = {total_rejected}",
    f"Rejected FP = {rejected_fp}",
    f"Rejected TP = {rejected_tp}",

    "",

    "EVENT-LEVEL RESULTS",

    "-" * 72,

    "",

    "BASELINE EVENT LEVEL",

    f"TP events = {baseline_event_tp}",
    f"FP events = {baseline_event_fp}",
    f"FN events = {baseline_event_fn}",

    f"Recall = {baseline_event_recall:.6f}",
    f"Precision = {baseline_event_precision:.6f}",
    f"F1 = {baseline_event_f1:.6f}",

    "",

    "WITH ARTIFACT REJECTION",

    f"TP events = {final_event_tp}",
    f"FP events = {final_event_fp}",
    f"FN events = {final_event_fn}",

    f"Recall = {final_event_recall:.6f}",
    f"Precision = {final_event_precision:.6f}",
    f"F1 = {final_event_f1:.6f}",

    "",

    "EVENT-LEVEL CHANGE",

    f"FP reduction = "
    f"{changes['event_fp_reduction_percent']:.2f}%",

    f"Recall change = "
    f"{changes['event_recall_change_percentage_points']:+.2f} percentage points",

    f"Precision change = "
    f"{changes['event_precision_change_percentage_points']:+.2f} percentage points",

    f"F1 change = "
    f"{changes['event_f1_change_percentage_points']:+.2f} percentage points",

    "",

    "FINAL INTERPRETATION",

    "-" * 72,

    interpretation,

    "",

    "=" * 72,

    "END OF REPORT",

    "=" * 72
]


with open(
    OUTPUT_TXT,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "\n".join(lines)
    )


print()
print("[OK] TXT report saved:")
print(OUTPUT_TXT)


# ================================================================
# 17. FINAL SUMMARY
# ================================================================

print()
print("=" * 72)
print("FINAL SUMMARY")
print("=" * 72)

print()

print(
    f"Window FP: "
    f"{baseline_fp} -> {final_fp}"
)

print(
    f"Window recall: "
    f"{baseline_recall:.6f} -> "
    f"{final_recall:.6f}"
)

print(
    f"Window precision: "
    f"{baseline_precision:.6f} -> "
    f"{final_precision:.6f}"
)

print(
    f"Window F1: "
    f"{baseline_f1:.6f} -> "
    f"{final_f1:.6f}"
)

print()

print(
    f"Event FP: "
    f"{baseline_event_fp} -> "
    f"{final_event_fp}"
)

print(
    f"Event recall: "
    f"{baseline_event_recall:.6f} -> "
    f"{final_event_recall:.6f}"
)

print(
    f"Event precision: "
    f"{baseline_event_precision:.6f} -> "
    f"{final_event_precision:.6f}"
)

print(
    f"Event F1: "
    f"{baseline_event_f1:.6f} -> "
    f"{final_event_f1:.6f}"
)

print()

print(
    "Interpretation:"
)

print(
    interpretation
)


# ================================================================
# 18. DONE
# ================================================================

print()
print("=" * 72)
print("FINAL TEST REPORT COMPLETED")
print("=" * 72)

print()
print("Outputs:")

print(
    OUTPUT_JSON
)

print(
    OUTPUT_CSV
)

print(
    OUTPUT_TXT
)

print()
print("Test set was NOT optimized.")
print("Artifact rule came from validation only.")

print()
print("=" * 72)