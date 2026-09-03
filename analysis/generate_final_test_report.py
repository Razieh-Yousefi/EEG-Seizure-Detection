# ================================================================
# generate_final_test_report.py
# FINAL STRICT TEST REPORT
# ================================================================

import os
import json
import csv
import numpy as np


# ================================================================
# 1. PATHS
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

WINDOW_JSON_FILE = os.path.join(
    RESULTS_DIR,
    "final_test_artifact_rejection_evaluation_strict.json"
)

WINDOW_NPZ_FILE = os.path.join(
    RESULTS_DIR,
    "final_test_artifact_rejection_scores_strict.npz"
)

EVENT_JSON_FILE = os.path.join(
    RESULTS_DIR,
    "final_test_patient_seizure_event_evaluation_strict.json"
)

EVENT_NPZ_FILE = os.path.join(
    RESULTS_DIR,
    "final_test_patient_seizure_event_results_strict.npz"
)

VALIDATION_JSON_FILE = os.path.join(
    RESULTS_DIR,
    "validation_artifact_rejection_optimization_v2.json"
)

VALIDATION_NPZ_FILE = os.path.join(
    RESULTS_DIR,
    "validation_artifact_scores_v2.npz"
)

TEST_INDICES_FILE = os.path.join(
    DATA_DIR,
    "test_indices.npy"
)

OUTPUT_JSON = os.path.join(
    RESULTS_DIR,
    "final_strict_test_report.json"
)

OUTPUT_CSV = os.path.join(
    RESULTS_DIR,
    "final_strict_test_report.csv"
)

OUTPUT_TXT = os.path.join(
    RESULTS_DIR,
    "final_strict_test_report.txt"
)


# ================================================================
# 2. EXPECTED VALUES
# ================================================================

EXPECTED_BASELINE_THRESHOLD = 0.95
EXPECTED_ARTIFACT_THRESHOLD = 0.525596
EXPECTED_TEST_WINDOWS = 3114

EXPECTED_WINDOW_BASELINE = {
    "tp": 78,
    "fp": 56,
    "tn": 2965,
    "fn": 15,
}

EXPECTED_WINDOW_STRICT = {
    "tp": 76,
    "fp": 42,
    "tn": 2979,
    "fn": 17,
}

EXPECTED_EVENT_BASELINE = {
    "tp": 77,
    "fp": 56,
    "fn": 15,
}

EXPECTED_EVENT_STRICT = {
    "tp": 75,
    "fp": 42,
    "fn": 17,
}


# ================================================================
# 3. HELPERS
# ================================================================

def require_file(path):

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"Required file not found:\n{path}"
        )


def get_metric(dictionary, *keys):

    for key in keys:

        if key in dictionary:

            return dictionary[key]

    raise KeyError(
        "Missing metric. Tried: "
        + ", ".join(keys)
    )


def get_optional_metric(
    dictionary,
    *keys,
    default=None
):

    for key in keys:

        if key in dictionary:

            return dictionary[key]

    return default


def safe_float_or_none(value):

    if value is None:

        return None

    try:

        value = float(value)

    except (
        TypeError,
        ValueError
    ):

        return None

    if not np.isfinite(value):

        return None

    return value


def calculate_window_metrics(
    TP,
    FP,
    TN,
    FN
):

    TP = int(TP)
    FP = int(FP)
    TN = int(TN)
    FN = int(FN)

    recall = (
        TP / (TP + FN)
        if TP + FN > 0
        else 0.0
    )

    specificity = (
        TN / (TN + FP)
        if TN + FP > 0
        else 0.0
    )

    precision = (
        TP / (TP + FP)
        if TP + FP > 0
        else 0.0
    )

    accuracy = (
        (TP + TN)
        /
        (TP + TN + FP + FN)
        if TP + TN + FP + FN > 0
        else 0.0
    )

    f1 = (
        2.0
        * precision
        * recall
        /
        (precision + recall)
        if precision + recall > 0
        else 0.0
    )

    return {
        "tp": TP,
        "fp": FP,
        "tn": TN,
        "fn": FN,
        "recall": float(recall),
        "sensitivity": float(recall),
        "specificity": float(specificity),
        "precision": float(precision),
        "accuracy": float(accuracy),
        "f1": float(f1),
    }


def calculate_event_metrics(
    TP,
    FP,
    FN,
    TN=None
):

    TP = int(TP)
    FP = int(FP)
    FN = int(FN)

    recall = (
        TP / (TP + FN)
        if TP + FN > 0
        else 0.0
    )

    precision = (
        TP / (TP + FP)
        if TP + FP > 0
        else 0.0
    )

    f1 = (
        2.0
        * precision
        * recall
        /
        (precision + recall)
        if precision + recall > 0
        else 0.0
    )

    result = {
        "tp": TP,
        "fp": FP,
        "fn": FN,
        "recall": float(recall),
        "sensitivity": float(recall),
        "precision": float(precision),
        "f1": float(f1),
    }

    if TN is not None:

        TN = int(TN)

        specificity = (
            TN / (TN + FP)
            if TN + FP > 0
            else 0.0
        )

        accuracy = (
            (TP + TN)
            /
            (TP + TN + FP + FN)
            if TP + TN + FP + FN > 0
            else 0.0
        )

        result["tn"] = TN
        result["specificity"] = float(
            specificity
        )
        result["accuracy"] = float(
            accuracy
        )

    return result


def normalize_window_metrics(metrics):

    TP = get_metric(
        metrics,
        "TP",
        "tp"
    )

    FP = get_metric(
        metrics,
        "FP",
        "fp"
    )

    TN = get_metric(
        metrics,
        "TN",
        "tn"
    )

    FN = get_metric(
        metrics,
        "FN",
        "fn"
    )

    result = calculate_window_metrics(
        TP,
        FP,
        TN,
        FN
    )

    for name in [
        "recall",
        "sensitivity",
        "specificity",
        "precision",
        "accuracy",
        "f1",
    ]:

        if name in metrics:

            result[name] = float(
                metrics[name]
            )

    return result


def normalize_event_metrics(metrics):

    TP = get_metric(
        metrics,
        "TP",
        "tp"
    )

    FP = get_metric(
        metrics,
        "FP",
        "fp"
    )

    FN = get_metric(
        metrics,
        "FN",
        "fn"
    )

    TN = get_optional_metric(
        metrics,
        "TN",
        "tn",
        default=None
    )

    result = calculate_event_metrics(
        TP,
        FP,
        FN,
        TN
    )

    for name in [
        "recall",
        "sensitivity",
        "specificity",
        "precision",
        "accuracy",
        "f1",
    ]:

        if name in metrics:

            result[name] = float(
                metrics[name]
            )

    return result


def print_window_metrics(
    title,
    metrics
):

    print()
    print(title)
    print("-" * 70)

    print(
        f"TP          = {metrics['tp']}"
    )

    print(
        f"FP          = {metrics['fp']}"
    )

    print(
        f"TN          = {metrics['tn']}"
    )

    print(
        f"FN          = {metrics['fn']}"
    )

    print(
        f"Recall      = {metrics['recall']:.6f}"
    )

    print(
        f"Specificity = {metrics['specificity']:.6f}"
    )

    print(
        f"Precision   = {metrics['precision']:.6f}"
    )

    print(
        f"Accuracy    = {metrics['accuracy']:.6f}"
    )

    print(
        f"F1          = {metrics['f1']:.6f}"
    )


def print_event_metrics(
    title,
    metrics
):

    print()
    print(title)
    print("-" * 70)

    print(
        f"TP          = {metrics['tp']}"
    )

    print(
        f"FP          = {metrics['fp']}"
    )

    if "tn" in metrics:

        print(
            f"TN          = {metrics['tn']}"
        )

    print(
        f"FN          = {metrics['fn']}"
    )

    print(
        f"Recall      = {metrics['recall']:.6f}"
    )

    if "specificity" in metrics:

        print(
            f"Specificity = {metrics['specificity']:.6f}"
        )

    print(
        f"Precision   = {metrics['precision']:.6f}"
    )

    if "accuracy" in metrics:

        print(
            f"Accuracy    = {metrics['accuracy']:.6f}"
        )

    print(
        f"F1          = {metrics['f1']:.6f}"
    )


def is_validation_based_source(
    source
):

    if source is None:

        return False

    text = str(
        source
    ).strip().lower()

    validation_tokens = [
        "validation",
        "valid",
        "frozen",
    ]

    return any(
        token in text
        for token in validation_tokens
    )


# ================================================================
# 4. START
# ================================================================

print()
print("=" * 76)
print("FINAL STRICT TEST REPORT")
print("=" * 76)

print()
print("Project:")
print(PROJECT_DIR)

print()
print("Methodological guarantees:")
print("  Test set used for optimization : FALSE")
print("  Test threshold fitting         : FALSE")
print("  Test feature fitting           : FALSE")
print("  Model modified                 : FALSE")
print("  Dataset modified               : FALSE")
print("  Artifact rule source           : VALIDATION-FROZEN")


# ================================================================
# 5. FILE CHECK
# ================================================================

print()
print("=" * 76)
print("CHECKING REQUIRED FILES")
print("=" * 76)

required_files = [
    WINDOW_JSON_FILE,
    WINDOW_NPZ_FILE,
    EVENT_JSON_FILE,
    EVENT_NPZ_FILE,
    VALIDATION_JSON_FILE,
    VALIDATION_NPZ_FILE,
    TEST_INDICES_FILE,
]

for path in required_files:

    require_file(path)

    print(
        "[OK]",
        path
    )


# ================================================================
# 6. LOAD WINDOW JSON
# ================================================================

print()
print("=" * 76)
print("LOADING STRICT WINDOW RESULTS")
print("=" * 76)

with open(
    WINDOW_JSON_FILE,
    "r",
    encoding="utf-8"
) as f:

    window_json = json.load(f)


print()
print("Window JSON keys:")
print(
    list(
        window_json.keys()
    )
)


required_window_keys = [
    "baseline",
    "artifact_rejection",
    "change",
]

for key in required_window_keys:

    if key not in window_json:

        raise RuntimeError(
            f"Missing window JSON key: {key}"
        )


# ================================================================
# 7. CORRECT STRICT WINDOW JSON MAPPING
# ================================================================

window_baseline = (
    window_json["baseline"]
)

window_strict = (
    window_json["artifact_rejection"]
)

window_change_json = (
    window_json["change"]
)


# ================================================================
# 8. WINDOW METRICS
# ================================================================

baseline_window_metrics = (
    normalize_window_metrics(
        window_baseline
    )
)

strict_window_metrics = (
    normalize_window_metrics(
        window_strict
    )
)

print_window_metrics(
    "WINDOW LEVEL - BASELINE",
    baseline_window_metrics
)

print_window_metrics(
    "WINDOW LEVEL - STRICT VALIDATION-FROZEN",
    strict_window_metrics
)


# ================================================================
# 9. LOAD WINDOW NPZ
# ================================================================

print()
print("=" * 76)
print("LOADING STRICT WINDOW NPZ")
print("=" * 76)

window_data = np.load(
    WINDOW_NPZ_FILE,
    allow_pickle=True
)

print()
print("Available arrays:")

for key in window_data.files:

    print(
        f"  {key:35s} "
        f"{window_data[key].shape}"
    )


required_npz_keys = [
    "test_indices",
    "labels",
    "probabilities",
    "artifact_score",
    "artifact_rejected",
    "baseline_positive",
    "final_positive",
]

for key in required_npz_keys:

    if key not in window_data.files:

        raise RuntimeError(
            f"Missing required NPZ array: "
            f"{key}"
        )


test_indices = np.asarray(
    window_data["test_indices"],
    dtype=np.int64
)

labels = np.asarray(
    window_data["labels"],
    dtype=np.int64
)

probabilities = np.asarray(
    window_data["probabilities"],
    dtype=np.float64
)

artifact_scores = np.asarray(
    window_data["artifact_score"],
    dtype=np.float64
)

artifact_rejected = np.asarray(
    window_data["artifact_rejected"],
    dtype=bool
)

baseline_positive = np.asarray(
    window_data["baseline_positive"],
    dtype=bool
)

strict_positive = np.asarray(
    window_data["final_positive"],
    dtype=bool
)


# ================================================================
# 10. ARRAY CHECKS
# ================================================================

print()
print("=" * 76)
print("VERIFYING WINDOW ARRAYS")
print("=" * 76)

n_windows = len(
    test_indices
)

if n_windows != EXPECTED_TEST_WINDOWS:

    raise RuntimeError(
        f"Unexpected test window count: "
        f"{n_windows}; expected "
        f"{EXPECTED_TEST_WINDOWS}"
    )


for name, array in {

    "labels":
        labels,

    "probabilities":
        probabilities,

    "artifact_scores":
        artifact_scores,

    "artifact_rejected":
        artifact_rejected,

    "baseline_positive":
        baseline_positive,

    "strict_positive":
        strict_positive,

}.items():

    if len(array) != n_windows:

        raise RuntimeError(
            f"{name} length mismatch."
        )

    print(
        f"[OK] {name}: "
        f"{len(array)}"
    )


if not np.all(
    np.isfinite(
        probabilities
    )
):

    raise RuntimeError(
        "Non-finite probability detected."
    )


if not np.all(
    np.isfinite(
        artifact_scores
    )
):

    raise RuntimeError(
        "Non-finite artifact score detected."
    )


# ================================================================
# 11. TEST INDEX CHECK
# ================================================================

print()
print("=" * 76)
print("VERIFYING TEST INDICES")
print("=" * 76)

reference_test_indices = np.asarray(
    np.load(
        TEST_INDICES_FILE
    ),
    dtype=np.int64
)

if not np.array_equal(
    reference_test_indices,
    test_indices
):

    raise RuntimeError(
        "Strict NPZ test_indices do not "
        "match data/test_indices.npy."
    )


print(
    "[OK] test_indices exactly match."
)


# ================================================================
# 12. RECOMPUTE WINDOW CONFUSION MATRICES
# ================================================================

print()
print("=" * 76)
print("RECOMPUTING WINDOW COUNTS FROM NPZ")
print("=" * 76)

npz_baseline = (
    calculate_window_metrics(

        np.sum(
            baseline_positive
            &
            (labels == 1)
        ),

        np.sum(
            baseline_positive
            &
            (labels == 0)
        ),

        np.sum(
            (~baseline_positive)
            &
            (labels == 0)
        ),

        np.sum(
            (~baseline_positive)
            &
            (labels == 1)
        ),
    )
)


npz_strict = (
    calculate_window_metrics(

        np.sum(
            strict_positive
            &
            (labels == 1)
        ),

        np.sum(
            strict_positive
            &
            (labels == 0)
        ),

        np.sum(
            (~strict_positive)
            &
            (labels == 0)
        ),

        np.sum(
            (~strict_positive)
            &
            (labels == 1)
        ),
    )
)


# ================================================================
# 13. WINDOW CROSS-CHECK
# ================================================================

print()
print("=" * 76)
print("WINDOW CROSS-CHECK")
print("=" * 76)

for key in [
    "tp",
    "fp",
    "tn",
    "fn",
]:

    if (
        npz_baseline[key]
        !=
        baseline_window_metrics[key]
    ):

        raise RuntimeError(
            f"Baseline cross-check failed: "
            f"{key}"
        )

    if (
        npz_strict[key]
        !=
        strict_window_metrics[key]
    ):

        raise RuntimeError(
            f"Strict cross-check failed: "
            f"{key}"
        )


print(
    "[OK] Window-level cross-check passed."
)


# ================================================================
# 14. LABEL COUNTS
# ================================================================

positive_windows = int(
    np.sum(
        labels == 1
    )
)

negative_windows = int(
    np.sum(
        labels == 0
    )
)

if (
    positive_windows
    +
    negative_windows
    !=
    n_windows
):

    raise RuntimeError(
        "Positive + negative window counts "
        "do not equal total."
    )


print()
print(
    "Positive windows:",
    positive_windows
)

print(
    "Negative windows:",
    negative_windows
)


# ================================================================
# 15. THRESHOLD + METHODOLOGY CHECKS
# ================================================================

print()
print("=" * 76)
print("VERIFYING FROZEN THRESHOLDS")
print("=" * 76)

baseline_threshold = float(
    window_json[
        "baseline_probability_threshold"
    ]
)

artifact_threshold = float(
    window_json[
        "artifact_score_threshold"
    ]
)

threshold_source = (
    window_json.get(
        "artifact_threshold_source"
    )
)


if abs(
    baseline_threshold
    -
    EXPECTED_BASELINE_THRESHOLD
) > 1e-9:

    raise RuntimeError(
        "Baseline threshold mismatch."
    )


if abs(
    artifact_threshold
    -
    EXPECTED_ARTIFACT_THRESHOLD
) > 1e-6:

    raise RuntimeError(
        "Artifact threshold mismatch."
    )


if threshold_source is None:

    raise RuntimeError(
        "artifact_threshold_source is missing "
        "from strict window JSON."
    )


if not is_validation_based_source(
    threshold_source
):

    raise RuntimeError(
        "Artifact threshold source does not "
        "appear to be validation-based. "
        f"Found: {threshold_source}"
    )


test_used_for_optimization = bool(
    window_json.get(
        "test_set_used_for_optimization",
        False
    )
)

test_labels_used_for_fitting = bool(
    window_json.get(
        "test_labels_used_for_parameter_fitting",
        False
    )
)

model_modified = bool(
    window_json.get(
        "model_modified",
        False
    )
)

dataset_modified = bool(
    window_json.get(
        "dataset_modified",
        False
    )
)


if test_used_for_optimization:

    raise RuntimeError(
        "Methodology violation: "
        "test set was used for optimization."
    )


if test_labels_used_for_fitting:

    raise RuntimeError(
        "Methodology violation: "
        "test labels were used for "
        "parameter fitting."
    )


if model_modified:

    raise RuntimeError(
        "Unexpected methodology state: "
        "model_modified=True"
    )


if dataset_modified:

    raise RuntimeError(
        "Unexpected methodology state: "
        "dataset_modified=True"
    )


print(
    "[OK] Baseline threshold:",
    baseline_threshold
)

print(
    "[OK] Artifact threshold:",
    artifact_threshold
)

print(
    "[OK] Threshold source:",
    threshold_source
)

print(
    "[OK] Threshold source is "
    "validation-based."
)

print(
    "[OK] Test set used for optimization: "
    "FALSE"
)

print(
    "[OK] Test labels used for fitting: "
    "FALSE"
)


# ================================================================
# 16. VALIDATION SCORE REPRODUCTION
# ================================================================

print()
print("=" * 76)
print("VALIDATION SCORE REPRODUCTION")
print("=" * 76)

validation_reproduction = window_json.get(
    "validation_score_reproduction",
    {}
)

if not isinstance(
    validation_reproduction,
    dict
):

    raise RuntimeError(
        "validation_score_reproduction must "
        "be a dictionary."
    )


print()
print("Validation reproduction metadata:")

print(
    json.dumps(
        validation_reproduction,
        indent=4,
        ensure_ascii=False
    )
)


# ------------------------------------------------
# Read reproduction status
# ------------------------------------------------

reproduction_passed = bool(
    validation_reproduction.get(
        "passed",
        False
    )
)


# ------------------------------------------------
# IMPORTANT FIX:
#
# The actual strict JSON uses:
#
#     max_absolute_difference
#     mean_absolute_difference
#
# rather than:
#
#     max_abs_difference
#     mean_abs_difference
#
# Support both names so the report generator
# remains compatible with either format.
# ------------------------------------------------

max_abs_difference_raw = (
    validation_reproduction.get(
        "max_absolute_difference",
        validation_reproduction.get(
            "max_abs_difference",
            None
        )
    )
)


mean_abs_difference_raw = (
    validation_reproduction.get(
        "mean_absolute_difference",
        validation_reproduction.get(
            "mean_abs_difference",
            None
        )
    )
)


max_abs_difference = safe_float_or_none(
    max_abs_difference_raw
)

mean_abs_difference = safe_float_or_none(
    mean_abs_difference_raw
)


if not reproduction_passed:

    raise RuntimeError(
        "Validation score reproduction did not "
        "report passed=True."
    )


if max_abs_difference is None:

    raise RuntimeError(
        "Validation score reproduction did not "
        "contain a valid maximum absolute "
        "difference."
    )


if mean_abs_difference is None:

    raise RuntimeError(
        "Validation score reproduction did not "
        "contain a valid mean absolute "
        "difference."
    )


if max_abs_difference > 1e-12:

    raise RuntimeError(
        "Validation score reproduction "
        "maximum difference is too large: "
        f"{max_abs_difference}"
    )


if mean_abs_difference > 1e-12:

    raise RuntimeError(
        "Validation score reproduction "
        "mean difference is too large: "
        f"{mean_abs_difference}"
    )


print(
    "[OK] Validation reproduction passed."
)

print(
    "Max abs difference:",
    max_abs_difference
)

print(
    "Mean abs difference:",
    mean_abs_difference
)


# ================================================================
# 17. LOAD VALIDATION JSON FOR CROSS-CHECK
# ================================================================

print()
print("=" * 76)
print("CROSS-CHECKING VALIDATION THRESHOLD")
print("=" * 76)

with open(
    VALIDATION_JSON_FILE,
    "r",
    encoding="utf-8"
) as f:

    validation_json = json.load(f)


# ------------------------------------------------
# Recursive search helper for threshold values
# ------------------------------------------------

def find_threshold_candidates(
    obj,
    path="root"
):

    candidates = []

    if isinstance(
        obj,
        dict
    ):

        for key, value in obj.items():

            key_lower = str(
                key
            ).lower()

            if any(
                token in key_lower
                for token in [
                    "artifact_threshold",
                    "artifact_score_threshold",
                ]
            ):

                candidates.append(
                    (
                        path
                        +
                        "."
                        +
                        str(key),
                        value
                    )
                )

            elif key_lower == "threshold":

                candidates.append(
                    (
                        path
                        +
                        "."
                        +
                        str(key),
                        value
                    )
                )

            candidates.extend(
                find_threshold_candidates(
                    value,
                    path
                    +
                    "."
                    +
                    str(key)
                )
            )

    elif isinstance(
        obj,
        list
    ):

        for index, value in enumerate(
            obj
        ):

            candidates.extend(
                find_threshold_candidates(
                    value,
                    path
                    +
                    f"[{index}]"
                )
            )

    return candidates


validation_threshold_candidates = (
    find_threshold_candidates(
        validation_json
    )
)


validation_threshold_value = None
validation_threshold_path = None

for path, candidate in (
    validation_threshold_candidates
):

    value = safe_float_or_none(
        candidate
    )

    if value is None:

        continue

    if abs(
        value
        -
        artifact_threshold
    ) <= 1e-6:

        validation_threshold_value = (
            value
        )

        validation_threshold_path = (
            path
        )

        break


if validation_threshold_value is not None:

    print(
        "[OK] Test artifact threshold matches "
        "validation result:"
    )

    print(
        "      threshold =",
        validation_threshold_value
    )

    print(
        "      source key =",
        validation_threshold_path
    )

else:

    print(
        "[INFO] Could not locate a canonical "
        "matching threshold key in validation JSON."
    )

    print(
        "[INFO] Strict threshold remains "
        "verified by:"
    )

    print(
        "      - expected frozen value"
    )

    print(
        "      - validation score reproduction"
    )


# ================================================================
# 18. WINDOW CHANGES
# ================================================================

print()
print("=" * 76)
print("WINDOW-LEVEL CHANGES")
print("=" * 76)

rejected_fp = int(
    np.sum(
        baseline_positive
        &
        (labels == 0)
        &
        artifact_rejected
    )
)

rejected_tp = int(
    np.sum(
        baseline_positive
        &
        (labels == 1)
        &
        artifact_rejected
    )
)

total_rejected_positive = int(
    np.sum(
        baseline_positive
        &
        artifact_rejected
    )
)

window_fp_reduction = (
    (
        baseline_window_metrics["fp"]
        -
        strict_window_metrics["fp"]
    )
    /
    max(
        baseline_window_metrics["fp"],
        1
    )
)

window_recall_change = (
    strict_window_metrics["recall"]
    -
    baseline_window_metrics["recall"]
)

window_specificity_change = (
    strict_window_metrics["specificity"]
    -
    baseline_window_metrics["specificity"]
)

window_precision_change = (
    strict_window_metrics["precision"]
    -
    baseline_window_metrics["precision"]
)

window_accuracy_change = (
    strict_window_metrics["accuracy"]
    -
    baseline_window_metrics["accuracy"]
)

window_f1_change = (
    strict_window_metrics["f1"]
    -
    baseline_window_metrics["f1"]
)


print(
    f"Rejected baseline FP: "
    f"{rejected_fp}"
)

print(
    f"Rejected baseline TP: "
    f"{rejected_tp}"
)

print(
    f"Total rejected model-positive: "
    f"{total_rejected_positive}"
)

print(
    f"FP reduction: "
    f"{window_fp_reduction * 100:.2f}%"
)

print(
    f"Recall change: "
    f"{window_recall_change * 100:.2f} pp"
)

print(
    f"Specificity change: "
    f"{window_specificity_change * 100:.2f} pp"
)

print(
    f"Precision change: "
    f"{window_precision_change * 100:.2f} pp"
)

print(
    f"Accuracy change: "
    f"{window_accuracy_change * 100:.2f} pp"
)

print(
    f"F1 change: "
    f"{window_f1_change * 100:.2f} pp"
)


# ================================================================
# 19. CROSS-CHECK WINDOW CHANGE JSON
# ================================================================

print()
print("=" * 76)
print("CROSS-CHECKING WINDOW CHANGE JSON")
print("=" * 76)


if "rejected_baseline_fp" in window_change_json:

    if int(
        window_change_json[
            "rejected_baseline_fp"
        ]
    ) != rejected_fp:

        raise RuntimeError(
            "rejected_baseline_fp mismatch."
        )


if "rejected_baseline_tp" in window_change_json:

    if int(
        window_change_json[
            "rejected_baseline_tp"
        ]
    ) != rejected_tp:

        raise RuntimeError(
            "rejected_baseline_tp mismatch."
        )


if (
    "total_rejected_model_positive"
    in window_change_json
):

    if int(
        window_change_json[
            "total_rejected_model_positive"
        ]
    ) != total_rejected_positive:

        raise RuntimeError(
            "total_rejected_model_positive "
            "mismatch."
        )


print(
    "[OK] Window change counts verified."
)


# ================================================================
# 20. LOAD EVENT JSON
# ================================================================

print()
print("=" * 76)
print("LOADING STRICT EVENT RESULTS")
print("=" * 76)

with open(
    EVENT_JSON_FILE,
    "r",
    encoding="utf-8"
) as f:

    event_json = json.load(f)


print()
print("Event JSON keys:")
print(
    list(
        event_json.keys()
    )
)


if "event_level" not in event_json:

    raise RuntimeError(
        "Event JSON missing event_level."
    )


event_level = (
    event_json["event_level"]
)


print()
print("Event-level keys:")
print(
    list(
        event_level.keys()
    )
)


if "baseline" not in event_level:

    raise RuntimeError(
        "Event-level baseline missing."
    )


# ================================================================
# 21. CORRECT EVENT JSON MAPPING
# ================================================================

# IMPORTANT:
#
# The strict event evaluator stores the strict
# validation-frozen result under:
#
#     "strict_validation_frozen"
#
# NOT:
#
#     "artifact_rejection"
#
# Therefore this report generator must use the
# actual JSON structure.

if "strict_validation_frozen" not in event_level:

    raise RuntimeError(
        "Event-level strict_validation_frozen "
        "missing."
    )


event_baseline = (
    event_level["baseline"]
)

event_strict = (
    event_level["strict_validation_frozen"]
)

event_change_json = (
    event_level.get(
        "change",
        {}
    )
)

print(
    "[OK] Event-level baseline loaded."
)

print(
    "[OK] Event-level strict "
    "validation-frozen result loaded."
)


# ================================================================
# 22. EVENT METRICS
# ================================================================

baseline_event_metrics = (
    normalize_event_metrics(
        event_baseline
    )
)

strict_event_metrics = (
    normalize_event_metrics(
        event_strict
    )
)

print_event_metrics(
    "EVENT LEVEL - BASELINE",
    baseline_event_metrics
)

print_event_metrics(
    "EVENT LEVEL - STRICT VALIDATION-FROZEN",
    strict_event_metrics
)


# ================================================================
# 23. TRUE EVENT COUNT
# ================================================================

true_event_count_raw = (
    event_level.get(
        "true_event_count",
        event_json.get(
            "true_event_count",
            None
        )
    )
)


if true_event_count_raw is None:

    true_event_count = int(
        baseline_event_metrics["tp"]
        +
        baseline_event_metrics["fn"]
    )

else:

    true_event_count = int(
        true_event_count_raw
    )


expected_true_events_from_baseline = int(
    baseline_event_metrics["tp"]
    +
    baseline_event_metrics["fn"]
)


if (
    true_event_count
    !=
    expected_true_events_from_baseline
):

    raise RuntimeError(
        "True event count does not match "
        "baseline TP + FN."
    )


strict_true_events = int(
    strict_event_metrics["tp"]
    +
    strict_event_metrics["fn"]
)


if (
    strict_true_events
    !=
    true_event_count
):

    raise RuntimeError(
        "Strict event TP + FN does not match "
        "true event count."
    )


print()
print(
    "True seizure events:",
    true_event_count
)


# ================================================================
# 24. EVENT CHANGES
# ================================================================

event_fp_reduction = (
    (
        baseline_event_metrics["fp"]
        -
        strict_event_metrics["fp"]
    )
    /
    max(
        baseline_event_metrics["fp"],
        1
    )
)

event_recall_change = (
    strict_event_metrics["recall"]
    -
    baseline_event_metrics["recall"]
)

event_precision_change = (
    strict_event_metrics["precision"]
    -
    baseline_event_metrics["precision"]
)

event_f1_change = (
    strict_event_metrics["f1"]
    -
    baseline_event_metrics["f1"]
)


print()
print("=" * 76)
print("EVENT-LEVEL CHANGES")
print("=" * 76)

print(
    f"FP reduction: "
    f"{event_fp_reduction * 100:.2f}%"
)

print(
    f"Recall change: "
    f"{event_recall_change * 100:.2f} pp"
)

print(
    f"Precision change: "
    f"{event_precision_change * 100:.2f} pp"
)

print(
    f"F1 change: "
    f"{event_f1_change * 100:.2f} pp"
)


# ================================================================
# 25. CROSS-CHECK EVENT CHANGE JSON
# ================================================================

print()
print("=" * 76)
print("CROSS-CHECKING EVENT CHANGE JSON")
print("=" * 76)

if isinstance(
    event_change_json,
    dict
):

    expected_change_values = {
        "fp_reduction": event_fp_reduction,
        "recall_change": event_recall_change,
        "precision_change": event_precision_change,
        "f1_change": event_f1_change,
    }

    for key, calculated_value in (
        expected_change_values.items()
    ):

        if key not in event_change_json:

            continue

        stored_value = safe_float_or_none(
            event_change_json[key]
        )

        if stored_value is None:

            continue

        if abs(
            stored_value
            -
            calculated_value
        ) > 1e-9:

            raise RuntimeError(
                f"Event change mismatch: "
                f"{key}; "
                f"stored={stored_value}, "
                f"calculated={calculated_value}"
            )

    print(
        "[OK] Event change values verified."
    )

else:

    print(
        "[INFO] Event change JSON is not a "
        "dictionary; count-based verification "
        "continues from event metrics."
    )


# ================================================================
# 26. LOAD EVENT NPZ
# ================================================================

print()
print("=" * 76)
print("LOADING EVENT NPZ")
print("=" * 76)

event_data = np.load(
    EVENT_NPZ_FILE,
    allow_pickle=True
)

print()

for key in event_data.files:

    print(
        f"{key:40s} "
        f"{event_data[key].shape}"
    )


# ================================================================
# 27. PATIENT SUMMARY
# ================================================================

# Actual strict event JSON uses:
#
#     patient_event_summary
#
# Keep backward compatibility with older names.

patient_summary = {}

if "patient_event_summary" in event_json:

    patient_summary = (
        event_json[
            "patient_event_summary"
        ]
    )

elif "patient_summary" in event_json:

    patient_summary = (
        event_json[
            "patient_summary"
        ]
    )

elif "patient_summary" in event_level:

    patient_summary = (
        event_level[
            "patient_summary"
        ]
    )


print()
print(
    "Patient summary entries:",
    len(patient_summary)
)


# ================================================================
# 28. FINAL EXPECTED RESULT CHECK
# ================================================================

print()
print("=" * 76)
print("FINAL SANITY CHECK")
print("=" * 76)


for key, expected in (
    EXPECTED_WINDOW_BASELINE.items()
):

    actual = (
        baseline_window_metrics[key]
    )

    if actual != expected:

        raise RuntimeError(
            f"Window baseline {key}: "
            f"got {actual}, "
            f"expected {expected}"
        )


for key, expected in (
    EXPECTED_WINDOW_STRICT.items()
):

    actual = (
        strict_window_metrics[key]
    )

    if actual != expected:

        raise RuntimeError(
            f"Window strict {key}: "
            f"got {actual}, "
            f"expected {expected}"
        )


for key, expected in (
    EXPECTED_EVENT_BASELINE.items()
):

    actual = (
        baseline_event_metrics[key]
    )

    if actual != expected:

        raise RuntimeError(
            f"Event baseline {key}: "
            f"got {actual}, "
            f"expected {expected}"
        )


for key, expected in (
    EXPECTED_EVENT_STRICT.items()
):

    actual = (
        strict_event_metrics[key]
    )

    if actual != expected:

        raise RuntimeError(
            f"Event strict {key}: "
            f"got {actual}, "
            f"expected {expected}"
        )


if positive_windows != 93:

    raise RuntimeError(
        f"Expected 93 positive windows, "
        f"got {positive_windows}"
    )


if negative_windows != 3021:

    raise RuntimeError(
        f"Expected 3021 negative windows, "
        f"got {negative_windows}"
    )


if true_event_count != 92:

    raise RuntimeError(
        f"Expected 92 true seizure events, "
        f"got {true_event_count}"
    )


if rejected_fp != 14:

    raise RuntimeError(
        f"Expected 14 rejected baseline FP, "
        f"got {rejected_fp}"
    )


if rejected_tp != 2:

    raise RuntimeError(
        f"Expected 2 rejected baseline TP, "
        f"got {rejected_tp}"
    )


if total_rejected_positive != 16:

    raise RuntimeError(
        "Expected 16 total rejected "
        f"model-positive windows, got "
        f"{total_rejected_positive}"
    )


print(
    "[OK] All expected strict results verified."
)


# ================================================================
# 29. BUILD REPORT
# ================================================================

report = {

    "report_type":
        "FINAL_STRICT_TEST_REPORT",

    "project":
        PROJECT_DIR,

    "methodology": {

        "test_used_for_optimization":
            False,

        "test_threshold_fitting":
            False,

        "test_feature_fitting":
            False,

        "artifact_rule_source":
            "validation_frozen",

        "artifact_threshold_source_raw":
            str(
                threshold_source
            ),

        "evaluation_type":
            "strict_validation_frozen",

        "strict_results_used":
            True,

        "previous_non_strict_results_used":
            False,

        "model_modified":
            False,

        "dataset_modified":
            False,
    },

    "settings": {

        "test_window_count":
            n_windows,

        "true_seizure_windows":
            positive_windows,

        "true_non_seizure_windows":
            negative_windows,

        "true_event_count":
            true_event_count,

        "baseline_probability_threshold":
            baseline_threshold,

        "artifact_threshold":
            artifact_threshold,
    },

    "validation_rule": {

        "artifact_threshold_source":
            str(
                threshold_source
            ),

        "artifact_threshold_source_is_validation_based":
            True,

        "validation_score_reproduction_passed":
            reproduction_passed,

        "validation_max_abs_difference":
            max_abs_difference,

        "validation_mean_abs_difference":
            mean_abs_difference,

        "validation_threshold_crosscheck":
            (
                validation_threshold_value
                if validation_threshold_value
                is not None
                else None
            ),

        "validation_threshold_crosscheck_path":
            validation_threshold_path,
    },

    "window_level": {

        "baseline":
            baseline_window_metrics,

        "strict_validation_frozen":
            strict_window_metrics,

        "change": {

            "fp_reduction":
                float(
                    window_fp_reduction
                ),

            "fp_reduction_percent":
                float(
                    window_fp_reduction
                    * 100
                ),

            "rejected_baseline_fp":
                rejected_fp,

            "rejected_baseline_tp":
                rejected_tp,

            "total_rejected_model_positive":
                total_rejected_positive,

            "recall_change":
                float(
                    window_recall_change
                ),

            "recall_change_percentage_points":
                float(
                    window_recall_change
                    * 100
                ),

            "specificity_change":
                float(
                    window_specificity_change
                ),

            "specificity_change_percentage_points":
                float(
                    window_specificity_change
                    * 100
                ),

            "precision_change":
                float(
                    window_precision_change
                ),

            "precision_change_percentage_points":
                float(
                    window_precision_change
                    * 100
                ),

            "accuracy_change":
                float(
                    window_accuracy_change
                ),

            "accuracy_change_percentage_points":
                float(
                    window_accuracy_change
                    * 100
                ),

            "f1_change":
                float(
                    window_f1_change
                ),

            "f1_change_percentage_points":
                float(
                    window_f1_change
                    * 100
                ),
        },
    },

    "event_level": {

        "true_event_count":
            true_event_count,

        "baseline":
            baseline_event_metrics,

        "strict_validation_frozen":
            strict_event_metrics,

        "change":
            {
                "fp_reduction":
                    float(
                        event_fp_reduction
                    ),

                "fp_reduction_percent":
                    float(
                        event_fp_reduction
                        * 100
                    ),

                "recall_change":
                    float(
                        event_recall_change
                    ),

                "recall_change_percentage_points":
                    float(
                        event_recall_change
                        * 100
                    ),

                "precision_change":
                    float(
                        event_precision_change
                    ),

                "precision_change_percentage_points":
                    float(
                        event_precision_change
                        * 100
                    ),

                "f1_change":
                    float(
                        event_f1_change
                    ),

                "f1_change_percentage_points":
                    float(
                        event_f1_change
                        * 100
                    ),
            },
    },

    "patient_summary":
        patient_summary,
}


# ================================================================
# 30. SAVE JSON
# ================================================================

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
print(
    OUTPUT_JSON
)


# ================================================================
# 31. SAVE CSV
# ================================================================

csv_rows = [

    {
        "level":
            "window",

        "method":
            "baseline",

        **baseline_window_metrics,
    },

    {
        "level":
            "window",

        "method":
            "strict_validation_frozen",

        **strict_window_metrics,
    },

    {
        "level":
            "event",

        "method":
            "baseline",

        **baseline_event_metrics,
    },

    {
        "level":
            "event",

        "method":
            "strict_validation_frozen",

        **strict_event_metrics,
    },
]


csv_fields = [
    "level",
    "method",
    "tp",
    "fp",
    "tn",
    "fn",
    "recall",
    "sensitivity",
    "specificity",
    "precision",
    "accuracy",
    "f1",
]


with open(
    OUTPUT_CSV,
    "w",
    newline="",
    encoding="utf-8-sig"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=csv_fields,
        extrasaction="ignore"
    )

    writer.writeheader()

    for row in csv_rows:

        writer.writerow(
            row
        )


print()
print("[OK] CSV saved:")
print(
    OUTPUT_CSV
)


# ================================================================
# 32. SAVE TXT
# ================================================================

with open(
    OUTPUT_TXT,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "=" * 76
        + "\n"
    )

    f.write(
        "FINAL STRICT TEST REPORT\n"
    )

    f.write(
        "=" * 76
        + "\n\n"
    )

    f.write(
        f"Project: {PROJECT_DIR}\n"
    )

    f.write(
        f"Test windows: {n_windows}\n"
    )

    f.write(
        f"True seizure windows: "
        f"{positive_windows}\n"
    )

    f.write(
        f"True non-seizure windows: "
        f"{negative_windows}\n"
    )

    f.write(
        f"True seizure events: "
        f"{true_event_count}\n"
    )

    f.write(
        f"Baseline threshold: "
        f"{baseline_threshold}\n"
    )

    f.write(
        f"Artifact threshold: "
        f"{artifact_threshold}\n"
    )

    f.write(
        "Artifact threshold source: "
        f"{threshold_source}\n"
    )

    f.write(
        "Artifact rule methodology: "
        "validation-frozen\n\n"
    )


    # ============================================================
    # WINDOW LEVEL
    # ============================================================

    f.write(
        "WINDOW LEVEL\n"
    )

    f.write(
        "-" * 76
        + "\n"
    )

    for method, metrics in [

        (
            "baseline",
            baseline_window_metrics
        ),

        (
            "strict_validation_frozen",
            strict_window_metrics
        ),

    ]:

        f.write(
            f"\n{method}\n"
        )

        for key in [
            "tp",
            "fp",
            "tn",
            "fn",
            "recall",
            "specificity",
            "precision",
            "accuracy",
            "f1",
        ]:

            f.write(
                f"{key}: "
                f"{metrics[key]}\n"
            )


    f.write(
        "\nWindow changes\n"
    )

    f.write(
        f"Rejected baseline FP: "
        f"{rejected_fp}\n"
    )

    f.write(
        f"Rejected baseline TP: "
        f"{rejected_tp}\n"
    )

    f.write(
        f"Total rejected model-positive: "
        f"{total_rejected_positive}\n"
    )

    f.write(
        f"FP reduction: "
        f"{window_fp_reduction * 100:.4f}%\n"
    )

    f.write(
        f"Recall change: "
        f"{window_recall_change * 100:.4f} pp\n"
    )

    f.write(
        f"Specificity change: "
        f"{window_specificity_change * 100:.4f} pp\n"
    )

    f.write(
        f"Precision change: "
        f"{window_precision_change * 100:.4f} pp\n"
    )

    f.write(
        f"Accuracy change: "
        f"{window_accuracy_change * 100:.4f} pp\n"
    )

    f.write(
        f"F1 change: "
        f"{window_f1_change * 100:.4f} pp\n"
    )


    # ============================================================
    # EVENT LEVEL
    # ============================================================

    f.write(
        "\nEVENT LEVEL\n"
    )

    f.write(
        "-" * 76
        + "\n"
    )

    for method, metrics in [

        (
            "baseline",
            baseline_event_metrics
        ),

        (
            "strict_validation_frozen",
            strict_event_metrics
        ),

    ]:

        f.write(
            f"\n{method}\n"
        )

        for key in [
            "tp",
            "fp",
            "fn",
            "recall",
            "precision",
            "f1",
        ]:

            f.write(
                f"{key}: "
                f"{metrics[key]}\n"
            )


    f.write(
        "\nEvent changes\n"
    )

    f.write(
        f"FP reduction: "
        f"{event_fp_reduction * 100:.4f}%\n"
    )

    f.write(
        f"Recall change: "
        f"{event_recall_change * 100:.4f} pp\n"
    )

    f.write(
        f"Precision change: "
        f"{event_precision_change * 100:.4f} pp\n"
    )

    f.write(
        f"F1 change: "
        f"{event_f1_change * 100:.4f} pp\n"
    )


    # ============================================================
    # VALIDATION REPRODUCTION
    # ============================================================

    f.write(
        "\nVALIDATION RULE VERIFICATION\n"
    )

    f.write(
        "-" * 76
        + "\n"
    )

    f.write(
        "Validation score reproduction "
        f"passed: {reproduction_passed}\n"
    )

    f.write(
        "Validation max abs difference: "
        f"{max_abs_difference}\n"
    )

    f.write(
        "Validation mean abs difference: "
        f"{mean_abs_difference}\n"
    )

    f.write(
        "Threshold source raw value: "
        f"{threshold_source}\n"
    )

    f.write(
        "Threshold source validation-based: "
        "TRUE\n"
    )

    f.write(
        "Validation threshold cross-check: "
        f"{validation_threshold_value}\n"
    )

    f.write(
        "Validation threshold cross-check path: "
        f"{validation_threshold_path}\n"
    )


    # ============================================================
    # METHODOLOGY
    # ============================================================

    f.write(
        "\nMETHODOLOGY\n"
    )

    f.write(
        "-" * 76
        + "\n"
    )

    f.write(
        "Test set used for optimization: FALSE\n"
    )

    f.write(
        "Test threshold fitting: FALSE\n"
    )

    f.write(
        "Test feature fitting: FALSE\n"
    )

    f.write(
        "Artifact rule source: "
        "VALIDATION-FROZEN\n"
    )

    f.write(
        "Strict validation-frozen results "
        "used: TRUE\n"
    )

    f.write(
        "Previous non-strict results "
        "used: FALSE\n"
    )

    f.write(
        "Model modified: FALSE\n"
    )

    f.write(
        "Dataset modified: FALSE\n"
    )


print()
print("[OK] TXT saved:")
print(
    OUTPUT_TXT
)


# ================================================================
# 33. FINAL SUMMARY
# ================================================================

print()
print("=" * 76)
print("FINAL STRICT TEST SUMMARY")
print("=" * 76)

print()
print("WINDOW LEVEL")

print(
    f"Baseline : "
    f"TP={baseline_window_metrics['tp']} "
    f"FP={baseline_window_metrics['fp']} "
    f"TN={baseline_window_metrics['tn']} "
    f"FN={baseline_window_metrics['fn']}"
)

print(
    f"Strict   : "
    f"TP={strict_window_metrics['tp']} "
    f"FP={strict_window_metrics['fp']} "
    f"TN={strict_window_metrics['tn']} "
    f"FN={strict_window_metrics['fn']}"
)

print(
    f"FP reduction = "
    f"{window_fp_reduction * 100:.2f}%"
)

print(
    f"Recall change = "
    f"{window_recall_change * 100:.2f} pp"
)

print(
    f"Precision change = "
    f"{window_precision_change * 100:.2f} pp"
)

print(
    f"F1 change = "
    f"{window_f1_change * 100:.2f} pp"
)


print()
print("EVENT LEVEL")

print(
    f"Baseline : "
    f"TP={baseline_event_metrics['tp']} "
    f"FP={baseline_event_metrics['fp']} "
    f"FN={baseline_event_metrics['fn']}"
)

print(
    f"Strict   : "
    f"TP={strict_event_metrics['tp']} "
    f"FP={strict_event_metrics['fp']} "
    f"FN={strict_event_metrics['fn']}"
)

print(
    f"FP reduction = "
    f"{event_fp_reduction * 100:.2f}%"
)

print(
    f"Recall change = "
    f"{event_recall_change * 100:.2f} pp"
)

print(
    f"Precision change = "
    f"{event_precision_change * 100:.2f} pp"
)

print(
    f"F1 change = "
    f"{event_f1_change * 100:.2f} pp"
)


print()
print("VALIDATION RULE")

print(
    f"Artifact threshold = "
    f"{artifact_threshold}"
)

print(
    f"Threshold source = "
    f"{threshold_source}"
)

print(
    f"Validation reproduction max diff = "
    f"{max_abs_difference}"
)

print(
    f"Validation reproduction mean diff = "
    f"{mean_abs_difference}"
)

print(
    f"Validation reproduction passed = "
    f"{reproduction_passed}"
)


print()
print("METHODOLOGY")

print(
    "Test optimization              = FALSE"
)

print(
    "Test threshold fitting         = FALSE"
)

print(
    "Test feature fitting           = FALSE"
)

print(
    "Artifact rule                  = VALIDATION-FROZEN"
)

print(
    "Previous non-strict results    = NOT USED"
)


print()
print("OUTPUTS:")

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
print("=" * 76)
print("DONE")
print("=" * 76)