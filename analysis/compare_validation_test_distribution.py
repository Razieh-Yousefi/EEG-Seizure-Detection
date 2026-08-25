# ================================================================
# compare_validation_test_distribution.py
#
# Compare validation and test probability distributions.
#
# PURPOSE:
#   - Compare model confidence between validation and test
#   - Compare seizure / non-seizure probabilities
#   - Compare FP / TP behavior
#   - Investigate validation-test distribution shift
#
# IMPORTANT:
#   - Does NOT modify model
#   - Does NOT modify dataset
#   - Does NOT change threshold
#   - Does NOT train anything
#   - Does NOT use validation to alter test predictions
# ================================================================

import os
import json
import numpy as np


# ================================================================
# 1. PATHS
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

TEST_FILE = os.path.join(
    RESULTS_DIR,
    "test_window_probabilities.npz"
)

VAL_FILE = os.path.join(
    RESULTS_DIR,
    "validation_window_probabilities.npz"
)

THRESHOLD_FILE = os.path.join(
    RESULTS_DIR,
    "validation_threshold_results.json"
)

OUTPUT_FILE = os.path.join(
    RESULTS_DIR,
    "validation_test_distribution_comparison.json"
)


# ================================================================
# 2. HEADER
# ================================================================

print()
print("=" * 70)
print("VALIDATION vs TEST PROBABILITY DISTRIBUTION ANALYSIS")
print("=" * 70)

print()
print("Project directory:")
print(PROJECT_DIR)

print()
print("Results directory:")
print(RESULTS_DIR)


# ================================================================
# 3. CHECK FILES
# ================================================================

print()
print("=" * 70)
print("1. CHECKING INPUT FILES")
print("=" * 70)

required_files = [
    TEST_FILE,
    VAL_FILE,
    THRESHOLD_FILE,
]

for path in required_files:

    if os.path.exists(path):
        print("[OK]", path)

    else:
        raise FileNotFoundError(path)


# ================================================================
# 4. LOAD DATA
# ================================================================

print()
print("=" * 70)
print("2. LOADING PROBABILITIES")
print("=" * 70)

test = np.load(
    TEST_FILE,
    allow_pickle=True
)

val = np.load(
    VAL_FILE,
    allow_pickle=True
)

test_prob = np.asarray(
    test["probabilities"],
    dtype=np.float64
)

test_labels = np.asarray(
    test["labels"],
    dtype=np.int64
)

test_patients = np.asarray(
    test["patients"]
)

val_prob = np.asarray(
    val["probabilities"],
    dtype=np.float64
)

val_labels = np.asarray(
    val["labels"],
    dtype=np.int64
)

val_patients = np.asarray(
    val["patients"]
)


print()
print("Validation samples:", len(val_prob))
print("Test samples:", len(test_prob))

print()
print("Validation labels:", len(val_labels))
print("Test labels:", len(test_labels))


# ================================================================
# 5. LOAD THRESHOLD
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


threshold = float(
    threshold_data["best_threshold"]
)

print()
print("Validation threshold:", threshold)


# ================================================================
# 6. VERIFY DATA
# ================================================================

print()
print("=" * 70)
print("4. VERIFYING DATA")
print("=" * 70)

if len(val_prob) != len(val_labels):
    raise RuntimeError(
        "Validation probability/label mismatch."
    )

if len(test_prob) != len(test_labels):
    raise RuntimeError(
        "Test probability/label mismatch."
    )

if not np.all(np.isfinite(val_prob)):
    raise RuntimeError(
        "Validation probabilities contain NaN/Inf."
    )

if not np.all(np.isfinite(test_prob)):
    raise RuntimeError(
        "Test probabilities contain NaN/Inf."
    )

print()
print("[OK] Validation arrays aligned.")
print("[OK] Test arrays aligned.")
print("[OK] All probabilities finite.")


# ================================================================
# 7. HELPER FUNCTIONS
# ================================================================

def describe(values):

    values = np.asarray(values, dtype=np.float64)

    if len(values) == 0:

        return {
            "count": 0,
            "mean": None,
            "median": None,
            "std": None,
            "min": None,
            "max": None,
            "q25": None,
            "q75": None,
            "q90": None,
            "q95": None,
            "q99": None,
        }

    return {
        "count": int(len(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "q25": float(np.percentile(values, 25)),
        "q75": float(np.percentile(values, 75)),
        "q90": float(np.percentile(values, 90)),
        "q95": float(np.percentile(values, 95)),
        "q99": float(np.percentile(values, 99)),
    }


def classification_stats(
    probabilities,
    labels,
    threshold
):

    predictions = (
        probabilities >= threshold
    ).astype(np.int64)

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
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "precision": float(precision),
        "f1": float(f1),
    }


# ================================================================
# 8. GLOBAL DISTRIBUTIONS
# ================================================================

print()
print("=" * 70)
print("5. GLOBAL PROBABILITY DISTRIBUTIONS")
print("=" * 70)

results = {

    "threshold": threshold,

    "validation": {
        "samples": int(len(val_prob)),
        "patients": int(len(np.unique(val_patients))),

        "probability_all":
            describe(val_prob),

        "probability_class_0":
            describe(
                val_prob[val_labels == 0]
            ),

        "probability_class_1":
            describe(
                val_prob[val_labels == 1]
            ),

        "classification":
            classification_stats(
                val_prob,
                val_labels,
                threshold
            ),
    },

    "test": {
        "samples": int(len(test_prob)),
        "patients": int(len(np.unique(test_patients))),

        "probability_all":
            describe(test_prob),

        "probability_class_0":
            describe(
                test_prob[test_labels == 0]
            ),

        "probability_class_1":
            describe(
                test_prob[test_labels == 1]
            ),

        "classification":
            classification_stats(
                test_prob,
                test_labels,
                threshold
            ),
    }
}


# ================================================================
# 9. PRINT GLOBAL RESULTS
# ================================================================

for name, probs, labels in [

    (
        "VALIDATION",
        val_prob,
        val_labels
    ),

    (
        "TEST",
        test_prob,
        test_labels
    ),

]:

    print()
    print("-" * 70)
    print(name)

    print()
    print("All probabilities:")
    print(
        "  mean   =",
        round(
            float(np.mean(probs)),
            6
        )
    )

    print(
        "  median =",
        round(
            float(np.median(probs)),
            6
        )
    )

    print(
        "  std    =",
        round(
            float(np.std(probs)),
            6
        )
    )

    print(
        "  q90    =",
        round(
            float(np.percentile(probs, 90)),
            6
        )
    )

    print(
        "  q95    =",
        round(
            float(np.percentile(probs, 95)),
            6
        )
    )

    print(
        "  q99    =",
        round(
            float(np.percentile(probs, 99)),
            6
        )
    )

    class0 = probs[labels == 0]
    class1 = probs[labels == 1]

    print()
    print("Class 0:")
    print(
        "  count  =",
        len(class0)
    )
    print(
        "  mean   =",
        round(
            float(np.mean(class0)),
            6
        )
    )
    print(
        "  median =",
        round(
            float(np.median(class0)),
            6
        )
    )

    print()
    print("Class 1:")
    print(
        "  count  =",
        len(class1)
    )
    print(
        "  mean   =",
        round(
            float(np.mean(class1)),
            6
        )
    )
    print(
        "  median =",
        round(
            float(np.median(class1)),
            6
        )
    )

    stats = results[
        name.lower()
    ]["classification"]

    print()
    print("Classification:")
    print(
        "  TP =",
        stats["tp"],
        "FP =",
        stats["fp"],
        "FN =",
        stats["fn"],
        "TN =",
        stats["tn"]
    )

    print(
        "  Sensitivity =",
        round(
            stats["sensitivity"],
            6
        )
    )

    print(
        "  Specificity =",
        round(
            stats["specificity"],
            6
        )
    )

    print(
        "  Precision   =",
        round(
            stats["precision"],
            6
        )
    )

    print(
        "  F1          =",
        round(
            stats["f1"],
            6
        )
    )


# ================================================================
# 10. FALSE POSITIVE ANALYSIS
# ================================================================

print()
print("=" * 70)
print("6. FALSE POSITIVE PROBABILITY ANALYSIS")
print("=" * 70)

val_pred = (
    val_prob >= threshold
).astype(np.int64)

test_pred = (
    test_prob >= threshold
).astype(np.int64)

val_fp_prob = val_prob[
    (val_labels == 0) &
    (val_pred == 1)
]

test_fp_prob = test_prob[
    (test_labels == 0) &
    (test_pred == 1)
]

val_tp_prob = val_prob[
    (val_labels == 1) &
    (val_pred == 1)
]

test_tp_prob = test_prob[
    (test_labels == 1) &
    (test_pred == 1)
]

results["validation"]["false_positive_probability"] = describe(
    val_fp_prob
)

results["test"]["false_positive_probability"] = describe(
    test_fp_prob
)

results["validation"]["true_positive_probability"] = describe(
    val_tp_prob
)

results["test"]["true_positive_probability"] = describe(
    test_tp_prob
)


print()
print("Validation FP probability:")
print(
    describe(val_fp_prob)
)

print()
print("Test FP probability:")
print(
    describe(test_fp_prob)
)

print()
print("Validation TP probability:")
print(
    describe(val_tp_prob)
)

print()
print("Test TP probability:")
print(
    describe(test_tp_prob)
)


# ================================================================
# 11. PROBABILITY BINS
# ================================================================

print()
print("=" * 70)
print("7. PROBABILITY BIN ANALYSIS")
print("=" * 70)

bins = [
    0.0,
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.56,
    0.6,
    0.7,
    0.8,
    0.9,
    0.95,
    0.99,
    1.0,
]

bin_results = {}

for i in range(len(bins) - 1):

    low = bins[i]
    high = bins[i + 1]

    val_mask = (
        (val_prob >= low) &
        (
            val_prob < high
            if high < 1.0
            else val_prob <= high
        )
    )

    test_mask = (
        (test_prob >= low) &
        (
            test_prob < high
            if high < 1.0
            else test_prob <= high
        )
    )

    val_count = int(
        np.sum(val_mask)
    )

    test_count = int(
        np.sum(test_mask)
    )

    bin_results[
        f"{low:.2f}-{high:.2f}"
    ] = {
        "validation_count": val_count,
        "test_count": test_count,
        "validation_fraction":
            float(
                val_count /
                len(val_prob)
            ),
        "test_fraction":
            float(
                test_count /
                len(test_prob)
            ),
    }


results["probability_bins"] = bin_results


print()
print(
    "BIN        | VALIDATION | TEST"
)
print("-" * 45)

for key, value in bin_results.items():

    print(
        f"{key:10s} | "
        f"{value['validation_count']:10d} | "
        f"{value['test_count']:4d}"
    )


# ================================================================
# 12. SUMMARY OF DISTRIBUTION SHIFT
# ================================================================

val_class0_mean = float(
    np.mean(
        val_prob[val_labels == 0]
    )
)

test_class0_mean = float(
    np.mean(
        test_prob[test_labels == 0]
    )
)

val_class1_mean = float(
    np.mean(
        val_prob[val_labels == 1]
    )
)

test_class1_mean = float(
    np.mean(
        test_prob[test_labels == 1]
    )
)

results["distribution_shift"] = {

    "class_0_mean_difference_test_minus_validation":
        test_class0_mean -
        val_class0_mean,

    "class_1_mean_difference_test_minus_validation":
        test_class1_mean -
        val_class1_mean,

    "overall_mean_difference_test_minus_validation":
        float(
            np.mean(test_prob) -
            np.mean(val_prob)
        ),

    "fp_mean_difference_test_minus_validation":
        (
            float(np.mean(test_fp_prob))
            -
            float(np.mean(val_fp_prob))
            if len(test_fp_prob) > 0
            and len(val_fp_prob) > 0
            else None
        ),

    "tp_mean_difference_test_minus_validation":
        (
            float(np.mean(test_tp_prob))
            -
            float(np.mean(val_tp_prob))
            if len(test_tp_prob) > 0
            and len(val_tp_prob) > 0
            else None
        ),
}


# ================================================================
# 13. PATIENT COUNTS
# ================================================================

print()
print("=" * 70)
print("8. PATIENT DISTRIBUTION")
print("=" * 70)

val_unique, val_counts = np.unique(
    val_patients,
    return_counts=True
)

test_unique, test_counts = np.unique(
    test_patients,
    return_counts=True
)

results["validation"]["patient_distribution"] = {
    str(patient): int(count)
    for patient, count
    in zip(
        val_unique,
        val_counts
    )
}

results["test"]["patient_distribution"] = {
    str(patient): int(count)
    for patient, count
    in zip(
        test_unique,
        test_counts
    )
}

print()
print(
    "Validation patients:",
    len(val_unique)
)

print(
    "Test patients:",
    len(test_unique)
)

print()
print(
    "Validation patient distribution:"
)

for patient, count in zip(
    val_unique,
    val_counts
):

    print(
        f"  {patient}: {count}"
    )

print()
print(
    "Test patient distribution:"
)

for patient, count in zip(
    test_unique,
    test_counts
):

    print(
        f"  {patient}: {count}"
    )


# ================================================================
# 14. FINAL INTERPRETATION
# ================================================================

val_stats = results[
    "validation"
]["classification"]

test_stats = results[
    "test"
]["classification"]

print()
print("=" * 70)
print("9. INTERPRETATION")
print("=" * 70)

print()

if test_stats["sensitivity"] > val_stats["sensitivity"]:

    print(
        "[OBSERVATION] Test sensitivity is higher "
        "than validation sensitivity."
    )

else:

    print(
        "[OBSERVATION] Validation sensitivity is "
        "higher than or equal to test sensitivity."
    )


if test_stats["precision"] < val_stats["precision"]:

    print(
        "[OBSERVATION] Test precision is lower "
        "than validation precision."
    )

else:

    print(
        "[OBSERVATION] Test precision is "
        "higher than or equal to validation precision."
    )


if (
    len(test_fp_prob) > 0 and
    len(val_fp_prob) > 0
):

    if (
        np.mean(test_fp_prob)
        >
        np.mean(val_fp_prob)
    ):

        print(
            "[OBSERVATION] Test false positives "
            "have higher average confidence."
        )

    else:

        print(
            "[OBSERVATION] Validation false positives "
            "have higher or equal average confidence."
        )


print()
print(
    "IMPORTANT:"
)

print(
    "This analysis does not modify the model, "
    "threshold, dataset, or test predictions."
)

print(
    "The purpose is to identify validation-test "
    "distribution differences before designing "
    "any post-processing filter."
)


# ================================================================
# 15. SAVE RESULTS
# ================================================================

print()
print("=" * 70)
print("10. SAVING RESULTS")
print("=" * 70)

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        results,
        f,
        indent=2
    )


print()
print("[OK] Saved:")
print(OUTPUT_FILE)

print()
print("=" * 70)
print("DISTRIBUTION ANALYSIS COMPLETED")
print("=" * 70)