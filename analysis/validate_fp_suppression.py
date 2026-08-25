# -*- coding: utf-8 -*-

"""
Validation-based FP suppression threshold selection.

IMPORTANT:
- Thresholds are selected ONLY on validation data.
- Test data is NOT used for threshold selection.
- The selected thresholds are saved for later final test evaluation.
"""

from pathlib import Path
import json

import numpy as np


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]

RESULTS_DIR = PROJECT_DIR / "results"

PROBABILITY_PATH = RESULTS_DIR / "val_window_probabilities.npz"

OUTPUT_JSON = RESULTS_DIR / "fp_suppression_thresholds.json"


# ============================================================
# SETTINGS
# ============================================================

BASE_PROBABILITY_THRESHOLD = 0.50

# We want high sensitivity for seizure detection.
MIN_RECALL = 0.95

# Candidate probability thresholds.
PROBABILITY_THRESHOLDS = [
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

# Percentiles used to derive artifact thresholds.
PERCENTILES = [
    50,
    60,
    70,
    75,
    80,
    85,
    90,
    95,
]


# ============================================================
# FEATURE NAMES
# ============================================================

FEATURE_NAMES = [
    "mean_high_frequency_ratio",
    "mean_zero_crossing_rate",
    "mean_beta_power",
    "mean_gamma_power",
    "mean_line_length",
]


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(labels, predictions):

    labels = np.asarray(labels).astype(int)
    predictions = np.asarray(predictions).astype(int)

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

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0.0
    )

    recall = (
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
        2 * precision * recall /
        (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "f1": float(f1),
    }


# ============================================================
# LOAD VALIDATION RESULTS
# ============================================================

print("=" * 70)
print("VALIDATION-BASED FP SUPPRESSION THRESHOLD SELECTION")
print("=" * 70)

print("\nLoading validation predictions...")

if not PROBABILITY_PATH.exists():

    raise FileNotFoundError(
        f"""
Validation probability file not found:

{PROBABILITY_PATH}

Expected file:
results/val_window_probabilities.npz

The validation prediction file must contain:
- labels
- probabilities
- feature arrays
- optionally patients
"""
    )


data = np.load(
    PROBABILITY_PATH,
    allow_pickle=True
)


print("\nAvailable arrays:")

for key in data.files:

    arr = np.asarray(
        data[key]
    )

    print(
        f"  {key}: shape={arr.shape}"
    )


# ============================================================
# REQUIRED ARRAYS
# ============================================================

required = [
    "labels",
    "probabilities",
]

for key in required:

    if key not in data.files:

        raise KeyError(
            f"Required array '{key}' "
            f"not found in validation file."
        )


labels = np.asarray(
    data["labels"]
).reshape(-1)

probabilities = np.asarray(
    data["probabilities"]
).reshape(-1)


if len(labels) != len(probabilities):

    raise ValueError(
        "labels and probabilities "
        "have different lengths."
    )


N = len(labels)


print("\nNumber of validation windows:", N)


# ============================================================
# LOAD FEATURES
# ============================================================

feature_arrays = {}

for feature_name in FEATURE_NAMES:

    if feature_name in data.files:

        feature_arrays[
            feature_name
        ] = np.asarray(
            data[feature_name]
        ).reshape(-1)

    else:

        print(
            f"\nWARNING:"
            f" '{feature_name}' "
            f"not found directly in NPZ."
        )


# ------------------------------------------------------------
# Support alternative naming
# ------------------------------------------------------------

alternative_names = {
    "mean_high_frequency_ratio":
        "high_frequency_ratio",

    "mean_zero_crossing_rate":
        "zero_crossing_rate",

    "mean_beta_power":
        "beta_power",

    "mean_gamma_power":
        "gamma_power",

    "mean_line_length":
        "line_length",
}


for feature_name in FEATURE_NAMES:

    if feature_name in feature_arrays:
        continue

    alternative = alternative_names[
        feature_name
    ]

    if alternative in data.files:

        feature_arrays[
            feature_name
        ] = np.asarray(
            data[alternative]
        ).reshape(-1)

        print(
            f"Using '{alternative}' "
            f"for '{feature_name}'."
        )


# ============================================================
# VALIDATE FEATURE LENGTHS
# ============================================================

for feature_name, values in feature_arrays.items():

    if len(values) != N:

        raise ValueError(
            f"Feature {feature_name} "
            f"has length {len(values)}, "
            f"expected {N}."
        )

    if not np.all(
        np.isfinite(values)
    ):

        raise ValueError(
            f"Feature {feature_name} "
            f"contains NaN/Inf."
        )


# ============================================================
# BASELINE
# ============================================================

baseline_predictions = (
    probabilities >=
    BASE_PROBABILITY_THRESHOLD
).astype(int)


baseline_metrics = calculate_metrics(
    labels,
    baseline_predictions
)


print("\n" + "=" * 70)
print("VALIDATION BASELINE")
print("=" * 70)

print(
    f"Threshold = "
    f"{BASE_PROBABILITY_THRESHOLD:.2f}"
)

print(
    f"TP={baseline_metrics['tp']} | "
    f"FP={baseline_metrics['fp']} | "
    f"FN={baseline_metrics['fn']} | "
    f"TN={baseline_metrics['tn']}"
)

print(
    f"Precision="
    f"{baseline_metrics['precision']:.4f} | "
    f"Recall="
    f"{baseline_metrics['recall']:.4f} | "
    f"F1="
    f"{baseline_metrics['f1']:.4f} | "
    f"Specificity="
    f"{baseline_metrics['specificity']:.4f}"
)


# ============================================================
# PROBABILITY THRESHOLD SWEEP
# ============================================================

print("\n" + "=" * 70)
print("PROBABILITY THRESHOLD SWEEP")
print("=" * 70)


probability_results = []


for threshold in PROBABILITY_THRESHOLDS:

    predictions = (
        probabilities >= threshold
    ).astype(int)

    metrics = calculate_metrics(
        labels,
        predictions
    )

    metrics["threshold"] = threshold

    probability_results.append(
        metrics
    )

    print(
        f"Threshold={threshold:.2f} | "
        f"FP={metrics['fp']:4d} | "
        f"FN={metrics['fn']:3d} | "
        f"TP={metrics['tp']:3d} | "
        f"Precision={metrics['precision']:.4f} | "
        f"Recall={metrics['recall']:.4f} | "
        f"F1={metrics['f1']:.4f}"
    )


# ============================================================
# SELECT PROBABILITY THRESHOLD
# ============================================================

valid_probability_results = [
    result
    for result in probability_results
    if result["recall"] >= MIN_RECALL
]


if not valid_probability_results:

    raise RuntimeError(
        "No probability threshold satisfies "
        f"minimum recall={MIN_RECALL:.3f}."
    )


best_probability = max(
    valid_probability_results,
    key=lambda x: (
        x["f1"],
        -x["fp"]
    )
)


print("\n" + "=" * 70)
print("SELECTED PROBABILITY THRESHOLD")
print("=" * 70)

print(
    f"Selected threshold: "
    f"{best_probability['threshold']:.2f}"
)

print(
    f"TP={best_probability['tp']} | "
    f"FP={best_probability['fp']} | "
    f"FN={best_probability['fn']}"
)

print(
    f"Recall="
    f"{best_probability['recall']:.4f} | "
    f"Precision="
    f"{best_probability['precision']:.4f} | "
    f"F1="
    f"{best_probability['f1']:.4f}"
)


# ============================================================
# ARTIFACT THRESHOLD CANDIDATES
# ============================================================

print("\n" + "=" * 70)
print("ARTIFACT THRESHOLD CANDIDATES")
print("=" * 70)


artifact_thresholds = {}


for feature_name in FEATURE_NAMES:

    if feature_name not in feature_arrays:

        print(
            f"\nSkipping {feature_name}: "
            f"feature unavailable."
        )

        continue

    values = feature_arrays[
        feature_name
    ]

    artifact_thresholds[
        feature_name
    ] = {}

    print(
        f"\n{feature_name}"
    )

    for percentile in PERCENTILES:

        threshold = float(
            np.percentile(
                values,
                percentile
            )
        )

        artifact_thresholds[
            feature_name
        ][str(percentile)] = threshold

        print(
            f"  p{percentile}: "
            f"{threshold:.6f}"
        )


# ============================================================
# ARTIFACT SUPPRESSION
# ============================================================

print("\n" + "=" * 70)
print("ARTIFACT SUPPRESSION SEARCH")
print("=" * 70)

print(
    "\nIMPORTANT:"
)

print(
    "This is validation-only threshold selection."
)

print(
    "No test data is used here."
)


artifact_results = []


# ------------------------------------------------------------
# We test one feature at a time.
#
# Rule:
# A positive model prediction is suppressed when
# the artifact feature exceeds its threshold.
#
# Example:
#
# probability >= probability_threshold
# AND
# gamma_power < artifact_threshold
#
# ------------------------------------------------------------

for feature_name in FEATURE_NAMES:

    if feature_name not in feature_arrays:

        continue

    values = feature_arrays[
        feature_name
    ]

    for percentile in [
        75,
        80,
        85,
        90,
        95,
    ]:

        artifact_threshold = (
            artifact_thresholds[
                feature_name
            ][str(percentile)]
        )

        for probability_threshold in [
            0.50,
            0.55,
            0.60,
            0.65,
            0.70,
            0.75,
            0.80,
            0.85,
            0.90,
        ]:

            model_positive = (
                probabilities >=
                probability_threshold
            )

            suppression = (
                values >=
                artifact_threshold
            )

            final_predictions = (
                model_positive &
                ~suppression
            ).astype(int)

            metrics = calculate_metrics(
                labels,
                final_predictions
            )

            result = {
                "feature":
                    feature_name,

                "percentile":
                    percentile,

                "artifact_threshold":
                    float(
                        artifact_threshold
                    ),

                "probability_threshold":
                    float(
                        probability_threshold
                    ),

                **metrics,
            }

            # FP reduction relative to
            # the corresponding probability threshold
            original_predictions = (
                probabilities >=
                probability_threshold
            ).astype(int)

            original_metrics = calculate_metrics(
                labels,
                original_predictions
            )

            result[
                "fp_reduction"
            ] = int(
                original_metrics["fp"]
                - metrics["fp"]
            )

            result[
                "fp_reduction_percent"
            ] = float(
                (
                    result["fp_reduction"]
                    /
                    max(
                        1,
                        original_metrics["fp"]
                    )
                ) * 100.0
            )

            artifact_results.append(
                result
            )


# ============================================================
# FILTER SAFE CANDIDATES
# ============================================================

safe_candidates = [
    result
    for result in artifact_results
    if result["recall"] >= MIN_RECALL
]


print(
    f"\nTotal candidate combinations: "
    f"{len(artifact_results)}"
)

print(
    f"Candidates with Recall >= "
    f"{MIN_RECALL:.2f}: "
    f"{len(safe_candidates)}"
)


if len(safe_candidates) == 0:

    raise RuntimeError(
        "No artifact suppression candidate "
        "satisfied the minimum recall."
    )


# ============================================================
# RANK CANDIDATES
# ============================================================

# Priority:
# 1. highest F1
# 2. lowest FP
# 3. highest recall

safe_candidates_sorted = sorted(
    safe_candidates,
    key=lambda x: (
        x["f1"],
        -x["fp"],
        x["recall"],
    ),
    reverse=True,
)


print("\n" + "=" * 70)
print("TOP VALIDATION SUPPRESSION CANDIDATES")
print("=" * 70)


for rank, result in enumerate(
    safe_candidates_sorted[:15],
    start=1
):

    print(
        f"{rank:2d}. "
        f"{result['feature']:30s} "
        f"p{result['percentile']} "
        f"Prob={result['probability_threshold']:.2f} | "
        f"FP={result['fp']:4d} | "
        f"FN={result['fn']:3d} | "
        f"TP={result['tp']:3d} | "
        f"Recall={result['recall']:.4f} | "
        f"Precision={result['precision']:.4f} | "
        f"F1={result['f1']:.4f} | "
        f"FP reduction="
        f"{result['fp_reduction_percent']:.1f}%"
    )


# ============================================================
# SELECT FINAL EXPLORATORY CANDIDATE
# ============================================================

best_suppression = (
    safe_candidates_sorted[0]
)


print("\n" + "=" * 70)
print("SELECTED VALIDATION SUPPRESSION")
print("=" * 70)

print(
    f"Feature: "
    f"{best_suppression['feature']}"
)

print(
    f"Percentile: "
    f"p{best_suppression['percentile']}"
)

print(
    f"Artifact threshold: "
    f"{best_suppression['artifact_threshold']:.6f}"
)

print(
    f"Probability threshold: "
    f"{best_suppression['probability_threshold']:.2f}"
)

print(
    f"TP={best_suppression['tp']} | "
    f"FP={best_suppression['fp']} | "
    f"FN={best_suppression['fn']} | "
    f"TN={best_suppression['tn']}"
)

print(
    f"Precision="
    f"{best_suppression['precision']:.4f}"
)

print(
    f"Recall="
    f"{best_suppression['recall']:.4f}"
)

print(
    f"Specificity="
    f"{best_suppression['specificity']:.4f}"
)

print(
    f"F1="
    f"{best_suppression['f1']:.4f}"
)

print(
    f"FP reduction="
    f"{best_suppression['fp_reduction_percent']:.2f}%"
)


# ============================================================
# SAVE CONFIGURATION
# ============================================================

output = {

    "description":
        "Thresholds selected using validation data only.",

    "warning":
        "Do not use these results as final test performance.",

    "minimum_recall":
        MIN_RECALL,

    "baseline": baseline_metrics,

    "selected_probability_threshold":
        best_probability,

    "selected_suppression":
        best_suppression,

    "artifact_thresholds":
        artifact_thresholds,

    "top_candidates":
        safe_candidates_sorted[:20],
}


with open(
    OUTPUT_JSON,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        output,
        f,
        indent=4
    )


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 70)
print("VALIDATION ANALYSIS COMPLETED")
print("=" * 70)

print(
    "\nThreshold configuration saved to:"
)

print(
    OUTPUT_JSON
)

print(
    "\nIMPORTANT:"
)

print(
    "The test set has NOT been used "
    "to select these thresholds."
)

print(
    "\nNext step:"
)

print(
    "Apply the selected configuration "
    "ONCE to the test set and compare "
    "baseline vs suppression."
)

print("\nDONE")