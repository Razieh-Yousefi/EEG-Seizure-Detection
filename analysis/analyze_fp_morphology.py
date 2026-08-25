# -*- coding: utf-8 -*-

"""
False Positive Morphology Analysis

This script compares EEG windows that were classified as:
    1. True Positives (TP)
    2. False Positives (FP)
    3. False Negatives (FN)

The goal is to identify signal characteristics that may cause
the model to confuse normal EEG activity with seizure activity.
"""

from pathlib import Path
import json

import numpy as np


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]

RESULTS_DIR = PROJECT_DIR / "results"

PROBABILITY_FILE = (
    RESULTS_DIR / "test_window_probabilities.npz"
)

OUTPUT_FILE = (
    RESULTS_DIR / "fp_morphology_analysis.json"
)


# ============================================================
# SETTINGS
# ============================================================

THRESHOLD = 0.50

# Number of examples to inspect from each category
MAX_EXAMPLES = 20


# ============================================================
# LOAD TEST RESULTS
# ============================================================

print("=" * 70)
print("FALSE POSITIVE MORPHOLOGY ANALYSIS")
print("=" * 70)

print("\nLoading:")
print(PROBABILITY_FILE)


if not PROBABILITY_FILE.exists():
    raise FileNotFoundError(
        f"File not found:\n{PROBABILITY_FILE}"
    )


data = np.load(
    PROBABILITY_FILE,
    allow_pickle=True
)


print("\nAvailable arrays:")

for key in data.files:
    print(f" - {key}")


# ============================================================
# LOAD REQUIRED ARRAYS
# ============================================================

labels = np.asarray(
    data["labels"]
).reshape(-1)

probabilities = np.asarray(
    data["probabilities"]
).reshape(-1)

test_indices = np.asarray(
    data["test_indices"]
)

patients = np.asarray(
    data["patients"]
)


# ============================================================
# VALIDATION
# ============================================================

n = len(labels)

if len(probabilities) != n:
    raise ValueError(
        "labels and probabilities have different lengths."
    )

if len(patients) != n:
    raise ValueError(
        "labels and patients have different lengths."
    )

if len(test_indices) != n:
    raise ValueError(
        "labels and test_indices have different lengths."
    )


print("\nNumber of test windows:", n)


# ============================================================
# CLASSIFICATION
# ============================================================

predictions = (
    probabilities >= THRESHOLD
).astype(int)


tp_mask = (
    (labels == 1) &
    (predictions == 1)
)

fp_mask = (
    (labels == 0) &
    (predictions == 1)
)

tn_mask = (
    (labels == 0) &
    (predictions == 0)
)

fn_mask = (
    (labels == 1) &
    (predictions == 0)
)


print("\n" + "=" * 70)
print("CLASS DISTRIBUTION")
print("=" * 70)

print(f"TP: {np.sum(tp_mask)}")
print(f"FP: {np.sum(fp_mask)}")
print(f"TN: {np.sum(tn_mask)}")
print(f"FN: {np.sum(fn_mask)}")


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def safe_float(value):
    """
    Convert a NumPy scalar into a Python float.
    """
    return float(value)


def summarize_probabilities(values):
    """
    Calculate probability statistics.
    """

    if len(values) == 0:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "std": None,
            "min": None,
            "max": None,
        }

    return {
        "count": int(len(values)),
        "mean": safe_float(np.mean(values)),
        "median": safe_float(np.median(values)),
        "std": safe_float(np.std(values)),
        "min": safe_float(np.min(values)),
        "max": safe_float(np.max(values)),
    }


# ============================================================
# PROBABILITY COMPARISON
# ============================================================

tp_probabilities = probabilities[tp_mask]
fp_probabilities = probabilities[fp_mask]
fn_probabilities = probabilities[fn_mask]


print("\n" + "=" * 70)
print("PROBABILITY COMPARISON")
print("=" * 70)

print("\nTrue Positives:")
print(
    summarize_probabilities(
        tp_probabilities
    )
)

print("\nFalse Positives:")
print(
    summarize_probabilities(
        fp_probabilities
    )
)

print("\nFalse Negatives:")
print(
    summarize_probabilities(
        fn_probabilities
    )
)


# ============================================================
# PATIENT-LEVEL FP ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("PATIENT-LEVEL FALSE POSITIVE ANALYSIS")
print("=" * 70)


fp_patients = patients[fp_mask]

unique_patients, fp_counts = np.unique(
    fp_patients,
    return_counts=True
)


patient_fp_results = []


for patient, count in sorted(
    zip(unique_patients, fp_counts),
    key=lambda x: x[1],
    reverse=True
):

    patient_mask = (
        fp_mask &
        (patients == patient)
    )

    patient_probabilities = (
        probabilities[patient_mask]
    )

    patient_fp_results.append(
        {
            "patient": str(patient),
            "fp_count": int(count),
            "mean_probability": safe_float(
                np.mean(
                    patient_probabilities
                )
            ),
            "max_probability": safe_float(
                np.max(
                    patient_probabilities
                )
            ),
        }
    )


for item in patient_fp_results:

    print(
        f"{item['patient']}: "
        f"FP={item['fp_count']} | "
        f"mean_prob={item['mean_probability']:.4f} | "
        f"max_prob={item['max_probability']:.4f}"
    )


# ============================================================
# PATIENT-LEVEL FP CONCENTRATION
# ============================================================

total_fp = int(np.sum(fp_mask))

top_5_fp = patient_fp_results[:5]

top_5_count = sum(
    item["fp_count"]
    for item in top_5_fp
)


if total_fp > 0:

    top_5_percentage = (
        top_5_count /
        total_fp *
        100
    )

else:

    top_5_percentage = 0.0


print("\nTop 5 patients account for:")
print(
    f"{top_5_count}/{total_fp} "
    f"FPs ({top_5_percentage:.2f}%)"
)


# ============================================================
# TEMPORAL INDEX ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("TEST INDEX ANALYSIS")
print("=" * 70)


def summarize_indices(indices):
    """
    Summarize the numerical distribution of test indices.
    """

    indices = np.asarray(indices)

    if len(indices) == 0:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
        }

    return {
        "count": int(len(indices)),
        "min": safe_float(np.min(indices)),
        "max": safe_float(np.max(indices)),
        "mean": safe_float(np.mean(indices)),
        "median": safe_float(np.median(indices)),
    }


tp_indices = test_indices[tp_mask]
fp_indices = test_indices[fp_mask]
fn_indices = test_indices[fn_mask]


print("\nTP indices:")
print(
    summarize_indices(
        tp_indices
    )
)

print("\nFP indices:")
print(
    summarize_indices(
        fp_indices
    )
)

print("\nFN indices:")
print(
    summarize_indices(
        fn_indices
    )
)


# ============================================================
# HIGH-CONFIDENCE FALSE POSITIVES
# ============================================================

print("\n" + "=" * 70)
print("HIGH-CONFIDENCE FALSE POSITIVES")
print("=" * 70)


high_confidence_threshold = 0.90

high_fp_mask = (
    fp_mask &
    (probabilities >= high_confidence_threshold)
)


high_fp_indices = np.where(
    high_fp_mask
)[0]


print(
    f"\nFPs with probability >= "
    f"{high_confidence_threshold}: "
    f"{len(high_fp_indices)}"
)


high_fp_examples = []


for index in high_fp_indices[
    :MAX_EXAMPLES
]:

    high_fp_examples.append(
        {
            "array_index": int(index),
            "patient": str(
                patients[index]
            ),
            "probability": safe_float(
                probabilities[index]
            ),
            "label": int(
                labels[index]
            ),
            "test_index": (
                test_indices[index].tolist()
                if hasattr(
                    test_indices[index],
                    "tolist"
                )
                else test_indices[index]
            ),
        }
    )


for example in high_fp_examples:

    print(
        f"Index={example['array_index']} | "
        f"Patient={example['patient']} | "
        f"Probability={example['probability']:.4f} | "
        f"TestIndex={example['test_index']}"
    )


# ============================================================
# SAVE RESULTS
# ============================================================

results = {

    "settings": {
        "threshold": THRESHOLD,
        "high_confidence_threshold":
            high_confidence_threshold,
        "max_examples":
            MAX_EXAMPLES,
    },

    "dataset": {
        "total_windows": int(n),
        "tp": int(np.sum(tp_mask)),
        "fp": int(np.sum(fp_mask)),
        "tn": int(np.sum(tn_mask)),
        "fn": int(np.sum(fn_mask)),
    },

    "probability_statistics": {

        "true_positive":
            summarize_probabilities(
                tp_probabilities
            ),

        "false_positive":
            summarize_probabilities(
                fp_probabilities
            ),

        "false_negative":
            summarize_probabilities(
                fn_probabilities
            ),
    },

    "patient_fp_analysis":
        patient_fp_results,

    "top_5_patient_fp_percentage":
        float(top_5_percentage),

    "index_statistics": {

        "true_positive":
            summarize_indices(
                tp_indices
            ),

        "false_positive":
            summarize_indices(
                fp_indices
            ),

        "false_negative":
            summarize_indices(
                fn_indices
            ),
    },

    "high_confidence_fp": {
        "threshold":
            high_confidence_threshold,

        "count":
            int(len(high_fp_indices)),

        "examples":
            high_fp_examples,
    },
}


with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        results,
        f,
        indent=4,
        ensure_ascii=False
    )


print("\n" + "=" * 70)
print("ANALYSIS COMPLETED")
print("=" * 70)

print(
    "\nResults saved to:"
)

print(
    OUTPUT_FILE
)