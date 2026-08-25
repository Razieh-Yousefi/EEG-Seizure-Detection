# -*- coding: utf-8 -*-

"""
False Positive Analysis and Post-Processing
CHB-MIT EEG Seizure Detection Project

هدف:
1. بررسی توزیع FPها
2. بررسی confidence مدل برای FPها
3. آزمایش thresholdهای مختلف
4. آزمایش minimum consecutive positive windows
5. مقایسه Precision / Recall / F1 / Sensitivity / Specificity
"""

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]

RESULTS_DIR = PROJECT_DIR / "results"

OUTPUT_FILE = RESULTS_DIR / "false_positive_analysis_results.json"

# فایل احتمالاتی که قبلاً در پروژه تولید کرده‌ایم
PROBABILITY_FILE = RESULTS_DIR / "test_window_probabilities.npz"


# ============================================================
# 2. SETTINGS
# ============================================================

# Thresholdهای مختلف برای بررسی
THRESHOLDS = [
    0.30,
    0.40,
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

# حداقل تعداد پنجره مثبت پشت سر هم
MIN_CONSECUTIVE_WINDOWS = [
    1,
    2,
    3,
    4,
    5,
]


# ============================================================
# 3. LOAD PROBABILITY DATA
# ============================================================

print("=" * 70)
print("FALSE POSITIVE ANALYSIS")
print("=" * 70)

print("\nProject directory:")
print(PROJECT_DIR)

print("\nProbability file:")
print(PROBABILITY_FILE)


if not PROBABILITY_FILE.exists():
    raise FileNotFoundError(
        f"\nProbability file not found:\n{PROBABILITY_FILE}\n\n"
        "Make sure test_window_probabilities.npz exists in results/."
    )


data = np.load(PROBABILITY_FILE)

print("\nAvailable arrays:")
for key in data.files:
    print(f" - {key}")


# ============================================================
# 4. DETECT ARRAY NAMES AUTOMATICALLY
# ============================================================

def find_array(keys, candidates):
    for candidate in candidates:
        if candidate in keys:
            return data[candidate]

    return None


probabilities = find_array(
    data.files,
    [
        "probabilities",
        "probs",
        "y_prob",
        "test_probabilities",
        "positive_probabilities",
    ],
)

labels = find_array(
    data.files,
    [
        "labels",
        "y_true",
        "y_test",
        "test_labels",
    ],
)


if probabilities is None:
    raise ValueError(
        "\nCould not find probability array in NPZ file.\n"
        f"Available arrays: {data.files}"
    )

if labels is None:
    raise ValueError(
        "\nCould not find label array in NPZ file.\n"
        f"Available arrays: {data.files}"
    )


probabilities = np.asarray(probabilities).reshape(-1)
labels = np.asarray(labels).reshape(-1)


if len(probabilities) != len(labels):
    raise ValueError(
        f"\nLength mismatch:\n"
        f"Probabilities: {len(probabilities)}\n"
        f"Labels: {len(labels)}"
    )


print("\nTotal samples:", len(labels))
print("Seizure samples:", int(np.sum(labels == 1)))
print("Non-seizure samples:", int(np.sum(labels == 0)))


# ============================================================
# 5. BASIC MODEL PERFORMANCE
# ============================================================

print("\n" + "=" * 70)
print("ROC-AUC / PR-AUC")
print("=" * 70)

roc_auc = roc_auc_score(labels, probabilities)
pr_auc = average_precision_score(labels, probabilities)

print(f"ROC-AUC: {roc_auc:.6f}")
print(f"PR-AUC : {pr_auc:.6f}")


# ============================================================
# 6. THRESHOLD ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("THRESHOLD ANALYSIS")
print("=" * 70)

threshold_results = []


for threshold in THRESHOLDS:

    predictions = (probabilities >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        labels,
        predictions,
        labels=[0, 1]
    ).ravel()

    precision = precision_score(
        labels,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        labels,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        labels,
        predictions,
        zero_division=0
    )

    sensitivity = recall

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0
    )

    accuracy = (
        (tp + tn) / (tp + tn + fp + fn)
    )

    result = {
        "threshold": threshold,
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }

    threshold_results.append(result)

    print(
        f"\nThreshold = {threshold:.2f}"
    )

    print(
        f"Precision   : {precision:.4f}"
    )

    print(
        f"Recall      : {recall:.4f}"
    )

    print(
        f"F1          : {f1:.4f}"
    )

    print(
        f"Sensitivity : {sensitivity:.4f}"
    )

    print(
        f"Specificity : {specificity:.4f}"
    )

    print(
        f"FP={fp} | FN={fn} | TP={tp} | TN={tn}"
    )


# ============================================================
# 7. BEST THRESHOLDS
# ============================================================

best_f1 = max(
    threshold_results,
    key=lambda x: x["f1"]
)

best_precision = max(
    threshold_results,
    key=lambda x: x["precision"]
)

# برای پروژه تشخیص تشنج، نمی‌خواهیم فقط Precision را زیاد کنیم
# چون ممکن است Sensitivity شدیداً افت کند.
#
# بنابراین یک معیار متعادل نیز تعریف می‌کنیم:
# Sensitivity >= 0.90
# سپس کمترین FP را انتخاب می‌کنیم.

high_sensitivity_candidates = [
    r for r in threshold_results
    if r["sensitivity"] >= 0.90
]


if high_sensitivity_candidates:

    best_low_fp_high_sensitivity = min(
        high_sensitivity_candidates,
        key=lambda x: x["fp"]
    )

else:

    best_low_fp_high_sensitivity = None


print("\n" + "=" * 70)
print("BEST THRESHOLD RESULTS")
print("=" * 70)

print("\nBest F1 threshold:")
print(best_f1)

print("\nBest precision threshold:")
print(best_precision)

print("\nBest low-FP threshold with sensitivity >= 0.90:")

if best_low_fp_high_sensitivity:
    print(best_low_fp_high_sensitivity)
else:
    print("No threshold satisfies sensitivity >= 0.90")


# ============================================================
# 8. FALSE POSITIVE CONFIDENCE ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("FALSE POSITIVE CONFIDENCE ANALYSIS")
print("=" * 70)

analysis_threshold = 0.50

predictions = (
    probabilities >= analysis_threshold
).astype(int)

fp_mask = (
    (labels == 0) &
    (predictions == 1)
)

tp_mask = (
    (labels == 1) &
    (predictions == 1)
)

fp_probabilities = probabilities[fp_mask]
tp_probabilities = probabilities[tp_mask]


print("\nThreshold:", analysis_threshold)

print("False positives:", len(fp_probabilities))
print("True positives :", len(tp_probabilities))


if len(fp_probabilities) > 0:

    print(
        "\nFP probability statistics:"
    )

    print(
        "Mean   :",
        float(np.mean(fp_probabilities))
    )

    print(
        "Median :",
        float(np.median(fp_probabilities))
    )

    print(
        "Min    :",
        float(np.min(fp_probabilities))
    )

    print(
        "Max    :",
        float(np.max(fp_probabilities))
    )


# ============================================================
# 9. FP CONFIDENCE BINS
# ============================================================

confidence_bins = [
    (0.50, 0.60),
    (0.60, 0.70),
    (0.70, 0.80),
    (0.80, 0.90),
    (0.90, 1.00),
]


fp_confidence_distribution = []


print("\nFP confidence distribution:")


for low, high in confidence_bins:

    mask = (
        fp_probabilities >= low
    ) & (
        fp_probabilities < high
    )

    count = int(np.sum(mask))

    result = {
        "lower": low,
        "upper": high,
        "count": count,
    }

    fp_confidence_distribution.append(result)

    print(
        f"{low:.2f} - {high:.2f}: {count}"
    )


# ============================================================
# 10. CONSECUTIVE WINDOW POST-PROCESSING
# ============================================================

print("\n" + "=" * 70)
print("CONSECUTIVE WINDOW POST-PROCESSING")
print("=" * 70)

print(
    "\nWARNING:"
)

print(
    "This analysis assumes the samples are temporally ordered."
)

print(
    "If test_window_probabilities.npz does not preserve temporal order,"
)

print(
    "the consecutive-window results should NOT be used."
)


def consecutive_filter(predictions, minimum_windows):
    """
    Keep positive predictions only if they belong
    to a run of at least minimum_windows consecutive positives.
    """

    predictions = np.asarray(predictions).astype(int)

    output = np.zeros_like(predictions)

    start = None

    for i, value in enumerate(predictions):

        if value == 1:

            if start is None:
                start = i

        else:

            if start is not None:

                run_length = i - start

                if run_length >= minimum_windows:
                    output[start:i] = 1

                start = None

    # Handle run extending to final sample
    if start is not None:

        run_length = len(predictions) - start

        if run_length >= minimum_windows:
            output[start:] = 1

    return output


consecutive_results = []

base_threshold = 0.50

base_predictions = (
    probabilities >= base_threshold
).astype(int)


for minimum_windows in MIN_CONSECUTIVE_WINDOWS:

    filtered_predictions = consecutive_filter(
        base_predictions,
        minimum_windows
    )

    tn, fp, fn, tp = confusion_matrix(
        labels,
        filtered_predictions,
        labels=[0, 1]
    ).ravel()

    precision = precision_score(
        labels,
        filtered_predictions,
        zero_division=0
    )

    recall = recall_score(
        labels,
        filtered_predictions,
        zero_division=0
    )

    f1 = f1_score(
        labels,
        filtered_predictions,
        zero_division=0
    )

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0
    )

    result = {
        "minimum_consecutive_windows": minimum_windows,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "sensitivity": float(recall),
        "specificity": float(specificity),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }

    consecutive_results.append(result)

    print(
        f"\nMinimum consecutive windows: "
        f"{minimum_windows}"
    )

    print(
        f"FP={fp} | FN={fn} | "
        f"TP={tp} | TN={tn}"
    )

    print(
        f"Precision={precision:.4f} | "
        f"Recall={recall:.4f} | "
        f"F1={f1:.4f} | "
        f"Specificity={specificity:.4f}"
    )


# ============================================================
# 11. SAVE RESULTS
# ============================================================

output = {
    "project": "CHB-MIT EEG Seizure Detection",

    "dataset": {
        "total_samples": int(len(labels)),
        "seizure_samples": int(np.sum(labels == 1)),
        "non_seizure_samples": int(np.sum(labels == 0)),
    },

    "auc": {
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
    },

    "threshold_analysis": threshold_results,

    "best_f1_threshold": best_f1,

    "best_precision_threshold": best_precision,

    "best_low_fp_high_sensitivity": (
        best_low_fp_high_sensitivity
    ),

    "false_positive_confidence": {
        "threshold": analysis_threshold,
        "count": int(len(fp_probabilities)),
        "mean_probability": (
            float(np.mean(fp_probabilities))
            if len(fp_probabilities) > 0
            else None
        ),
        "median_probability": (
            float(np.median(fp_probabilities))
            if len(fp_probabilities) > 0
            else None
        ),
        "distribution": fp_confidence_distribution,
    },

    "consecutive_window_analysis": {
        "base_threshold": base_threshold,
        "results": consecutive_results,
    },
}


RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        output,
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

print("\nDONE")