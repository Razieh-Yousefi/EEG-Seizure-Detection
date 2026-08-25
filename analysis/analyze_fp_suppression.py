# -*- coding: utf-8 -*-

"""
Evaluate false-positive suppression strategies using
signal-derived artifact features.

This script does NOT modify the trained model.

It evaluates:
1. Baseline model predictions.
2. Probability thresholding.
3. High-frequency suppression.
4. Zero-crossing-rate suppression.
5. Beta/gamma suppression.
6. Combined artifact suppression.

Important:
Thresholds are evaluated on the available test predictions
only as an exploratory analysis. They must NOT be used as
final model-selection thresholds without validation-set tuning.
"""

from pathlib import Path
import json

import numpy as np


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_DIR / "data"
RESULTS_DIR = PROJECT_DIR / "results"

X_PATH = DATA_DIR / "X_chbmit_full.npy"
NORMALIZATION_PATH = DATA_DIR / "normalization_params.npz"
PROBABILITY_PATH = RESULTS_DIR / "test_window_probabilities.npz"

OUTPUT_JSON = RESULTS_DIR / "fp_suppression_analysis.json"


# ============================================================
# SETTINGS
# ============================================================

BASELINE_THRESHOLD = 0.50

HIGH_CONFIDENCE_THRESHOLD = 0.90

EXPECTED_CHANNELS = 23
EXPECTED_SAMPLES = 1280

SAMPLING_FREQUENCY = 256.0

EPSILON = 1e-12


# ============================================================
# FREQUENCY BANDS
# ============================================================

BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
    "high_frequency": (20.0, 45.0),
}


# ============================================================
# FEATURE FUNCTIONS
# ============================================================

def calculate_zero_crossing_rate(signal):
    """Calculate zero-crossing rate."""

    centered = signal - np.mean(signal)

    signs = np.sign(centered)

    crossings = np.sum(
        signs[:-1] != signs[1:]
    )

    return float(
        crossings /
        max(1, len(signal) - 1)
    )


def calculate_spectral_features(window):
    """
    Calculate frequency-domain features for all channels.

    Returns aggregate mean features across channels.
    """

    high_frequency_ratios = []
    beta_powers = []
    gamma_powers = []

    for channel in window:

        signal = channel.astype(
            np.float64,
            copy=False
        )

        signal = signal - np.mean(signal)

        power = np.abs(
            np.fft.rfft(signal)
        ) ** 2

        frequencies = np.fft.rfftfreq(
            len(signal),
            d=1.0 / SAMPLING_FREQUENCY
        )

        total_mask = (
            frequencies >= 0.5
        ) & (
            frequencies <= 45.0
        )

        beta_mask = (
            frequencies >= BANDS["beta"][0]
        ) & (
            frequencies < BANDS["beta"][1]
        )

        gamma_mask = (
            frequencies >= BANDS["gamma"][0]
        ) & (
            frequencies < BANDS["gamma"][1]
        )

        high_mask = (
            frequencies >= BANDS["high_frequency"][0]
        ) & (
            frequencies <= BANDS["high_frequency"][1]
        )

        total_power = np.sum(
            power[total_mask]
        )

        beta_power = np.sum(
            power[beta_mask]
        )

        gamma_power = np.sum(
            power[gamma_mask]
        )

        high_power = np.sum(
            power[high_mask]
        )

        if total_power <= EPSILON:
            total_power = EPSILON

        high_frequency_ratios.append(
            high_power / total_power
        )

        beta_powers.append(
            beta_power / total_power
        )

        gamma_powers.append(
            gamma_power / total_power
        )

    return {
        "mean_high_frequency_ratio":
            float(
                np.mean(
                    high_frequency_ratios
                )
            ),

        "max_high_frequency_ratio":
            float(
                np.max(
                    high_frequency_ratios
                )
            ),

        "mean_beta_power":
            float(
                np.mean(
                    beta_powers
                )
            ),

        "max_beta_power":
            float(
                np.max(
                    beta_powers
                )
            ),

        "mean_gamma_power":
            float(
                np.mean(
                    gamma_powers
                )
            ),

        "max_gamma_power":
            float(
                np.max(
                    gamma_powers
                )
            ),
    }


def calculate_morphology_features(window):
    """Calculate morphology-related features."""

    zcr_values = []
    line_lengths = []

    for channel in window:

        zcr_values.append(
            calculate_zero_crossing_rate(
                channel
            )
        )

        line_lengths.append(
            float(
                np.sum(
                    np.abs(
                        np.diff(channel)
                    )
                )
            )
        )

    return {
        "mean_zero_crossing_rate":
            float(
                np.mean(zcr_values)
            ),

        "max_zero_crossing_rate":
            float(
                np.max(zcr_values)
            ),

        "mean_line_length":
            float(
                np.mean(line_lengths)
            ),

        "max_line_length":
            float(
                np.max(line_lengths)
            ),
    }


def extract_features(window):
    """Extract all artifact-related features."""

    frequency_features = (
        calculate_spectral_features(
            window
        )
    )

    morphology_features = (
        calculate_morphology_features(
            window
        )
    )

    return {
        **frequency_features,
        **morphology_features,
    }


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("FALSE POSITIVE SUPPRESSION ANALYSIS")
print("=" * 70)

print("\nLoading files...")

if not X_PATH.exists():

    raise FileNotFoundError(
        f"X file not found:\n{X_PATH}"
    )

if not NORMALIZATION_PATH.exists():

    raise FileNotFoundError(
        "Normalization file not found:\n"
        f"{NORMALIZATION_PATH}"
    )

if not PROBABILITY_PATH.exists():

    raise FileNotFoundError(
        "Probability file not found:\n"
        f"{PROBABILITY_PATH}"
    )


X = np.load(
    X_PATH,
    mmap_mode="r"
)

normalization = np.load(
    NORMALIZATION_PATH
)

channel_mean = np.asarray(
    normalization["channel_mean"],
    dtype=np.float32
)

channel_std = np.asarray(
    normalization["channel_std"],
    dtype=np.float32
)

prediction_data = np.load(
    PROBABILITY_PATH,
    allow_pickle=True
)


# ============================================================
# VALIDATE DATA
# ============================================================

print(
    "\nX shape:",
    X.shape
)

if X.ndim != 3:

    raise ValueError(
        "X must be a 3D array."
    )

if X.shape[1] != EXPECTED_CHANNELS:

    raise ValueError(
        f"Expected {EXPECTED_CHANNELS} channels, "
        f"got {X.shape[1]}"
    )

if X.shape[2] != EXPECTED_SAMPLES:

    raise ValueError(
        f"Expected {EXPECTED_SAMPLES} samples, "
        f"got {X.shape[2]}"
    )


# ============================================================
# LOAD PREDICTIONS
# ============================================================

test_indices = np.asarray(
    prediction_data["test_indices"]
).reshape(-1)

patients = np.asarray(
    prediction_data["patients"]
).reshape(-1)

labels = np.asarray(
    prediction_data["labels"]
).reshape(-1)

probabilities = np.asarray(
    prediction_data["probabilities"]
).reshape(-1)


n = len(labels)

if len(test_indices) != n:
    raise ValueError(
        "test_indices length mismatch."
    )

if len(patients) != n:
    raise ValueError(
        "patients length mismatch."
    )

if len(probabilities) != n:
    raise ValueError(
        "probabilities length mismatch."
    )


print(
    "Number of test windows:",
    n
)


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def load_normalized_window(sample_index):
    """Load and normalize one EEG window."""

    sample_index = int(
        sample_index
    )

    raw_window = np.asarray(
        X[sample_index],
        dtype=np.float32
    )

    normalized_window = (
        raw_window -
        channel_mean[:, None]
    ) / channel_std[:, None]

    return normalized_window


print("\nExtracting signal features...")

feature_names = [
    "mean_high_frequency_ratio",
    "max_high_frequency_ratio",
    "mean_beta_power",
    "max_beta_power",
    "mean_gamma_power",
    "max_gamma_power",
    "mean_zero_crossing_rate",
    "max_zero_crossing_rate",
    "mean_line_length",
    "max_line_length",
]


feature_matrix = np.zeros(
    (n, len(feature_names)),
    dtype=np.float32
)


for i in range(n):

    if (i + 1) % 250 == 0:

        print(
            f"Processed {i + 1}/{n}"
        )

    window = load_normalized_window(
        test_indices[i]
    )

    features = extract_features(
        window
    )

    feature_matrix[i] = [
        features[name]
        for name in feature_names
    ]


feature_index = {
    name: i
    for i, name in enumerate(
        feature_names
    )
}


# ============================================================
# BASELINE METRICS
# ============================================================

def calculate_metrics(
    labels,
    predictions
):
    """Calculate binary classification metrics."""

    labels = np.asarray(
        labels
    ).astype(int)

    predictions = np.asarray(
        predictions
    ).astype(int)

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

    tn = int(
        np.sum(
            (labels == 0) &
            (predictions == 0)
        )
    )

    fn = int(
        np.sum(
            (labels == 1) &
            (predictions == 0)
        )
    )

    precision = (
        tp /
        max(1, tp + fp)
    )

    recall = (
        tp /
        max(1, tp + fn)
    )

    specificity = (
        tn /
        max(1, tn + fp)
    )

    f1 = (
        2.0 *
        precision *
        recall /
        max(
            EPSILON,
            precision + recall
        )
    )

    fpr = (
        fp /
        max(1, fp + tn)
    )

    accuracy = (
        (tp + tn) /
        max(1, tp + tn + fp + fn)
    )

    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": float(precision),
        "recall": float(recall),
        "sensitivity": float(recall),
        "specificity": float(specificity),
        "f1": float(f1),
        "fpr": float(fpr),
        "accuracy": float(accuracy),
    }


# ============================================================
# BASELINE
# ============================================================

baseline_predictions = (
    probabilities >= BASELINE_THRESHOLD
).astype(int)

baseline_metrics = calculate_metrics(
    labels,
    baseline_predictions
)


print("\n" + "=" * 70)
print("BASELINE")
print("=" * 70)

print(
    f"Threshold = {BASELINE_THRESHOLD:.2f}"
)

print(
    f"TP={baseline_metrics['tp']} | "
    f"FP={baseline_metrics['fp']} | "
    f"FN={baseline_metrics['fn']} | "
    f"TN={baseline_metrics['tn']}"
)

print(
    f"Precision={baseline_metrics['precision']:.4f} | "
    f"Recall={baseline_metrics['recall']:.4f} | "
    f"F1={baseline_metrics['f1']:.4f} | "
    f"Specificity={baseline_metrics['specificity']:.4f}"
)


# ============================================================
# THRESHOLD SWEEP
# ============================================================

print("\n" + "=" * 70)
print("PROBABILITY THRESHOLD SWEEP")
print("=" * 70)


threshold_results = []

for threshold in np.arange(
    0.50,
    0.951,
    0.05
):

    predictions = (
        probabilities >= threshold
    ).astype(int)

    metrics = calculate_metrics(
        labels,
        predictions
    )

    metrics["threshold"] = float(
        threshold
    )

    threshold_results.append(
        metrics
    )

    print(
        f"Threshold={threshold:.2f} | "
        f"FP={metrics['fp']:3d} | "
        f"FN={metrics['fn']:3d} | "
        f"TP={metrics['tp']:3d} | "
        f"Precision={metrics['precision']:.4f} | "
        f"Recall={metrics['recall']:.4f} | "
        f"F1={metrics['f1']:.4f}"
    )


# ============================================================
# ARTIFACT THRESHOLDS
# ============================================================

hf = feature_matrix[
    :,
    feature_index[
        "mean_high_frequency_ratio"
    ]
]

zcr = feature_matrix[
    :,
    feature_index[
        "mean_zero_crossing_rate"
    ]
]

beta = feature_matrix[
    :,
    feature_index[
        "mean_beta_power"
    ]
]

gamma = feature_matrix[
    :,
    feature_index[
        "mean_gamma_power"
    ]
]

line_length = feature_matrix[
    :,
    feature_index[
        "mean_line_length"
    ]
]


# ============================================================
# DATA-DRIVEN CANDIDATE THRESHOLDS
# ============================================================

print("\n" + "=" * 70)
print("ARTIFACT THRESHOLD CANDIDATES")
print("=" * 70)

print(
    "\nThese thresholds are exploratory."
)

print(
    "Final thresholds should be selected "
    "using validation data."
)


def percentile_values(values):

    return {
        "p50": float(
            np.percentile(values, 50)
        ),

        "p60": float(
            np.percentile(values, 60)
        ),

        "p70": float(
            np.percentile(values, 70)
        ),

        "p75": float(
            np.percentile(values, 75)
        ),

        "p80": float(
            np.percentile(values, 80)
        ),

        "p85": float(
            np.percentile(values, 85)
        ),

        "p90": float(
            np.percentile(values, 90)
        ),

        "p95": float(
            np.percentile(values, 95)
        ),
    }


candidate_thresholds = {
    "mean_high_frequency_ratio":
        percentile_values(hf),

    "mean_zero_crossing_rate":
        percentile_values(zcr),

    "mean_beta_power":
        percentile_values(beta),

    "mean_gamma_power":
        percentile_values(gamma),

    "mean_line_length":
        percentile_values(line_length),
}


for name, values in candidate_thresholds.items():

    print(
        f"\n{name}"
    )

    for percentile, value in values.items():

        print(
            f"  {percentile}: {value:.6f}"
        )


# ============================================================
# SINGLE-FEATURE SUPPRESSION
# ============================================================

print("\n" + "=" * 70)
print("SINGLE-FEATURE SUPPRESSION")
print("=" * 70)


single_feature_results = []


candidate_percentiles = [
    "p75",
    "p80",
    "p85",
    "p90",
    "p95",
]


def evaluate_suppression(
    feature_name,
    feature_values,
    threshold,
    base_predictions
):
    """
    Suppress positive predictions when an artifact
    feature exceeds a threshold.
    """

    predictions = base_predictions.copy()

    suppression_mask = (
        feature_values >= threshold
    )

    predictions[
        suppression_mask
    ] = 0

    metrics = calculate_metrics(
        labels,
        predictions
    )

    metrics.update({
        "feature": feature_name,
        "threshold": float(threshold),
        "suppressed_predictions": int(
            np.sum(
                base_predictions &
                suppression_mask
            )
        ),
    })

    return metrics


for feature_name, values in [
    (
        "mean_high_frequency_ratio",
        hf
    ),
    (
        "mean_zero_crossing_rate",
        zcr
    ),
    (
        "mean_beta_power",
        beta
    ),
    (
        "mean_gamma_power",
        gamma
    ),
    (
        "mean_line_length",
        line_length
    ),
]:

    for percentile in candidate_percentiles:

        threshold = candidate_thresholds[
            feature_name
        ][percentile]

        result = evaluate_suppression(
            feature_name,
            values,
            threshold,
            baseline_predictions
        )

        result["percentile"] = percentile

        single_feature_results.append(
            result
        )

        print(
            f"{feature_name:30s} "
            f"{percentile} | "
            f"FP={result['fp']:3d} | "
            f"FN={result['fn']:3d} | "
            f"TP={result['tp']:3d} | "
            f"Precision={result['precision']:.4f} | "
            f"Recall={result['recall']:.4f} | "
            f"F1={result['f1']:.4f}"
        )


# ============================================================
# COMBINED SUPPRESSION
# ============================================================

print("\n" + "=" * 70)
print("COMBINED ARTIFACT SUPPRESSION")
print("=" * 70)


combined_results = []


combined_percentiles = [
    ("p80", "p80"),
    ("p85", "p85"),
    ("p90", "p90"),
    ("p90", "p95"),
    ("p95", "p95"),
]


for hf_percentile, zcr_percentile in combined_percentiles:

    hf_threshold = candidate_thresholds[
        "mean_high_frequency_ratio"
    ][hf_percentile]

    zcr_threshold = candidate_thresholds[
        "mean_zero_crossing_rate"
    ][zcr_percentile]

    suppression_mask = (
        (hf >= hf_threshold) &
        (zcr >= zcr_threshold)
    )

    predictions = (
        baseline_predictions.copy()
    )

    predictions[
        suppression_mask
    ] = 0

    metrics = calculate_metrics(
        labels,
        predictions
    )

    metrics.update({
        "high_frequency_percentile":
            hf_percentile,

        "high_frequency_threshold":
            float(hf_threshold),

        "zcr_percentile":
            zcr_percentile,

        "zcr_threshold":
            float(zcr_threshold),

        "suppressed_predictions":
            int(
                np.sum(
                    baseline_predictions &
                    suppression_mask
                )
            ),
    })

    combined_results.append(
        metrics
    )

    print(
        f"HF={hf_percentile} "
        f"ZCR={zcr_percentile} | "
        f"FP={metrics['fp']:3d} | "
        f"FN={metrics['fn']:3d} | "
        f"TP={metrics['tp']:3d} | "
        f"Precision={metrics['precision']:.4f} | "
        f"Recall={metrics['recall']:.4f} | "
        f"F1={metrics['f1']:.4f}"
    )


# ============================================================
# HIGH-CONFIDENCE FP ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("HIGH-CONFIDENCE FP ARTIFACT PROFILE")
print("=" * 70)


high_confidence_fp_mask = (
    (labels == 0) &
    (probabilities >= HIGH_CONFIDENCE_THRESHOLD)
)

high_confidence_fp_count = int(
    np.sum(
        high_confidence_fp_mask
    )
)


print(
    "High-confidence FP count:",
    high_confidence_fp_count
)


high_confidence_profile = {}


for feature_name in feature_names:

    values = feature_matrix[
        high_confidence_fp_mask,
        feature_index[feature_name]
    ]

    if len(values) == 0:

        high_confidence_profile[
            feature_name
        ] = {
            "count": 0
        }

        continue

    high_confidence_profile[
        feature_name
    ] = {

        "count": int(
            len(values)
        ),

        "mean": float(
            np.mean(values)
        ),

        "median": float(
            np.median(values)
        ),

        "std": float(
            np.std(values)
        ),

        "p90": float(
            np.percentile(
                values,
                90
            )
        ),
    }

    print(
        f"{feature_name:30s} "
        f"mean={np.mean(values):.6f} "
        f"median={np.median(values):.6f}"
    )


# ============================================================
# PATIENT-LEVEL SUPPRESSION ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("PATIENT-LEVEL BASELINE")
print("=" * 70)


patient_results = {}


for patient in np.unique(
    patients
):

    mask = (
        patients == patient
    )

    patient_metrics = calculate_metrics(
        labels[mask],
        baseline_predictions[mask]
    )

    patient_results[
        str(patient)
    ] = patient_metrics

    print(
        f"{patient}: "
        f"TP={patient_metrics['tp']} | "
        f"FP={patient_metrics['fp']} | "
        f"FN={patient_metrics['fn']} | "
        f"TN={patient_metrics['tn']}"
    )


# ============================================================
# FIND PROMISING EXPLORATORY RESULTS
# ============================================================

print("\n" + "=" * 70)
print("PROMISING EXPLORATORY RESULTS")
print("=" * 70)


all_suppression_results = (
    single_feature_results +
    combined_results
)


promising = []

for result in all_suppression_results:

    recall = result["recall"]

    fp = result["fp"]

    if (
        recall >= 0.90 and
        fp < baseline_metrics["fp"]
    ):

        promising.append(
            result
        )


promising.sort(
    key=lambda x: (
        x["fp"],
        -x["recall"]
    )
)


if len(promising) == 0:

    print(
        "No suppression configuration "
        "reduced FP while maintaining "
        "recall >= 0.90."
    )

else:

    for result in promising[:10]:

        if "feature" in result:

            name = (
                f"{result['feature']} "
                f"{result['percentile']}"
            )

        else:

            name = (
                f"HF={result['high_frequency_percentile']} "
                f"ZCR={result['zcr_percentile']}"
            )

        print(
            f"{name}: "
            f"FP={result['fp']} | "
            f"FN={result['fn']} | "
            f"TP={result['tp']} | "
            f"Recall={result['recall']:.4f} | "
            f"Precision={result['precision']:.4f} | "
            f"F1={result['f1']:.4f}"
        )


# ============================================================
# SAVE RESULTS
# ============================================================

results = {

    "warning": (
        "All suppression thresholds in this file "
        "are exploratory and were evaluated on the "
        "test set. Final thresholds must be selected "
        "using validation data."
    ),

    "settings": {
        "baseline_threshold":
            BASELINE_THRESHOLD,

        "high_confidence_threshold":
            HIGH_CONFIDENCE_THRESHOLD,

        "sampling_frequency":
            SAMPLING_FREQUENCY,
    },

    "baseline": baseline_metrics,

    "probability_thresholds":
        threshold_results,

    "artifact_threshold_candidates":
        candidate_thresholds,

    "single_feature_suppression":
        single_feature_results,

    "combined_suppression":
        combined_results,

    "high_confidence_fp_profile":
        high_confidence_profile,

    "patient_baseline":
        patient_results,

    "promising_results":
        promising[:20],
}


with open(
    OUTPUT_JSON,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        results,
        f,
        indent=4
    )


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n" + "=" * 70)
print("ANALYSIS COMPLETED")
print("=" * 70)

print(
    "\nResults saved to:"
)

print(
    OUTPUT_JSON
)

print("\nDONE")