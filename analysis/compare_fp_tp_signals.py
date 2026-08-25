# -*- coding: utf-8 -*-

"""
Compare EEG signal characteristics between true positives
and false positives.

This script:
1. Loads model predictions.
2. Identifies TP and FP windows.
3. Maps test indices back to X_chbmit_full.npy.
4. Applies the same channel-wise normalization used during training.
5. Extracts signal-level features.
6. Compares FP and TP distributions.
7. Performs patient-level FP analysis.
8. Saves numerical results and visualization plots.
"""

from pathlib import Path
import json

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_DIR / "data"
RESULTS_DIR = PROJECT_DIR / "results"

X_PATH = DATA_DIR / "X_chbmit_full.npy"
NORMALIZATION_PATH = DATA_DIR / "normalization_params.npz"
PROBABILITY_PATH = RESULTS_DIR / "test_window_probabilities.npz"

OUTPUT_JSON = RESULTS_DIR / "fp_tp_signal_comparison.json"

OUTPUT_DIR = RESULTS_DIR / "fp_tp_signal_plots"


# ============================================================
# SETTINGS
# ============================================================

THRESHOLD = 0.50

HIGH_CONFIDENCE_THRESHOLD = 0.90

MAX_EXAMPLES_PER_CLASS = 20

EXPECTED_CHANNELS = 23
EXPECTED_SAMPLES = 1280


# ============================================================
# FEATURE FUNCTIONS
# ============================================================

def calculate_rms(signal):
    """Calculate root mean square."""
    return float(
        np.sqrt(
            np.mean(
                np.square(signal)
            )
        )
    )


def calculate_peak_to_peak(signal):
    """Calculate peak-to-peak amplitude."""
    return float(
        np.max(signal) -
        np.min(signal)
    )


def calculate_line_length(signal):
    """Calculate waveform line length."""
    return float(
        np.sum(
            np.abs(
                np.diff(signal)
            )
        )
    )


def calculate_zero_crossing_rate(signal):
    """Calculate zero crossing rate."""
    centered = signal - np.mean(signal)

    signs = np.sign(centered)

    crossings = np.sum(
        signs[:-1] != signs[1:]
    )

    return float(
        crossings /
        max(1, len(signal) - 1)
    )


def calculate_signal_features(window):
    """
    Calculate signal-level features across all channels.

    Input:
        window: shape (channels, samples)

    Returns:
        Dictionary of aggregate features.
    """

    channel_rms = []
    channel_std = []
    channel_variance = []
    channel_ptp = []
    channel_max_abs = []
    channel_line_length = []
    channel_zcr = []

    for channel in window:

        channel_rms.append(
            calculate_rms(channel)
        )

        channel_std.append(
            float(np.std(channel))
        )

        channel_variance.append(
            float(np.var(channel))
        )

        channel_ptp.append(
            calculate_peak_to_peak(channel)
        )

        channel_max_abs.append(
            float(np.max(np.abs(channel)))
        )

        channel_line_length.append(
            calculate_line_length(channel)
        )

        channel_zcr.append(
            calculate_zero_crossing_rate(channel)
        )

    return {
        "mean_rms": float(np.mean(channel_rms)),
        "max_rms": float(np.max(channel_rms)),

        "mean_std": float(np.mean(channel_std)),
        "max_std": float(np.max(channel_std)),

        "mean_variance":
            float(np.mean(channel_variance)),

        "max_variance":
            float(np.max(channel_variance)),

        "mean_peak_to_peak":
            float(np.mean(channel_ptp)),

        "max_peak_to_peak":
            float(np.max(channel_ptp)),

        "mean_max_abs":
            float(np.mean(channel_max_abs)),

        "max_max_abs":
            float(np.max(channel_max_abs)),

        "mean_line_length":
            float(np.mean(channel_line_length)),

        "max_line_length":
            float(np.max(channel_line_length)),

        "mean_zero_crossing_rate":
            float(np.mean(channel_zcr)),

        "max_zero_crossing_rate":
            float(np.max(channel_zcr)),
    }


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("FP vs TP EEG SIGNAL COMPARISON")
print("=" * 70)

print("\nLoading files...")

if not X_PATH.exists():
    raise FileNotFoundError(
        f"X file not found:\n{X_PATH}"
    )

if not NORMALIZATION_PATH.exists():
    raise FileNotFoundError(
        f"Normalization file not found:\n"
        f"{NORMALIZATION_PATH}"
    )

if not PROBABILITY_PATH.exists():
    raise FileNotFoundError(
        f"Probability file not found:\n"
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
# VALIDATE DATASET
# ============================================================

print("\nX shape:", X.shape)

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

if channel_mean.shape != (
    EXPECTED_CHANNELS,
):
    raise ValueError(
        "Invalid channel mean shape."
    )

if channel_std.shape != (
    EXPECTED_CHANNELS,
):
    raise ValueError(
        "Invalid channel std shape."
    )


# ============================================================
# LOAD PREDICTION ARRAYS
# ============================================================

test_indices = np.asarray(
    prediction_data["test_indices"]
)

patients = np.asarray(
    prediction_data["patients"]
)

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


# ============================================================
# CREATE PREDICTIONS
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

fn_mask = (
    (labels == 1) &
    (predictions == 0)
)


tp_indices = np.where(
    tp_mask
)[0]

fp_indices = np.where(
    fp_mask
)[0]

fn_indices = np.where(
    fn_mask
)[0]


print("\n" + "=" * 70)
print("CLASSIFICATION")
print("=" * 70)

print("TP:", len(tp_indices))
print("FP:", len(fp_indices))
print("FN:", len(fn_indices))


# ============================================================
# SIGNAL EXTRACTION
# ============================================================

def load_normalized_window(sample_index):
    """
    Load one EEG window and apply the exact normalization
    used by the training Dataset.
    """

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


def extract_examples(indices, name):
    """
    Extract signal features for selected test examples.
    """

    results = []

    print(
        f"\nExtracting {name} examples..."
    )

    for position in indices:

        sample_index = int(
            test_indices[position]
        )

        window = load_normalized_window(
            sample_index
        )

        features = calculate_signal_features(
            window
        )

        result = {
            "array_position": int(position),
            "sample_index": sample_index,
            "patient": str(
                patients[position]
            ),
            "label": int(
                labels[position]
            ),
            "probability": float(
                probabilities[position]
            ),
            "features": features,
        }

        results.append(result)

    return results


# ============================================================
# EXTRACT ALL TP AND FP FEATURES
# ============================================================

tp_feature_results = extract_examples(
    tp_indices,
    "TP"
)

fp_feature_results = extract_examples(
    fp_indices,
    "FP"
)


# ============================================================
# FEATURE SUMMARY
# ============================================================

FEATURE_NAMES = [
    "mean_rms",
    "max_rms",
    "mean_std",
    "max_std",
    "mean_variance",
    "max_variance",
    "mean_peak_to_peak",
    "max_peak_to_peak",
    "mean_max_abs",
    "max_max_abs",
    "mean_line_length",
    "max_line_length",
    "mean_zero_crossing_rate",
    "max_zero_crossing_rate",
]


def summarize_feature_group(results):
    """
    Calculate mean, median, std, min, and max
    for each signal feature.
    """

    summary = {}

    for feature_name in FEATURE_NAMES:

        values = np.array([
            item["features"][feature_name]
            for item in results
        ])

        if len(values) == 0:
            summary[feature_name] = {
                "count": 0
            }
            continue

        summary[feature_name] = {
            "count": int(len(values)),
            "mean": float(
                np.mean(values)
            ),
            "median": float(
                np.median(values)
            ),
            "std": float(
                np.std(values)
            ),
            "min": float(
                np.min(values)
            ),
            "max": float(
                np.max(values)
            ),
        }

    return summary


tp_summary = summarize_feature_group(
    tp_feature_results
)

fp_summary = summarize_feature_group(
    fp_feature_results
)


# ============================================================
# PRINT COMPARISON
# ============================================================

print("\n" + "=" * 70)
print("SIGNAL FEATURE COMPARISON")
print("=" * 70)

for feature_name in FEATURE_NAMES:

    tp_mean = tp_summary[
        feature_name
    ]["mean"]

    fp_mean = fp_summary[
        feature_name
    ]["mean"]

    ratio = (
        fp_mean / tp_mean
        if tp_mean != 0
        else np.nan
    )

    print(
        f"\n{feature_name}"
    )

    print(
        f"  TP mean: {tp_mean:.6f}"
    )

    print(
        f"  FP mean: {fp_mean:.6f}"
    )

    print(
        f"  FP/TP ratio: {ratio:.4f}"
    )


# ============================================================
# PATIENT-LEVEL FEATURE ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("PATIENT-LEVEL FP FEATURE ANALYSIS")
print("=" * 70)


unique_fp_patients = np.unique(
    patients[fp_mask]
)

patient_results = {}


for patient in unique_fp_patients:

    patient_fp_positions = fp_indices[
        patients[fp_indices] == patient
    ]

    patient_features = extract_examples(
        patient_fp_positions,
        f"FP - {patient}"
    )

    patient_summary = summarize_feature_group(
        patient_features
    )

    patient_results[str(patient)] = {
        "count": int(
            len(patient_fp_positions)
        ),
        "features": patient_summary,
    }

    print(
        f"\n{patient}: "
        f"{len(patient_fp_positions)} FP windows"
    )

    for feature_name in [
        "mean_rms",
        "mean_peak_to_peak",
        "mean_line_length",
        "mean_zero_crossing_rate",
    ]:

        value = patient_summary[
            feature_name
        ]["mean"]

        print(
            f"  {feature_name}: "
            f"{value:.6f}"
        )


# ============================================================
# HIGH-CONFIDENCE FP ANALYSIS
# ============================================================

high_fp_positions = fp_indices[
    probabilities[fp_indices]
    >= HIGH_CONFIDENCE_THRESHOLD
]


print("\n" + "=" * 70)
print("HIGH-CONFIDENCE FP SIGNAL ANALYSIS")
print("=" * 70)

print(
    f"High-confidence FP count: "
    f"{len(high_fp_positions)}"
)


high_fp_features = extract_examples(
    high_fp_positions,
    "High-confidence FP"
)


high_fp_summary = summarize_feature_group(
    high_fp_features
)


# ============================================================
# VISUALIZATION
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


def get_feature_values(results, feature):
    """
    Extract one feature across examples.
    """

    return np.array([
        item["features"][feature]
        for item in results
    ])


PLOT_FEATURES = [
    "mean_rms",
    "mean_std",
    "mean_peak_to_peak",
    "mean_line_length",
    "mean_zero_crossing_rate",
]


for feature in PLOT_FEATURES:

    tp_values = get_feature_values(
        tp_feature_results,
        feature
    )

    fp_values = get_feature_values(
        fp_feature_results,
        feature
    )

    plt.figure(
        figsize=(8, 5)
    )

    plt.boxplot(
        [
            tp_values,
            fp_values
        ],
        labels=[
            "TP",
            "FP"
        ]
    )

    plt.title(
        f"TP vs FP: {feature}"
    )

    plt.ylabel(
        feature
    )

    plt.grid(
        axis="y",
        alpha=0.3
    )

    output_path = (
        OUTPUT_DIR /
        f"{feature}_comparison.png"
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=150
    )

    plt.close()


# ============================================================
# EXAMPLE WAVEFORM PLOTS
# ============================================================

def save_waveform_plot(
    sample_index,
    patient,
    probability,
    label,
    classification,
    output_path
):
    """
    Save a visualization of all EEG channels
    for one selected window.
    """

    window = load_normalized_window(
        sample_index
    )

    time = np.arange(
        window.shape[1]
    )

    fig, axes = plt.subplots(
        EXPECTED_CHANNELS,
        1,
        figsize=(14, 18),
        sharex=True
    )

    for channel_index, ax in enumerate(
        axes
    ):

        ax.plot(
            time,
            window[channel_index],
            linewidth=0.6
        )

        ax.set_ylabel(
            f"Ch {channel_index + 1}",
            fontsize=7
        )

        ax.grid(
            alpha=0.2
        )

    fig.suptitle(
        f"{classification} | "
        f"Patient={patient} | "
        f"Probability={probability:.4f} | "
        f"Sample={sample_index}"
    )

    axes[-1].set_xlabel(
        "Sample"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=150
    )

    plt.close()


# ============================================================
# SAVE SELECTED TP WAVEFORMS
# ============================================================

for counter, position in enumerate(
    tp_indices[:5]
):

    sample_index = int(
        test_indices[position]
    )

    save_waveform_plot(
        sample_index=sample_index,
        patient=str(
            patients[position]
        ),
        probability=float(
            probabilities[position]
        ),
        label=int(
            labels[position]
        ),
        classification="TRUE POSITIVE",
        output_path=(
            OUTPUT_DIR /
            f"TP_{counter + 1}.png"
        )
    )


# ============================================================
# SAVE HIGH-CONFIDENCE FP WAVEFORMS
# ============================================================

for counter, position in enumerate(
    high_fp_positions[:10]
):

    sample_index = int(
        test_indices[position]
    )

    save_waveform_plot(
        sample_index=sample_index,
        patient=str(
            patients[position]
        ),
        probability=float(
            probabilities[position]
        ),
        label=int(
            labels[position]
        ),
        classification="HIGH CONFIDENCE FALSE POSITIVE",
        output_path=(
            OUTPUT_DIR /
            f"FP_high_confidence_{counter + 1}.png"
        )
    )


# ============================================================
# SAVE JSON RESULTS
# ============================================================

results = {

    "settings": {
        "threshold": THRESHOLD,
        "high_confidence_threshold":
            HIGH_CONFIDENCE_THRESHOLD,
    },

    "classification": {
        "tp": int(len(tp_indices)),
        "fp": int(len(fp_indices)),
        "fn": int(len(fn_indices)),
    },

    "tp_signal_summary": tp_summary,

    "fp_signal_summary": fp_summary,

    "high_confidence_fp_summary":
        high_fp_summary,

    "patient_fp_analysis":
        patient_results,

    "tp_examples":
        tp_feature_results[
            :MAX_EXAMPLES_PER_CLASS
        ],

    "fp_examples":
        fp_feature_results[
            :MAX_EXAMPLES_PER_CLASS
        ],

    "high_confidence_fp_examples":
        high_fp_features[
            :MAX_EXAMPLES_PER_CLASS
        ],
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
    "\nJSON results:"
)

print(
    OUTPUT_JSON
)

print(
    "\nPlots:"
)

print(
    OUTPUT_DIR
)

print("\nGenerated plots:")

for path in sorted(
    OUTPUT_DIR.glob("*.png")
):

    print(
        f" - {path.name}"
    )

print("\nDONE")