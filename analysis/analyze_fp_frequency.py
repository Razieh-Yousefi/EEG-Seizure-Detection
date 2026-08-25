# -*- coding: utf-8 -*-

"""
Frequency-domain analysis of true positives and false positives.

This script:
1. Loads test predictions.
2. Identifies TP and FP windows.
3. Loads the original EEG windows.
4. Applies the same channel-wise normalization used during training.
5. Computes frequency-domain features using Welch PSD.
6. Compares TP and FP spectral characteristics.
7. Focuses on channels identified by the channel-wise analysis.
8. Analyzes high-confidence false positives.
9. Saves numerical results and plots.
"""

from pathlib import Path
import json

import numpy as np
import matplotlib.pyplot as plt

try:
    from scipy.signal import welch
except ImportError:
    raise ImportError(
        "scipy is required. Install it with: pip install scipy"
    )


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_DIR / "data"
RESULTS_DIR = PROJECT_DIR / "results"

X_PATH = DATA_DIR / "X_chbmit_full.npy"
NORMALIZATION_PATH = DATA_DIR / "normalization_params.npz"
PROBABILITY_PATH = RESULTS_DIR / "test_window_probabilities.npz"

OUTPUT_JSON = RESULTS_DIR / "fp_tp_frequency_analysis.json"
OUTPUT_DIR = RESULTS_DIR / "fp_tp_frequency_plots"


# ============================================================
# SETTINGS
# ============================================================

THRESHOLD = 0.50
HIGH_CONFIDENCE_THRESHOLD = 0.90

EXPECTED_CHANNELS = 23
EXPECTED_SAMPLES = 1280

SAMPLING_FREQUENCY = 256.0

N_PER_SEGMENT = 256

FOCUS_CHANNELS = [
    4,
    16,
    12,
    18,
    8,
]


# ============================================================
# FEATURE DEFINITIONS
# ============================================================

BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 80.0),
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def load_normalized_window(X, channel_mean, channel_std, sample_index):
    """
    Load one EEG window and apply training normalization.
    """

    sample_index = int(sample_index)

    raw_window = np.asarray(
        X[sample_index],
        dtype=np.float32
    )

    normalized_window = (
        raw_window -
        channel_mean[:, None]
    ) / channel_std[:, None]

    return normalized_window


def calculate_psd(signal):
    """
    Calculate Welch power spectral density.
    """

    frequencies, power = welch(
        signal,
        fs=SAMPLING_FREQUENCY,
        nperseg=N_PER_SEGMENT,
        noverlap=N_PER_SEGMENT // 2
    )

    return frequencies, power


def band_power(frequencies, power, low, high):
    """
    Calculate absolute power inside a frequency band.
    """

    mask = (
        (frequencies >= low) &
        (frequencies < high)
    )

    if not np.any(mask):
        return 0.0

    return float(
        np.trapezoid(
            power[mask],
            frequencies[mask]
        )
    )


def spectral_features(signal):
    """
    Calculate spectral features for one channel.
    """

    frequencies, power = calculate_psd(signal)

    total_power = float(
        np.trapezoid(
            power,
            frequencies
        )
    )

    features = {
        "total_power": total_power
    }

    band_values = {}

    for band_name, (
        low,
        high
    ) in BANDS.items():

        value = band_power(
            frequencies,
            power,
            low,
            high
        )

        band_values[band_name] = value

        features[
            f"{band_name}_power"
        ] = value

        if total_power > 0:

            features[
                f"{band_name}_relative_power"
            ] = float(
                value / total_power
            )

        else:

            features[
                f"{band_name}_relative_power"
            ] = 0.0

    dominant_index = int(
        np.argmax(power)
    )

    dominant_frequency = float(
        frequencies[dominant_index]
    )

    features[
        "dominant_frequency"
    ] = dominant_frequency

    if total_power > 0:

        spectral_centroid = float(
            np.sum(
                frequencies * power
            ) /
            np.sum(power)
        )

    else:

        spectral_centroid = 0.0

    features[
        "spectral_centroid"
    ] = spectral_centroid

    high_frequency_mask = (
        frequencies >= 30.0
    )

    if np.any(high_frequency_mask):

        high_frequency_power = float(
            np.trapezoid(
                power[
                    high_frequency_mask
                ],
                frequencies[
                    high_frequency_mask
                ]
            )
        )

    else:

        high_frequency_power = 0.0

    features[
        "high_frequency_power"
    ] = high_frequency_power

    if total_power > 0:

        features[
            "high_frequency_ratio"
        ] = float(
            high_frequency_power /
            total_power
        )

    else:

        features[
            "high_frequency_ratio"
        ] = 0.0

    return features


def analyze_window(
    window,
    channels
):
    """
    Calculate spectral features for selected channels.
    """

    results = {}

    for channel_number in channels:

        channel_index = (
            channel_number - 1
        )

        signal = window[
            channel_index
        ]

        results[
            str(channel_number)
        ] = spectral_features(
            signal
        )

    return results


def summarize_channel_features(
    examples,
    channels
):
    """
    Calculate group statistics for spectral features.
    """

    feature_names = [
        "total_power",
        "delta_power",
        "theta_power",
        "alpha_power",
        "beta_power",
        "gamma_power",
        "delta_relative_power",
        "theta_relative_power",
        "alpha_relative_power",
        "beta_relative_power",
        "gamma_relative_power",
        "dominant_frequency",
        "spectral_centroid",
        "high_frequency_power",
        "high_frequency_ratio",
    ]

    summary = {}

    for channel in channels:

        channel_key = str(channel)

        summary[
            channel_key
        ] = {}

        for feature_name in feature_names:

            values = []

            for example in examples:

                channel_features = (
                    example["channels"][
                        channel_key
                    ]
                )

                values.append(
                    channel_features[
                        feature_name
                    ]
                )

            values = np.asarray(
                values,
                dtype=np.float64
            )

            if len(values) == 0:
                continue

            summary[
                channel_key
            ][feature_name] = {
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
                "min": float(
                    np.min(values)
                ),
                "max": float(
                    np.max(values)
                ),
            }

    return summary


def print_comparison(
    tp_summary,
    fp_summary,
    channels
):
    """
    Print TP versus FP spectral comparisons.
    """

    print("\n" + "=" * 70)
    print("SPECTRAL COMPARISON")
    print("=" * 70)

    features = [
        "dominant_frequency",
        "spectral_centroid",
        "high_frequency_ratio",
        "delta_relative_power",
        "theta_relative_power",
        "alpha_relative_power",
        "beta_relative_power",
        "gamma_relative_power",
    ]

    for channel in channels:

        channel_key = str(channel)

        print(
            f"\nCHANNEL {channel}"
        )

        for feature in features:

            tp_value = tp_summary[
                channel_key
            ][feature]["mean"]

            fp_value = fp_summary[
                channel_key
            ][feature]["mean"]

            if tp_value != 0:

                relative_difference = (
                    fp_value - tp_value
                ) / abs(tp_value)

            else:

                relative_difference = np.nan

            print(
                f"{feature:25s} "
                f"TP={tp_value:.6f} "
                f"FP={fp_value:.6f} "
                f"diff={relative_difference:+.4f}"
            )


def save_psd_plot(
    tp_examples,
    fp_examples,
    channel,
    output_path
):
    """
    Plot mean normalized PSD for TP and FP.
    """

    channel_index = channel - 1

    tp_psds = []
    fp_psds = []

    frequencies = None

    for example in tp_examples:

        signal = example[
            "window"
        ][channel_index]

        freq, power = calculate_psd(
            signal
        )

        frequencies = freq

        total = np.trapezoid(
            power,
            freq
        )

        if total > 0:

            power = power / total

        tp_psds.append(
            power
        )

    for example in fp_examples:

        signal = example[
            "window"
        ][channel_index]

        freq, power = calculate_psd(
            signal
        )

        frequencies = freq

        total = np.trapezoid(
            power,
            freq
        )

        if total > 0:

            power = power / total

        fp_psds.append(
            power
        )

    plt.figure(
        figsize=(10, 6)
    )

    if len(tp_psds) > 0:

        tp_mean = np.mean(
            np.asarray(tp_psds),
            axis=0
        )

        plt.plot(
            frequencies,
            tp_mean,
            label="TP"
        )

    if len(fp_psds) > 0:

        fp_mean = np.mean(
            np.asarray(fp_psds),
            axis=0
        )

        plt.plot(
            frequencies,
            fp_mean,
            label="FP"
        )

    plt.xlim(
        0,
        80
    )

    plt.xlabel(
        "Frequency (Hz)"
    )

    plt.ylabel(
        "Normalized PSD"
    )

    plt.title(
        f"TP vs FP Mean PSD - Channel {channel}"
    )

    plt.legend()

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=150
    )

    plt.close()


# ============================================================
# START
# ============================================================

print("=" * 70)
print("FP vs TP FREQUENCY-DOMAIN ANALYSIS")
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


# ============================================================
# LOAD DATA
# ============================================================

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


print(
    "\nX shape:",
    X.shape
)


# ============================================================
# VALIDATION
# ============================================================

if X.ndim != 3:

    raise ValueError(
        "X must be 3D."
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
        "Invalid normalization mean shape."
    )

if channel_std.shape != (
    EXPECTED_CHANNELS,
):

    raise ValueError(
        "Invalid normalization std shape."
    )


# ============================================================
# LOAD PREDICTIONS
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

high_fp_mask = (
    fp_mask &
    (
        probabilities >=
        HIGH_CONFIDENCE_THRESHOLD
    )
)


tp_positions = np.where(
    tp_mask
)[0]

fp_positions = np.where(
    fp_mask
)[0]

high_fp_positions = np.where(
    high_fp_mask
)[0]


print("\n" + "=" * 70)
print("CLASSIFICATION")
print("=" * 70)

print(
    "TP:",
    len(tp_positions)
)

print(
    "FP:",
    len(fp_positions)
)

print(
    "High-confidence FP:",
    len(high_fp_positions)
)


# ============================================================
# EXTRACT SPECTRAL FEATURES
# ============================================================

def extract_examples(
    positions,
    name
):
    """
    Extract selected windows and spectral features.
    """

    results = []

    print(
        f"\nExtracting {name}..."
    )

    for position in positions:

        sample_index = int(
            test_indices[position]
        )

        window = load_normalized_window(
            X,
            channel_mean,
            channel_std,
            sample_index
        )

        channel_features = analyze_window(
            window,
            FOCUS_CHANNELS
        )

        results.append({
            "position": int(position),
            "sample_index": sample_index,
            "patient": str(
                patients[position]
            ),
            "probability": float(
                probabilities[position]
            ),
            "label": int(
                labels[position]
            ),
            "channels": channel_features,
            "window": window,
        })

    return results


tp_examples = extract_examples(
    tp_positions,
    "TP windows"
)

fp_examples = extract_examples(
    fp_positions,
    "FP windows"
)

high_fp_examples = extract_examples(
    high_fp_positions,
    "high-confidence FP windows"
)


# ============================================================
# SUMMARIZE
# ============================================================

tp_summary = summarize_channel_features(
    tp_examples,
    FOCUS_CHANNELS
)

fp_summary = summarize_channel_features(
    fp_examples,
    FOCUS_CHANNELS
)

high_fp_summary = summarize_channel_features(
    high_fp_examples,
    FOCUS_CHANNELS
)


# ============================================================
# PRINT RESULTS
# ============================================================

print_comparison(
    tp_summary,
    fp_summary,
    FOCUS_CHANNELS
)


# ============================================================
# HIGH-CONFIDENCE COMPARISON
# ============================================================

print("\n" + "=" * 70)
print("HIGH-CONFIDENCE FP VS TP")
print("=" * 70)

for channel in FOCUS_CHANNELS:

    channel_key = str(channel)

    print(
        f"\nCHANNEL {channel}"
    )

    for feature in [
        "dominant_frequency",
        "spectral_centroid",
        "high_frequency_ratio",
        "beta_relative_power",
        "gamma_relative_power",
    ]:

        tp_value = tp_summary[
            channel_key
        ][feature]["mean"]

        fp_value = high_fp_summary[
            channel_key
        ][feature]["mean"]

        if tp_value != 0:

            difference = (
                fp_value - tp_value
            ) / abs(tp_value)

        else:

            difference = np.nan

        print(
            f"{feature:25s} "
            f"TP={tp_value:.6f} "
            f"High-FP={fp_value:.6f} "
            f"diff={difference:+.4f}"
        )


# ============================================================
# OUTPUT DIRECTORY
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# PSD PLOTS
# ============================================================

print("\n" + "=" * 70)
print("GENERATING PSD PLOTS")
print("=" * 70)

for channel in FOCUS_CHANNELS:

    output_path = (
        OUTPUT_DIR /
        f"channel_{channel}_mean_psd.png"
    )

    save_psd_plot(
        tp_examples,
        fp_examples,
        channel,
        output_path
    )

    print(
        f"Saved: {output_path.name}"
    )


# ============================================================
# SAVE JSON
# ============================================================

def clean_for_json(value):
    """
    Convert NumPy values to JSON-compatible values.
    """

    if isinstance(
        value,
        np.ndarray
    ):

        return value.tolist()

    if isinstance(
        value,
        np.floating
    ):

        return float(value)

    if isinstance(
        value,
        np.integer
    ):

        return int(value)

    if isinstance(
        value,
        dict
    ):

        return {
            key: clean_for_json(
                item
            )
            for key, item in value.items()
        }

    if isinstance(
        value,
        list
    ):

        return [
            clean_for_json(item)
            for item in value
        ]

    return value


json_results = {
    "settings": {
        "threshold": THRESHOLD,
        "high_confidence_threshold":
            HIGH_CONFIDENCE_THRESHOLD,
        "sampling_frequency":
            SAMPLING_FREQUENCY,
        "n_per_segment":
            N_PER_SEGMENT,
        "focus_channels":
            FOCUS_CHANNELS,
    },

    "classification": {
        "tp": int(
            len(tp_positions)
        ),
        "fp": int(
            len(fp_positions)
        ),
        "high_confidence_fp": int(
            len(high_fp_positions)
        ),
    },

    "tp_summary": tp_summary,

    "fp_summary": fp_summary,

    "high_confidence_fp_summary":
        high_fp_summary,
}


with open(
    OUTPUT_JSON,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        clean_for_json(
            json_results
        ),
        file,
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

print(
    "\nPlots saved to:"
)

print(
    OUTPUT_DIR
)

print("\nDONE")