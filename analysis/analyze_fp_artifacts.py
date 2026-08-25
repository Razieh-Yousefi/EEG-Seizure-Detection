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

OUTPUT_JSON = RESULTS_DIR / "fp_artifact_analysis.json"
OUTPUT_DIR = RESULTS_DIR / "fp_artifact_plots"


# ============================================================
# SETTINGS
# ============================================================

THRESHOLD = 0.50
HIGH_CONFIDENCE_THRESHOLD = 0.90

EXPECTED_CHANNELS = 23
EXPECTED_SAMPLES = 1280

TOP_CHANNELS = 8


# ============================================================
# FEATURE FUNCTIONS
# ============================================================

def calculate_rms(signal):
    return float(
        np.sqrt(
            np.mean(
                np.square(signal)
            )
        )
    )


def calculate_zero_crossing_rate(signal):
    centered = signal - np.mean(signal)

    signs = np.sign(centered)

    crossings = np.sum(
        signs[:-1] != signs[1:]
    )

    return float(
        crossings /
        max(1, len(signal) - 1)
    )


def calculate_line_length(signal):
    return float(
        np.sum(
            np.abs(
                np.diff(signal)
            )
        )
    )


def calculate_high_frequency_ratio(
    signal,
    sampling_rate=256.0,
    cutoff_frequency=20.0
):
    """
    Calculate the fraction of spectral power above
    the specified cutoff frequency.
    """

    signal = signal - np.mean(signal)

    frequencies = np.fft.rfftfreq(
        len(signal),
        d=1.0 / sampling_rate
    )

    spectrum = np.abs(
        np.fft.rfft(signal)
    ) ** 2

    total_power = np.sum(spectrum)

    if total_power <= 0:
        return 0.0

    high_frequency_power = np.sum(
        spectrum[
            frequencies >= cutoff_frequency
        ]
    )

    return float(
        high_frequency_power /
        total_power
    )


def calculate_band_power(
    signal,
    low_frequency,
    high_frequency,
    sampling_rate=256.0
):
    """
    Calculate relative spectral power inside a frequency band.
    """

    signal = signal - np.mean(signal)

    frequencies = np.fft.rfftfreq(
        len(signal),
        d=1.0 / sampling_rate
    )

    spectrum = np.abs(
        np.fft.rfft(signal)
    ) ** 2

    total_power = np.sum(spectrum)

    if total_power <= 0:
        return 0.0

    mask = (
        (frequencies >= low_frequency) &
        (frequencies < high_frequency)
    )

    band_power = np.sum(
        spectrum[mask]
    )

    return float(
        band_power /
        total_power
    )


def calculate_channel_features(window):
    """
    Calculate artifact-related features for every channel.
    """

    channel_features = []

    for channel_index in range(
        window.shape[0]
    ):

        signal = window[channel_index]

        rms = calculate_rms(signal)

        zcr = calculate_zero_crossing_rate(
            signal
        )

        line_length = calculate_line_length(
            signal
        )

        high_frequency_ratio = (
            calculate_high_frequency_ratio(
                signal
            )
        )

        beta_power = calculate_band_power(
            signal,
            13.0,
            30.0
        )

        gamma_power = calculate_band_power(
            signal,
            30.0,
            80.0
        )

        channel_features.append(
            {
                "channel": channel_index + 1,
                "rms": rms,
                "zero_crossing_rate": zcr,
                "line_length": line_length,
                "high_frequency_ratio":
                    high_frequency_ratio,
                "beta_power": beta_power,
                "gamma_power": gamma_power,
            }
        )

    return channel_features


# ============================================================
# WINDOW-LEVEL SUMMARY
# ============================================================

def summarize_window_features(
    channel_features
):
    rms_values = np.array([
        item["rms"]
        for item in channel_features
    ])

    zcr_values = np.array([
        item["zero_crossing_rate"]
        for item in channel_features
    ])

    line_values = np.array([
        item["line_length"]
        for item in channel_features
    ])

    high_frequency_values = np.array([
        item["high_frequency_ratio"]
        for item in channel_features
    ])

    beta_values = np.array([
        item["beta_power"]
        for item in channel_features
    ])

    gamma_values = np.array([
        item["gamma_power"]
        for item in channel_features
    ])

    return {
        "mean_rms":
            float(np.mean(rms_values)),

        "max_rms":
            float(np.max(rms_values)),

        "mean_zero_crossing_rate":
            float(np.mean(zcr_values)),

        "max_zero_crossing_rate":
            float(np.max(zcr_values)),

        "mean_line_length":
            float(np.mean(line_values)),

        "max_line_length":
            float(np.max(line_values)),

        "mean_high_frequency_ratio":
            float(np.mean(
                high_frequency_values
            )),

        "max_high_frequency_ratio":
            float(np.max(
                high_frequency_values
            )),

        "mean_beta_power":
            float(np.mean(beta_values)),

        "max_beta_power":
            float(np.max(beta_values)),

        "mean_gamma_power":
            float(np.mean(gamma_values)),

        "max_gamma_power":
            float(np.max(gamma_values)),
    }


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("FP ARTIFACT AND MORPHOLOGY ANALYSIS")
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
# VALIDATION
# ============================================================

print("\nX shape:", X.shape)

if X.ndim != 3:
    raise ValueError(
        "X must be 3D."
    )

if X.shape[1] != EXPECTED_CHANNELS:
    raise ValueError(
        f"Expected {EXPECTED_CHANNELS} channels."
    )

if X.shape[2] != EXPECTED_SAMPLES:
    raise ValueError(
        f"Expected {EXPECTED_SAMPLES} samples."
    )


# ============================================================
# LOAD PREDICTION ARRAYS
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


if not (
    len(test_indices)
    == len(patients)
    == len(labels)
    == len(probabilities)
):
    raise ValueError(
        "Prediction arrays have different lengths."
    )


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

high_fp_mask = (
    fp_mask &
    (probabilities >= HIGH_CONFIDENCE_THRESHOLD)
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
# NORMALIZED WINDOW LOADING
# ============================================================

def load_normalized_window(
    sample_index
):
    raw_window = np.asarray(
        X[int(sample_index)],
        dtype=np.float32
    )

    normalized_window = (
        raw_window -
        channel_mean[:, None]
    ) / channel_std[:, None]

    return normalized_window


# ============================================================
# EXTRACT WINDOW FEATURES
# ============================================================

def analyze_position(position):

    sample_index = int(
        test_indices[position]
    )

    window = load_normalized_window(
        sample_index
    )

    channel_features = (
        calculate_channel_features(
            window
        )
    )

    summary = summarize_window_features(
        channel_features
    )

    return {
        "position": int(position),
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
        "summary": summary,
        "channels": channel_features,
    }


# ============================================================
# EXTRACT TP / FP FEATURES
# ============================================================

print("\nExtracting TP windows...")

tp_results = [
    analyze_position(position)
    for position in tp_positions
]


print("Extracting FP windows...")

fp_results = [
    analyze_position(position)
    for position in fp_positions
]


print(
    "Extracting high-confidence FP windows..."
)

high_fp_results = [
    analyze_position(position)
    for position in high_fp_positions
]


# ============================================================
# GROUP SUMMARY
# ============================================================

SUMMARY_FEATURES = [
    "mean_rms",
    "max_rms",
    "mean_zero_crossing_rate",
    "max_zero_crossing_rate",
    "mean_line_length",
    "max_line_length",
    "mean_high_frequency_ratio",
    "max_high_frequency_ratio",
    "mean_beta_power",
    "max_beta_power",
    "mean_gamma_power",
    "max_gamma_power",
]


def summarize_group(
    results
):

    output = {}

    for feature in SUMMARY_FEATURES:

        values = np.array([
            item["summary"][feature]
            for item in results
        ])

        if len(values) == 0:
            output[feature] = {
                "count": 0
            }

            continue

        output[feature] = {
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

    return output


tp_summary = summarize_group(
    tp_results
)

fp_summary = summarize_group(
    fp_results
)

high_fp_summary = summarize_group(
    high_fp_results
)


# ============================================================
# PRINT SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("ARTIFACT FEATURE COMPARISON")
print("=" * 70)

for feature in SUMMARY_FEATURES:

    tp_mean = tp_summary[
        feature
    ]["mean"]

    fp_mean = fp_summary[
        feature
    ]["mean"]

    high_fp_mean = high_fp_summary[
        feature
    ]["mean"]

    print(
        f"\n{feature}"
    )

    print(
        f"TP      = {tp_mean:.6f}"
    )

    print(
        f"FP      = {fp_mean:.6f}"
    )

    print(
        f"High-FP = {high_fp_mean:.6f}"
    )


# ============================================================
# CHANNEL CONCENTRATION ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("CHANNEL CONCENTRATION ANALYSIS")
print("=" * 70)


def calculate_channel_mean(
    results,
    feature
):

    if len(results) == 0:
        return np.zeros(
            EXPECTED_CHANNELS
        )

    values = np.array([
        [
            channel[feature]
            for channel in item["channels"]
        ]
        for item in results
    ])

    return np.mean(
        values,
        axis=0
    )


channel_features = [
    "rms",
    "zero_crossing_rate",
    "line_length",
    "high_frequency_ratio",
    "beta_power",
    "gamma_power",
]


channel_comparison = {}


for feature in channel_features:

    tp_values = calculate_channel_mean(
        tp_results,
        feature
    )

    fp_values = calculate_channel_mean(
        fp_results,
        feature
    )

    high_fp_values = calculate_channel_mean(
        high_fp_results,
        feature
    )

    relative_difference = (
        fp_values - tp_values
    ) / (
        np.abs(tp_values) + 1e-8
    )

    ranking = np.argsort(
        np.abs(relative_difference)
    )[::-1]

    print(
        f"\n{feature.upper()}"
    )

    for channel_index in ranking[
        :TOP_CHANNELS
    ]:

        print(
            f"Channel {channel_index + 1:02d} | "
            f"TP={tp_values[channel_index]:.6f} | "
            f"FP={fp_values[channel_index]:.6f} | "
            f"High-FP={high_fp_values[channel_index]:.6f} | "
            f"relative={relative_difference[channel_index]:+.4f}"
        )

    channel_comparison[
        feature
    ] = {
        "tp_mean": tp_values.tolist(),
        "fp_mean": fp_values.tolist(),
        "high_fp_mean":
            high_fp_values.tolist(),
        "relative_difference":
            relative_difference.tolist(),
    }


# ============================================================
# HIGH-FREQUENCY DOMINANCE
# ============================================================

print("\n" + "=" * 70)
print("HIGH-FREQUENCY DOMINANCE")
print("=" * 70)


def calculate_dominant_channel(
    result,
    feature
):

    values = np.array([
        channel[feature]
        for channel in result["channels"]
    ])

    return int(
        np.argmax(values)
    ) + 1


fp_dominant_channels = []

high_fp_dominant_channels = []


for result in fp_results:

    fp_dominant_channels.append(
        calculate_dominant_channel(
            result,
            "high_frequency_ratio"
        )
    )


for result in high_fp_results:

    high_fp_dominant_channels.append(
        calculate_dominant_channel(
            result,
            "high_frequency_ratio"
        )
    )


def count_channels(values):

    counts = {}

    for channel in values:

        channel = int(channel)

        counts[str(channel)] = (
            counts.get(
                str(channel),
                0
            ) + 1
        )

    return dict(
        sorted(
            counts.items(),
            key=lambda item: item[1],
            reverse=True
        )
    )


fp_dominant_counts = count_channels(
    fp_dominant_channels
)

high_fp_dominant_counts = count_channels(
    high_fp_dominant_channels
)


print("\nFP dominant high-frequency channels:")

for channel, count in list(
    fp_dominant_counts.items()
)[:10]:

    print(
        f"Channel {int(channel):02d}: "
        f"{count}"
    )


print(
    "\nHigh-confidence FP dominant "
    "high-frequency channels:"
)

for channel, count in list(
    high_fp_dominant_counts.items()
)[:10]:

    print(
        f"Channel {int(channel):02d}: "
        f"{count}"
    )


# ============================================================
# PATIENT ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("PATIENT-LEVEL ARTIFACT ANALYSIS")
print("=" * 70)


patient_results = {}


unique_patients = np.unique(
    patients[fp_mask]
)


for patient in unique_patients:

    patient_positions = fp_positions[
        patients[fp_positions]
        == patient
    ]

    patient_high_positions = (
        high_fp_positions[
            patients[high_fp_positions]
            == patient
        ]
    )

    patient_fp_results = [
        analyze_position(position)
        for position in patient_positions
    ]

    patient_high_results = [
        analyze_position(position)
        for position in patient_high_positions
    ]

    patient_summary = summarize_group(
        patient_fp_results
    )

    patient_high_summary = summarize_group(
        patient_high_results
    )

    patient_results[
        str(patient)
    ] = {
        "fp_count": int(
            len(patient_positions)
        ),
        "high_confidence_fp_count":
            int(
                len(patient_high_positions)
            ),
        "fp_summary":
            patient_summary,
        "high_confidence_fp_summary":
            patient_high_summary,
    }

    print(
        f"\n{patient}"
    )

    print(
        f"FP count: "
        f"{len(patient_positions)}"
    )

    print(
        f"High-confidence FP count: "
        f"{len(patient_high_positions)}"
    )

    if len(patient_fp_results) > 0:

        print(
            "Mean high-frequency ratio: "
            f"{patient_summary['mean_high_frequency_ratio']['mean']:.6f}"
        )

        print(
            "Mean ZCR: "
            f"{patient_summary['mean_zero_crossing_rate']['mean']:.6f}"
        )


# ============================================================
# VISUALIZATION
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


def plot_feature(
    feature,
    title,
    filename
):

    tp_values = np.array([
        item["summary"][feature]
        for item in tp_results
    ])

    fp_values = np.array([
        item["summary"][feature]
        for item in fp_results
    ])

    high_values = np.array([
        item["summary"][feature]
        for item in high_fp_results
    ])

    plt.figure(
        figsize=(8, 5)
    )

    plt.boxplot(
        [
            tp_values,
            fp_values,
            high_values
        ],
        tick_labels=[
            "TP",
            "FP",
            "High-FP"
        ]
    )

    plt.title(title)

    plt.ylabel(feature)

    plt.grid(
        axis="y",
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / filename,
        dpi=150
    )

    plt.close()


plot_feature(
    "mean_high_frequency_ratio",
    "High-Frequency Ratio: TP vs FP",
    "high_frequency_ratio_comparison.png"
)

plot_feature(
    "mean_zero_crossing_rate",
    "Zero-Crossing Rate: TP vs FP",
    "zero_crossing_comparison.png"
)

plot_feature(
    "mean_line_length",
    "Line Length: TP vs FP",
    "line_length_comparison.png"
)

plot_feature(
    "mean_rms",
    "RMS: TP vs FP",
    "rms_comparison.png"
)


# ============================================================
# SAVE JSON
# ============================================================

results = {

    "settings": {
        "threshold": THRESHOLD,
        "high_confidence_threshold":
            HIGH_CONFIDENCE_THRESHOLD,
    },

    "classification": {
        "tp": int(len(tp_positions)),
        "fp": int(len(fp_positions)),
        "high_confidence_fp":
            int(len(high_fp_positions)),
    },

    "tp_summary":
        tp_summary,

    "fp_summary":
        fp_summary,

    "high_confidence_fp_summary":
        high_fp_summary,

    "channel_comparison":
        channel_comparison,

    "fp_dominant_high_frequency_channels":
        fp_dominant_counts,

    "high_fp_dominant_high_frequency_channels":
        high_fp_dominant_counts,

    "patient_analysis":
        patient_results,
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

print(
    "\nPlots saved to:"
)

print(
    OUTPUT_DIR
)

print("\nDONE")