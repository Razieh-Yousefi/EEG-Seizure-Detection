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

OUTPUT_JSON = RESULTS_DIR / "fp_tp_channel_analysis.json"
OUTPUT_DIR = RESULTS_DIR / "fp_tp_channel_plots"


# ============================================================
# SETTINGS
# ============================================================

THRESHOLD = 0.50

EXPECTED_CHANNELS = 23
EXPECTED_SAMPLES = 1280

TOP_CHANNELS = 10


# ============================================================
# CHECK FILES
# ============================================================

print("=" * 70)
print("FP vs TP CHANNEL-WISE ANALYSIS")
print("=" * 70)

print("\nLoading files...")

for path in [
    X_PATH,
    NORMALIZATION_PATH,
    PROBABILITY_PATH
]:

    if not path.exists():

        raise FileNotFoundError(
            f"File not found:\n{path}"
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


print("\nX shape:", X.shape)


# ============================================================
# VALIDATION
# ============================================================

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
# LOAD PREDICTION DATA
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
).reshape(-1
)


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


tp_positions = np.where(
    tp_mask
)[0]

fp_positions = np.where(
    fp_mask
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


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def load_normalized_window(
    sample_index
):

    sample_index = int(
        sample_index
    )

    raw = np.asarray(
        X[sample_index],
        dtype=np.float32
    )

    normalized = (
        raw -
        channel_mean[:, None]
    ) / channel_std[:, None]

    return normalized


def calculate_channel_features(
    window
):

    features = {

        "rms": [],
        "std": [],
        "variance": [],
        "peak_to_peak": [],
        "max_abs": [],
        "line_length": [],
        "zero_crossing_rate": [],

    }


    for channel in window:

        centered = (
            channel -
            np.mean(channel)
        )

        signs = np.sign(
            centered
        )

        crossings = np.sum(
            signs[:-1] !=
            signs[1:]
        )

        zcr = (
            crossings /
            max(
                1,
                len(channel) - 1
            )
        )


        features["rms"].append(
            np.sqrt(
                np.mean(
                    channel ** 2
                )
            )
        )

        features["std"].append(
            np.std(channel)
        )

        features["variance"].append(
            np.var(channel)
        )

        features["peak_to_peak"].append(
            np.max(channel) -
            np.min(channel)
        )

        features["max_abs"].append(
            np.max(
                np.abs(channel)
            )
        )

        features["line_length"].append(
            np.sum(
                np.abs(
                    np.diff(channel)
                )
            )
        )

        features[
            "zero_crossing_rate"
        ].append(
            zcr
        )


    return {
        key: np.asarray(
            value,
            dtype=np.float64
        )
        for key, value
        in features.items()
    }


# ============================================================
# COLLECT CHANNEL FEATURES
# ============================================================

FEATURE_NAMES = [

    "rms",
    "std",
    "variance",
    "peak_to_peak",
    "max_abs",
    "line_length",
    "zero_crossing_rate",

]


def collect_features(
    positions,
    name
):

    collected = {
        feature: []
        for feature
        in FEATURE_NAMES
    }


    print(
        f"\nExtracting {name}..."
    )


    for position in positions:

        sample_index = int(
            test_indices[position]
        )

        window = load_normalized_window(
            sample_index
        )

        features = calculate_channel_features(
            window
        )


        for feature in FEATURE_NAMES:

            collected[
                feature
            ].append(
                features[feature]
            )


    return {
        feature: np.stack(
            values,
            axis=0
        )
        for feature, values
        in collected.items()
    }


tp_features = collect_features(
    tp_positions,
    "TP windows"
)

fp_features = collect_features(
    fp_positions,
    "FP windows"
)


# ============================================================
# CHANNEL-WISE COMPARISON
# ============================================================

print("\n" + "=" * 70)
print("CHANNEL-WISE COMPARISON")
print("=" * 70)


channel_results = {}


for feature in FEATURE_NAMES:

    tp_values = tp_features[
        feature
    ]

    fp_values = fp_features[
        feature
    ]


    tp_mean = np.mean(
        tp_values,
        axis=0
    )

    fp_mean = np.mean(
        fp_values,
        axis=0
    )


    difference = (
        fp_mean -
        tp_mean
    )


    relative_difference = (
        difference /
        (
            np.abs(tp_mean) +
            1e-8
        )
    )


    channel_results[
        feature
    ] = {

        "tp_mean": tp_mean.tolist(),

        "fp_mean": fp_mean.tolist(),

        "difference": difference.tolist(),

        "relative_difference":
            relative_difference.tolist(),

    }


# ============================================================
# PRINT TOP CHANNELS
# ============================================================

for feature in FEATURE_NAMES:

    relative_difference = np.asarray(
        channel_results[
            feature
        ][
            "relative_difference"
        ]
    )


    ranking = np.argsort(
        np.abs(
            relative_difference
        )
    )[::-1]


    print(
        f"\n{feature.upper()}"
    )


    for channel_index in ranking[
        :TOP_CHANNELS
    ]:

        print(
            f"Channel {channel_index + 1:02d} | "
            f"relative difference = "
            f"{relative_difference[channel_index]:+.4f}"
        )


# ============================================================
# PATIENT-LEVEL CHANNEL ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("PATIENT-LEVEL CHANNEL ANALYSIS")
print("=" * 70)


unique_patients = np.unique(
    patients[fp_positions]
)


patient_results = {}


for patient in unique_patients:

    patient_positions = fp_positions[
        patients[fp_positions] ==
        patient
    ]


    print(
        f"\nPatient {patient}: "
        f"{len(patient_positions)} FP windows"
    )


    patient_features = collect_features(
        patient_positions,
        f"FP - {patient}"
    )


    patient_summary = {}


    for feature in FEATURE_NAMES:

        values = patient_features[
            feature
        ]


        patient_summary[
            feature
        ] = {

            "mean_by_channel":
                np.mean(
                    values,
                    axis=0
                ).tolist(),

            "median_by_channel":
                np.median(
                    values,
                    axis=0
                ).tolist(),

        }


    patient_results[
        str(patient)
    ] = {

        "count":
            int(
                len(patient_positions)
            ),

        "features":
            patient_summary,

    }


# ============================================================
# VISUALIZATION
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# CHANNEL HEATMAPS
# ============================================================

for feature in [

    "rms",
    "variance",
    "peak_to_peak",
    "line_length",
    "zero_crossing_rate",

]:

    tp_mean = np.mean(
        tp_features[feature],
        axis=0
    )

    fp_mean = np.mean(
        fp_features[feature],
        axis=0
    )


    ratio = (
        fp_mean /
        (
            tp_mean +
            1e-8
        )
    )


    plt.figure(
        figsize=(14, 5)
    )


    plt.bar(
        np.arange(
            EXPECTED_CHANNELS
        ),
        ratio
    )


    plt.axhline(
        1.0,
        linestyle="--",
        linewidth=1
    )


    plt.xticks(
        np.arange(
            EXPECTED_CHANNELS
        ),
        [
            f"Ch {i + 1}"
            for i in range(
                EXPECTED_CHANNELS
            )
        ],
        rotation=45
    )


    plt.ylabel(
        "FP / TP ratio"
    )


    plt.title(
        f"FP vs TP Channel Ratio: {feature}"
    )


    plt.grid(
        axis="y",
        alpha=0.3
    )


    plt.tight_layout()


    output_path = (
        OUTPUT_DIR /
        f"{feature}_channel_ratio.png"
    )


    plt.savefig(
        output_path,
        dpi=150
    )


    plt.close()


# ============================================================
# ZERO CROSSING CHANNEL ANALYSIS
# ============================================================

zcr_tp = np.mean(
    tp_features[
        "zero_crossing_rate"
    ],
    axis=0
)

zcr_fp = np.mean(
    fp_features[
        "zero_crossing_rate"
    ],
    axis=0
)


plt.figure(
    figsize=(14, 5)
)


plt.plot(
    np.arange(
        EXPECTED_CHANNELS
    ),
    zcr_tp,
    marker="o",
    label="TP"
)


plt.plot(
    np.arange(
        EXPECTED_CHANNELS
    ),
    zcr_fp,
    marker="o",
    label="FP"
)


plt.xticks(
    np.arange(
        EXPECTED_CHANNELS
    ),
    [
        f"Ch {i + 1}"
        for i in range(
            EXPECTED_CHANNELS
        )
    ],
    rotation=45
)


plt.ylabel(
    "Zero Crossing Rate"
)


plt.title(
    "TP vs FP Zero Crossing Rate by Channel"
)


plt.legend()


plt.grid(
    alpha=0.3
)


plt.tight_layout()


plt.savefig(
    OUTPUT_DIR /
    "zero_crossing_rate_by_channel.png",
    dpi=150
)


plt.close()


# ============================================================
# SAVE RESULTS
# ============================================================

results = {

    "settings": {

        "threshold":
            THRESHOLD,

        "expected_channels":
            EXPECTED_CHANNELS,

        "expected_samples":
            EXPECTED_SAMPLES,

    },


    "classification": {

        "tp":
            int(len(tp_positions)),

        "fp":
            int(len(fp_positions)),

    },


    "channel_results":
        channel_results,


    "patient_results":
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