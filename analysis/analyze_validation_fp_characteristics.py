# ================================================================
# analyze_validation_fp_characteristics.py
#
# Validation-only analysis of signal characteristics:
#
# Compares:
#   - Validation True Positives (TP)
#   - Validation False Positives (FP)
#
# Features:
#   - RMS
#   - Peak-to-Peak
#   - Standard Deviation
#   - Variance
#   - Line Length
#   - Zero Crossing Rate
#   - Per-channel RMS
#   - Per-channel Peak-to-Peak
#
# IMPORTANT:
#   - Does NOT modify model
#   - Does NOT modify dataset
#   - Does NOT use test probabilities for optimization
#   - Does NOT change threshold
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

DATA_DIR = os.path.join(
    PROJECT_DIR,
    "data"
)

RESULTS_DIR = os.path.join(
    PROJECT_DIR,
    "results"
)

VAL_PROB_FILE = os.path.join(
    RESULTS_DIR,
    "validation_window_probabilities.npz"
)

THRESHOLD_FILE = os.path.join(
    RESULTS_DIR,
    "validation_threshold_results.json"
)

X_FILE = os.path.join(
    DATA_DIR,
    "X_chbmit_full.npy"
)

OUTPUT_FILE = os.path.join(
    RESULTS_DIR,
    "validation_fp_characteristics.json"
)


# ================================================================
# 2. HEADER
# ================================================================

print()
print("=" * 70)
print("VALIDATION FP CHARACTERISTICS ANALYSIS")
print("=" * 70)

print()
print("Project directory:")
print(PROJECT_DIR)

print()
print("Data directory:")
print(DATA_DIR)

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
    VAL_PROB_FILE,
    THRESHOLD_FILE,
    X_FILE,
]

for path in required_files:

    if os.path.exists(path):

        print("[OK]", path)

    else:

        raise FileNotFoundError(
            f"\nRequired file not found:\n{path}"
        )


# ================================================================
# 4. LOAD VALIDATION PROBABILITIES
# ================================================================

print()
print("=" * 70)
print("2. LOADING VALIDATION PROBABILITIES")
print("=" * 70)

val_data = np.load(
    VAL_PROB_FILE,
    allow_pickle=True
)

validation_indices = np.asarray(
    val_data["validation_indices"],
    dtype=np.int64
)

labels = np.asarray(
    val_data["labels"],
    dtype=np.int64
)

probabilities = np.asarray(
    val_data["probabilities"],
    dtype=np.float32
)

patients = np.asarray(
    val_data["patients"]
)

print()
print("Validation samples:", len(probabilities))
print("Probability shape:", probabilities.shape)
print("Labels shape:", labels.shape)
print("Patients shape:", patients.shape)


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
# 6. VERIFY ALIGNMENT
# ================================================================

print()
print("=" * 70)
print("4. VERIFYING ARRAY ALIGNMENT")
print("=" * 70)

n = len(probabilities)

if len(validation_indices) != n:
    raise RuntimeError(
        "Validation index count mismatch."
    )

if len(labels) != n:
    raise RuntimeError(
        "Label count mismatch."
    )

if len(patients) != n:
    raise RuntimeError(
        "Patient count mismatch."
    )

if not np.all(np.isfinite(probabilities)):
    raise RuntimeError(
        "Probabilities contain NaN or Inf."
    )

if not np.all(
    np.isin(labels, [0, 1])
):
    raise RuntimeError(
        "Labels contain values other than 0 and 1."
    )

print()
print("[OK] Arrays are aligned.")
print("[OK] Probabilities are finite.")


# ================================================================
# 7. CLASSIFICATION
# ================================================================

predictions = (
    probabilities >= threshold
).astype(np.int64)

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

tn_mask = (
    (labels == 0) &
    (predictions == 0)
)

tp_positions = np.where(tp_mask)[0]
fp_positions = np.where(fp_mask)[0]
fn_positions = np.where(fn_mask)[0]
tn_positions = np.where(tn_mask)[0]


# ================================================================
# 8. BASELINE
# ================================================================

print()
print("=" * 70)
print("5. VALIDATION BASELINE")
print("=" * 70)

tp = int(np.sum(tp_mask))
fp = int(np.sum(fp_mask))
fn = int(np.sum(fn_mask))
tn = int(np.sum(tn_mask))

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
    2 * precision * sensitivity /
    (precision + sensitivity)
    if (precision + sensitivity) > 0
    else 0.0
)

print()
print("TP:", tp)
print("FP:", fp)
print("FN:", fn)
print("TN:", tn)

print()
print("Sensitivity:", f"{sensitivity:.6f}")
print("Specificity:", f"{specificity:.6f}")
print("Precision:", f"{precision:.6f}")
print("F1:", f"{f1:.6f}")


# ================================================================
# 9. LOAD RAW EEG
# ================================================================

print()
print("=" * 70)
print("6. LOADING RAW EEG")
print("=" * 70)

X = np.load(
    X_FILE,
    mmap_mode="r"
)

print()
print("Full X shape:", X.shape)

if X.ndim != 3:
    raise RuntimeError(
        "Expected X to have shape "
        "(samples, channels, time)."
    )

n_channels = X.shape[1]
n_time = X.shape[2]

print()
print("Channels:", n_channels)
print("Time points:", n_time)


# ================================================================
# 10. FEATURE FUNCTIONS
# ================================================================

def calculate_rms(signal):

    signal = np.asarray(
        signal,
        dtype=np.float64
    )

    return float(
        np.sqrt(
            np.mean(
                signal * signal
            )
        )
    )


def calculate_ptp(signal):

    signal = np.asarray(
        signal,
        dtype=np.float64
    )

    return float(
        np.max(signal) -
        np.min(signal)
    )


def calculate_std(signal):

    signal = np.asarray(
        signal,
        dtype=np.float64
    )

    return float(
        np.std(signal)
    )


def calculate_variance(signal):

    signal = np.asarray(
        signal,
        dtype=np.float64
    )

    return float(
        np.var(signal)
    )


def calculate_line_length(signal):

    signal = np.asarray(
        signal,
        dtype=np.float64
    )

    if len(signal) < 2:
        return 0.0

    return float(
        np.sum(
            np.abs(
                np.diff(signal)
            )
        )
    )


def calculate_zero_crossing_rate(signal):

    signal = np.asarray(
        signal,
        dtype=np.float64
    )

    if len(signal) < 2:
        return 0.0

    signs = np.sign(signal)

    crossings = np.sum(
        signs[:-1] *
        signs[1:] < 0
    )

    return float(
        crossings /
        (len(signal) - 1)
    )


# ================================================================
# 11. FEATURE EXTRACTION
# ================================================================

def extract_features(
    sample_indices,
    name
):

    print()
    print(
        f"Extracting features for {name}..."
    )

    if len(sample_indices) == 0:

        return {
            "count": 0,
            "global_rms": [],
            "global_ptp": [],
            "global_std": [],
            "global_variance": [],
            "global_line_length": [],
            "global_zero_crossing_rate": [],
            "channel_rms": [],
            "channel_ptp": [],
        }

    global_rms = []
    global_ptp = []
    global_std = []
    global_variance = []
    global_line_length = []
    global_zcr = []

    channel_rms = []
    channel_ptp = []

    total = len(sample_indices)

    for counter, position in enumerate(
        sample_indices,
        start=1
    ):

        original_index = int(
            validation_indices[position]
        )

        sample = np.asarray(
            X[original_index],
            dtype=np.float64
        )

        if sample.shape != (
            n_channels,
            n_time
        ):

            raise RuntimeError(
                f"Unexpected sample shape: "
                f"{sample.shape}"
            )

        # --------------------------------------------------------
        # Global signal
        # --------------------------------------------------------

        flattened = sample.reshape(-1)

        global_rms.append(
            calculate_rms(
                flattened
            )
        )

        global_ptp.append(
            calculate_ptp(
                flattened
            )
        )

        global_std.append(
            calculate_std(
                flattened
            )
        )

        global_variance.append(
            calculate_variance(
                flattened
            )
        )

        global_line_length.append(
            calculate_line_length(
                flattened
            )
        )

        global_zcr.append(
            calculate_zero_crossing_rate(
                flattened
            )
        )

        # --------------------------------------------------------
        # Per-channel features
        # --------------------------------------------------------

        sample_channel_rms = []
        sample_channel_ptp = []

        for channel in range(
            n_channels
        ):

            channel_signal = sample[
                channel
            ]

            sample_channel_rms.append(
                calculate_rms(
                    channel_signal
                )
            )

            sample_channel_ptp.append(
                calculate_ptp(
                    channel_signal
                )
            )

        channel_rms.append(
            sample_channel_rms
        )

        channel_ptp.append(
            sample_channel_ptp
        )

        if (
            counter == 1
            or counter % 50 == 0
            or counter == total
        ):

            print(
                f"Processed "
                f"{counter}/{total}"
            )

    return {
        "count": int(total),

        "global_rms": global_rms,

        "global_ptp": global_ptp,

        "global_std": global_std,

        "global_variance": global_variance,

        "global_line_length":
            global_line_length,

        "global_zero_crossing_rate":
            global_zcr,

        "channel_rms": channel_rms,

        "channel_ptp": channel_ptp,
    }


# ================================================================
# 12. EXTRACT TP FEATURES
# ================================================================

print()
print("=" * 70)
print("7. EXTRACTING TP FEATURES")
print("=" * 70)

tp_features = extract_features(
    tp_positions,
    "VALIDATION TP"
)


# ================================================================
# 13. EXTRACT FP FEATURES
# ================================================================

print()
print("=" * 70)
print("8. EXTRACTING FP FEATURES")
print("=" * 70)

fp_features = extract_features(
    fp_positions,
    "VALIDATION FP"
)


# ================================================================
# 14. SUMMARY FUNCTION
# ================================================================

def summarize_array(values):

    arr = np.asarray(
        values,
        dtype=np.float64
    )

    if arr.size == 0:

        return {
            "count": 0
        }

    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "q25": float(np.percentile(arr, 25)),
        "q75": float(np.percentile(arr, 75)),
        "q90": float(np.percentile(arr, 90)),
        "q95": float(np.percentile(arr, 95)),
        "max": float(np.max(arr)),
    }


# ================================================================
# 15. GLOBAL COMPARISON
# ================================================================

print()
print("=" * 70)
print("9. GLOBAL FEATURE COMPARISON")
print("=" * 70)


global_feature_names = [
    "global_rms",
    "global_ptp",
    "global_std",
    "global_variance",
    "global_line_length",
    "global_zero_crossing_rate",
]


global_comparison = {}


for feature_name in global_feature_names:

    tp_summary = summarize_array(
        tp_features[feature_name]
    )

    fp_summary = summarize_array(
        fp_features[feature_name]
    )

    global_comparison[
        feature_name
    ] = {
        "tp": tp_summary,
        "fp": fp_summary,
    }

    print()
    print(
        feature_name
    )

    if tp_summary.get("count", 0) > 0:

        print(
            "  TP mean   :",
            f"{tp_summary['mean']:.8f}"
        )

        print(
            "  TP median :",
            f"{tp_summary['median']:.8f}"
        )

    if fp_summary.get("count", 0) > 0:

        print(
            "  FP mean   :",
            f"{fp_summary['mean']:.8f}"
        )

        print(
            "  FP median :",
            f"{fp_summary['median']:.8f}"
        )


# ================================================================
# 16. CHANNEL COMPARISON
# ================================================================

print()
print("=" * 70)
print("10. CHANNEL-LEVEL COMPARISON")
print("=" * 70)

tp_channel_rms = np.asarray(
    tp_features["channel_rms"],
    dtype=np.float64
)

fp_channel_rms = np.asarray(
    fp_features["channel_rms"],
    dtype=np.float64
)

tp_channel_ptp = np.asarray(
    tp_features["channel_ptp"],
    dtype=np.float64
)

fp_channel_ptp = np.asarray(
    fp_features["channel_ptp"],
    dtype=np.float64
)


channel_comparison = []


for channel in range(
    n_channels
):

    if len(tp_channel_rms) > 0:

        tp_rms_mean = float(
            np.mean(
                tp_channel_rms[:, channel]
            )
        )

        tp_rms_median = float(
            np.median(
                tp_channel_rms[:, channel]
            )
        )

    else:

        tp_rms_mean = None
        tp_rms_median = None

    if len(fp_channel_rms) > 0:

        fp_rms_mean = float(
            np.mean(
                fp_channel_rms[:, channel]
            )
        )

        fp_rms_median = float(
            np.median(
                fp_channel_rms[:, channel]
            )
        )

    else:

        fp_rms_mean = None
        fp_rms_median = None

    if len(tp_channel_ptp) > 0:

        tp_ptp_mean = float(
            np.mean(
                tp_channel_ptp[:, channel]
            )
        )

        tp_ptp_median = float(
            np.median(
                tp_channel_ptp[:, channel]
            )
        )

    else:

        tp_ptp_mean = None
        tp_ptp_median = None

    if len(fp_channel_ptp) > 0:

        fp_ptp_mean = float(
            np.mean(
                fp_channel_ptp[:, channel]
            )
        )

        fp_ptp_median = float(
            np.median(
                fp_channel_ptp[:, channel]
            )
        )

    else:

        fp_ptp_mean = None
        fp_ptp_median = None

    rms_ratio = (
        fp_rms_mean / tp_rms_mean
        if (
            fp_rms_mean is not None
            and tp_rms_mean is not None
            and tp_rms_mean != 0
        )
        else None
    )

    ptp_ratio = (
        fp_ptp_mean / tp_ptp_mean
        if (
            fp_ptp_mean is not None
            and tp_ptp_mean is not None
            and tp_ptp_mean != 0
        )
        else None
    )

    channel_comparison.append(
        {
            "channel": int(channel),

            "tp_rms_mean":
                tp_rms_mean,

            "tp_rms_median":
                tp_rms_median,

            "fp_rms_mean":
                fp_rms_mean,

            "fp_rms_median":
                fp_rms_median,

            "rms_ratio_fp_tp":
                rms_ratio,

            "tp_ptp_mean":
                tp_ptp_mean,

            "tp_ptp_median":
                tp_ptp_median,

            "fp_ptp_mean":
                fp_ptp_mean,

            "fp_ptp_median":
                fp_ptp_median,

            "ptp_ratio_fp_tp":
                ptp_ratio,
        }
    )


# ================================================================
# 17. TOP CHANNELS
# ================================================================

print()
print("=" * 70)
print("11. CHANNEL RANKING")
print("=" * 70)


valid_rms_channels = [
    item
    for item in channel_comparison
    if item["rms_ratio_fp_tp"] is not None
]

valid_ptp_channels = [
    item
    for item in channel_comparison
    if item["ptp_ratio_fp_tp"] is not None
]


top_rms_channels = sorted(
    valid_rms_channels,
    key=lambda x: x[
        "rms_ratio_fp_tp"
    ],
    reverse=True
)


top_ptp_channels = sorted(
    valid_ptp_channels,
    key=lambda x: x[
        "ptp_ratio_fp_tp"
    ],
    reverse=True
)


print()
print(
    "Top channels by FP/TP RMS ratio:"
)

for item in top_rms_channels[:10]:

    print(
        f"Channel {item['channel']:2d} | "
        f"ratio={item['rms_ratio_fp_tp']:.4f} | "
        f"TP={item['tp_rms_mean']:.6f} | "
        f"FP={item['fp_rms_mean']:.6f}"
    )


print()
print(
    "Top channels by FP/TP PTP ratio:"
)

for item in top_ptp_channels[:10]:

    print(
        f"Channel {item['channel']:2d} | "
        f"ratio={item['ptp_ratio_fp_tp']:.4f} | "
        f"TP={item['tp_ptp_mean']:.6f} | "
        f"FP={item['fp_ptp_mean']:.6f}"
    )


# ================================================================
# 18. SIMPLE EFFECT-SIZE-LIKE SEPARATION
# ================================================================

def standardized_difference(
    tp_values,
    fp_values
):

    tp_arr = np.asarray(
        tp_values,
        dtype=np.float64
    )

    fp_arr = np.asarray(
        fp_values,
        dtype=np.float64
    )

    if (
        len(tp_arr) == 0
        or len(fp_arr) == 0
    ):

        return None

    tp_mean = np.mean(tp_arr)
    fp_mean = np.mean(fp_arr)

    tp_var = np.var(
        tp_arr,
        ddof=1
    ) if len(tp_arr) > 1 else 0.0

    fp_var = np.var(
        fp_arr,
        ddof=1
    ) if len(fp_arr) > 1 else 0.0

    pooled = np.sqrt(
        (
            tp_var +
            fp_var
        ) / 2.0
    )

    if pooled == 0:

        return 0.0

    return float(
        (
            fp_mean -
            tp_mean
        ) / pooled
    )


separation = {}


for feature_name in global_feature_names:

    separation[
        feature_name
    ] = standardized_difference(
        tp_features[feature_name],
        fp_features[feature_name]
    )


# ================================================================
# 19. PROBABILITY COMPARISON
# ================================================================

print()
print("=" * 70)
print("12. PROBABILITY COMPARISON")
print("=" * 70)

tp_probabilities = probabilities[
    tp_positions
]

fp_probabilities = probabilities[
    fp_positions
]

tp_probability_summary = summarize_array(
    tp_probabilities
)

fp_probability_summary = summarize_array(
    fp_probabilities
)

print()
print("TP probability:")
print(
    json.dumps(
        tp_probability_summary,
        indent=2
    )
)

print()
print("FP probability:")
print(
    json.dumps(
        fp_probability_summary,
        indent=2
    )
)


# ================================================================
# 20. SAVE RAW FEATURE ARRAYS
# ================================================================

RAW_FEATURE_FILE = os.path.join(
    RESULTS_DIR,
    "validation_fp_feature_arrays.npz"
)

np.savez(
    RAW_FEATURE_FILE,

    tp_indices=
        validation_indices[
            tp_positions
        ],

    fp_indices=
        validation_indices[
            fp_positions
        ],

    tp_probabilities=
        tp_probabilities,

    fp_probabilities=
        fp_probabilities,

    tp_global_rms=
        np.asarray(
            tp_features["global_rms"],
            dtype=np.float32
        ),

    fp_global_rms=
        np.asarray(
            fp_features["global_rms"],
            dtype=np.float32
        ),

    tp_global_ptp=
        np.asarray(
            tp_features["global_ptp"],
            dtype=np.float32
        ),

    fp_global_ptp=
        np.asarray(
            fp_features["global_ptp"],
            dtype=np.float32
        ),

    tp_global_std=
        np.asarray(
            tp_features["global_std"],
            dtype=np.float32
        ),

    fp_global_std=
        np.asarray(
            fp_features["global_std"],
            dtype=np.float32
        ),

    tp_global_variance=
        np.asarray(
            tp_features["global_variance"],
            dtype=np.float32
        ),

    fp_global_variance=
        np.asarray(
            fp_features["global_variance"],
            dtype=np.float32
        ),

    tp_line_length=
        np.asarray(
            tp_features["global_line_length"],
            dtype=np.float32
        ),

    fp_line_length=
        np.asarray(
            fp_features["global_line_length"],
            dtype=np.float32
        ),

    tp_zero_crossing_rate=
        np.asarray(
            tp_features[
                "global_zero_crossing_rate"
            ],
            dtype=np.float32
        ),

    fp_zero_crossing_rate=
        np.asarray(
            fp_features[
                "global_zero_crossing_rate"
            ],
            dtype=np.float32
        ),

    tp_channel_rms=
        tp_channel_rms.astype(
            np.float32
        ),

    fp_channel_rms=
        fp_channel_rms.astype(
            np.float32
        ),

    tp_channel_ptp=
        tp_channel_ptp.astype(
            np.float32
        ),

    fp_channel_ptp=
        fp_channel_ptp.astype(
            np.float32
        ),
)

print()
print(
    "[OK] Raw feature arrays saved:"
)

print(
    RAW_FEATURE_FILE
)


# ================================================================
# 21. SAVE JSON SUMMARY
# ================================================================

results = {

    "analysis": (
        "Validation-only comparison "
        "of TP and FP signal characteristics."
    ),

    "validation_threshold":
        threshold,

    "validation_samples":
        int(n),

    "channels":
        int(n_channels),

    "time_points":
        int(n_time),

    "baseline": {

        "tp": tp,

        "fp": fp,

        "fn": fn,

        "tn": tn,

        "sensitivity":
            sensitivity,

        "specificity":
            specificity,

        "precision":
            precision,

        "f1":
            f1,
    },

    "tp_count":
        int(len(tp_positions)),

    "fp_count":
        int(len(fp_positions)),

    "global_feature_comparison":
        global_comparison,

    "channel_comparison":
        channel_comparison,

    "top_rms_channels":
        top_rms_channels[:10],

    "top_ptp_channels":
        top_ptp_channels[:10],

    "feature_separation":
        separation,

    "tp_probability":
        tp_probability_summary,

    "fp_probability":
        fp_probability_summary,

    "note": (
        "This analysis uses validation data only. "
        "No model, dataset, threshold, or test "
        "prediction was modified."
    ),
}


# ================================================================
# 22. SAVE
# ================================================================

print()
print("=" * 70)
print("13. SAVING RESULTS")
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
print("[OK] JSON results saved:")

print(
    OUTPUT_FILE
)


# ================================================================
# 23. FINAL
# ================================================================

print()
print("=" * 70)
print("VALIDATION FP CHARACTERISTICS ANALYSIS COMPLETED")
print("=" * 70)

print()
print("Model was NOT modified.")
print("Dataset was NOT modified.")
print("Threshold was NOT modified.")
print("Test data was NOT used for optimization.")

print()
print("Output files:")

print(
    OUTPUT_FILE
)

print(
    RAW_FEATURE_FILE
)

print()
print("=" * 70)