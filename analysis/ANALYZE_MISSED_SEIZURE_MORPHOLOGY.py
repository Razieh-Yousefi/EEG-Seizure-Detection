# ============================================================
# ANALYZE_MISSED_SEIZURE_MORPHOLOGY.py
#
# Purpose:
# Direct morphological analysis of the missed seizure
# compared with detected seizures.
#
# IMPORTANT:
# - Model is NOT modified.
# - Model is NOT retrained.
# - Threshold is NOT changed.
# - Original X_test/y_test files are NOT modified.
# ============================================================

import os
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_DIR = r"C:\Users\rezay\Desktop\EEG_Seizure_Project"

ANALYSIS_DIR = os.path.join(
    PROJECT_DIR,
    "missed_seizure_analysis"
)

OUTPUT_DIR = os.path.join(
    ANALYSIS_DIR,
    "morphology_analysis"
)

X_PATH = os.path.join(
    PROJECT_DIR,
    "X_test_full.npy"
)

Y_PATH = os.path.join(
    PROJECT_DIR,
    "y_test_full.npy"
)

MODEL_PATH = os.path.join(
    PROJECT_DIR,
    "eeg_seizure_cnn_full_model.keras"
)

SALIENCY_CHANNEL_PATH = os.path.join(
    ANALYSIS_DIR,
    "saliency",
    "missed_seizure_2815_channel_saliency.csv"
)

SALIENCY_TIME_PATH = os.path.join(
    ANALYSIS_DIR,
    "saliency",
    "missed_seizure_2815_time_saliency.csv"
)

COUNTERFACTUAL_PATH = os.path.join(
    ANALYSIS_DIR,
    "counterfactual_analysis",
    "channel_ablation_results.csv"
)

TARGET_INDEX = 2815
THRESHOLD = 0.5

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def safe_float(value):
    try:
        return float(value)
    except Exception:
        return np.nan


def zero_crossing_rate(signal):
    signal = np.asarray(signal, dtype=np.float64)

    if len(signal) < 2:
        return 0.0

    centered = signal - np.mean(signal)

    return np.mean(
        np.diff(np.signbit(centered)) != 0
    )


def rms(signal):
    signal = np.asarray(signal, dtype=np.float64)
    return np.sqrt(np.mean(signal ** 2))


def line_length(signal):
    signal = np.asarray(signal, dtype=np.float64)

    if len(signal) < 2:
        return 0.0

    return np.sum(np.abs(np.diff(signal)))


def crest_factor(signal):
    signal = np.asarray(signal, dtype=np.float64)

    r = rms(signal)

    if r == 0:
        return 0.0

    return np.max(np.abs(signal)) / r


def hjorth_activity(signal):
    signal = np.asarray(signal, dtype=np.float64)

    return np.var(signal)


def hjorth_mobility(signal):
    signal = np.asarray(signal, dtype=np.float64)

    activity = np.var(signal)

    if activity <= 0:
        return 0.0

    diff_signal = np.diff(signal)

    return np.sqrt(
        np.var(diff_signal) / activity
    )


def hjorth_complexity(signal):
    signal = np.asarray(signal, dtype=np.float64)

    mobility = hjorth_mobility(signal)

    if mobility == 0:
        return 0.0

    diff_signal = np.diff(signal)

    mobility_diff = hjorth_mobility(diff_signal)

    return mobility_diff / mobility


def extract_features(signal):

    signal = np.asarray(
        signal,
        dtype=np.float64
    )

    mean_value = np.mean(signal)

    std_value = np.std(signal)

    rms_value = rms(signal)

    max_abs = np.max(
        np.abs(signal)
    )

    min_value = np.min(signal)

    max_value = np.max(signal)

    peak_to_peak = (
        max_value - min_value
    )

    mean_absolute = np.mean(
        np.abs(signal)
    )

    zcr = zero_crossing_rate(signal)

    ll = line_length(signal)

    cf = crest_factor(signal)

    activity = hjorth_activity(signal)

    mobility = hjorth_mobility(signal)

    complexity = hjorth_complexity(signal)

    return {
        "Mean": mean_value,
        "STD": std_value,
        "RMS": rms_value,
        "Mean_Absolute": mean_absolute,
        "Max_Absolute": max_abs,
        "Peak_to_Peak": peak_to_peak,
        "Zero_Crossing_Rate": zcr,
        "Line_Length": ll,
        "Crest_Factor": cf,
        "Hjorth_Activity": activity,
        "Hjorth_Mobility": mobility,
        "Hjorth_Complexity": complexity
    }


def normalize_for_distance(
    missed_value,
    detected_values
):

    detected_values = np.asarray(
        detected_values,
        dtype=np.float64
    )

    mean_val = np.nanmean(
        detected_values
    )

    std_val = np.nanstd(
        detected_values
    )

    if std_val == 0 or np.isnan(std_val):
        return 0.0

    return abs(
        missed_value - mean_val
    ) / std_val


# ============================================================
# HEADER
# ============================================================

print()
print("=" * 75)
print("MISSED SEIZURE MORPHOLOGY ANALYSIS")
print("=" * 75)

print("Project directory:")
print(PROJECT_DIR)

print()
print("Analysis directory:")
print(ANALYSIS_DIR)

print()
print("Output directory:")
print(OUTPUT_DIR)

print()
print("Target missed seizure:")
print(f"Index = {TARGET_INDEX}")

print(f"Threshold = {THRESHOLD}")

print()
print("Model changed: NO")
print("Model retrained: NO")
print("Threshold changed: NO")

# ============================================================
# 1. CHECK FILES
# ============================================================

print()
print("=" * 75)
print("1. CHECKING REQUIRED FILES")
print("=" * 75)

required_files = [
    X_PATH,
    Y_PATH,
    MODEL_PATH
]

for path in required_files:

    print(path)

    if not os.path.exists(path):

        raise FileNotFoundError(
            "\nREQUIRED FILE NOT FOUND:\n" + path
        )

    print("[FOUND]")

# ============================================================
# 2. LOAD DATA
# ============================================================

print()
print("=" * 75)
print("2. LOADING DATA")
print("=" * 75)

X_test = np.load(
    X_PATH,
    allow_pickle=False
)

y_test = np.load(
    Y_PATH,
    allow_pickle=False
)

print(
    "X_test shape:",
    X_test.shape
)

print(
    "y_test shape:",
    y_test.shape
)

if TARGET_INDEX >= len(X_test):

    raise IndexError(
        "TARGET_INDEX is outside X_test."
    )

target_label = int(
    y_test[TARGET_INDEX]
)

target_sample = np.asarray(
    X_test[TARGET_INDEX],
    dtype=np.float64
)

print(
    "Target label:",
    target_label
)

print(
    "Target sample shape:",
    target_sample.shape
)

if target_label != 1:

    print()
    print(
        "WARNING: Target sample does not have label 1."
    )

# ============================================================
# 3. DETERMINE DATA FORMAT
# ============================================================

print()
print("=" * 75)
print("3. DETERMINING EEG DATA FORMAT")
print("=" * 75)

if target_sample.ndim != 2:

    raise ValueError(
        "Expected target sample to be 2-dimensional."
    )

shape_a, shape_b = target_sample.shape

if shape_a == 23:

    channel_axis = 0
    num_channels = shape_a
    num_time_points = shape_b

elif shape_b == 23:

    channel_axis = 1
    num_channels = shape_b
    num_time_points = shape_a

else:

    raise ValueError(
        "Could not identify the 23-channel dimension."
    )

print(
    "Channel axis:",
    channel_axis
)

print(
    "Number of channels:",
    num_channels
)

print(
    "Number of time points:",
    num_time_points
)

# Convert target to [channels, time]
if channel_axis == 0:

    target_ct = target_sample

else:

    target_ct = target_sample.T

# ============================================================
# 4. FIND DETECTED SEIZURES
# ============================================================

print()
print("=" * 75)
print("4. IDENTIFYING DETECTED SEIZURES")
print("=" * 75)

try:

    import tensorflow as tf

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

    print("CNN model loaded successfully.")

except Exception as e:

    print(
        "WARNING: CNN could not be loaded."
    )

    print(e)

    model = None


def prepare_model_input(sample):

    sample = np.asarray(
        sample,
        dtype=np.float32
    )

    expected = model.input_shape

    if len(expected) != 3:

        raise ValueError(
            "Unexpected CNN input shape."
        )

    expected_t = expected[1]
    expected_c = expected[2]

    if (
        sample.shape ==
        (expected_t, expected_c)
    ):

        return sample[np.newaxis, ...]

    if (
        sample.shape ==
        (expected_c, expected_t)
    ):

        return sample.T[np.newaxis, ...]

    raise ValueError(
        f"Cannot convert sample shape "
        f"{sample.shape} to model input."
    )


def predict_probability(sample):

    if model is None:
        return np.nan

    model_input = prepare_model_input(
        sample
    )

    prediction = model.predict(
        model_input,
        verbose=0
    )

    return float(
        np.asarray(prediction).reshape(-1)[0]
    )


all_probabilities = []

if model is not None:

    print(
        "Calculating probabilities for "
        "all test samples..."
    )

    for i in range(len(X_test)):

        try:

            p = predict_probability(
                X_test[i]
            )

        except Exception:

            p = np.nan

        all_probabilities.append(p)

    all_probabilities = np.asarray(
        all_probabilities,
        dtype=np.float64
    )

else:

    all_probabilities = np.full(
        len(X_test),
        np.nan
    )

target_probability = (
    all_probabilities[TARGET_INDEX]
)

detected_indices = np.where(
    (y_test == 1)
    &
    (all_probabilities >= THRESHOLD)
)[0]

detected_indices = detected_indices[
    detected_indices != TARGET_INDEX
]

print(
    "Number of detected seizures:",
    len(detected_indices)
)

print(
    "Detected seizure indices:"
)

print(
    detected_indices.tolist()
)

# ============================================================
# 5. EXTRACT MORPHOLOGY FEATURES
# ============================================================

print()
print("=" * 75)
print("5. EXTRACTING MORPHOLOGICAL FEATURES")
print("=" * 75)

feature_rows = []

# ---- missed seizure ----

for channel in range(num_channels):

    signal = target_ct[channel]

    features = extract_features(
        signal
    )

    row = {
        "Sample_Type": "Missed",
        "Sample_Index": TARGET_INDEX,
        "Channel": channel + 1
    }

    row.update(features)

    feature_rows.append(row)

# ---- detected seizures ----

for idx in detected_indices:

    sample = np.asarray(
        X_test[idx],
        dtype=np.float64
    )

    if sample.shape[0] == num_channels:

        sample_ct = sample

    else:

        sample_ct = sample.T

    for channel in range(num_channels):

        signal = sample_ct[channel]

        features = extract_features(
            signal
        )

        row = {
            "Sample_Type": "Detected",
            "Sample_Index": int(idx),
            "Channel": channel + 1
        }

        row.update(features)

        feature_rows.append(row)

feature_df = pd.DataFrame(
    feature_rows
)

feature_path = os.path.join(
    OUTPUT_DIR,
    "morphology_all_features.csv"
)

feature_df.to_csv(
    feature_path,
    index=False
)

print("Saved:")
print(feature_path)

# ============================================================
# 6. CHANNEL-LEVEL COMPARISON
# ============================================================

print()
print("=" * 75)
print("6. CHANNEL-LEVEL MORPHOLOGY COMPARISON")
print("=" * 75)

feature_names = [
    "Mean",
    "STD",
    "RMS",
    "Mean_Absolute",
    "Max_Absolute",
    "Peak_to_Peak",
    "Zero_Crossing_Rate",
    "Line_Length",
    "Crest_Factor",
    "Hjorth_Activity",
    "Hjorth_Mobility",
    "Hjorth_Complexity"
]

channel_rows = []

for channel in range(1, num_channels + 1):

    missed_rows = feature_df[
        (feature_df["Sample_Type"] == "Missed")
        &
        (feature_df["Channel"] == channel)
    ]

    detected_rows = feature_df[
        (feature_df["Sample_Type"] == "Detected")
        &
        (feature_df["Channel"] == channel)
    ]

    for feature in feature_names:

        missed_value = safe_float(
            missed_rows[feature].iloc[0]
        )

        detected_values = (
            detected_rows[feature]
            .values
        )

        detected_mean = np.nanmean(
            detected_values
        )

        detected_std = np.nanstd(
            detected_values
        )

        difference = (
            missed_value -
            detected_mean
        )

        standardized_distance = (
            normalize_for_distance(
                missed_value,
                detected_values
            )
        )

        channel_rows.append({
            "Channel": channel,
            "Feature": feature,
            "Missed_Value": missed_value,
            "Detected_Mean": detected_mean,
            "Detected_STD": detected_std,
            "Difference": difference,
            "Standardized_Distance":
                standardized_distance
        })

channel_comparison_df = pd.DataFrame(
    channel_rows
)

channel_comparison_path = os.path.join(
    OUTPUT_DIR,
    "morphology_channel_comparison.csv"
)

channel_comparison_df.to_csv(
    channel_comparison_path,
    index=False
)

print("Saved:")
print(channel_comparison_path)

# ============================================================
# 7. FEATURE-LEVEL SUMMARY
# ============================================================

print()
print("=" * 75)
print("7. FEATURE-LEVEL MORPHOLOGY SUMMARY")
print("=" * 75)

feature_summary_rows = []

for feature in feature_names:

    missed_values = feature_df[
        feature_df["Sample_Type"] == "Missed"
    ][feature].values

    detected_values = feature_df[
        feature_df["Sample_Type"] == "Detected"
    ][feature].values

    missed_mean = np.nanmean(
        missed_values
    )

    detected_mean = np.nanmean(
        detected_values
    )

    detected_std = np.nanstd(
        detected_values
    )

    difference = (
        missed_mean -
        detected_mean
    )

    if detected_std > 0:

        z_distance = abs(
            difference
        ) / detected_std

    else:

        z_distance = 0.0

    feature_summary_rows.append({
        "Feature": feature,
        "Missed_Mean": missed_mean,
        "Detected_Mean": detected_mean,
        "Detected_STD": detected_std,
        "Difference": difference,
        "Absolute_Difference":
            abs(difference),
        "Standardized_Distance":
            z_distance
    })

feature_summary_df = pd.DataFrame(
    feature_summary_rows
)

feature_summary_df = (
    feature_summary_df
    .sort_values(
        "Standardized_Distance",
        ascending=False
    )
    .reset_index(drop=True)
)

feature_summary_path = os.path.join(
    OUTPUT_DIR,
    "morphology_feature_comparison.csv"
)

feature_summary_df.to_csv(
    feature_summary_path,
    index=False
)

print(
    feature_summary_df.to_string(
        index=False
    )
)

print()
print("Saved:")
print(feature_summary_path)

# ============================================================
# 8. DISTANCE OF MISSED SEIZURE TO DETECTED SEIZURES
# ============================================================

print()
print("=" * 75)
print("8. DISTANCE TO DETECTED SEIZURES")
print("=" * 75)

distance_rows = []

for idx in detected_indices:

    sample = np.asarray(
        X_test[idx],
        dtype=np.float64
    )

    if sample.shape[0] == num_channels:

        sample_ct = sample

    else:

        sample_ct = sample.T

    distances = []

    for channel in range(num_channels):

        missed_signal = (
            target_ct[channel]
        )

        detected_signal = (
            sample_ct[channel]
        )

        missed_features = extract_features(
            missed_signal
        )

        detected_features = extract_features(
            detected_signal
        )

        feature_distances = []

        for feature in feature_names:

            a = missed_features[feature]
            b = detected_features[feature]

            if (
                np.isfinite(a)
                and
                np.isfinite(b)
            ):

                scale = abs(b)

                if scale < 1e-12:
                    scale = 1.0

                feature_distances.append(
                    abs(a - b) / scale
                )

        if len(feature_distances) > 0:

            distances.append(
                np.mean(feature_distances)
            )

    overall_distance = (
        np.mean(distances)
        if distances
        else np.nan
    )

    distance_rows.append({
        "Detected_Seizure_Index":
            int(idx),
        "Morphology_Distance":
            overall_distance,
        "Detected_Probability":
            all_probabilities[idx]
    })

distance_df = pd.DataFrame(
    distance_rows
)

distance_df = distance_df.sort_values(
    "Morphology_Distance"
)

distance_df["Morphology_Rank"] = (
    np.arange(1, len(distance_df) + 1)
)

distance_path = os.path.join(
    OUTPUT_DIR,
    "morphology_distance_to_detected.csv"
)

distance_df.to_csv(
    distance_path,
    index=False
)

print(
    distance_df.head(10).to_string(
        index=False
    )
)

print()
print("Saved:")
print(distance_path)

# ============================================================
# 9. SPECIAL ANALYSIS OF CHANNEL 17
# ============================================================

print()
print("=" * 75)
print("9. CHANNEL 17 DETAILED ANALYSIS")
print("=" * 75)

SPECIAL_CHANNEL = 17

channel17_rows = []

missed_signal_17 = (
    target_ct[SPECIAL_CHANNEL - 1]
)

missed_features_17 = extract_features(
    missed_signal_17
)

for idx in detected_indices:

    sample = np.asarray(
        X_test[idx],
        dtype=np.float64
    )

    if sample.shape[0] == num_channels:

        sample_ct = sample

    else:

        sample_ct = sample.T

    detected_signal_17 = (
        sample_ct[SPECIAL_CHANNEL - 1]
    )

    detected_features_17 = (
        extract_features(
            detected_signal_17
        )
    )

    row = {
        "Detected_Seizure_Index":
            int(idx),
        "Detected_Probability":
            all_probabilities[idx]
    }

    for feature in feature_names:

        row[
            "Missed_" + feature
        ] = missed_features_17[feature]

        row[
            "Detected_" + feature
        ] = detected_features_17[feature]

        row[
            "Difference_" + feature
        ] = (
            missed_features_17[feature]
            -
            detected_features_17[feature]
        )

    channel17_rows.append(row)

channel17_df = pd.DataFrame(
    channel17_rows
)

channel17_path = os.path.join(
    OUTPUT_DIR,
    "channel_17_detailed_analysis.csv"
)

channel17_df.to_csv(
    channel17_path,
    index=False
)

print(
    "Channel 17 morphology comparison completed."
)

print("Saved:")
print(channel17_path)

# ============================================================
# 10. TEMPORAL MORPHOLOGY AROUND SALIENCY PEAKS
# ============================================================

print()
print("=" * 75)
print("10. TEMPORAL MORPHOLOGY AROUND SALIENCY PEAKS")
print("=" * 75)

temporal_results = []

if os.path.exists(SALIENCY_TIME_PATH):

    saliency_df = pd.read_csv(
        SALIENCY_TIME_PATH
    )

    print(
        "Saliency columns:",
        saliency_df.columns.tolist()
    )

    if "Mean_Saliency" in saliency_df.columns:

        saliency_column = "Mean_Saliency"

    elif "Saliency" in saliency_df.columns:

        saliency_column = "Saliency"

    else:

        saliency_column = (
            saliency_df.columns[-1]
        )

    top_saliency_times = (
        saliency_df
        .sort_values(
            saliency_column,
            ascending=False
        )
        .head(20)
    )

    window = 25

    for _, row in top_saliency_times.iterrows():

        center = int(
            row["Sample"]
        )

        start = max(
            0,
            center - window
        )

        end = min(
            num_time_points,
            center + window + 1
        )

        for channel in range(num_channels):

            signal = target_ct[channel]

            segment = signal[start:end]

            if len(segment) < 2:
                continue

            features = extract_features(
                segment
            )

            temporal_results.append({
                "Saliency_Sample":
                    center,
                "Saliency_Value":
                    row[saliency_column],
                "Window_Start":
                    start,
                "Window_End":
                    end - 1,
                "Channel":
                    channel + 1,
                **features
            })

else:

    print(
        "Saliency time file not found."
    )

temporal_df = pd.DataFrame(
    temporal_results
)

temporal_path = os.path.join(
    OUTPUT_DIR,
    "top_temporal_morphology_regions.csv"
)

temporal_df.to_csv(
    temporal_path,
    index=False
)

print("Saved:")
print(temporal_path)

# ============================================================
# 11. CHANNEL 17 VS SALIENCY
# ============================================================

print()
print("=" * 75)
print("11. CHANNEL 17 VS SALIENCY / COUNTERFACTUAL")
print("=" * 75)

saliency_channel_value = np.nan
saliency_channel_rank = np.nan

if os.path.exists(
    SALIENCY_CHANNEL_PATH
):

    sc_df = pd.read_csv(
        SALIENCY_CHANNEL_PATH
    )

    if "Mean_Saliency" in sc_df.columns:

        sc_df = sc_df.sort_values(
            "Mean_Saliency",
            ascending=False
        ).reset_index(
            drop=True
        )

        match = sc_df[
            sc_df["Channel"] ==
            SPECIAL_CHANNEL
        ]

        if not match.empty:

            saliency_channel_value = (
                float(
                    match[
                        "Mean_Saliency"
                    ].iloc[0]
                )
            )

            saliency_channel_rank = (
                int(
                    match.index[0] + 1
                )
            )

counterfactual_change = np.nan
counterfactual_probability = np.nan

if os.path.exists(
    COUNTERFACTUAL_PATH
):

    cf_df = pd.read_csv(
        COUNTERFACTUAL_PATH
    )

    match = cf_df[
        cf_df["Channel"] ==
        SPECIAL_CHANNEL
    ]

    if not match.empty:

        counterfactual_change = float(
            match[
                "Probability_Change"
            ].iloc[0]
        )

        counterfactual_probability = float(
            match[
                "Counterfactual_Probability"
            ].iloc[0]
        )

channel17_summary = pd.DataFrame([
    {
        "Channel": SPECIAL_CHANNEL,
        "Baseline_Probability":
            target_probability,
        "Counterfactual_Probability":
            counterfactual_probability,
        "Counterfactual_Probability_Change":
            counterfactual_change,
        "Mean_Saliency":
            saliency_channel_value,
        "Saliency_Rank":
            saliency_channel_rank
    }
])

channel17_summary_path = os.path.join(
    OUTPUT_DIR,
    "channel_17_integrated_summary.csv"
)

channel17_summary.to_csv(
    channel17_summary_path,
    index=False
)

print(
    channel17_summary.to_string(
        index=False
    )
)

print()
print("Saved:")
print(channel17_summary_path)

# ============================================================
# 12. TOP MORPHOLOGY DIFFERENCES
# ============================================================

print()
print("=" * 75)
print("12. TOP MORPHOLOGY DIFFERENCES")
print("=" * 75)

top_morphology = (
    channel_comparison_df
    .sort_values(
        "Standardized_Distance",
        ascending=False
    )
    .head(30)
)

top_morphology_path = os.path.join(
    OUTPUT_DIR,
    "top_morphology_differences.csv"
)

top_morphology.to_csv(
    top_morphology_path,
    index=False
)

print(
    top_morphology.to_string(
        index=False
    )
)

print()
print("Saved:")
print(top_morphology_path)

# ============================================================
# 13. SUMMARY
# ============================================================

print()
print("=" * 75)
print("13. MORPHOLOGY SUMMARY")
print("=" * 75)

if len(feature_summary_df) > 0:

    top_feature = (
        feature_summary_df.iloc[0]
    )

else:

    top_feature = None

if len(distance_df) > 0:

    closest_detected = (
        distance_df.iloc[0]
    )

else:

    closest_detected = None

summary_rows = []

if top_feature is not None:

    summary_rows.append({
        "Evidence_Type":
            "Morphology_Feature",
        "Feature_or_Channel":
            top_feature["Feature"],
        "Evidence_Value":
            top_feature[
                "Standardized_Distance"
            ],
        "Interpretation":
            "Largest standardized morphology difference"
    })

summary_rows.append({
    "Evidence_Type":
        "Counterfactual_Channel",
    "Feature_or_Channel":
        "Channel 17",
    "Evidence_Value":
        abs(counterfactual_change)
        if np.isfinite(counterfactual_change)
        else np.nan,
    "Interpretation":
        "Largest counterfactual channel effect"
})

if closest_detected is not None:

    summary_rows.append({
        "Evidence_Type":
            "Closest_Detected_Seizure",
        "Feature_or_Channel":
            int(
                closest_detected[
                    "Detected_Seizure_Index"
                ]
            ),
        "Evidence_Value":
            closest_detected[
                "Morphology_Distance"
            ],
        "Interpretation":
            "Detected seizure with smallest morphology distance"
    })

summary_df = pd.DataFrame(
    summary_rows
)

summary_path = os.path.join(
    OUTPUT_DIR,
    "morphology_summary.csv"
)

summary_df.to_csv(
    summary_path,
    index=False
)

print(
    summary_df.to_string(
        index=False
    )
)

print()
print("Saved:")
print(summary_path)

# ============================================================
# 14. CREATE REPORT
# ============================================================

print()
print("=" * 75)
print("14. CREATING MORPHOLOGY REPORT")
print("=" * 75)

report_path = os.path.join(
    OUTPUT_DIR,
    "morphology_analysis_report.txt"
)

with open(
    report_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "MISSED SEIZURE MORPHOLOGY ANALYSIS\n"
    )

    f.write(
        "=" * 75 + "\n\n"
    )

    f.write(
        f"Target index: {TARGET_INDEX}\n"
    )

    f.write(
        f"Target label: {target_label}\n"
    )

    f.write(
        f"Model probability: "
        f"{target_probability:.8f}\n"
    )

    f.write(
        f"Threshold: {THRESHOLD}\n"
    )

    f.write(
        "Model changed: NO\n"
    )

    f.write(
        "Model retrained: NO\n"
    )

    f.write(
        "Threshold changed: NO\n\n"
    )

    f.write(
        f"Detected seizures used for comparison: "
        f"{len(detected_indices)}\n\n"
    )

    f.write(
        "MOST IMPORTANT MORPHOLOGY FEATURE\n"
    )

    f.write(
        "-" * 50 + "\n"
    )

    if top_feature is not None:

        f.write(
            f"Feature: "
            f"{top_feature['Feature']}\n"
        )

        f.write(
            f"Standardized distance: "
            f"{top_feature['Standardized_Distance']:.6f}\n"
        )

    f.write("\n")

    f.write(
        "CHANNEL 17\n"
    )

    f.write(
        "-" * 50 + "\n"
    )

    f.write(
        "Channel 17 was selected because "
        "counterfactual ablation produced "
        "the strongest probability reversal.\n\n"
    )

    if np.isfinite(
        counterfactual_change
    ):

        f.write(
            f"Counterfactual probability: "
            f"{counterfactual_probability:.8f}\n"
        )

        f.write(
            f"Probability change: "
            f"{counterfactual_change:+.8f}\n"
        )

    if np.isfinite(
        saliency_channel_value
    ):

        f.write(
            f"Saliency value: "
            f"{saliency_channel_value:.8f}\n"
        )

        f.write(
            f"Saliency rank: "
            f"{saliency_channel_rank}\n"
        )

    f.write("\n")

    if closest_detected is not None:

        f.write(
            "CLOSEST DETECTED SEIZURE\n"
        )

        f.write(
            "-" * 50 + "\n"
        )

        f.write(
            f"Index: "
            f"{int(closest_detected['Detected_Seizure_Index'])}\n"
        )

        f.write(
            f"Probability: "
            f"{closest_detected['Detected_Probability']:.8f}\n"
        )

        f.write(
            f"Morphology distance: "
            f"{closest_detected['Morphology_Distance']:.6f}\n"
        )

    f.write("\n")

    f.write(
        "INTERPRETATION\n"
    )

    f.write(
        "-" * 50 + "\n"
    )

    f.write(
        "This analysis describes morphological "
        "differences between the missed seizure "
        "and detected seizures.\n\n"
    )

    f.write(
        "It does not modify, retrain, or recalibrate "
        "the CNN model.\n"
    )

print()
print("Report saved:")
print(report_path)

# ============================================================
# FINAL
# ============================================================

print()
print("=" * 75)
print("FINAL SUMMARY")
print("=" * 75)

print(
    f"Missed seizure index: {TARGET_INDEX}"
)

print(
    f"Model probability: "
    f"{target_probability:.8f}"
)

print(
    f"Threshold: {THRESHOLD}"
)

print(
    f"Detected seizures compared: "
    f"{len(detected_indices)}"
)

if top_feature is not None:

    print(
        f"Top morphology feature: "
        f"{top_feature['Feature']}"
    )

    print(
        f"Top morphology standardized distance: "
        f"{top_feature['Standardized_Distance']:.6f}"
    )

print(
    "Special counterfactual channel: 17"
)

print(
    "Model changed: NO"
)

print(
    "Model retrained: NO"
)

print(
    "Threshold changed: NO"
)

print()
print(
    "Morphology analysis completed successfully."
)

print()
print(
    "Output directory:"
)

print(OUTPUT_DIR)

print()
print("DONE.")