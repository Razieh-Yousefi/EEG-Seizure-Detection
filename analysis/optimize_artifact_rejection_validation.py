# ================================================================
# optimize_artifact_rejection_validation.py
#
# VALIDATION-ONLY ARTIFACT REJECTION OPTIMIZATION
#
# IMPORTANT:
# - Model is NOT modified.
# - Dataset is NOT modified.
# - Test set is NOT used.
# - Sampling rate = 256 Hz.
# - Band powers are RELATIVE powers.
# - line_length is NOT used as an artifact feature.
# - The artifact rule is optimized ONLY on validation data.
# ================================================================

import os
import json
import csv
import numpy as np


# ================================================================
# 1. PROJECT PATHS
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

VALIDATION_PROB_FILE = os.path.join(
    RESULTS_DIR,
    "validation_window_probabilities.npz"
)

X_FILE = os.path.join(
    DATA_DIR,
    "X_chbmit_full.npy"
)

NORMALIZATION_FILE = os.path.join(
    DATA_DIR,
    "normalization_params.npz"
)

OUTPUT_JSON = os.path.join(
    RESULTS_DIR,
    "validation_artifact_rejection_optimization_v2.json"
)

OUTPUT_CSV = os.path.join(
    RESULTS_DIR,
    "validation_artifact_rejection_candidates_v2.csv"
)

OUTPUT_NPZ = os.path.join(
    RESULTS_DIR,
    "validation_artifact_scores_v2.npz"
)


# ================================================================
# 2. CONFIGURATION
# ================================================================

FS = 256.0

BASELINE_THRESHOLD = 0.95

MIN_VALIDATION_RECALL = 0.40

MAX_RECALL_DROP = 0.08

MIN_FP_REDUCTION = 0.05

# Candidate feature weights.
#
# These features are intentionally chosen because the previous
# validation analysis showed that false positives tended to have:
#
# - higher high-frequency activity
# - higher beta relative power
# - higher gamma relative power
# - higher zero-crossing rate
#
# line_length is deliberately excluded because it was higher in
# validation seizures and could therefore suppress real seizures.

FEATURE_WEIGHTS = {
    "mean_high_frequency_ratio": 0.30,
    "mean_beta_relative_power": 0.25,
    "mean_gamma_relative_power": 0.25,
    "mean_zero_crossing_rate": 0.20
}


# ================================================================
# 3. BAND DEFINITIONS
# ================================================================

BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0)
}


# ================================================================
# 4. PRINT HEADER
# ================================================================

print()
print("=" * 72)
print("VALIDATION-ONLY ARTIFACT REJECTION OPTIMIZATION V2")
print("=" * 72)

print()
print("Project:")
print(PROJECT_DIR)

print()
print("Sampling rate:")
print(f"{FS:.1f} Hz")

print()
print("Baseline threshold:")
print(BASELINE_THRESHOLD)

print()
print("Minimum allowed validation recall:")
print(MIN_VALIDATION_RECALL)

print()
print("Maximum allowed recall drop:")
print(MAX_RECALL_DROP)

print()
print("Minimum FP reduction:")
print(MIN_FP_REDUCTION)


# ================================================================
# 5. CHECK FILES
# ================================================================

print()
print("=" * 72)
print("CHECKING REQUIRED FILES")
print("=" * 72)

required_files = [
    VALIDATION_PROB_FILE,
    X_FILE,
    NORMALIZATION_FILE
]

for path in required_files:

    if os.path.exists(path):

        print("[OK]", path)

    else:

        print("[MISSING]", path)

        raise FileNotFoundError(path)


os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)


# ================================================================
# 6. LOAD VALIDATION PROBABILITIES
# ================================================================

print()
print("=" * 72)
print("LOADING VALIDATION PROBABILITIES")
print("=" * 72)

validation_data = np.load(
    VALIDATION_PROB_FILE,
    allow_pickle=True
)

validation_indices = np.asarray(
    validation_data["validation_indices"],
    dtype=np.int64
)

labels = np.asarray(
    validation_data["labels"],
    dtype=np.int64
)

probabilities = np.asarray(
    validation_data["probabilities"],
    dtype=np.float32
)

patients = np.asarray(
    validation_data["patients"]
)


print()
print("Validation windows:", len(validation_indices))
print("Labels:", len(labels))
print("Probabilities:", len(probabilities))
print("Patients:", len(patients))


if not (
    len(validation_indices)
    == len(labels)
    == len(probabilities)
    == len(patients)
):

    raise RuntimeError(
        "Validation arrays are not aligned."
    )


if not np.all(
    np.isfinite(probabilities)
):

    raise RuntimeError(
        "Validation probabilities contain NaN/Inf."
    )


# ================================================================
# 7. LOAD EEG DATA
# ================================================================

print()
print("=" * 72)
print("LOADING EEG DATA")
print("=" * 72)

X = np.load(
    X_FILE,
    mmap_mode="r"
)

print()
print("X shape:", X.shape)

if X.ndim != 3:

    raise ValueError(
        "Expected X to have shape "
        "(windows, channels, samples)."
    )

N_CHANNELS = X.shape[1]

N_SAMPLES = X.shape[2]

WINDOW_SECONDS = N_SAMPLES / FS

print()
print("Channels:", N_CHANNELS)
print("Samples per window:", N_SAMPLES)
print("Window duration:", f"{WINDOW_SECONDS:.3f} sec")


# ================================================================
# 8. LOAD NORMALIZATION
# ================================================================

print()
print("=" * 72)
print("LOADING NORMALIZATION")
print("=" * 72)

norm_data = np.load(
    NORMALIZATION_FILE
)

channel_mean = np.asarray(
    norm_data["channel_mean"],
    dtype=np.float32
)

channel_std = np.asarray(
    norm_data["channel_std"],
    dtype=np.float32
)


if channel_mean.shape != (
    N_CHANNELS,
):

    raise ValueError(
        "channel_mean shape mismatch."
    )


if channel_std.shape != (
    N_CHANNELS,
):

    raise ValueError(
        "channel_std shape mismatch."
    )


if not np.all(
    np.isfinite(channel_mean)
):

    raise ValueError(
        "channel_mean contains NaN/Inf."
    )


if not np.all(
    np.isfinite(channel_std)
):

    raise ValueError(
        "channel_std contains NaN/Inf."
    )


if np.any(
    channel_std <= 0
):

    raise ValueError(
        "channel_std contains zero/negative values."
    )


# ================================================================
# 9. BASELINE CLASSIFICATION
# ================================================================

print()
print("=" * 72)
print("BASELINE VALIDATION CLASSIFICATION")
print("=" * 72)

baseline_pred = (
    probabilities >= BASELINE_THRESHOLD
)

TP_baseline = int(
    np.sum(
        (baseline_pred == 1)
        & (labels == 1)
    )
)

FP_baseline = int(
    np.sum(
        (baseline_pred == 1)
        & (labels == 0)
    )
)

TN_baseline = int(
    np.sum(
        (baseline_pred == 0)
        & (labels == 0)
    )
)

FN_baseline = int(
    np.sum(
        (baseline_pred == 0)
        & (labels == 1)
    )
)

baseline_recall = (
    TP_baseline
    / max(
        TP_baseline + FN_baseline,
        1
    )
)

baseline_specificity = (
    TN_baseline
    / max(
        TN_baseline + FP_baseline,
        1
    )
)

baseline_precision = (
    TP_baseline
    / max(
        TP_baseline + FP_baseline,
        1
    )
)

print()
print("Threshold:", BASELINE_THRESHOLD)

print(
    "TP:",
    TP_baseline
)

print(
    "FP:",
    FP_baseline
)

print(
    "TN:",
    TN_baseline
)

print(
    "FN:",
    FN_baseline
)

print(
    f"Recall: {baseline_recall:.6f}"
)

print(
    f"Specificity: {baseline_specificity:.6f}"
)

print(
    f"Precision: {baseline_precision:.6f}"
)


# ================================================================
# 10. FEATURE FUNCTIONS
# ================================================================

def compute_zero_crossing_rate(signal):

    if len(signal) < 2:

        return 0.0

    signs = np.signbit(
        signal
    )

    crossings = np.count_nonzero(
        signs[1:] != signs[:-1]
    )

    return (
        crossings
        / max(
            len(signal) - 1,
            1
        )
    )


def compute_relative_band_power(
    signal,
    freqs,
    psd,
    low,
    high
):

    mask = (
        (freqs >= low)
        & (freqs < high)
    )

    if not np.any(mask):

        return 0.0

    band_power = np.trapezoid(
        psd[mask],
        freqs[mask]
    )

    total_mask = (
        (freqs >= 0.5)
        & (freqs < 45.0)
    )

    if not np.any(total_mask):

        return 0.0

    total_power = np.trapezoid(
        psd[total_mask],
        freqs[total_mask]
    )

    if total_power <= 0:

        return 0.0

    return float(
        band_power
        / total_power
    )


def compute_window_features(
    window
):

    # ------------------------------------------------------------
    # Convert to float32.
    # ------------------------------------------------------------

    signal = np.asarray(
        window,
        dtype=np.float32
    )

    # ------------------------------------------------------------
    # Remove channel offsets.
    #
    # This is only for feature extraction.
    # Original EEG data is NOT modified.
    # ------------------------------------------------------------

    signal = (
        signal
        - np.mean(
            signal,
            axis=1,
            keepdims=True
        )
    )

    n_channels = signal.shape[0]

    n_samples = signal.shape[1]

    # ------------------------------------------------------------
    # Frequency axis.
    # ------------------------------------------------------------

    freqs = np.fft.rfftfreq(
        n_samples,
        d=1.0 / FS
    )

    # ------------------------------------------------------------
    # Hann window.
    # ------------------------------------------------------------

    taper = np.hanning(
        n_samples
    ).astype(
        np.float32
    )

    tapered = (
        signal
        * taper[None, :]
    )

    # ------------------------------------------------------------
    # FFT.
    # ------------------------------------------------------------

    fft_values = np.fft.rfft(
        tapered,
        axis=1
    )

    psd = (
        np.abs(
            fft_values
        ) ** 2
    )

    # ------------------------------------------------------------
    # RMS.
    # ------------------------------------------------------------

    rms = np.sqrt(
        np.mean(
            signal ** 2,
            axis=1
        )
        + 1e-12
    )

    # ------------------------------------------------------------
    # High-frequency ratio.
    #
    # High frequency = 30-45 Hz.
    # Total = 0.5-45 Hz.
    # ------------------------------------------------------------

    total_mask = (
        (freqs >= 0.5)
        & (freqs < 45.0)
    )

    high_mask = (
        (freqs >= 30.0)
        & (freqs < 45.0)
    )

    if np.any(total_mask):

        total_power = np.trapezoid(
            psd[:, total_mask],
            freqs[total_mask],
            axis=1
        )

    else:

        total_power = np.ones(
            n_channels,
            dtype=np.float64
        )

    if np.any(high_mask):

        high_power = np.trapezoid(
            psd[:, high_mask],
            freqs[high_mask],
            axis=1
        )

    else:

        high_power = np.zeros(
            n_channels,
            dtype=np.float64
        )

    high_frequency_ratio = (
        high_power
        / np.maximum(
            total_power,
            1e-12
        )
    )

    # ------------------------------------------------------------
    # Relative beta power.
    # ------------------------------------------------------------

    beta_mask = (
        (freqs >= 13.0)
        & (freqs < 30.0)
    )

    if np.any(beta_mask):

        beta_power = np.trapezoid(
            psd[:, beta_mask],
            freqs[beta_mask],
            axis=1
        )

    else:

        beta_power = np.zeros(
            n_channels,
            dtype=np.float64
        )

    beta_relative_power = (
        beta_power
        / np.maximum(
            total_power,
            1e-12
        )
    )

    # ------------------------------------------------------------
    # Relative gamma power.
    # ------------------------------------------------------------

    gamma_mask = (
        (freqs >= 30.0)
        & (freqs < 45.0)
    )

    if np.any(gamma_mask):

        gamma_power = np.trapezoid(
            psd[:, gamma_mask],
            freqs[gamma_mask],
            axis=1
        )

    else:

        gamma_power = np.zeros(
            n_channels,
            dtype=np.float64
        )

    gamma_relative_power = (
        gamma_power
        / np.maximum(
            total_power,
            1e-12
        )
    )

    # ------------------------------------------------------------
    # Zero crossing rate.
    # ------------------------------------------------------------

    zcr = np.zeros(
        n_channels,
        dtype=np.float64
    )

    for ch in range(
        n_channels
    ):

        zcr[ch] = (
            compute_zero_crossing_rate(
                signal[ch]
            )
        )

    # ------------------------------------------------------------
    # Aggregate window features.
    # ------------------------------------------------------------

    features = {

        "mean_rms":
            float(
                np.mean(rms)
            ),

        "max_rms":
            float(
                np.max(rms)
            ),

        "mean_high_frequency_ratio":
            float(
                np.mean(
                    high_frequency_ratio
                )
            ),

        "max_high_frequency_ratio":
            float(
                np.max(
                    high_frequency_ratio
                )
            ),

        "mean_beta_relative_power":
            float(
                np.mean(
                    beta_relative_power
                )
            ),

        "max_beta_relative_power":
            float(
                np.max(
                    beta_relative_power
                )
            ),

        "mean_gamma_relative_power":
            float(
                np.mean(
                    gamma_relative_power
                )
            ),

        "max_gamma_relative_power":
            float(
                np.max(
                    gamma_relative_power
                )
            ),

        "mean_zero_crossing_rate":
            float(
                np.mean(zcr)
            ),

        "max_zero_crossing_rate":
            float(
                np.max(zcr)
            )
    }

    return features


# ================================================================
# 11. EXTRACT FEATURES
# ================================================================

print()
print("=" * 72)
print("EXTRACTING VALIDATION FEATURES")
print("=" * 72)

feature_names = [
    "mean_high_frequency_ratio",
    "mean_beta_relative_power",
    "mean_gamma_relative_power",
    "mean_zero_crossing_rate"
]

feature_matrix = np.zeros(
    (
        len(validation_indices),
        len(feature_names)
    ),
    dtype=np.float64
)


for i, dataset_index in enumerate(
    validation_indices
):

    window = np.asarray(
        X[dataset_index],
        dtype=np.float32
    )

    # ------------------------------------------------------------
    # Exact training normalization.
    #
    # Used only for feature extraction.
    # ------------------------------------------------------------

    window = (
        window
        - channel_mean[:, None]
    ) / channel_std[:, None]

    features = compute_window_features(
        window
    )

    for j, feature_name in enumerate(
        feature_names
    ):

        feature_matrix[i, j] = (
            features[feature_name]
        )

    if (
        i == 0
        or (i + 1) % 100 == 0
        or i + 1 == len(validation_indices)
    ):

        print(
            f"Processed "
            f"{i + 1}/{len(validation_indices)}"
        )


# ================================================================
# 12. VERIFY FEATURES
# ================================================================

if not np.all(
    np.isfinite(feature_matrix)
):

    raise RuntimeError(
        "Feature matrix contains NaN/Inf."
    )


# ================================================================
# 13. FEATURE DISTRIBUTIONS
# ================================================================

print()
print("=" * 72)
print("VALIDATION FEATURE DISTRIBUTIONS")
print("=" * 72)

for j, feature_name in enumerate(
    feature_names
):

    values = feature_matrix[:, j]

    seizure_values = values[
        labels == 1
    ]

    nonseizure_values = values[
        labels == 0
    ]

    print()
    print(feature_name)

    print(
        f"All mean   = "
        f"{np.mean(values):.8f}"
    )

    print(
        f"Seizure mean = "
        f"{np.mean(seizure_values):.8f}"
    )

    print(
        f"Non-seizure mean = "
        f"{np.mean(nonseizure_values):.8f}"
    )


# ================================================================
# 14. ROBUST FEATURE NORMALIZATION
# ================================================================

print()
print("=" * 72)
print("BUILDING ROBUST FEATURE SCORES")
print("=" * 72)

robust_scores = np.zeros(
    len(validation_indices),
    dtype=np.float64
)

feature_details = {}

for j, feature_name in enumerate(
    feature_names
):

    values = feature_matrix[:, j]

    # ------------------------------------------------------------
    # Robust reference distribution is calculated from
    # NON-SEIZURE validation windows only.
    #
    # This is intentional: artifact rejection is designed to
    # identify unusual non-seizure windows.
    # ------------------------------------------------------------

    reference = values[
        labels == 0
    ]

    q05 = np.percentile(
        reference,
        5
    )

    q95 = np.percentile(
        reference,
        95
    )

    scale = (
        q95 - q05
    )

    if scale <= 1e-12:

        scale = 1.0

    # ------------------------------------------------------------
    # Higher feature value -> stronger artifact evidence.
    # ------------------------------------------------------------

    normalized = (
        values - q05
    ) / scale

    normalized = np.clip(
        normalized,
        0.0,
        1.0
    )

    weight = FEATURE_WEIGHTS[
        feature_name
    ]

    robust_scores += (
        weight
        * normalized
    )

    feature_details[
        feature_name
    ] = {

        "q05_nonseizure":
            float(q05),

        "q95_nonseizure":
            float(q95),

        "scale":
            float(scale),

        "weight":
            float(weight)
    }


# ================================================================
# 15. ARTIFACT SCORE
# ================================================================

artifact_scores = np.asarray(
    robust_scores,
    dtype=np.float64
)

artifact_scores = np.clip(
    artifact_scores,
    0.0,
    1.0
)


print()
print("=" * 72)
print("ARTIFACT SCORE DISTRIBUTION")
print("=" * 72)

all_scores = artifact_scores

seizure_scores = artifact_scores[
    labels == 1
]

nonseizure_scores = artifact_scores[
    labels == 0
]

baseline_tp_scores = artifact_scores[
    baseline_pred
    & (labels == 1)
]

baseline_fp_scores = artifact_scores[
    baseline_pred
    & (labels == 0)
]

print()
print(
    f"All: "
    f"{np.mean(all_scores):.6f}"
)

print(
    f"Seizure: "
    f"{np.mean(seizure_scores):.6f}"
)

print(
    f"Non-seizure: "
    f"{np.mean(nonseizure_scores):.6f}"
)

print(
    f"FP baseline: "
    f"{np.mean(baseline_fp_scores):.6f}"
)

print(
    f"TP baseline: "
    f"{np.mean(baseline_tp_scores):.6f}"
)


# ================================================================
# 16. SEARCH THRESHOLDS
# ================================================================

print()
print("=" * 72)
print("SEARCHING ARTIFACT THRESHOLDS")
print("=" * 72)

candidate_thresholds = np.unique(
    np.round(
        artifact_scores[
            baseline_pred
        ],
        6
    )
)

candidate_thresholds = np.sort(
    candidate_thresholds
)


candidates = []


for threshold in candidate_thresholds:

    # ------------------------------------------------------------
    # A baseline-positive window is rejected if its artifact
    # score is greater than or equal to the candidate threshold.
    # ------------------------------------------------------------

    rejection_mask = (
        baseline_pred
        & (
            artifact_scores
            >= threshold
        )
    )

    final_pred = (
        baseline_pred
        & (~rejection_mask)
    )

    TP = int(
        np.sum(
            (final_pred == 1)
            & (labels == 1)
        )
    )

    FP = int(
        np.sum(
            (final_pred == 1)
            & (labels == 0)
        )
    )

    TN = int(
        np.sum(
            (final_pred == 0)
            & (labels == 0)
        )
    )

    FN = int(
        np.sum(
            (final_pred == 0)
            & (labels == 1)
        )
    )

    recall = (
        TP
        / max(
            TP + FN,
            1
        )
    )

    specificity = (
        TN
        / max(
            TN + FP,
            1
        )
    )

    precision = (
        TP
        / max(
            TP + FP,
            1
        )
    )

    fp_reduction = (
        (
            FP_baseline - FP
        )
        / max(
            FP_baseline,
            1
        )
    )

    recall_drop = (
        baseline_recall
        - recall
    )

    if (
        recall >= MIN_VALIDATION_RECALL
        and recall_drop <= MAX_RECALL_DROP
        and fp_reduction >= MIN_FP_REDUCTION
    ):

        candidates.append({

            "score_threshold":
                float(threshold),

            "TP":
                TP,

            "FP":
                FP,

            "TN":
                TN,

            "FN":
                FN,

            "recall":
                float(recall),

            "specificity":
                float(specificity),

            "precision":
                float(precision),

            "fp_reduction":
                float(fp_reduction),

            "recall_drop":
                float(recall_drop),

            "rejected_fp":
                int(
                    FP_baseline - FP
                ),

            "rejected_tp":
                int(
                    TP_baseline - TP
                )
        })


# ================================================================
# 17. RANK CANDIDATES
# ================================================================

# Primary objective:
# maximize FP reduction.
#
# Secondary objective:
# minimize TP loss.
#
# Tertiary objective:
# maximize recall.

candidates = sorted(
    candidates,
    key=lambda x: (
        -x["fp_reduction"],
        x["rejected_tp"],
        -x["recall"]
    )
)


print()
print("=" * 72)
print("ACCEPTABLE VALIDATION CANDIDATES")
print("=" * 72)


if len(candidates) == 0:

    print()
    print(
        "NO ACCEPTABLE VALIDATION CANDIDATE FOUND."
    )

else:

    for rank, candidate in enumerate(
        candidates[:20],
        start=1
    ):

        print()

        print(
            f"{rank:02d}. "
            f"score_threshold="
            f"{candidate['score_threshold']:.6f} "
            f"| TP={candidate['TP']} "
            f"| FP={candidate['FP']} "
            f"| Recall="
            f"{candidate['recall']:.6f} "
            f"| Specificity="
            f"{candidate['specificity']:.6f} "
            f"| FP reduction="
            f"{candidate['fp_reduction']:.4f} "
            f"| Recall drop="
            f"{candidate['recall_drop']:.4f}"
        )


# ================================================================
# 18. SELECT BEST CANDIDATE
# ================================================================

if len(candidates) > 0:

    best_candidate = candidates[0]

else:

    best_candidate = None


# ================================================================
# 19. BASELINE VS CANDIDATE
# ================================================================

print()
print("=" * 72)
print("BASELINE VS BEST VALIDATION CANDIDATE")
print("=" * 72)

print()
print("Baseline:")

print(
    f"TP={TP_baseline} "
    f"| FP={FP_baseline} "
    f"| Recall={baseline_recall:.6f} "
    f"| Specificity={baseline_specificity:.6f}"
)


if best_candidate is None:

    print()
    print(
        "No acceptable candidate."
    )

else:

    print()
    print("Candidate:")

    print(
        f"Artifact threshold="
        f"{best_candidate['score_threshold']:.6f}"
    )

    print(
        f"TP={best_candidate['TP']} "
        f"| FP={best_candidate['FP']} "
        f"| Recall="
        f"{best_candidate['recall']:.6f} "
        f"| Specificity="
        f"{best_candidate['specificity']:.6f}"
    )

    print()
    print(
        f"FP reduction: "
        f"{best_candidate['fp_reduction'] * 100:.2f}%"
    )

    print(
        f"Recall change: "
        f"{best_candidate['recall_drop'] * -100:.2f}%"
    )

    print()
    print(
        "Rejected FP windows:",
        best_candidate["rejected_fp"]
    )

    print(
        "Rejected TP windows:",
        best_candidate["rejected_tp"]
    )


# ================================================================
# 20. PATIENT-LEVEL DIAGNOSTIC CHECK
# ================================================================

print()
print("=" * 72)
print("PATIENT-LEVEL DIAGNOSTIC CHECK")
print("=" * 72)


if best_candidate is not None:

    best_threshold = (
        best_candidate[
            "score_threshold"
        ]
    )

    rejection_mask = (
        baseline_pred
        & (
            artifact_scores
            >= best_threshold
        )
    )

    rejected_fp_mask = (
        rejection_mask
        & (labels == 0)
    )

    rejected_tp_mask = (
        rejection_mask
        & (labels == 1)
    )

    rejected_fp_patients = (
        patients[
            rejected_fp_mask
        ]
    )

    rejected_tp_patients = (
        patients[
            rejected_tp_mask
        ]
    )

    unique_fp_patients = np.unique(
        rejected_fp_patients
    )

    unique_tp_patients = np.unique(
        rejected_tp_patients
    )

    print()
    print(
        "Baseline FP windows:",
        FP_baseline
    )

    print(
        "Rejected FP windows:",
        int(
            np.sum(
                rejected_fp_mask
            )
        )
    )

    print(
        "Rejected TP windows:",
        int(
            np.sum(
                rejected_tp_mask
            )
        )
    )

    print()
    print(
        "Patients with rejected FP:",
        len(
            unique_fp_patients
        )
    )

    print(
        "Patients with rejected TP:",
        len(
            unique_tp_patients
        )
    )


# ================================================================
# 21. SAVE JSON
# ================================================================

print()
print("=" * 72)
print("SAVING RESULTS")
print("=" * 72)


json_result = {

    "analysis_type":
        "validation_only_artifact_rejection_v2",

    "project_dir":
        PROJECT_DIR,

    "sampling_rate_hz":
        FS,

    "window_samples":
        int(N_SAMPLES),

    "window_seconds":
        float(WINDOW_SECONDS),

    "baseline_threshold":
        BASELINE_THRESHOLD,

    "minimum_validation_recall":
        MIN_VALIDATION_RECALL,

    "maximum_recall_drop":
        MAX_RECALL_DROP,

    "minimum_fp_reduction":
        MIN_FP_REDUCTION,

    "feature_weights":
        FEATURE_WEIGHTS,

    "feature_reference_statistics":
        feature_details,

    "baseline": {

        "TP":
            TP_baseline,

        "FP":
            FP_baseline,

        "TN":
            TN_baseline,

        "FN":
            FN_baseline,

        "recall":
            float(baseline_recall),

        "specificity":
            float(baseline_specificity),

        "precision":
            float(baseline_precision)
    },

    "best_candidate":
        best_candidate,

    "candidate_count":
        len(candidates),

    "all_acceptable_candidates":
        candidates,

    "warnings": [

        "Validation-only optimization.",

        "Test set was not used.",

        "Model was not modified.",

        "Dataset was not modified.",

        "Artifact rule must be evaluated on untouched test data.",

        "This rule must not be considered final until test evaluation."
    ]
}


with open(
    OUTPUT_JSON,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        json_result,
        f,
        indent=4
    )


print()
print(
    "[OK] JSON saved:"
)

print(
    OUTPUT_JSON
)


# ================================================================
# 22. SAVE CSV
# ================================================================

csv_fields = [

    "rank",
    "score_threshold",
    "TP",
    "FP",
    "TN",
    "FN",
    "recall",
    "specificity",
    "precision",
    "fp_reduction",
    "recall_drop",
    "rejected_fp",
    "rejected_tp"
]


with open(
    OUTPUT_CSV,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=csv_fields
    )

    writer.writeheader()

    for rank, candidate in enumerate(
        candidates,
        start=1
    ):

        row = {
            "rank":
                rank
        }

        row.update(
            candidate
        )

        writer.writerow(
            row
        )


print()
print(
    "[OK] CSV saved:"
)

print(
    OUTPUT_CSV
)


# ================================================================
# 23. SAVE WINDOW-LEVEL DATA
# ================================================================

save_dict = {

    "validation_indices":
        validation_indices,

    "patients":
        patients,

    "labels":
        labels,

    "probabilities":
        probabilities,

    "baseline_predictions":
        baseline_pred.astype(
            np.int8
        ),

    "artifact_scores":
        artifact_scores,

    "feature_names":
        np.asarray(
            feature_names
        )
}


for j, feature_name in enumerate(
    feature_names
):

    save_dict[
        feature_name
    ] = feature_matrix[
        :, j
    ]


if best_candidate is not None:

    best_threshold = (
        best_candidate[
            "score_threshold"
        ]
    )

    best_rejection_mask = (
        baseline_pred
        & (
            artifact_scores
            >= best_threshold
        )
    )

    best_final_predictions = (
        baseline_pred
        & (~best_rejection_mask)
    )

    save_dict[
        "best_rejection_mask"
    ] = best_rejection_mask.astype(
        np.int8
    )

    save_dict[
        "best_final_predictions"
    ] = best_final_predictions.astype(
        np.int8
    )

    save_dict[
        "best_threshold"
    ] = np.asarray(
        best_threshold,
        dtype=np.float64
    )


np.savez(
    OUTPUT_NPZ,
    **save_dict
)


print()
print(
    "[OK] Window-level diagnostic data saved:"
)

print(
    OUTPUT_NPZ
)


# ================================================================
# 24. FINAL SUMMARY
# ================================================================

print()
print("=" * 72)
print("VALIDATION ARTIFACT OPTIMIZATION V2 COMPLETED")
print("=" * 72)


if best_candidate is not None:

    print()
    print(
        "Best validation candidate:"
    )

    print(
        "Artifact score threshold =",
        f"{best_candidate['score_threshold']:.6f}"
    )

    print(
        "Baseline FP =",
        FP_baseline
    )

    print(
        "Candidate FP =",
        best_candidate["FP"]
    )

    print(
        "FP reduction =",
        f"{best_candidate['fp_reduction'] * 100:.2f}%"
    )

    print(
        "Baseline recall =",
        f"{baseline_recall:.6f}"
    )

    print(
        "Candidate recall =",
        f"{best_candidate['recall']:.6f}"
    )

else:

    print()
    print(
        "NO ACCEPTABLE CANDIDATE WAS FOUND."
    )


print()
print("IMPORTANT:")
print(
    "This candidate was optimized on validation only."
)

print(
    "It MUST be evaluated on the untouched test set "
    "before being considered a final rule."
)

print()
print(
    "Model was NOT modified."
)

print(
    "Dataset was NOT modified."
)

print(
    "Test set was NOT used."
)

print()
print("=" * 72)
print("DONE")
print("=" * 72)