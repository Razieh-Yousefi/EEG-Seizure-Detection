# ================================================================
# analyze_final_test_residual_errors.py
#
# FINAL TEST-SET RESIDUAL ERROR ANALYSIS
#
# Purpose:
#   Analyze remaining false positives and false negatives after
#   validation-selected artifact rejection.
#
# IMPORTANT:
#   - NO optimization
#   - NO threshold fitting
#   - NO test-set tuning
#   - Test labels are used ONLY for retrospective error analysis
#   - Model is NOT modified
#   - Dataset is NOT modified
#
# Inputs:
#   results/final_test_artifact_rejection_scores.npz
#   results/final_test_patient_seizure_event_results.npz
#   results/final_test_artifact_rejection_evaluation.json
#   data/X_chbmit_full.npy
#   data/patients_chbmit_full.npy
#   data/test_indices.npy
#
# Outputs:
#   results/final_test_residual_error_analysis.json
#   results/final_test_residual_error_windows.csv
#   results/final_test_residual_error_summary.csv
#   results/final_test_residual_error_analysis.npz
#   results/final_test_residual_figures/
# ================================================================

import os
import csv
import json
import math

import numpy as np
import matplotlib.pyplot as plt


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

SCORES_FILE = os.path.join(
    RESULTS_DIR,
    "final_test_artifact_rejection_scores.npz"
)

EVENT_RESULTS_FILE = os.path.join(
    RESULTS_DIR,
    "final_test_patient_seizure_event_results.npz"
)

ARTIFACT_EVAL_FILE = os.path.join(
    RESULTS_DIR,
    "final_test_artifact_rejection_evaluation.json"
)

X_FILE = os.path.join(
    DATA_DIR,
    "X_chbmit_full.npy"
)

PATIENTS_FILE = os.path.join(
    DATA_DIR,
    "patients_chbmit_full.npy"
)

TEST_INDICES_FILE = os.path.join(
    DATA_DIR,
    "test_indices.npy"
)

OUTPUT_JSON = os.path.join(
    RESULTS_DIR,
    "final_test_residual_error_analysis.json"
)

OUTPUT_WINDOWS_CSV = os.path.join(
    RESULTS_DIR,
    "final_test_residual_error_windows.csv"
)

OUTPUT_SUMMARY_CSV = os.path.join(
    RESULTS_DIR,
    "final_test_residual_error_summary.csv"
)

OUTPUT_NPZ = os.path.join(
    RESULTS_DIR,
    "final_test_residual_error_analysis.npz"
)

FIGURES_DIR = os.path.join(
    RESULTS_DIR,
    "final_test_residual_figures"
)


# ================================================================
# 2. CONFIGURATION
# ================================================================

SAMPLING_RATE = 256.0

BASELINE_THRESHOLD = 0.95

# This value is NOT optimized here.
# It came from validation.
ARTIFACT_THRESHOLD = 0.525596

N_FFT = 256

EPS = 1e-12


# ================================================================
# 3. HEADER
# ================================================================

print()
print("=" * 72)
print("FINAL TEST-SET RESIDUAL ERROR ANALYSIS")
print("=" * 72)

print()
print("Project:")
print(PROJECT_DIR)

print()
print("Baseline probability threshold:")
print(BASELINE_THRESHOLD)

print()
print("Validation-selected artifact threshold:")
print(ARTIFACT_THRESHOLD)

print()
print("Sampling rate:")
print(SAMPLING_RATE)

print()
print("=" * 72)
print("IMPORTANT")
print("=" * 72)

print()
print("No optimization is performed.")
print("No threshold is fitted on the test set.")
print("Test labels are used only for retrospective error analysis.")
print("Model is NOT modified.")
print("Dataset is NOT modified.")


# ================================================================
# 4. CHECK FILES
# ================================================================

print()
print("=" * 72)
print("1. CHECKING REQUIRED FILES")
print("=" * 72)

required_files = [
    SCORES_FILE,
    EVENT_RESULTS_FILE,
    ARTIFACT_EVAL_FILE,
    X_FILE,
    PATIENTS_FILE,
    TEST_INDICES_FILE,
]

for path in required_files:

    if os.path.exists(path):

        print(
            "[OK]",
            path
        )

    else:

        print(
            "[MISSING]",
            path
        )

        raise FileNotFoundError(
            path
        )


os.makedirs(
    FIGURES_DIR,
    exist_ok=True
)


# ================================================================
# 5. LOAD TEST SCORES
# ================================================================

print()
print("=" * 72)
print("2. LOADING TEST RESULTS")
print("=" * 72)

scores = np.load(
    SCORES_FILE,
    allow_pickle=True
)

print()
print("Available arrays:")

for key in scores.files:

    print(
        f"{key:35s}: {scores[key].shape}"
    )


test_indices = np.asarray(
    scores["test_indices"],
    dtype=np.int64
)

labels = np.asarray(
    scores["labels"],
    dtype=np.int64
)

probabilities = np.asarray(
    scores["probabilities"],
    dtype=np.float64
)

artifact_scores = np.asarray(
    scores["artifact_score"],
    dtype=np.float64
)

artifact_rejected = np.asarray(
    scores["artifact_rejected"],
    dtype=bool
)

baseline_positive = np.asarray(
    scores["baseline_positive"],
    dtype=bool
)

final_positive = np.asarray(
    scores["final_positive"],
    dtype=bool
)


# ================================================================
# 6. VERIFY ALIGNMENT
# ================================================================

print()
print("=" * 72)
print("3. VERIFYING ALIGNMENT")
print("=" * 72)

n = len(
    test_indices
)

arrays_to_check = {
    "labels": labels,
    "probabilities": probabilities,
    "artifact_scores": artifact_scores,
    "artifact_rejected": artifact_rejected,
    "baseline_positive": baseline_positive,
    "final_positive": final_positive,
}

for name, arr in arrays_to_check.items():

    if len(arr) != n:

        raise RuntimeError(
            f"{name} length mismatch."
        )


if np.any(
    test_indices < 0
):

    raise RuntimeError(
        "Negative test indices detected."
    )


if not np.all(
    np.isfinite(probabilities)
):

    raise RuntimeError(
        "Non-finite probabilities detected."
    )


if not np.all(
    np.isfinite(artifact_scores)
):

    raise RuntimeError(
        "Non-finite artifact scores detected."
    )


print()
print(
    "[OK] All window-level arrays aligned."
)

print(
    "Test windows:",
    n
)


# ================================================================
# 7. LOAD PATIENTS
# ================================================================

print()
print("=" * 72)
print("4. LOADING PATIENT INFORMATION")
print("=" * 72)

patients_full = np.load(
    PATIENTS_FILE,
    allow_pickle=True
)

test_indices_reference = np.load(
    TEST_INDICES_FILE
)


if len(test_indices_reference) != n:

    print()
    print(
        "[WARNING] data/test_indices.npy length differs "
        "from final score file."
    )


patients = np.asarray(
    patients_full[
        test_indices
    ]
)


print()
print(
    "[OK] Patient mapping completed."
)

print(
    "Unique patients:",
    len(
        np.unique(
            patients
        )
    )
)


# ================================================================
# 8. LOAD EEG DATA
# ================================================================

print()
print("=" * 72)
print("5. LOADING EEG DATA")
print("=" * 72)

X = np.load(
    X_FILE,
    mmap_mode="r"
)

print()
print(
    "X shape:",
    X.shape
)

n_channels = X.shape[1]

samples_per_window = X.shape[2]

window_duration = (
    samples_per_window
    /
    SAMPLING_RATE
)

print(
    "Channels:",
    n_channels
)

print(
    "Samples/window:",
    samples_per_window
)

print(
    "Window duration:",
    f"{window_duration:.3f} sec"
)


# ================================================================
# 9. CONFUSION-MATRIX RESIDUAL GROUPS
# ================================================================

print()
print("=" * 72)
print("6. IDENTIFYING RESIDUAL ERROR GROUPS")
print("=" * 72)


# Final classification
#
# TP = final positive + seizure
# FP = final positive + non-seizure
# TN = final negative + non-seizure
# FN = final negative + seizure

TP_mask = (
    final_positive
    &
    (labels == 1)
)

FP_mask = (
    final_positive
    &
    (labels == 0)
)

TN_mask = (
    (~final_positive)
    &
    (labels == 0)
)

FN_mask = (
    (~final_positive)
    &
    (labels == 1)
)


# Baseline groups
baseline_TP_mask = (
    baseline_positive
    &
    (labels == 1)
)

baseline_FP_mask = (
    baseline_positive
    &
    (labels == 0)
)


print()
print(
    "Final TP:",
    int(TP_mask.sum())
)

print(
    "Final FP:",
    int(FP_mask.sum())
)

print(
    "Final TN:",
    int(TN_mask.sum())
)

print(
    "Final FN:",
    int(FN_mask.sum())
)


# ================================================================
# 10. ERROR TRANSITIONS
# ================================================================

print()
print("=" * 72)
print("7. ANALYZING ERROR TRANSITIONS")
print("=" * 72)


rejected_FP_mask = (
    baseline_FP_mask
    &
    (~final_positive)
)

rejected_TP_mask = (
    baseline_TP_mask
    &
    (~final_positive)
)

remaining_FP_mask = FP_mask

remaining_FN_mask = FN_mask


print()
print(
    "Baseline FP:",
    int(
        baseline_FP_mask.sum()
    )
)

print(
    "Rejected FP:",
    int(
        rejected_FP_mask.sum()
    )
)

print(
    "Remaining FP:",
    int(
        remaining_FP_mask.sum()
    )
)

print(
    "Baseline TP:",
    int(
        baseline_TP_mask.sum()
    )
)

print(
    "Rejected TP:",
    int(
        rejected_TP_mask.sum()
    )
)

print(
    "Final FN:",
    int(
        remaining_FN_mask.sum()
    )
)


# ================================================================
# 11. FEATURE EXTRACTION
# ================================================================

print()
print("=" * 72)
print("8. EXTRACTING RESIDUAL-ERROR FEATURES")
print("=" * 72)


def safe_mean(
    values
):

    if len(values) == 0:

        return np.nan

    return float(
        np.mean(
            values
        )
    )


def safe_median(
    values
):

    if len(values) == 0:

        return np.nan

    return float(
        np.median(
            values
        )
    )


def safe_std(
    values
):

    if len(values) == 0:

        return np.nan

    return float(
        np.std(
            values
        )
    )


def percentile(
    values,
    q
):

    if len(values) == 0:

        return np.nan

    return float(
        np.percentile(
            values,
            q
        )
    )


def extract_features(
    signal
):

    signal = np.asarray(
        signal,
        dtype=np.float64
    )

    # ------------------------------------------------------------
    # Basic amplitude
    # ------------------------------------------------------------

    abs_signal = np.abs(
        signal
    )

    rms = np.sqrt(
        np.mean(
            signal ** 2,
            axis=1
        )
    )

    mean_abs = np.mean(
        abs_signal,
        axis=1
    )

    peak_to_peak = (
        np.max(
            signal,
            axis=1
        )
        -
        np.min(
            signal,
            axis=1
        )
    )

    # ------------------------------------------------------------
    # Line length
    # ------------------------------------------------------------

    differences = np.diff(
        signal,
        axis=1
    )

    line_length = np.mean(
        np.abs(
            differences
        ),
        axis=1
    )

    # ------------------------------------------------------------
    # Zero crossing rate
    # ------------------------------------------------------------

    signs = np.signbit(
        signal
    )

    zero_crossings = np.diff(
        signs.astype(
            np.int8
        ),
        axis=1
    ) != 0

    zcr = np.mean(
        zero_crossings,
        axis=1
    )

    # ------------------------------------------------------------
    # Frequency-domain features
    # ------------------------------------------------------------

    freqs = np.fft.rfftfreq(
        signal.shape[1],
        d=1.0 / SAMPLING_RATE
    )

    fft_values = np.fft.rfft(
        signal,
        axis=1
    )

    power = (
        np.abs(
            fft_values
        ) ** 2
    )

    total_power = np.sum(
        power,
        axis=1
    ) + EPS

    def band_power(
        low,
        high
    ):

        mask = (
            (freqs >= low)
            &
            (freqs < high)
        )

        if not np.any(mask):

            return np.zeros(
                signal.shape[0]
            )

        return np.sum(
            power[
                :,
                mask
            ],
            axis=1
        )

    delta = band_power(
        0.5,
        4.0
    )

    theta = band_power(
        4.0,
        8.0
    )

    alpha = band_power(
        8.0,
        13.0
    )

    beta = band_power(
        13.0,
        30.0
    )

    gamma = band_power(
        30.0,
        45.0
    )

    high_frequency = band_power(
        45.0,
        min(
            100.0,
            SAMPLING_RATE / 2.0
        )
    )

    return {

        "rms":
            float(
                np.mean(
                    rms
                )
            ),

        "mean_abs":
            float(
                np.mean(
                    mean_abs
                )
            ),

        "peak_to_peak":
            float(
                np.mean(
                    peak_to_peak
                )
            ),

        "line_length":
            float(
                np.mean(
                    line_length
                )
            ),

        "zero_crossing_rate":
            float(
                np.mean(
                    zcr
                )
            ),

        "delta_relative":
            float(
                np.mean(
                    delta
                    /
                    total_power
                )
            ),

        "theta_relative":
            float(
                np.mean(
                    theta
                    /
                    total_power
                )
            ),

        "alpha_relative":
            float(
                np.mean(
                    alpha
                    /
                    total_power
                )
            ),

        "beta_relative":
            float(
                np.mean(
                    beta
                    /
                    total_power
                )
            ),

        "gamma_relative":
            float(
                np.mean(
                    gamma
                    /
                    total_power
                )
            ),

        "high_frequency_relative":
            float(
                np.mean(
                    high_frequency
                    /
                    total_power
                )
            ),
    }


# ================================================================
# 12. EXTRACT FEATURES FOR ALL TEST WINDOWS
# ================================================================

feature_names = [
    "rms",
    "mean_abs",
    "peak_to_peak",
    "line_length",
    "zero_crossing_rate",
    "delta_relative",
    "theta_relative",
    "alpha_relative",
    "beta_relative",
    "gamma_relative",
    "high_frequency_relative",
]


feature_matrix = np.zeros(
    (
        n,
        len(feature_names)
    ),
    dtype=np.float64
)


for i, index in enumerate(
    test_indices
):

    if (
        i == 0
        or
        (i + 1) % 100 == 0
        or
        (i + 1) == n
    ):

        print(
            f"Processed {i + 1}/{n}"
        )

    signal = X[
        index
    ]

    features = extract_features(
        signal
    )

    for j, name in enumerate(
        feature_names
    ):

        feature_matrix[
            i,
            j
        ] = features[
            name
        ]


# ================================================================
# 13. FEATURE MATRIX VALIDATION
# ================================================================

if not np.all(
    np.isfinite(
        feature_matrix
    )
):

    raise RuntimeError(
        "Feature matrix contains NaN/Inf."
    )


# ================================================================
# 14. FEATURE SUMMARY
# ================================================================

print()
print("=" * 72)
print("9. FEATURE DISTRIBUTIONS BY FINAL ERROR GROUP")
print("=" * 72)


group_masks = {

    "TP":
        TP_mask,

    "FP":
        FP_mask,

    "TN":
        TN_mask,

    "FN":
        FN_mask,

    "Rejected_FP":
        rejected_FP_mask,

    "Rejected_TP":
        rejected_TP_mask,
}


group_statistics = {}


for group_name, mask in group_masks.items():

    group_statistics[
        group_name
    ] = {}

    print()
    print(
        group_name,
        "n=",
        int(
            mask.sum()
        )
    )

    for j, feature_name in enumerate(
        feature_names
    ):

        values = feature_matrix[
            mask,
            j
        ]

        stats = {

            "n":
                int(
                    len(values)
                ),

            "mean":
                safe_mean(
                    values
                ),

            "median":
                safe_median(
                    values
                ),

            "std":
                safe_std(
                    values
                ),

            "p25":
                percentile(
                    values,
                    25
                ),

            "p75":
                percentile(
                    values,
                    75
                ),

            "min":
                (
                    float(
                        np.min(
                            values
                        )
                    )
                    if len(values) > 0
                    else np.nan
                ),

            "max":
                (
                    float(
                        np.max(
                            values
                        )
                    )
                    if len(values) > 0
                    else np.nan
                ),
        }

        group_statistics[
            group_name
        ][
            feature_name
        ] = stats

        print(
            f"{feature_name:28s}"
            f" mean={stats['mean']:.8f}"
            f" median={stats['median']:.8f}"
        )


# ================================================================
# 15. ERROR-WINDOW TABLE
# ================================================================

print()
print("=" * 72)
print("10. BUILDING RESIDUAL ERROR TABLE")
print("=" * 72)


error_mask = (
    remaining_FP_mask
    |
    remaining_FN_mask
)


error_indices = np.where(
    error_mask
)[0]


rows = []


for position in error_indices:

    group = (
        "FP"
        if remaining_FP_mask[
            position
        ]
        else
        "FN"
    )

    row = {

        "position":
            int(
                position
            ),

        "dataset_index":
            int(
                test_indices[
                    position
                ]
            ),

        "patient":
            str(
                patients[
                    position
                ]
            ),

        "label":
            int(
                labels[
                    position
                ]
            ),

        "probability":
            float(
                probabilities[
                    position
                ]
            ),

        "artifact_score":
            float(
                artifact_scores[
                    position
                ]
            ),

        "artifact_rejected":
            bool(
                artifact_rejected[
                    position
                ]
            ),

        "baseline_positive":
            bool(
                baseline_positive[
                    position
                ]
            ),

        "final_positive":
            bool(
                final_positive[
                    position
                ]
            ),

        "error_type":
            group,
    }

    for j, feature_name in enumerate(
        feature_names
    ):

        row[
            feature_name
        ] = float(
            feature_matrix[
                position,
                j
            ]
        )

    rows.append(
        row
    )


# Sort by error type and probability
rows.sort(
    key=lambda row: (
        row["error_type"],
        -row["probability"]
    )
)


# ================================================================
# 16. SAVE ERROR CSV
# ================================================================

if len(rows) > 0:

    fieldnames = list(
        rows[0].keys()
    )

else:

    fieldnames = [
        "position",
        "dataset_index",
        "patient",
        "label",
        "probability",
        "artifact_score",
        "artifact_rejected",
        "baseline_positive",
        "final_positive",
        "error_type",
    ] + feature_names


with open(
    OUTPUT_WINDOWS_CSV,
    "w",
    newline="",
    encoding="utf-8-sig"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames
    )

    writer.writeheader()

    for row in rows:

        writer.writerow(
            row
        )


print()
print(
    "[OK] Error-window CSV saved:"
)

print(
    OUTPUT_WINDOWS_CSV
)


# ================================================================
# 17. PATIENT-LEVEL ERROR SUMMARY
# ================================================================

print()
print("=" * 72)
print("11. PATIENT-LEVEL RESIDUAL ERROR SUMMARY")
print("=" * 72)


unique_patients = np.unique(
    patients
)


patient_rows = []


for patient in unique_patients:

    patient_mask = (
        patients == patient
    )

    patient_FP = (
        patient_mask
        &
        remaining_FP_mask
    )

    patient_FN = (
        patient_mask
        &
        remaining_FN_mask
    )

    patient_TP = (
        patient_mask
        &
        TP_mask
    )

    patient_TN = (
        patient_mask
        &
        TN_mask
    )

    baseline_patient_FP = (
        patient_mask
        &
        baseline_FP_mask
    )

    rejected_patient_FP = (
        patient_mask
        &
        rejected_FP_mask
    )

    rejected_patient_TP = (
        patient_mask
        &
        rejected_TP_mask
    )

    patient_row = {

        "patient":
            str(
                patient
            ),

        "total_windows":
            int(
                patient_mask.sum()
            ),

        "TP":
            int(
                patient_TP.sum()
            ),

        "FP":
            int(
                patient_FP.sum()
            ),

        "TN":
            int(
                patient_TN.sum()
            ),

        "FN":
            int(
                patient_FN.sum()
            ),

        "baseline_FP":
            int(
                baseline_patient_FP.sum()
            ),

        "rejected_FP":
            int(
                rejected_patient_FP.sum()
            ),

        "rejected_TP":
            int(
                rejected_patient_TP.sum()
            ),
    }

    patient_rows.append(
        patient_row
    )


patient_rows.sort(
    key=lambda row: (
        -row["FP"],
        -row["FN"]
    )
)


with open(
    OUTPUT_SUMMARY_CSV,
    "w",
    newline="",
    encoding="utf-8-sig"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=list(
            patient_rows[0].keys()
        )
        if patient_rows
        else [
            "patient",
            "total_windows",
            "TP",
            "FP",
            "TN",
            "FN",
            "baseline_FP",
            "rejected_FP",
            "rejected_TP",
        ]
    )

    writer.writeheader()

    for row in patient_rows:

        writer.writerow(
            row
        )


print()
print(
    "[OK] Patient summary saved:"
)

print(
    OUTPUT_SUMMARY_CSV
)


# ================================================================
# 18. RESIDUAL ERROR PROBABILITY DISTRIBUTION
# ================================================================

print()
print("=" * 72)
print("12. RESIDUAL ERROR PROBABILITY ANALYSIS")
print("=" * 72)


fp_probabilities = probabilities[
    remaining_FP_mask
]

fn_probabilities = probabilities[
    remaining_FN_mask
]

tp_probabilities = probabilities[
    TP_mask
]


print()
print(
    "Remaining FP probability:"
)

print(
    "mean =",
    safe_mean(
        fp_probabilities
    )
)

print(
    "median =",
    safe_median(
        fp_probabilities
    )
)

print(
    "min =",
    (
        np.min(
            fp_probabilities
        )
        if len(fp_probabilities)
        else np.nan
    )
)

print(
    "max =",
    (
        np.max(
            fp_probabilities
        )
        if len(fp_probabilities)
        else np.nan
    )
)


print()
print(
    "Remaining FN probability:"
)

print(
    "mean =",
    safe_mean(
        fn_probabilities
    )
)

print(
    "median =",
    safe_median(
        fn_probabilities
    )
)

print(
    "min =",
    (
        np.min(
            fn_probabilities
        )
        if len(fn_probabilities)
        else np.nan
    )
)

print(
    "max =",
    (
        np.max(
            fn_probabilities
        )
        if len(fn_probabilities)
        else np.nan
    )
)


# ================================================================
# 19. HARD FALSE POSITIVES
# ================================================================

print()
print("=" * 72)
print("13. HIGHEST-CONFIDENCE RESIDUAL FALSE POSITIVES")
print("=" * 72)


fp_positions = np.where(
    remaining_FP_mask
)[0]


fp_order = fp_positions[
    np.argsort(
        -probabilities[
            fp_positions
        ]
    )
]


top_fp = fp_order[
    :min(
        20,
        len(fp_order)
    )
]


for rank, position in enumerate(
    top_fp,
    start=1
):

    print(
        f"{rank:02d}. "
        f"dataset_index={int(test_indices[position])} | "
        f"patient={patients[position]} | "
        f"prob={probabilities[position]:.6f} | "
        f"artifact={artifact_scores[position]:.6f}"
    )


# ================================================================
# 20. HARD FALSE NEGATIVES
# ================================================================

print()
print("=" * 72)
print("14. HIGHEST-CONFIDENCE RESIDUAL FALSE NEGATIVES")
print("=" * 72)


fn_positions = np.where(
    remaining_FN_mask
)[0]


fn_order = fn_positions[
    np.argsort(
        probabilities[
            fn_positions
        ]
    )
]


top_fn = fn_order[
    :min(
        20,
        len(fn_order)
    )
]


for rank, position in enumerate(
    top_fn,
    start=1
):

    print(
        f"{rank:02d}. "
        f"dataset_index={int(test_indices[position])} | "
        f"patient={patients[position]} | "
        f"prob={probabilities[position]:.6f} | "
        f"artifact={artifact_scores[position]:.6f}"
    )


# ================================================================
# 21. FIGURE 1
# PROBABILITY DISTRIBUTIONS
# ================================================================

print()
print("=" * 72)
print("15. CREATING PROBABILITY DISTRIBUTION FIGURE")
print("=" * 72)


fig, ax = plt.subplots(
    figsize=(10, 6)
)


bins = np.linspace(
    0,
    1,
    31
)


if len(tp_probabilities) > 0:

    ax.hist(
        tp_probabilities,
        bins=bins,
        alpha=0.45,
        label="Final TP"
    )


if len(fp_probabilities) > 0:

    ax.hist(
        fp_probabilities,
        bins=bins,
        alpha=0.60,
        label="Residual FP"
    )


if len(fn_probabilities) > 0:

    ax.hist(
        fn_probabilities,
        bins=bins,
        alpha=0.60,
        label="Residual FN"
    )


ax.axvline(
    BASELINE_THRESHOLD,
    linestyle="--",
    linewidth=2,
    label="Baseline threshold"
)


ax.set_xlabel(
    "Model Seizure Probability"
)

ax.set_ylabel(
    "Number of Windows"
)

ax.set_title(
    "Final Test Residual Error Probability Distribution"
)

ax.set_xlim(
    0,
    1
)

ax.grid(
    axis="y",
    alpha=0.25
)

ax.legend()


fig.tight_layout()


path = os.path.join(
    FIGURES_DIR,
    "01_residual_probability_distribution.png"
)


fig.savefig(
    path,
    dpi=200,
    bbox_inches="tight"
)


plt.close(
    fig
)


print(
    "[OK]",
    path
)


# ================================================================
# 22. FIGURE 2
# ARTIFACT SCORE DISTRIBUTION
# ================================================================

print()
print("=" * 72)
print("16. CREATING ARTIFACT SCORE FIGURE")
print("=" * 72)


fig, ax = plt.subplots(
    figsize=(10, 6)
)


if len(fp_positions) > 0:

    ax.hist(
        artifact_scores[
            fp_positions
        ],
        bins=np.linspace(
            0,
            1,
            31
        ),
        alpha=0.60,
        label="Residual FP"
    )


if len(fn_positions) > 0:

    ax.hist(
        artifact_scores[
            fn_positions
        ],
        bins=np.linspace(
            0,
            1,
            31
        ),
        alpha=0.60,
        label="Residual FN"
    )


ax.axvline(
    ARTIFACT_THRESHOLD,
    linestyle="--",
    linewidth=2,
    label=(
        "Validation threshold "
        f"{ARTIFACT_THRESHOLD:.6f}"
    )
)


ax.set_xlabel(
    "Artifact Score"
)

ax.set_ylabel(
    "Number of Windows"
)

ax.set_title(
    "Artifact Scores of Residual Test Errors"
)

ax.set_xlim(
    0,
    1
)

ax.grid(
    axis="y",
    alpha=0.25
)

ax.legend()


fig.tight_layout()


path = os.path.join(
    FIGURES_DIR,
    "02_residual_artifact_score_distribution.png"
)


fig.savefig(
    path,
    dpi=200,
    bbox_inches="tight"
)


plt.close(
    fig
)


print(
    "[OK]",
    path
)


# ================================================================
# 23. FIGURE 3
# FEATURE COMPARISON FP VS FN
# ================================================================

print()
print("=" * 72)
print("17. CREATING RESIDUAL FEATURE COMPARISON")
print("=" * 72)


comparison_features = [
    "rms",
    "line_length",
    "zero_crossing_rate",
    "delta_relative",
    "theta_relative",
    "alpha_relative",
    "beta_relative",
    "gamma_relative",
    "high_frequency_relative",
]


fp_means = []

fn_means = []


for feature_name in comparison_features:

    j = feature_names.index(
        feature_name
    )

    fp_means.append(
        safe_mean(
            feature_matrix[
                FP_mask,
                j
            ]
        )
    )

    fn_means.append(
        safe_mean(
            feature_matrix[
                FN_mask,
                j
            ]
        )
    )


x = np.arange(
    len(comparison_features)
)

width = 0.35


fig, ax = plt.subplots(
    figsize=(13, 6)
)


bars_fp = ax.bar(
    x - width / 2,
    fp_means,
    width,
    label="Residual FP"
)


bars_fn = ax.bar(
    x + width / 2,
    fn_means,
    width,
    label="Residual FN"
)


ax.set_xticks(
    x
)

ax.set_xticklabels(
    comparison_features,
    rotation=35,
    ha="right"
)

ax.set_ylabel(
    "Mean Feature Value"
)

ax.set_title(
    "Residual False Positive vs False Negative Features"
)

ax.grid(
    axis="y",
    alpha=0.25
)

ax.legend()


fig.tight_layout()


path = os.path.join(
    FIGURES_DIR,
    "03_residual_feature_comparison.png"
)


fig.savefig(
    path,
    dpi=200,
    bbox_inches="tight"
)


plt.close(
    fig
)


print(
    "[OK]",
    path
)


# ================================================================
# 24. FIGURE 4
# PATIENT FP/FN
# ================================================================

print()
print("=" * 72)
print("18. CREATING PATIENT ERROR FIGURE")
print("=" * 72)


patient_labels = [
    row["patient"]
    for row in patient_rows
]


patient_fp_counts = [
    row["FP"]
    for row in patient_rows
]


patient_fn_counts = [
    row["FN"]
    for row in patient_rows
]


x = np.arange(
    len(patient_labels)
)


fig, ax = plt.subplots(
    figsize=(11, 6)
)


bars_fp = ax.bar(
    x - width / 2,
    patient_fp_counts,
    width,
    label="Residual FP"
)


bars_fn = ax.bar(
    x + width / 2,
    patient_fn_counts,
    width,
    label="Residual FN"
)


ax.set_xticks(
    x
)

ax.set_xticklabels(
    patient_labels
)

ax.set_ylabel(
    "Number of Windows"
)

ax.set_title(
    "Residual Errors by Test Patient"
)

ax.grid(
    axis="y",
    alpha=0.25
)

ax.legend()


fig.tight_layout()


path = os.path.join(
    FIGURES_DIR,
    "04_residual_errors_by_patient.png"
)


fig.savefig(
    path,
    dpi=200,
    bbox_inches="tight"
)


plt.close(
    fig
)


print(
    "[OK]",
    path
)


# ================================================================
# 25. BUILD SUMMARY OBJECT
# ================================================================

print()
print("=" * 72)
print("19. BUILDING FINAL ANALYSIS SUMMARY")
print("=" * 72)


summary = {

    "project":
        PROJECT_DIR,

    "test_windows":
        int(n),

    "baseline_probability_threshold":
        BASELINE_THRESHOLD,

    "validation_selected_artifact_threshold":
        ARTIFACT_THRESHOLD,

    "window_duration_seconds":
        float(
            window_duration
        ),

    "final_confusion_matrix": {

        "TP":
            int(
                TP_mask.sum()
            ),

        "FP":
            int(
                FP_mask.sum()
            ),

        "TN":
            int(
                TN_mask.sum()
            ),

        "FN":
            int(
                FN_mask.sum()
            ),
    },

    "baseline_confusion_matrix": {

        "TP":
            int(
                baseline_TP_mask.sum()
            ),

        "FP":
            int(
                baseline_FP_mask.sum()
            ),

        "TN":
            int(
                (
                    (~baseline_positive)
                    &
                    (labels == 0)
                ).sum()
            ),

        "FN":
            int(
                (
                    (~baseline_positive)
                    &
                    (labels == 1)
                ).sum()
            ),
    },

    "artifact_transition": {

        "baseline_fp":
            int(
                baseline_FP_mask.sum()
            ),

        "rejected_fp":
            int(
                rejected_FP_mask.sum()
            ),

        "remaining_fp":
            int(
                remaining_FP_mask.sum()
            ),

        "baseline_tp":
            int(
                baseline_TP_mask.sum()
            ),

        "rejected_tp":
            int(
                rejected_TP_mask.sum()
            ),
    },

    "residual_error_counts": {

        "remaining_false_positives":
            int(
                remaining_FP_mask.sum()
            ),

        "remaining_false_negatives":
            int(
                remaining_FN_mask.sum()
            ),
    },

    "residual_probability_statistics": {

        "FP_mean":
            safe_mean(
                fp_probabilities
            ),

        "FP_median":
            safe_median(
                fp_probabilities
            ),

        "FP_p25":
            percentile(
                fp_probabilities,
                25
            ),

        "FP_p75":
            percentile(
                fp_probabilities,
                75
            ),

        "FN_mean":
            safe_mean(
                fn_probabilities
            ),

        "FN_median":
            safe_median(
                fn_probabilities
            ),

        "FN_p25":
            percentile(
                fn_probabilities,
                25
            ),

        "FN_p75":
            percentile(
                fn_probabilities,
                75
            ),
    },

    "patient_count":
        int(
            len(
                unique_patients
            )
        ),

    "patient_summary":
        patient_rows,

    "feature_statistics":
        group_statistics,
}


# ================================================================
# 26. SAVE JSON
# ================================================================

with open(
    OUTPUT_JSON,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        summary,
        f,
        indent=4,
        ensure_ascii=False,
        allow_nan=True
    )


print()
print(
    "[OK] JSON saved:"
)

print(
    OUTPUT_JSON
)


# ================================================================
# 27. SAVE NPZ
# ================================================================

np.savez(
    OUTPUT_NPZ,

    test_indices=
        test_indices,

    labels=
        labels,

    patients=
        patients,

    probabilities=
        probabilities.astype(
            np.float32
        ),

    artifact_scores=
        artifact_scores.astype(
            np.float32
        ),

    baseline_positive=
        baseline_positive,

    final_positive=
        final_positive,

    artifact_rejected=
        artifact_rejected,

    TP_mask=
        TP_mask,

    FP_mask=
        FP_mask,

    TN_mask=
        TN_mask,

    FN_mask=
        FN_mask,

    rejected_FP_mask=
        rejected_FP_mask,

    rejected_TP_mask=
        rejected_TP_mask,

    feature_names=
        np.asarray(
            feature_names
        ),

    feature_matrix=
        feature_matrix.astype(
            np.float32
        ),
)


print()
print(
    "[OK] NPZ saved:"
)

print(
    OUTPUT_NPZ
)


# ================================================================
# 28. FINAL INTERPRETATION
# ================================================================

fp_count = int(
    remaining_FP_mask.sum()
)

fn_count = int(
    remaining_FN_mask.sum()
)

rejected_fp_count = int(
    rejected_FP_mask.sum()
)

rejected_tp_count = int(
    rejected_TP_mask.sum()
)


print()
print("=" * 72)
print("20. FINAL RESIDUAL ERROR SUMMARY")
print("=" * 72)

print()
print(
    "Baseline FP:",
    int(
        baseline_FP_mask.sum()
    )
)

print(
    "Rejected FP:",
    rejected_fp_count
)

print(
    "Remaining FP:",
    fp_count
)

print()
print(
    "Baseline TP:",
    int(
        baseline_TP_mask.sum()
    )
)

print(
    "Rejected TP:",
    rejected_tp_count
)

print()
print(
    "Remaining FN:",
    fn_count
)

print()
print(
    "Residual FP mean probability:",
    f"{safe_mean(fp_probabilities):.6f}"
)

print(
    "Residual FN mean probability:",
    f"{safe_mean(fn_probabilities):.6f}"
)


if fp_count > 0:

    fp_high_confidence = int(
        np.sum(
            fp_probabilities >= 0.99
        )
    )

    print()
    print(
        "Residual FP with probability >= 0.99:",
        fp_high_confidence
    )


if fn_count > 0:

    fn_low_confidence = int(
        np.sum(
            fn_probabilities < 0.50
        )
    )

    print()
    print(
        "Residual FN with probability < 0.50:",
        fn_low_confidence
    )


print()
print(
    "Figures saved to:"
)

print(
    FIGURES_DIR
)

print()
print("=" * 72)
print("FINAL TEST RESIDUAL ERROR ANALYSIS COMPLETED")
print("=" * 72)

print()
print(
    "No optimization was performed."
)

print(
    "No test threshold was fitted."
)

print(
    "Artifact threshold remained:",
    ARTIFACT_THRESHOLD
)

print()
print(
    "Outputs:"
)

print(
    OUTPUT_JSON
)

print(
    OUTPUT_WINDOWS_CSV
)

print(
    OUTPUT_SUMMARY_CSV
)

print(
    OUTPUT_NPZ
)

print(
    FIGURES_DIR
)

print()
print("=" * 72)
print("DONE")
print("=" * 72)