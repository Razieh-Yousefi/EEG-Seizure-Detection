# ================================================================
# analyze_fp_signal_characteristics.py
#
# Signal-level analysis of false positives.
#
# PURPOSE:
# - Compare EEG signal characteristics of TP / FP / TN windows.
# - Detect abnormal amplitude / energy patterns in false positives.
# - Identify channels that are disproportionately involved in FPs.
# - Compare FP characteristics against true seizures.
#
# IMPORTANT:
# - Does NOT modify model.
# - Does NOT modify X/y.
# - Does NOT modify threshold.
# - Does NOT optimize threshold.
# ================================================================

import os
import json
import numpy as np


# ================================================================
# 1. CONFIGURATION
# ================================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

X_FILE = os.path.join(
    BASE_DIR,
    "X_chbmit_full.npy"
)

Y_FILE = os.path.join(
    BASE_DIR,
    "y_chbmit_full.npy"
)

PATIENTS_FILE = os.path.join(
    BASE_DIR,
    "patients_chbmit_full.npy"
)

TEST_INDICES_FILE = os.path.join(
    BASE_DIR,
    "test_indices.npy"
)

THRESHOLD_FILE = os.path.join(
    BASE_DIR,
    "validation_threshold_results.json"
)

PROB_FILE = os.path.join(
    BASE_DIR,
    "test_window_probabilities.npz"
)

OUTPUT_FILE = os.path.join(
    BASE_DIR,
    "fp_signal_characteristics_results.json"
)

# Number of channels
N_CHANNELS = 23


# ================================================================
# 2. HEADER
# ================================================================

print()
print("=" * 70)
print("FALSE-POSITIVE SIGNAL CHARACTERISTICS ANALYSIS")
print("=" * 70)

print()
print("Base directory:")
print(BASE_DIR)


# ================================================================
# 3. CHECK FILES
# ================================================================

print()
print("=" * 70)
print("1. CHECKING INPUT FILES")
print("=" * 70)

required_files = [
    X_FILE,
    Y_FILE,
    PATIENTS_FILE,
    TEST_INDICES_FILE,
    THRESHOLD_FILE,
    PROB_FILE,
]

for path in required_files:

    if os.path.exists(path):
        print("[OK]", path)

    else:
        raise FileNotFoundError(
            f"Required file not found:\n{path}"
        )


# ================================================================
# 4. LOAD DATA
# ================================================================

print()
print("=" * 70)
print("2. LOADING DATA")
print("=" * 70)

X = np.load(
    X_FILE,
    mmap_mode="r"
)

y = np.load(
    Y_FILE
)

patients = np.load(
    PATIENTS_FILE,
    allow_pickle=True
)

test_indices = np.load(
    TEST_INDICES_FILE
)

print()
print("X shape:")
print(X.shape)

print()
print("y shape:")
print(y.shape)

print()
print("patients shape:")
print(patients.shape)

print()
print("test indices:")
print(test_indices.shape)


if X.ndim != 3:
    raise ValueError(
        "Expected X to have shape "
        "(samples, channels, time)."
    )

if X.shape[1] != N_CHANNELS:
    raise ValueError(
        f"Expected {N_CHANNELS} channels, "
        f"but found {X.shape[1]}."
    )


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


def find_threshold(obj):

    if isinstance(obj, dict):

        preferred_keys = [
            "selected_threshold",
            "validation_threshold",
            "best_threshold",
            "threshold",
        ]

        for key in preferred_keys:

            if key in obj:

                value = obj[key]

                if isinstance(
                    value,
                    (int, float)
                ):

                    return float(value)

        for value in obj.values():

            result = find_threshold(value)

            if result is not None:
                return result

    elif isinstance(obj, list):

        for item in obj:

            result = find_threshold(item)

            if result is not None:
                return result

    return None


THRESHOLD = find_threshold(
    threshold_data
)

if THRESHOLD is None:

    raise RuntimeError(
        "Could not find validation threshold."
    )

print()
print(
    f"Validation threshold: {THRESHOLD:.4f}"
)


# ================================================================
# 6. LOAD TEST PROBABILITIES
# ================================================================

print()
print("=" * 70)
print("4. LOADING TEST PROBABILITIES")
print("=" * 70)

prob_data = np.load(
    PROB_FILE
)

probabilities = np.asarray(
    prob_data["probabilities"],
    dtype=np.float32
)

saved_indices = np.asarray(
    prob_data["test_indices"]
)

saved_labels = np.asarray(
    prob_data["labels"]
)

saved_patients = np.asarray(
    prob_data["patients"]
)


# ================================================================
# 7. VERIFY ALIGNMENT
# ================================================================

print()
print("=" * 70)
print("5. VERIFYING DATA ALIGNMENT")
print("=" * 70)

if len(probabilities) != len(test_indices):
    raise ValueError(
        "Probability count does not match "
        "test index count."
    )

if not np.array_equal(
    saved_indices,
    test_indices
):
    raise ValueError(
        "Saved test indices do not match "
        "current test_indices.npy."
    )

y_test = y[test_indices]

patients_test = patients[test_indices]

if not np.array_equal(
    saved_labels,
    y_test
):
    raise ValueError(
        "Saved labels do not match "
        "current test labels."
    )

if not np.array_equal(
    saved_patients,
    patients_test
):
    raise ValueError(
        "Saved patients do not match "
        "current test patients."
    )

print()
print("[OK] Alignment verified.")

print(
    "Test samples:",
    len(test_indices)
)


# ================================================================
# 8. CREATE PREDICTIONS
# ================================================================

print()
print("=" * 70)
print("6. CREATING TEST GROUPS")
print("=" * 70)

predictions = (
    probabilities >= THRESHOLD
).astype(np.int64)

tp_mask = (
    (predictions == 1)
    & (y_test == 1)
)

fp_mask = (
    (predictions == 1)
    & (y_test == 0)
)

tn_mask = (
    (predictions == 0)
    & (y_test == 0)
)

fn_mask = (
    (predictions == 0)
    & (y_test == 1)
)

print()
print("TP:", int(tp_mask.sum()))
print("FP:", int(fp_mask.sum()))
print("TN:", int(tn_mask.sum()))
print("FN:", int(fn_mask.sum()))


# ================================================================
# 9. LOAD TEST WINDOWS
# ================================================================

print()
print("=" * 70)
print("7. LOADING TEST EEG WINDOWS")
print("=" * 70)

X_test = np.asarray(
    X[test_indices],
    dtype=np.float32
)

print()
print("X_test shape:")
print(X_test.shape)


# ================================================================
# 10. SIGNAL FEATURE EXTRACTION
# ================================================================

print()
print("=" * 70)
print("8. EXTRACTING SIGNAL FEATURES")
print("=" * 70)

print()
print("Calculating:")
print("- Global RMS")
print("- Global standard deviation")
print("- Global peak-to-peak amplitude")
print("- Maximum absolute amplitude")
print("- Per-channel RMS")
print("- Per-channel peak-to-peak amplitude")


# ----------------------------------------------------------------
# Global features
# ----------------------------------------------------------------

global_rms = np.sqrt(
    np.mean(
        X_test ** 2,
        axis=(1, 2)
    )
)

global_std = np.std(
    X_test,
    axis=(1, 2)
)

global_ptp = (
    np.max(
        X_test,
        axis=(1, 2)
    )
    -
    np.min(
        X_test,
        axis=(1, 2)
    )
)

global_max_abs = np.max(
    np.abs(X_test),
    axis=(1, 2)
)


# ----------------------------------------------------------------
# Per-channel features
# ----------------------------------------------------------------

channel_rms = np.sqrt(
    np.mean(
        X_test ** 2,
        axis=2
    )
)

channel_ptp = (
    np.max(
        X_test,
        axis=2
    )
    -
    np.min(
        X_test,
        axis=2
    )
)

channel_std = np.std(
    X_test,
    axis=2
)


print()
print("[OK] Signal features extracted.")


# ================================================================
# 11. GROUP STATISTICS
# ================================================================

def summarize_feature(
    values,
    mask
):

    v = values[mask]

    if len(v) == 0:

        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
            "std": None,
        }

    return {
        "count": int(len(v)),
        "min": float(np.min(v)),
        "max": float(np.max(v)),
        "mean": float(np.mean(v)),
        "median": float(np.median(v)),
        "std": float(np.std(v)),
    }


print()
print("=" * 70)
print("9. GLOBAL SIGNAL CHARACTERISTICS")
print("=" * 70)


groups = {
    "TP": tp_mask,
    "FP": fp_mask,
    "TN": tn_mask,
    "FN": fn_mask,
}


global_statistics = {}

for name, mask in groups.items():

    print()
    print("-" * 70)
    print(name)
    print("-" * 70)

    stats = {
        "rms": summarize_feature(
            global_rms,
            mask
        ),

        "std": summarize_feature(
            global_std,
            mask
        ),

        "peak_to_peak": summarize_feature(
            global_ptp,
            mask
        ),

        "max_abs": summarize_feature(
            global_max_abs,
            mask
        ),
    }

    global_statistics[name] = stats

    if stats["rms"]["count"] > 0:

        print(
            "RMS mean   :",
            f"{stats['rms']['mean']:.6f}"
        )

        print(
            "RMS median :",
            f"{stats['rms']['median']:.6f}"
        )

        print(
            "STD mean   :",
            f"{stats['std']['mean']:.6f}"
        )

        print(
            "PTP mean   :",
            f"{stats['peak_to_peak']['mean']:.6f}"
        )

        print(
            "MaxAbs mean:",
            f"{stats['max_abs']['mean']:.6f}"
        )


# ================================================================
# 12. CHANNEL-LEVEL ANALYSIS
# ================================================================

print()
print("=" * 70)
print("10. CHANNEL-LEVEL ANALYSIS")
print("=" * 70)


channel_statistics = {}


for group_name, mask in groups.items():

    if int(mask.sum()) == 0:
        continue

    print()
    print(
        f"Analyzing {group_name}..."
    )

    rms_values = channel_rms[mask]

    ptp_values = channel_ptp[mask]

    std_values = channel_std[mask]

    channel_statistics[group_name] = {
        "rms_mean_per_channel": [
            float(x)
            for x in np.mean(
                rms_values,
                axis=0
            )
        ],

        "rms_median_per_channel": [
            float(x)
            for x in np.median(
                rms_values,
                axis=0
            )
        ],

        "ptp_mean_per_channel": [
            float(x)
            for x in np.mean(
                ptp_values,
                axis=0
            )
        ],

        "std_mean_per_channel": [
            float(x)
            for x in np.mean(
                std_values,
                axis=0
            )
        ],
    }


print()
print("[OK] Channel analysis completed.")


# ================================================================
# 13. FP / TP CHANNEL RATIO
# ================================================================

print()
print("=" * 70)
print("11. FP VS TP CHANNEL COMPARISON")
print("=" * 70)

fp_rms_mean = np.mean(
    channel_rms[fp_mask],
    axis=0
)

tp_rms_mean = np.mean(
    channel_rms[tp_mask],
    axis=0
)

fp_ptp_mean = np.mean(
    channel_ptp[fp_mask],
    axis=0
)

tp_ptp_mean = np.mean(
    channel_ptp[tp_mask],
    axis=0
)


eps = 1e-8

rms_ratio = (
    fp_rms_mean
    /
    (tp_rms_mean + eps)
)

ptp_ratio = (
    fp_ptp_mean
    /
    (tp_ptp_mean + eps)
)


print()
print("Channels with highest FP/TP RMS ratio:")

top_rms_channels = np.argsort(
    rms_ratio
)[::-1][:10]

for rank, ch in enumerate(
    top_rms_channels,
    start=1
):

    print(
        f"{rank:02d}. "
        f"Channel {ch:02d} | "
        f"ratio={rms_ratio[ch]:.4f} | "
        f"FP_RMS={fp_rms_mean[ch]:.6f} | "
        f"TP_RMS={tp_rms_mean[ch]:.6f}"
    )


print()
print("Channels with highest FP/TP PTP ratio:")

top_ptp_channels = np.argsort(
    ptp_ratio
)[::-1][:10]

for rank, ch in enumerate(
    top_ptp_channels,
    start=1
):

    print(
        f"{rank:02d}. "
        f"Channel {ch:02d} | "
        f"ratio={ptp_ratio[ch]:.4f} | "
        f"FP_PTP={fp_ptp_mean[ch]:.6f} | "
        f"TP_PTP={tp_ptp_mean[ch]:.6f}"
    )


# ================================================================
# 14. FP WINDOWS WITH EXTREME SIGNAL AMPLITUDE
# ================================================================

print()
print("=" * 70)
print("12. EXTREME-AMPLITUDE FALSE POSITIVES")
print("=" * 70)

fp_positions = np.where(
    fp_mask
)[0]

if len(fp_positions) > 0:

    top_fp_amplitude_positions = fp_positions[
        np.argsort(
            global_max_abs[fp_positions]
        )[::-1][:20]
    ]

    for rank, pos in enumerate(
        top_fp_amplitude_positions,
        start=1
    ):

        print(
            f"{rank:02d}. "
            f"{patients_test[pos]} | "
            f"original_index="
            f"{int(test_indices[pos])} | "
            f"prob="
            f"{probabilities[pos]:.6f} | "
            f"RMS="
            f"{global_rms[pos]:.6f} | "
            f"PTP="
            f"{global_ptp[pos]:.6f} | "
            f"MaxAbs="
            f"{global_max_abs[pos]:.6f}"
        )


# ================================================================
# 15. FP PATIENT SUMMARY
# ================================================================

print()
print("=" * 70)
print("13. PATIENT-LEVEL SIGNAL SUMMARY")
print("=" * 70)

patient_signal_results = {}

unique_patients = np.unique(
    patients_test
)

for patient in unique_patients:

    patient_mask = (
        patients_test == patient
    )

    patient_fp = (
        patient_mask
        & fp_mask
    )

    patient_tp = (
        patient_mask
        & tp_mask
    )

    patient_name = str(patient)

    result = {
        "fp_count": int(
            patient_fp.sum()
        ),

        "tp_count": int(
            patient_tp.sum()
        ),

        "fp_rms_mean": None,
        "tp_rms_mean": None,

        "fp_ptp_mean": None,
        "tp_ptp_mean": None,

        "fp_max_abs_mean": None,
        "tp_max_abs_mean": None,
    }

    if patient_fp.sum() > 0:

        result["fp_rms_mean"] = float(
            np.mean(
                global_rms[patient_fp]
            )
        )

        result["fp_ptp_mean"] = float(
            np.mean(
                global_ptp[patient_fp]
            )
        )

        result["fp_max_abs_mean"] = float(
            np.mean(
                global_max_abs[patient_fp]
            )
        )

    if patient_tp.sum() > 0:

        result["tp_rms_mean"] = float(
            np.mean(
                global_rms[patient_tp]
            )
        )

        result["tp_ptp_mean"] = float(
            np.mean(
                global_ptp[patient_tp]
            )
        )

        result["tp_max_abs_mean"] = float(
            np.mean(
                global_max_abs[patient_tp]
            )
        )

    patient_signal_results[
        patient_name
    ] = result

    print()
    print("-" * 70)
    print(patient_name)
    print("-" * 70)

    print(
        "FP count:",
        result["fp_count"]
    )

    print(
        "FP RMS mean:",
        (
            f"{result['fp_rms_mean']:.6f}"
            if result["fp_rms_mean"] is not None
            else "N/A"
        )
    )

    print(
        "FP PTP mean:",
        (
            f"{result['fp_ptp_mean']:.6f}"
            if result["fp_ptp_mean"] is not None
            else "N/A"
        )
    )


# ================================================================
# 16. SAVE RESULTS
# ================================================================

print()
print("=" * 70)
print("14. SAVING SIGNAL ANALYSIS")
print("=" * 70)


results = {

    "validation_threshold": float(
        THRESHOLD
    ),

    "test_samples": int(
        len(test_indices)
    ),

    "channels": int(
        X.shape[1]
    ),

    "global_statistics":
        global_statistics,

    "channel_statistics":
        channel_statistics,

    "fp_vs_tp_channel_comparison": {

        "rms_ratio_fp_over_tp": [
            float(x)
            for x in rms_ratio
        ],

        "ptp_ratio_fp_over_tp": [
            float(x)
            for x in ptp_ratio
        ],

        "top_rms_channels": [
            {
                "channel": int(ch),
                "ratio": float(
                    rms_ratio[ch]
                )
            }
            for ch in top_rms_channels
        ],

        "top_ptp_channels": [
            {
                "channel": int(ch),
                "ratio": float(
                    ptp_ratio[ch]
                )
            }
            for ch in top_ptp_channels
        ],
    },

    "patient_results":
        patient_signal_results,

    "note": (
        "Diagnostic signal-level analysis only. "
        "No model, dataset, or threshold was modified."
    ),
}


with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        results,
        f,
        indent=2,
        ensure_ascii=False
    )


print()
print(
    "[OK] Signal analysis saved:"
)

print(
    OUTPUT_FILE
)


# ================================================================
# 17. FINAL
# ================================================================

print()
print("=" * 70)
print("FALSE-POSITIVE SIGNAL CHARACTERISTICS ANALYSIS COMPLETED")
print("=" * 70)

print()
print("No model modification.")
print("No dataset modification.")
print("No threshold modification.")
print("No threshold optimization.")

print()
print("=" * 70)