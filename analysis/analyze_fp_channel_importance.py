# ================================================================
# analyze_fp_channel_importance.py
#
# Channel importance analysis for false positives.
#
# PURPOSE:
# - Compare EEG channels between TP and FP.
# - Identify channels contributing to FP patterns.
# - Rank channels by FP/TP activity difference.
#
# NO MODEL MODIFICATION
# NO DATA MODIFICATION
# NO THRESHOLD OPTIMIZATION
# ================================================================

import os
import json
import numpy as np


# ================================================================
# CONFIGURATION
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

TEST_INDICES_FILE = os.path.join(
    BASE_DIR,
    "test_indices.npy"
)

PROB_FILE = os.path.join(
    BASE_DIR,
    "test_window_probabilities.npz"
)

THRESHOLD_FILE = os.path.join(
    BASE_DIR,
    "validation_threshold_results.json"
)

OUTPUT_FILE = os.path.join(
    BASE_DIR,
    "fp_channel_importance_results.json"
)


# ================================================================
# HEADER
# ================================================================

print()
print("=" * 70)
print("FALSE-POSITIVE CHANNEL IMPORTANCE ANALYSIS")
print("=" * 70)

print()
print("Base directory:")
print(BASE_DIR)


# ================================================================
# LOAD FILES
# ================================================================

print()
print("=" * 70)
print("1. LOADING DATA")
print("=" * 70)


X = np.load(
    X_FILE
)

y = np.load(
    Y_FILE
)

test_indices = np.load(
    TEST_INDICES_FILE
)


prob_data = np.load(
    PROB_FILE
)

probabilities = prob_data[
    "probabilities"
]


print()
print("X shape:")
print(X.shape)

print("y shape:")
print(y.shape)

print("test samples:")
print(len(test_indices))


# ================================================================
# LOAD THRESHOLD
# ================================================================

with open(
    THRESHOLD_FILE,
    "r",
    encoding="utf-8"
) as f:

    threshold_data = json.load(f)


def find_threshold(obj):

    if isinstance(obj, dict):

        for k in [
            "selected_threshold",
            "validation_threshold",
            "threshold",
            "best_threshold"
        ]:

            if k in obj:

                return float(obj[k])

        for v in obj.values():

            r = find_threshold(v)

            if r is not None:
                return r

    elif isinstance(obj, list):

        for item in obj:

            r = find_threshold(item)

            if r is not None:
                return r

    return None


threshold = find_threshold(
    threshold_data
)


print()
print(
    "Threshold:",
    threshold
)


# ================================================================
# CREATE TEST GROUPS
# ================================================================

print()
print("=" * 70)
print("2. CREATING TP / FP GROUPS")
print("=" * 70)


y_test = y[
    test_indices
]


X_test = X[
    test_indices
]


predictions = (
    probabilities >= threshold
).astype(int)


tp_mask = (
    (predictions == 1)
    &
    (y_test == 1)
)


fp_mask = (
    (predictions == 1)
    &
    (y_test == 0)
)


print()
print(
    "TP:",
    tp_mask.sum()
)

print(
    "FP:",
    fp_mask.sum()
)


TP = X_test[
    tp_mask
]

FP = X_test[
    fp_mask
]


# ================================================================
# CHANNEL FEATURE EXTRACTION
# ================================================================

print()
print("=" * 70)
print("3. CHANNEL FEATURE EXTRACTION")
print("=" * 70)


def channel_statistics(data):

    # data:
    # samples x channels x time

    rms = np.sqrt(
        np.mean(
            data ** 2,
            axis=2
        )
    )

    ptp = (
        np.max(
            data,
            axis=2
        )
        -
        np.min(
            data,
            axis=2
        )
    )

    return {

        "rms_mean":
            np.mean(
                rms,
                axis=0
            ),

        "ptp_mean":
            np.mean(
                ptp,
                axis=0
            )
    }


tp_stats = channel_statistics(
    TP
)

fp_stats = channel_statistics(
    FP
)


# ================================================================
# COMPARE CHANNELS
# ================================================================

print()
print("=" * 70)
print("4. CHANNEL RANKING")
print("=" * 70)


rms_ratio = (
    fp_stats["rms_mean"]
    /
    (tp_stats["rms_mean"] + 1e-12)
)


ptp_ratio = (
    fp_stats["ptp_mean"]
    /
    (tp_stats["ptp_mean"] + 1e-12)
)


ranking = np.argsort(
    rms_ratio
)[::-1]


print()
print(
    "Top channels by FP/TP RMS ratio:"
)


for i, ch in enumerate(
    ranking[:10],
    1
):

    print(
        f"{i:02d}. "
        f"Channel {ch:02d} | "
        f"ratio={rms_ratio[ch]:.4f} | "
        f"FP={fp_stats['rms_mean'][ch]:.6f} | "
        f"TP={tp_stats['rms_mean'][ch]:.6f}"
    )


# ================================================================
# SAVE RESULTS
# ================================================================

print()
print("=" * 70)
print("5. SAVING RESULTS")
print("=" * 70)


results = {

    "threshold": float(threshold),

    "tp_count": int(len(TP)),

    "fp_count": int(len(FP)),

    "channel_count": int(X.shape[1]),

    "rms_ratio_fp_tp":
        rms_ratio.tolist(),

    "ptp_ratio_fp_tp":
        ptp_ratio.tolist(),

    "fp_channel_rms":
        fp_stats["rms_mean"].tolist(),

    "tp_channel_rms":
        tp_stats["rms_mean"].tolist(),

    "fp_channel_ptp":
        fp_stats["ptp_mean"].tolist(),

    "tp_channel_ptp":
        tp_stats["ptp_mean"].tolist(),

    "top_rms_channels":
        ranking[:10].tolist()
}


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
print(
    "[OK] Saved:"
)

print(
    OUTPUT_FILE
)


print()
print("=" * 70)
print("CHANNEL IMPORTANCE ANALYSIS COMPLETED")
print("=" * 70)

print()
print(
    "Model was NOT modified."
)

print(
    "Dataset was NOT modified."
)

print(
    "Threshold was NOT modified."
)