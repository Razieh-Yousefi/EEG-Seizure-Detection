# ================================================================
# visualize_fp_tp_examples.py
#
# Visualization of representative EEG examples:
# - High confidence false positives
# - Medium confidence false positives
# - True positives
#
# PURPOSE:
# Visual inspection of EEG patterns causing FP errors.
#
# Does NOT modify:
# - Model
# - Dataset
# - Threshold
# ================================================================

import os
import numpy as np
import matplotlib.pyplot as plt


# ================================================================
# CONFIGURATION
# ================================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

NPZ_FILE = os.path.join(
    BASE_DIR,
    "fp_tp_examples.npz"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "fp_tp_visualizations"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


SAMPLE_RATE = 256


# ================================================================
# HEADER
# ================================================================

print()
print("=" * 70)
print("FP / TP EEG VISUALIZATION")
print("=" * 70)


# ================================================================
# LOAD DATA
# ================================================================

if not os.path.exists(NPZ_FILE):

    raise FileNotFoundError(
        NPZ_FILE
    )


data = np.load(
    NPZ_FILE,
    allow_pickle=True
)


print()
print("Loaded:")

for key in data.files:

    print(
        key,
        data[key].shape
    )


# ================================================================
# PLOT FUNCTION
# ================================================================

def plot_eeg(
    signal,
    title,
    filename,
    probability=None,
    label=""
):

    channels, samples = signal.shape

    time = (
        np.arange(samples)
        / SAMPLE_RATE
    )


    plt.figure(
        figsize=(14, 10)
    )


    offset = 0


    for ch in range(channels):

        channel_signal = signal[ch]

        scale = (
            np.max(
                np.abs(channel_signal)
            )
            * 4
        )

        if scale == 0:
            scale = 1


        plt.plot(
            time,
            channel_signal + offset
        )


        offset += scale


    plt.xlabel(
        "Time (seconds)"
    )

    plt.ylabel(
        "EEG channels (offset)"
    )


    full_title = title

    if probability is not None:

        full_title += (
            f" | probability={probability:.4f}"
        )


    plt.title(
        full_title
    )


    plt.grid(
        True
    )


    path = os.path.join(
        OUTPUT_DIR,
        filename
    )


    plt.tight_layout()


    plt.savefig(
        path,
        dpi=200
    )


    plt.close()


    print(
        "[OK]",
        path
    )



# ================================================================
# HIGH CONFIDENCE FP
# ================================================================

print()
print("=" * 70)
print("HIGH CONFIDENCE FALSE POSITIVES")
print("=" * 70)


high_fp = data[
    "high_confidence_fp_signals"
]


for i in range(
    min(5, len(high_fp))
):

    plot_eeg(
        high_fp[i],
        f"High Confidence FP #{i+1}",
        f"high_fp_{i+1}.png",
        label="FP"
    )



# ================================================================
# MEDIUM CONFIDENCE FP
# ================================================================

print()
print("=" * 70)
print("MEDIUM CONFIDENCE FALSE POSITIVES")
print("=" * 70)


medium_fp = data[
    "medium_confidence_fp_signals"
]


for i in range(
    min(5, len(medium_fp))
):

    plot_eeg(
        medium_fp[i],
        f"Medium Confidence FP #{i+1}",
        f"medium_fp_{i+1}.png",
        label="FP"
    )



# ================================================================
# TRUE POSITIVE
# ================================================================

print()
print("=" * 70)
print("TRUE POSITIVES")
print("=" * 70)


tp = data[
    "tp_signals"
]


for i in range(
    min(5, len(tp))
):

    plot_eeg(
        tp[i],
        f"True Positive #{i+1}",
        f"tp_{i+1}.png",
        label="TP"
    )



# ================================================================
# FINAL
# ================================================================

print()
print("=" * 70)
print("VISUALIZATION COMPLETED")
print("=" * 70)

print()
print(
    "Images saved in:"
)

print(
    OUTPUT_DIR
)