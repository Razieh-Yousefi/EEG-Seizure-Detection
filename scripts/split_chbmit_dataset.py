import os
import numpy as np


# ============================================================
# CHB-MIT TRAIN-ONLY NORMALIZATION PARAMETER CALCULATION
# ============================================================
#
# هدف:
#   محاسبه mean و std فقط از TRAIN samples
#
# نکته مهم:
#   کل X در RAM بارگذاری نمی‌شود.
#   X به صورت memory-mapped خوانده می‌شود.
#
# خروجی:
#   normalization_params.npz
#
# شامل:
#   channel_mean
#   channel_std
#
# این فایل بعداً هنگام آموزش مدل استفاده خواهد شد.
#
# ============================================================


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

X_PATH = os.path.join(
    BASE_DIR,
    "X_chbmit_full.npy"
)

Y_PATH = os.path.join(
    BASE_DIR,
    "y_chbmit_full.npy"
)

TRAIN_INDICES_PATH = os.path.join(
    BASE_DIR,
    "train_indices.npy"
)

VAL_INDICES_PATH = os.path.join(
    BASE_DIR,
    "val_indices.npy"
)

TEST_INDICES_PATH = os.path.join(
    BASE_DIR,
    "test_indices.npy"
)

OUTPUT_PATH = os.path.join(
    BASE_DIR,
    "normalization_params.npz"
)


# ============================================================
# DATASET CONFIGURATION
# ============================================================

EXPECTED_CHANNELS = 23
EXPECTED_SAMPLES_PER_WINDOW = 1280

# تعداد windowهایی که در هر مرحله پردازش می‌شوند.
#
# این مقدار عمداً محدود است تا RAM زیادی مصرف نشود.
#
CHUNK_SIZE = 256


# ============================================================
# PRINT HEADER
# ============================================================

print()
print("=" * 70)
print("CHB-MIT TRAIN-ONLY NORMALIZATION")
print("=" * 70)

print()
print("Base directory:")
print(BASE_DIR)

print()
print("Normalization strategy:")
print("Mean/std calculated ONLY from training data.")

print()
print("X file:")
print(X_PATH)

print()
print("Output:")
print(OUTPUT_PATH)


# ============================================================
# 1. CHECK INPUT FILES
# ============================================================

print()
print("=" * 70)
print("1. CHECKING INPUT FILES")
print("=" * 70)


required_files = {
    "X": X_PATH,
    "y": Y_PATH,
    "train_indices": TRAIN_INDICES_PATH,
    "val_indices": VAL_INDICES_PATH,
    "test_indices": TEST_INDICES_PATH,
}


for name, path in required_files.items():

    if not os.path.exists(path):

        print()
        print(
            f"[FATAL] Missing {name} file:"
        )

        print(path)

        raise FileNotFoundError(
            path
        )

    print(
        f"[OK] {name}: {path}"
    )


# ============================================================
# 2. LOAD X AS MEMORY-MAPPED ARRAY
# ============================================================

print()
print("=" * 70)
print("2. LOADING X AS MEMORY-MAPPED ARRAY")
print("=" * 70)

print()
print("Loading X metadata...")

X = np.load(
    X_PATH,
    mmap_mode="r"
)

print()
print("X shape:")
print(X.shape)

print()
print("X dtype:")
print(X.dtype)


# ============================================================
# 3. CHECK X SHAPE
# ============================================================

print()
print("=" * 70)
print("3. CHECKING X SHAPE")
print("=" * 70)


if X.ndim != 3:

    raise ValueError(
        f"Expected X to have 3 dimensions, "
        f"got {X.ndim}"
    )


number_of_samples = X.shape[0]
number_of_channels = X.shape[1]
samples_per_window = X.shape[2]


print()
print(
    "Number of windows:",
    number_of_samples
)

print(
    "Number of channels:",
    number_of_channels
)

print(
    "Samples per window:",
    samples_per_window
)


if number_of_channels != EXPECTED_CHANNELS:

    raise ValueError(
        f"Expected {EXPECTED_CHANNELS} channels, "
        f"got {number_of_channels}"
    )


if samples_per_window != EXPECTED_SAMPLES_PER_WINDOW:

    raise ValueError(
        f"Expected {EXPECTED_SAMPLES_PER_WINDOW} "
        f"samples per window, "
        f"got {samples_per_window}"
    )


print()
print("[OK] X shape is valid.")


# ============================================================
# 4. LOAD LABELS
# ============================================================

print()
print("=" * 70)
print("4. LOADING LABELS")
print("=" * 70)


y = np.load(
    Y_PATH
)


print()
print("y shape:")
print(y.shape)

print()
print("y dtype:")
print(y.dtype)


if len(y) != number_of_samples:

    raise ValueError(
        "X and y have different sample counts."
    )


print()
print("[OK] X and y have matching sample counts.")


# ============================================================
# 5. LOAD SPLIT INDICES
# ============================================================

print()
print("=" * 70)
print("5. LOADING SPLIT INDICES")
print("=" * 70)


print()
print("Loading train indices...")

train_indices = np.load(
    TRAIN_INDICES_PATH
)

print(
    "Train samples:",
    len(train_indices)
)


print()
print("Loading validation indices...")

val_indices = np.load(
    VAL_INDICES_PATH
)

print(
    "Validation samples:",
    len(val_indices)
)


print()
print("Loading test indices...")

test_indices = np.load(
    TEST_INDICES_PATH
)

print(
    "Test samples:",
    len(test_indices)
)


# ============================================================
# 6. SPLIT CONSISTENCY CHECK
# ============================================================

print()
print("=" * 70)
print("6. CHECKING SPLIT CONSISTENCY")
print("=" * 70)


all_indices = np.concatenate(
    [
        train_indices,
        val_indices,
        test_indices
    ]
)


print()
print(
    "Total split indices:",
    len(all_indices)
)

print(
    "Expected samples:",
    number_of_samples
)


if len(all_indices) != number_of_samples:

    raise ValueError(
        "Split indices do not cover the full dataset."
    )


unique_indices = np.unique(
    all_indices
)


print(
    "Unique split indices:",
    len(unique_indices)
)


if len(unique_indices) != number_of_samples:

    raise ValueError(
        "Duplicate or missing indices detected."
    )


if not np.array_equal(
    unique_indices,
    np.arange(number_of_samples)
):

    raise ValueError(
        "Split indices are not exactly "
        "the complete dataset index range."
    )


print()
print(
    "[OK] All samples belong to exactly one split."
)


# ============================================================
# 7. VERIFY SPLIT LABEL COUNTS
# ============================================================

print()
print("=" * 70)
print("7. SPLIT LABEL CHECK")
print("=" * 70)


train_labels = y[
    train_indices
]

val_labels = y[
    val_indices
]

test_labels = y[
    test_indices
]


train_seizure = int(
    np.sum(train_labels == 1)
)

train_non_seizure = int(
    np.sum(train_labels == 0)
)


val_seizure = int(
    np.sum(val_labels == 1)
)

val_non_seizure = int(
    np.sum(val_labels == 0)
)


test_seizure = int(
    np.sum(test_labels == 1)
)

test_non_seizure = int(
    np.sum(test_labels == 0)
)


print()
print("TRAIN")
print(
    "Samples:",
    len(train_indices)
)
print(
    "Seizure:",
    train_seizure
)
print(
    "Non-seizure:",
    train_non_seizure
)


print()
print("VALIDATION")
print(
    "Samples:",
    len(val_indices)
)
print(
    "Seizure:",
    val_seizure
)
print(
    "Non-seizure:",
    val_non_seizure
)


print()
print("TEST")
print(
    "Samples:",
    len(test_indices)
)
print(
    "Seizure:",
    test_seizure
)
print(
    "Non-seizure:",
    test_non_seizure
)


# ============================================================
# 8. CHECK TRAIN SET
# ============================================================

if len(train_indices) == 0:

    raise ValueError(
        "Training split is empty."
    )


if train_seizure == 0:

    print()
    print(
        "[WARNING] Training set contains "
        "no seizure windows."
    )


else:

    print()
    print(
        "[OK] Training set contains seizure windows."
    )


# ============================================================
# 9. PREPARE ACCUMULATORS
# ============================================================

print()
print("=" * 70)
print("8. PREPARING NORMALIZATION CALCULATION")
print("=" * 70)


#
# We calculate channel-wise mean and std.
#
# Each channel contains:
#
#   number_of_train_windows * 1280
#
# values.
#
# We process them chunk by chunk.
#
# Instead of storing all values in memory,
# we accumulate:
#
#   sum(x)
#   sum(x^2)
#
# using float64.
#
# This keeps RAM usage low.
#


channel_sum = np.zeros(
    number_of_channels,
    dtype=np.float64
)

channel_sum_squared = np.zeros(
    number_of_channels,
    dtype=np.float64
)


total_values_per_channel = 0


# ============================================================
# 10. CALCULATE TRAIN MEAN AND STD
# ============================================================

print()
print("=" * 70)
print("9. CALCULATING TRAIN-ONLY MEAN / STD")
print("=" * 70)

print()
print(
    "Training windows:",
    len(train_indices)
)

print(
    "Chunk size:",
    CHUNK_SIZE
)

print()
print(
    "IMPORTANT:"
)

print(
    "Only TRAIN samples are used."
)

print(
    "Validation and Test samples are NOT used."
)

print()


number_of_train_samples = len(
    train_indices
)

number_of_chunks = (
    number_of_train_samples
    + CHUNK_SIZE
    - 1
) // CHUNK_SIZE


for chunk_number, start in enumerate(
    range(
        0,
        number_of_train_samples,
        CHUNK_SIZE
    ),
    start=1
):

    end = min(
        start + CHUNK_SIZE,
        number_of_train_samples
    )


    indices_chunk = train_indices[
        start:end
    ]


    #
    # IMPORTANT:
    #
    # X is memory-mapped.
    # Only this small chunk is loaded.
    #

    chunk = np.asarray(
        X[
            indices_chunk
        ],
        dtype=np.float64
    )


    #
    # Shape:
    #
    # (chunk_windows, 23, 1280)
    #


    #
    # Sum over windows and time.
    #
    # Result:
    #
    # (23,)
    #

    channel_sum += np.sum(
        chunk,
        axis=(0, 2),
        dtype=np.float64
    )


    channel_sum_squared += np.sum(
        chunk * chunk,
        axis=(0, 2),
        dtype=np.float64
    )


    total_values_per_channel += (
        chunk.shape[0]
        * chunk.shape[2]
    )


    if (
        chunk_number == 1
        or
        chunk_number % 10 == 0
        or
        chunk_number == number_of_chunks
    ):

        percentage = (
            100.0
            * end
            / number_of_train_samples
        )


        print(
            f"Processed "
            f"{end}/{number_of_train_samples} "
            f"train windows "
            f"({percentage:.1f}%)"
        )


    #
    # Explicitly delete the chunk.
    #

    del chunk
    del indices_chunk


# ============================================================
# 11. CALCULATE MEAN
# ============================================================

print()
print("=" * 70)
print("10. CALCULATING CHANNEL MEAN")
print("=" * 70)


if total_values_per_channel == 0:

    raise ValueError(
        "No training values were processed."
    )


channel_mean = (
    channel_sum
    / total_values_per_channel
)


# ============================================================
# 12. CALCULATE VARIANCE
# ============================================================

channel_variance = (
    channel_sum_squared
    / total_values_per_channel
    -
    channel_mean * channel_mean
)


#
# Floating point calculations can sometimes
# produce extremely small negative values.
#
# Example:
#
# -1e-18
#
# These should be treated as zero.
#

channel_variance = np.maximum(
    channel_variance,
    0.0
)


# ============================================================
# 13. CALCULATE STD
# ============================================================

channel_std = np.sqrt(
    channel_variance
)


# ============================================================
# 14. CHECK STD
# ============================================================

print()
print("=" * 70)
print("11. CHECKING STANDARD DEVIATION")
print("=" * 70)


print()
print(
    "Minimum std:",
    float(np.min(channel_std))
)

print(
    "Maximum std:",
    float(np.max(channel_std))
)


zero_std_channels = np.where(
    channel_std <= 1e-12
)[0]


if len(zero_std_channels) > 0:

    print()
    print(
        "[WARNING] Channels with near-zero std:"
    )

    print(
        zero_std_channels
    )


    #
    # Prevent division by zero later.
    #

    channel_std[
        zero_std_channels
    ] = 1.0


else:

    print()
    print(
        "[OK] No zero-variance channels."
    )


# ============================================================
# 15. PRINT NORMALIZATION PARAMETERS
# ============================================================

print()
print("=" * 70)
print("12. TRAIN NORMALIZATION PARAMETERS")
print("=" * 70)


print()


for channel_index in range(
    number_of_channels
):

    print(
        f"Channel {channel_index + 1:02d}: "
        f"mean={channel_mean[channel_index]:.12e}, "
        f"std={channel_std[channel_index]:.12e}"
    )


# ============================================================
# 16. GLOBAL STATISTICS
# ============================================================

print()
print("=" * 70)
print("13. GLOBAL NORMALIZATION STATISTICS")
print("=" * 70)


print()
print(
    "Mean of channel means:",
    float(np.mean(channel_mean))
)

print(
    "Std of channel means:",
    float(np.std(channel_mean))
)

print(
    "Minimum channel mean:",
    float(np.min(channel_mean))
)

print(
    "Maximum channel mean:",
    float(np.max(channel_mean))
)

print()
print(
    "Mean of channel stds:",
    float(np.mean(channel_std))
)

print(
    "Minimum channel std:",
    float(np.min(channel_std))
)

print(
    "Maximum channel std:",
    float(np.max(channel_std))
)


# ============================================================
# 17. VALIDATE PARAMETERS
# ============================================================

print()
print("=" * 70)
print("14. VALIDATING NORMALIZATION PARAMETERS")
print("=" * 70)


if not np.all(
    np.isfinite(channel_mean)
):

    raise ValueError(
        "channel_mean contains NaN or Inf."
    )


if not np.all(
    np.isfinite(channel_std)
):

    raise ValueError(
        "channel_std contains NaN or Inf."
    )


if np.any(
    channel_std <= 0
):

    raise ValueError(
        "channel_std contains zero or negative values."
    )


print()
print(
    "[OK] Mean contains no NaN/Inf."
)

print(
    "[OK] Std contains no NaN/Inf."
)

print(
    "[OK] All standard deviations are positive."
)


# ============================================================
# 18. SAVE NORMALIZATION PARAMETERS
# ============================================================

print()
print("=" * 70)
print("15. SAVING NORMALIZATION PARAMETERS")
print("=" * 70)


#
# Save only tiny parameter arrays.
#
# No 2.3GB normalized dataset is created.
#


np.savez(
    OUTPUT_PATH,
    channel_mean=channel_mean.astype(
        np.float32
    ),
    channel_std=channel_std.astype(
        np.float32
    ),
    n_train_samples=np.asarray(
        len(train_indices),
        dtype=np.int64
    ),
    n_channels=np.asarray(
        number_of_channels,
        dtype=np.int64
    ),
    samples_per_window=np.asarray(
        samples_per_window,
        dtype=np.int64
    )
)


print()
print(
    "[OK] Normalization parameters saved:"
)

print(
    OUTPUT_PATH
)


# ============================================================
# 19. VERIFY SAVED FILE
# ============================================================

print()
print("=" * 70)
print("16. VERIFYING SAVED FILE")
print("=" * 70)


if not os.path.exists(
    OUTPUT_PATH
):

    raise FileNotFoundError(
        "Normalization file was not created."
    )


saved = np.load(
    OUTPUT_PATH
)


saved_mean = saved[
    "channel_mean"
]

saved_std = saved[
    "channel_std"
]

saved_train_samples = int(
    saved[
        "n_train_samples"
    ]
)

saved_channels = int(
    saved[
        "n_channels"
    ]
)

saved_samples_per_window = int(
    saved[
        "samples_per_window"
    ]
)


print()
print(
    "Saved mean shape:",
    saved_mean.shape
)

print(
    "Saved std shape:",
    saved_std.shape
)

print(
    "Saved train samples:",
    saved_train_samples
)

print(
    "Saved channels:",
    saved_channels
)

print(
    "Saved samples per window:",
    saved_samples_per_window
)


if not np.allclose(
    saved_mean,
    channel_mean.astype(
        np.float32
    )
):

    raise ValueError(
        "Saved mean does not match calculated mean."
    )


if not np.allclose(
    saved_std,
    channel_std.astype(
        np.float32
    )
):

    raise ValueError(
        "Saved std does not match calculated std."
    )


if saved_train_samples != len(
    train_indices
):

    raise ValueError(
        "Saved train sample count is incorrect."
    )


if saved_channels != number_of_channels:

    raise ValueError(
        "Saved channel count is incorrect."
    )


if saved_samples_per_window != samples_per_window:

    raise ValueError(
        "Saved window size is incorrect."
    )


print()
print(
    "[OK] Saved normalization parameters "
    "verified successfully."
)


# ============================================================
# 20. TEST NORMALIZATION ON A SMALL TRAIN CHUNK
# ============================================================

print()
print("=" * 70)
print("17. TESTING NORMALIZATION")
print("=" * 70)


#
# We do NOT create a normalized copy of the whole dataset.
#
# We only test a small number of training windows.
#


TEST_CHUNK_SIZE = min(
    32,
    len(train_indices)
)


test_indices = train_indices[
    :TEST_CHUNK_SIZE
]


test_chunk = np.asarray(
    X[
        test_indices
    ],
    dtype=np.float32
)


#
# Broadcast:
#
# channel_mean:
#     (23,)
#
# reshape:
#     (1, 23, 1)
#
# so that normalization is applied
# independently to every channel.
#


mean_for_broadcast = (
    saved_mean
    .reshape(
        1,
        number_of_channels,
        1
    )
)


std_for_broadcast = (
    saved_std
    .reshape(
        1,
        number_of_channels,
        1
    )
)


normalized_test_chunk = (
    test_chunk
    -
    mean_for_broadcast
) / std_for_broadcast


print()
print(
    "Test chunk shape:",
    normalized_test_chunk.shape
)

print(
    "Normalized dtype:",
    normalized_test_chunk.dtype
)


print()
print(
    "Normalized min:",
    float(
        np.min(
            normalized_test_chunk
        )
    )
)

print(
    "Normalized max:",
    float(
        np.max(
            normalized_test_chunk
        )
    )
)

print(
    "Normalized mean:",
    float(
        np.mean(
            normalized_test_chunk
        )
    )
)

print(
    "Normalized std:",
    float(
        np.std(
            normalized_test_chunk
        )
    )
)


#
# Check NaN/Inf
#


normalized_nan_count = int(
    np.isnan(
        normalized_test_chunk
    ).sum()
)


normalized_inf_count = int(
    np.isinf(
        normalized_test_chunk
    ).sum()
)


print()
print(
    "Normalized NaN values:",
    normalized_nan_count
)

print(
    "Normalized Inf values:",
    normalized_inf_count
)


if normalized_nan_count != 0:

    raise ValueError(
        "Normalization produced NaN values."
    )


if normalized_inf_count != 0:

    raise ValueError(
        "Normalization produced Inf values."
    )


print()
print(
    "[OK] Normalization test produced "
    "no NaN or Inf values."
)


# ============================================================
# 21. CLEANUP
# ============================================================

del test_chunk
del normalized_test_chunk
del mean_for_broadcast
del std_for_broadcast
del saved


# ============================================================
# 22. FINAL SUMMARY
# ============================================================

print()
print("=" * 70)
print("FINAL NORMALIZATION SUMMARY")
print("=" * 70)

print()
print(
    "Dataset:",
    "CHB-MIT"
)

print(
    "Total windows:",
    number_of_samples
)

print(
    "Training windows used for normalization:",
    len(train_indices)
)

print(
    "Validation windows:",
    len(val_indices)
)

print(
    "Test windows:",
    len(test_indices)
)

print()
print(
    "Channels:",
    number_of_channels
)

print(
    "Samples per window:",
    samples_per_window
)

print()
print(
    "Normalization:",
    "channel-wise z-score"
)

print(
    "Mean source:",
    "TRAIN ONLY"
)

print(
    "Std source:",
    "TRAIN ONLY"
)

print()
print(
    "Original X file modified:",
    "NO"
)

print(
    "Large normalized X file created:",
    "NO"
)

print()
print(
    "Normalization parameter file:"
)

print(
    OUTPUT_PATH
)

print()
print("=" * 70)
print("[SUCCESS] TRAIN-ONLY NORMALIZATION COMPLETED")
print("=" * 70)

print()
print(
    "No data leakage detected."
)

print(
    "Validation and Test data were NOT used "
    "to calculate normalization parameters."
)

print()
print("=" * 70)
print("DONE")
print("=" * 70)