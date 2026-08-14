import os
import numpy as np


# ============================================================
# CHB-MIT TRAIN-ONLY NORMALIZATION CALCULATION
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

X_PATH = os.path.join(
    BASE_DIR,
    "X_chbmit_full.npy"
)

TRAIN_INDICES_PATH = os.path.join(
    BASE_DIR,
    "train_indices.npy"
)

NORMALIZATION_PATH = os.path.join(
    BASE_DIR,
    "normalization_params.npz"
)


# ============================================================
# EXPECTED DATA SHAPE
# ============================================================

EXPECTED_CHANNELS = 23
EXPECTED_SAMPLES_PER_WINDOW = 1280


# ============================================================
# MEMORY SETTINGS
# ============================================================

# Number of windows processed in each batch.
# This keeps RAM usage low.
BATCH_SIZE = 64


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("CHB-MIT TRAIN-ONLY NORMALIZATION")
    print("=" * 70)

    print()
    print("Base directory:")
    print(BASE_DIR)

    print()
    print("X path:")
    print(X_PATH)

    print()
    print("Train indices:")
    print(TRAIN_INDICES_PATH)

    print()
    print("Normalization output:")
    print(NORMALIZATION_PATH)

    # ========================================================
    # 1. CHECK FILES
    # ========================================================

    print()
    print("=" * 70)
    print("1. CHECKING INPUT FILES")
    print("=" * 70)

    required_files = [
        X_PATH,
        TRAIN_INDICES_PATH
    ]

    for path in required_files:

        if not os.path.exists(path):

            raise FileNotFoundError(
                f"Required file not found:\n{path}"
            )

        print(
            "[OK]",
            path
        )

    # ========================================================
    # 2. LOAD X AS MEMORY MAP
    # ========================================================

    print()
    print("=" * 70)
    print("2. LOADING X AS MEMORY-MAPPED ARRAY")
    print("=" * 70)

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

    # ========================================================
    # 3. VALIDATE X
    # ========================================================

    print()
    print("=" * 70)
    print("3. VALIDATING X")
    print("=" * 70)

    if X.ndim != 3:

        raise ValueError(
            f"X must be 3D. Got {X.ndim}D."
        )

    if X.shape[1] != EXPECTED_CHANNELS:

        raise ValueError(
            f"Expected {EXPECTED_CHANNELS} channels, "
            f"got {X.shape[1]}."
        )

    if X.shape[2] != EXPECTED_SAMPLES_PER_WINDOW:

        raise ValueError(
            f"Expected {EXPECTED_SAMPLES_PER_WINDOW} "
            f"samples per window, "
            f"got {X.shape[2]}."
        )

    print(
        "[OK] X shape is valid."
    )

    # ========================================================
    # 4. LOAD TRAIN INDICES
    # ========================================================

    print()
    print("=" * 70)
    print("4. LOADING TRAIN INDICES")
    print("=" * 70)

    train_indices = np.load(
        TRAIN_INDICES_PATH
    )

    train_indices = np.asarray(
        train_indices,
        dtype=np.int64
    )

    print()
    print(
        "Train samples:",
        len(train_indices)
    )

    if len(train_indices) == 0:

        raise ValueError(
            "Train split contains zero samples."
        )

    # ========================================================
    # 5. VALIDATE TRAIN INDICES
    # ========================================================

    print()
    print("=" * 70)
    print("5. VALIDATING TRAIN INDICES")
    print("=" * 70)

    if np.any(train_indices < 0):

        raise ValueError(
            "Negative train index detected."
        )

    if np.any(
        train_indices >= len(X)
    ):

        raise ValueError(
            "Train index exceeds X size."
        )

    print(
        "[OK] Train indices are valid."
    )

    # ========================================================
    # 6. CHECK DUPLICATES
    # ========================================================

    print()
    print("=" * 70)
    print("6. CHECKING TRAIN INDICES")
    print("=" * 70)

    unique_indices = np.unique(
        train_indices
    )

    print(
        "Train indices:",
        len(train_indices)
    )

    print(
        "Unique train indices:",
        len(unique_indices)
    )

    if len(unique_indices) != len(train_indices):

        raise ValueError(
            "Duplicate train indices detected."
        )

    print(
        "[OK] Train indices are unique."
    )

    # ========================================================
    # 7. PREPARE ONLINE STATISTICS
    # ========================================================

    print()
    print("=" * 70)
    print("7. PREPARING STATISTICS")
    print("=" * 70)

    print()
    print(
        "Normalization will be calculated using "
        "TRAIN data only."
    )

    print(
        "Validation and test data will NOT be used."
    )

    # --------------------------------------------------------
    # We calculate channel-wise statistics.
    #
    # Each channel contains:
    #
    # number_of_train_windows * 1280
    #
    # values.
    #
    # We use float64 accumulators for numerical stability,
    # while X remains float32 on disk.
    # --------------------------------------------------------

    channel_sum = np.zeros(
        EXPECTED_CHANNELS,
        dtype=np.float64
    )

    channel_sum_squared = np.zeros(
        EXPECTED_CHANNELS,
        dtype=np.float64
    )

    total_values_per_channel = 0

    # ========================================================
    # 8. PROCESS TRAIN DATA IN SMALL BATCHES
    # ========================================================

    print()
    print("=" * 70)
    print("8. CALCULATING TRAIN-ONLY MEAN AND STD")
    print("=" * 70)

    total_train = len(
        train_indices
    )

    processed = 0

    while processed < total_train:

        batch_end = min(
            processed + BATCH_SIZE,
            total_train
        )

        batch_indices = train_indices[
            processed:batch_end
        ]

        # ----------------------------------------------------
        # Read only this batch from disk
        # ----------------------------------------------------

        batch = np.asarray(
            X[batch_indices],
            dtype=np.float64
        )

        # ----------------------------------------------------
        # Shape:
        #
        # (batch, 23, 1280)
        #
        # Sum over batch and time.
        # Result:
        #
        # (23,)
        # ----------------------------------------------------

        batch_sum = np.sum(
            batch,
            axis=(0, 2),
            dtype=np.float64
        )

        batch_sum_squared = np.sum(
            batch * batch,
            axis=(0, 2),
            dtype=np.float64
        )

        channel_sum += batch_sum

        channel_sum_squared += (
            batch_sum_squared
        )

        values_in_batch = (
            batch.shape[0]
            * batch.shape[2]
        )

        total_values_per_channel += (
            values_in_batch
        )

        processed = batch_end

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if (
            processed % 512 == 0
            or processed == total_train
        ):

            percentage = (
                processed
                / total_train
                * 100.0
            )

            print(
                f"Processed "
                f"{processed}/{total_train} "
                f"train windows "
                f"({percentage:.2f}%)"
            )

        # ----------------------------------------------------
        # Explicitly release batch memory
        # ----------------------------------------------------

        del batch

    # ========================================================
    # 9. CALCULATE MEAN
    # ========================================================

    print()
    print("=" * 70)
    print("9. CALCULATING MEAN")
    print("=" * 70)

    channel_mean = (
        channel_sum
        / total_values_per_channel
    )

    # ========================================================
    # 10. CALCULATE VARIANCE
    # ========================================================

    print()
    print("=" * 70)
    print("10. CALCULATING VARIANCE")
    print("=" * 70)

    channel_variance = (
        channel_sum_squared
        / total_values_per_channel
        - channel_mean * channel_mean
    )

    # Numerical protection against tiny negative
    # floating-point errors.

    channel_variance = np.maximum(
        channel_variance,
        0.0
    )

    # ========================================================
    # 11. CALCULATE STD
    # ========================================================

    print()
    print("=" * 70)
    print("11. CALCULATING STANDARD DEVIATION")
    print("=" * 70)

    channel_std = np.sqrt(
        channel_variance
    )

    # ========================================================
    # 12. VALIDATE STATISTICS
    # ========================================================

    print()
    print("=" * 70)
    print("12. VALIDATING NORMALIZATION PARAMETERS")
    print("=" * 70)

    if not np.all(
        np.isfinite(channel_mean)
    ):

        raise ValueError(
            "Mean contains NaN or Inf."
        )

    if not np.all(
        np.isfinite(channel_std)
    ):

        raise ValueError(
            "Std contains NaN or Inf."
        )

    if np.any(
        channel_std <= 0
    ):

        raise ValueError(
            "One or more channels have "
            "zero or negative standard deviation."
        )

    print(
        "[OK] Mean contains no NaN/Inf."
    )

    print(
        "[OK] Std contains no NaN/Inf."
    )

    print(
        "[OK] All standard deviations are positive."
    )

    # ========================================================
    # 13. CONVERT TO FLOAT32 FOR STORAGE
    # ========================================================

    channel_mean = np.asarray(
        channel_mean,
        dtype=np.float32
    )

    channel_std = np.asarray(
        channel_std,
        dtype=np.float32
    )

    # ========================================================
    # 14. DISPLAY PARAMETERS
    # ========================================================

    print()
    print("=" * 70)
    print("13. NORMALIZATION PARAMETERS")
    print("=" * 70)

    print()

    for channel_index in range(
        EXPECTED_CHANNELS
    ):

        print(
            f"Channel {channel_index + 1:02d}: "
            f"mean={channel_mean[channel_index]:.10f} "
            f"std={channel_std[channel_index]:.10f}"
        )

    # ========================================================
    # 15. SAVE PARAMETERS
    # ========================================================

    print()
    print("=" * 70)
    print("14. SAVING NORMALIZATION PARAMETERS")
    print("=" * 70)

    np.savez(
        NORMALIZATION_PATH,
        channel_mean=channel_mean,
        channel_std=channel_std
    )

    print()
    print(
        "[OK] Normalization parameters saved:"
    )

    print(
        NORMALIZATION_PATH
    )

    # ========================================================
    # 16. VERIFY SAVED FILE
    # ========================================================

    print()
    print("=" * 70)
    print("15. VERIFYING SAVED FILE")
    print("=" * 70)

    saved = np.load(
        NORMALIZATION_PATH
    )

    saved_mean = np.asarray(
        saved["channel_mean"]
    )

    saved_std = np.asarray(
        saved["channel_std"]
    )

    if saved_mean.shape != (
        EXPECTED_CHANNELS,
    ):

        raise RuntimeError(
            "Saved mean has incorrect shape."
        )

    if saved_std.shape != (
        EXPECTED_CHANNELS,
    ):

        raise RuntimeError(
            "Saved std has incorrect shape."
        )

    if not np.allclose(
        saved_mean,
        channel_mean
    ):

        raise RuntimeError(
            "Saved mean does not match "
            "calculated mean."
        )

    if not np.allclose(
        saved_std,
        channel_std
    ):

        raise RuntimeError(
            "Saved std does not match "
            "calculated std."
        )

    print(
        "[OK] Saved normalization parameters "
        "verified."
    )

    # ========================================================
    # 17. FINAL SUMMARY
    # ========================================================

    print()
    print("=" * 70)
    print("FINAL NORMALIZATION SUMMARY")
    print("=" * 70)

    print()
    print(
        "Total dataset windows:",
        len(X)
    )

    print(
        "Train windows used:",
        len(train_indices)
    )

    print(
        "Channels:",
        EXPECTED_CHANNELS
    )

    print(
        "Samples per window:",
        EXPECTED_SAMPLES_PER_WINDOW
    )

    print()
    print(
        "Normalization source:"
    )

    print(
        "TRAIN ONLY"
    )

    print()
    print(
        "Validation data used:",
        "NO"
    )

    print(
        "Test data used:",
        "NO"
    )

    print()
    print(
        "X original file modified:",
        "NO"
    )

    print(
        "Large normalized X copy created:",
        "NO"
    )

    print()
    print(
        "Normalization file:"
    )

    print(
        NORMALIZATION_PATH
    )

    print()
    print("=" * 70)
    print("[SUCCESS] NORMALIZATION PARAMETERS CREATED")
    print("=" * 70)

    print()
    print(
        "The PyTorch Dataset can now use "
        "normalization_params.npz."
    )

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()