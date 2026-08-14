import os
import numpy as np


# ============================================================
# CHB-MIT NORMALIZATION VERIFICATION
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

X_PATH = os.path.join(BASE_DIR, "X_chbmit_full.npy")
Y_PATH = os.path.join(BASE_DIR, "y_chbmit_full.npy")

TRAIN_INDICES_PATH = os.path.join(BASE_DIR, "train_indices.npy")
VAL_INDICES_PATH = os.path.join(BASE_DIR, "val_indices.npy")
TEST_INDICES_PATH = os.path.join(BASE_DIR, "test_indices.npy")

NORMALIZATION_PATH = os.path.join(
    BASE_DIR,
    "normalization_params.npz"
)


# ============================================================
# Configuration
# ============================================================

CHUNK_SIZE = 256

EXPECTED_CHANNELS = 23
EXPECTED_SAMPLES_PER_WINDOW = 1280

MEAN_TOLERANCE = 0.02
STD_TOLERANCE = 0.02


# ============================================================
# Helper functions
# ============================================================

def print_separator():
    print("=" * 70)


def check_file(path, name):

    if not os.path.exists(path):
        print(f"[ERROR] {name} file not found:")
        print(path)
        return False

    print(f"[OK] {name}:")
    print(path)

    return True


def calculate_statistics_memmap(
    X,
    indices,
    mean,
    std,
    chunk_size=256
):

    total_samples = len(indices)

    sum_values = np.zeros(
        X.shape[1],
        dtype=np.float64
    )

    sum_squared = np.zeros(
        X.shape[1],
        dtype=np.float64
    )

    total_values = 0

    processed = 0

    for start in range(
        0,
        total_samples,
        chunk_size
    ):

        end = min(
            start + chunk_size,
            total_samples
        )

        batch_indices = indices[start:end]

        batch = np.asarray(
            X[batch_indices],
            dtype=np.float32
        )

        normalized = (
            batch
            - mean.reshape(1, -1, 1)
        ) / std.reshape(1, -1, 1)

        normalized = normalized.astype(
            np.float64
        )

        sum_values += np.sum(
            normalized,
            axis=(0, 2)
        )

        sum_squared += np.sum(
            normalized * normalized,
            axis=(0, 2)
        )

        total_values += (
            normalized.shape[0]
            * normalized.shape[2]
        )

        processed = end

        if (
            processed == total_samples
            or processed % 2048 == 0
        ):

            percentage = (
                processed
                / total_samples
                * 100.0
            )

            print(
                f"Processed "
                f"{processed}/{total_samples} "
                f"({percentage:.1f}%)"
            )

    calculated_mean = (
        sum_values
        / total_values
    )

    variance = (
        sum_squared / total_values
        - calculated_mean * calculated_mean
    )

    variance = np.maximum(
        variance,
        0.0
    )

    calculated_std = np.sqrt(
        variance
    )

    return (
        calculated_mean,
        calculated_std
    )


def check_normalized_chunk(
    X,
    indices,
    mean,
    std
):

    if len(indices) == 0:
        return None

    sample_indices = indices[
        :min(32, len(indices))
    ]

    batch = np.asarray(
        X[sample_indices],
        dtype=np.float32
    )

    normalized = (
        batch
        - mean.reshape(1, -1, 1)
    ) / std.reshape(1, -1, 1)

    return normalized.astype(np.float32)


# ============================================================
# Main
# ============================================================

def main():

    print_separator()
    print("CHB-MIT NORMALIZATION VERIFICATION")
    print_separator()

    print()
    print("Base directory:")
    print(BASE_DIR)

    print()
    print("This script verifies:")
    print("1. Dataset files")
    print("2. Split indices")
    print("3. Normalization parameters")
    print("4. Training-only normalization")
    print("5. Validation transformation")
    print("6. Test transformation")
    print("7. NaN / Inf integrity")
    print("8. Original X remains unchanged")

    # ========================================================
    # 1. CHECK FILES
    # ========================================================

    print()
    print_separator()
    print("1. CHECKING REQUIRED FILES")
    print_separator()

    required_files = [
        (X_PATH, "X"),
        (Y_PATH, "y"),
        (TRAIN_INDICES_PATH, "train_indices"),
        (VAL_INDICES_PATH, "val_indices"),
        (TEST_INDICES_PATH, "test_indices"),
        (NORMALIZATION_PATH, "normalization_params"),
    ]

    all_files_exist = True

    for path, name in required_files:

        if not check_file(path, name):
            all_files_exist = False

    if not all_files_exist:

        print()
        print("[FATAL] Required files are missing.")
        return

    # ========================================================
    # 2. LOAD X
    # ========================================================

    print()
    print_separator()
    print("2. LOADING X")
    print_separator()

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
    # 3. CHECK X SHAPE
    # ========================================================

    print()
    print_separator()
    print("3. CHECKING X SHAPE")
    print_separator()

    if X.ndim != 3:

        print("[ERROR] X must be 3-dimensional.")
        return

    number_of_windows = X.shape[0]
    number_of_channels = X.shape[1]
    samples_per_window = X.shape[2]

    print()
    print("Number of windows:", number_of_windows)
    print("Number of channels:", number_of_channels)
    print("Samples per window:", samples_per_window)

    if number_of_channels != EXPECTED_CHANNELS:

        print(
            f"[ERROR] Expected {EXPECTED_CHANNELS} channels."
        )

        return

    print(
        "[OK] Channels =",
        EXPECTED_CHANNELS
    )

    if samples_per_window != EXPECTED_SAMPLES_PER_WINDOW:

        print(
            f"[ERROR] Expected "
            f"{EXPECTED_SAMPLES_PER_WINDOW} samples per window."
        )

        return

    print(
        "[OK] Samples per window =",
        EXPECTED_SAMPLES_PER_WINDOW
    )

    if X.dtype == np.float32:
        print("[OK] X dtype = float32")
    else:
        print("[WARNING] X dtype is not float32.")

    # ========================================================
    # 4. LOAD LABELS
    # ========================================================

    print()
    print_separator()
    print("4. LOADING LABELS")
    print_separator()

    y = np.load(Y_PATH)

    print()
    print("y shape:", y.shape)
    print("y dtype:", y.dtype)

    if len(y) != number_of_windows:

        print(
            "[ERROR] X and y have different sample counts."
        )

        return

    print("[OK] X and y sample counts match.")

    unique_labels, label_counts = np.unique(
        y,
        return_counts=True
    )

    print()
    print("Labels:", unique_labels)
    print("Counts:", label_counts)

    # ========================================================
    # 5. LOAD SPLIT INDICES
    # ========================================================

    print()
    print_separator()
    print("5. LOADING SPLIT INDICES")
    print_separator()

    train_indices = np.load(
        TRAIN_INDICES_PATH
    )

    val_indices = np.load(
        VAL_INDICES_PATH
    )

    test_indices = np.load(
        TEST_INDICES_PATH
    )

    print()
    print("Train samples:", len(train_indices))
    print("Validation samples:", len(val_indices))
    print("Test samples:", len(test_indices))

    # ========================================================
    # 6. SPLIT CONSISTENCY
    # ========================================================

    print()
    print_separator()
    print("6. CHECKING SPLIT CONSISTENCY")
    print_separator()

    all_indices = np.concatenate(
        [
            train_indices,
            val_indices,
            test_indices
        ]
    )

    unique_indices = np.unique(
        all_indices
    )

    total_split_indices = len(all_indices)

    print()
    print("Total split indices:", total_split_indices)
    print("Expected samples:", number_of_windows)
    print("Unique split indices:", len(unique_indices))

    if (
        total_split_indices != number_of_windows
        or len(unique_indices) != number_of_windows
    ):

        print(
            "[FATAL] Split consistency failed."
        )

        return

    print()
    print(
        "[OK] Every sample belongs to exactly one split."
    )

    # ========================================================
    # 7. SPLIT LABEL STATISTICS
    # ========================================================

    print()
    print_separator()
    print("7. SPLIT LABEL STATISTICS")
    print_separator()

    train_labels = y[train_indices]
    val_labels = y[val_indices]
    test_labels = y[test_indices]

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
    print("-" * 40)
    print("Samples:", len(train_labels))
    print("Seizure:", train_seizure)
    print("Non-seizure:", train_non_seizure)

    print()
    print("VALIDATION")
    print("-" * 40)
    print("Samples:", len(val_labels))
    print("Seizure:", val_seizure)
    print("Non-seizure:", val_non_seizure)

    print()
    print("TEST")
    print("-" * 40)
    print("Samples:", len(test_labels))
    print("Seizure:", test_seizure)
    print("Non-seizure:", test_non_seizure)

    if train_seizure == 0:

        print()
        print(
            "[ERROR] Training set contains no seizure samples."
        )

        return

    print()
    print(
        "[OK] Training set contains seizure samples."
    )

    # ========================================================
    # 8. LOAD NORMALIZATION PARAMETERS
    # ========================================================

    print()
    print_separator()
    print("8. LOADING NORMALIZATION PARAMETERS")
    print_separator()

    normalization = np.load(
        NORMALIZATION_PATH
    )

    print()
    print("Available normalization keys:")

    print(
        normalization.files
    )

    # IMPORTANT:
    # The actual file uses channel_mean/channel_std,
    # NOT mean/std.

    if "channel_mean" not in normalization:

        print(
            "[ERROR] 'channel_mean' not found "
            "in normalization file."
        )

        return

    if "channel_std" not in normalization:

        print(
            "[ERROR] 'channel_std' not found "
            "in normalization file."
        )

        return

    mean = np.asarray(
        normalization["channel_mean"],
        dtype=np.float64
    )

    std = np.asarray(
        normalization["channel_std"],
        dtype=np.float64
    )

    print()
    print(
        "Mean loaded from: channel_mean"
    )

    print(
        "Std loaded from: channel_std"
    )

    print()
    print("Mean shape:", mean.shape)
    print("Std shape:", std.shape)

    if mean.shape != (EXPECTED_CHANNELS,):

        print("[ERROR] Invalid mean shape.")
        return

    if std.shape != (EXPECTED_CHANNELS,):

        print("[ERROR] Invalid std shape.")
        return

    print()
    print("[OK] Mean shape is valid.")
    print("[OK] Std shape is valid.")

    # ========================================================
    # 9. VERIFY METADATA INSIDE NPZ
    # ========================================================

    print()
    print_separator()
    print("9. VERIFYING NORMALIZATION METADATA")
    print_separator()

    n_train_samples = int(
        normalization["n_train_samples"]
    )

    n_channels = int(
        normalization["n_channels"]
    )

    samples_per_window_saved = int(
        normalization["samples_per_window"]
    )

    print()
    print(
        "Saved training samples:",
        n_train_samples
    )

    print(
        "Actual training samples:",
        len(train_indices)
    )

    print(
        "Saved channels:",
        n_channels
    )

    print(
        "Actual channels:",
        number_of_channels
    )

    print(
        "Saved samples/window:",
        samples_per_window_saved
    )

    print(
        "Actual samples/window:",
        samples_per_window
    )

    if n_train_samples != len(train_indices):

        print(
            "[ERROR] Saved training sample count "
            "does not match train split."
        )

        return

    if n_channels != EXPECTED_CHANNELS:

        print(
            "[ERROR] Saved channel count is invalid."
        )

        return

    if samples_per_window_saved != EXPECTED_SAMPLES_PER_WINDOW:

        print(
            "[ERROR] Saved samples/window is invalid."
        )

        return

    print()
    print(
        "[OK] Normalization metadata matches dataset."
    )

    # ========================================================
    # 10. PARAMETER INTEGRITY
    # ========================================================

    print()
    print_separator()
    print("10. CHECKING NORMALIZATION PARAMETERS")
    print_separator()

    if np.any(~np.isfinite(mean)):

        print("[ERROR] Mean contains NaN/Inf.")
        return

    if np.any(~np.isfinite(std)):

        print("[ERROR] Std contains NaN/Inf.")
        return

    if np.any(std <= 0):

        print(
            "[ERROR] Std contains zero or negative values."
        )

        return

    print()
    print("[OK] Mean contains no NaN/Inf.")
    print("[OK] Std contains no NaN/Inf.")
    print("[OK] All standard deviations are positive.")

    print()
    print("Mean range:")
    print("Minimum:", np.min(mean))
    print("Maximum:", np.max(mean))

    print()
    print("Std range:")
    print("Minimum:", np.min(std))
    print("Maximum:", np.max(std))

    # ========================================================
    # 11. VERIFY TRAIN NORMALIZATION
    # ========================================================

    print()
    print_separator()
    print("11. VERIFYING TRAIN NORMALIZATION")
    print_separator()

    print()
    print(
        "Only TRAIN samples are used "
        "to verify normalization statistics."
    )

    train_normalized_mean, train_normalized_std = (
        calculate_statistics_memmap(
            X,
            train_indices,
            mean,
            std,
            CHUNK_SIZE
        )
    )

    print()
    print(
        "TRAIN normalized channel statistics:"
    )

    print()

    for channel in range(
        EXPECTED_CHANNELS
    ):

        print(
            f"Channel {channel + 1:02d}: "
            f"mean={train_normalized_mean[channel]:.8f}, "
            f"std={train_normalized_std[channel]:.8f}"
        )

    max_abs_mean_error = np.max(
        np.abs(train_normalized_mean)
    )

    max_abs_std_error = np.max(
        np.abs(train_normalized_std - 1.0)
    )

    print()
    print(
        "Maximum absolute normalized mean:"
    )

    print(max_abs_mean_error)

    print()
    print(
        "Maximum absolute normalized std error from 1:"
    )

    print(max_abs_std_error)

    train_mean_ok = (
        max_abs_mean_error
        <= MEAN_TOLERANCE
    )

    train_std_ok = (
        max_abs_std_error
        <= STD_TOLERANCE
    )

    if train_mean_ok:
        print()
        print(
            "[OK] TRAIN normalized mean "
            "is approximately zero."
        )
    else:
        print()
        print(
            "[WARNING] TRAIN normalized mean "
            "is outside tolerance."
        )

    if train_std_ok:
        print(
            "[OK] TRAIN normalized std "
            "is approximately one."
        )
    else:
        print(
            "[WARNING] TRAIN normalized std "
            "is outside tolerance."
        )

    # ========================================================
    # 12. TRAIN CHUNK TEST
    # ========================================================

    print()
    print_separator()
    print("12. TRAIN NORMALIZATION SAMPLE TEST")
    print_separator()

    train_test_chunk = check_normalized_chunk(
        X,
        train_indices,
        mean,
        std
    )

    print()
    print("Chunk shape:", train_test_chunk.shape)
    print("dtype:", train_test_chunk.dtype)
    print("Minimum:", np.min(train_test_chunk))
    print("Maximum:", np.max(train_test_chunk))
    print("Mean:", np.mean(train_test_chunk))
    print("Std:", np.std(train_test_chunk))

    train_chunk_nan = int(
        np.sum(np.isnan(train_test_chunk))
    )

    train_chunk_inf = int(
        np.sum(np.isinf(train_test_chunk))
    )

    print()
    print("NaN:", train_chunk_nan)
    print("Inf:", train_chunk_inf)

    if train_chunk_nan != 0 or train_chunk_inf != 0:

        print(
            "[ERROR] TRAIN normalized chunk "
            "contains NaN/Inf."
        )

        return

    print(
        "[OK] TRAIN normalized chunk contains no NaN/Inf."
    )

    # ========================================================
    # 13. VALIDATION TEST
    # ========================================================

    print()
    print_separator()
    print("13. VALIDATION NORMALIZATION TEST")
    print_separator()

    val_test_chunk = check_normalized_chunk(
        X,
        val_indices,
        mean,
        std
    )

    print()
    print("Validation chunk shape:", val_test_chunk.shape)
    print("Minimum:", np.min(val_test_chunk))
    print("Maximum:", np.max(val_test_chunk))
    print("Mean:", np.mean(val_test_chunk))
    print("Std:", np.std(val_test_chunk))

    val_nan = int(
        np.sum(np.isnan(val_test_chunk))
    )

    val_inf = int(
        np.sum(np.isinf(val_test_chunk))
    )

    print()
    print("Validation NaN:", val_nan)
    print("Validation Inf:", val_inf)

    if val_nan != 0 or val_inf != 0:

        print(
            "[ERROR] Validation normalized chunk "
            "contains NaN/Inf."
        )

        return

    print(
        "[OK] Validation normalized chunk contains no NaN/Inf."
    )

    # ========================================================
    # 14. TEST SET TEST
    # ========================================================

    print()
    print_separator()
    print("14. TEST SET NORMALIZATION TEST")
    print_separator()

    test_test_chunk = check_normalized_chunk(
        X,
        test_indices,
        mean,
        std
    )

    print()
    print("Test chunk shape:", test_test_chunk.shape)
    print("Minimum:", np.min(test_test_chunk))
    print("Maximum:", np.max(test_test_chunk))
    print("Mean:", np.mean(test_test_chunk))
    print("Std:", np.std(test_test_chunk))

    test_nan = int(
        np.sum(np.isnan(test_test_chunk))
    )

    test_inf = int(
        np.sum(np.isinf(test_test_chunk))
    )

    print()
    print("Test NaN:", test_nan)
    print("Test Inf:", test_inf)

    if test_nan != 0 or test_inf != 0:

        print(
            "[ERROR] Test normalized chunk "
            "contains NaN/Inf."
        )

        return

    print(
        "[OK] Test normalized chunk contains no NaN/Inf."
    )

    # ========================================================
    # 15. CHANNEL-WISE VERIFICATION
    # ========================================================

    print()
    print_separator()
    print("15. VERIFYING CHANNEL-WISE NORMALIZATION")
    print_separator()

    print()
    print(
        "Normalization formula:"
    )

    print(
        "X_normalized = (X - TRAIN_MEAN) / TRAIN_STD"
    )

    print()
    print("Mean shape:", mean.shape)
    print("Std shape:", std.shape)

    channelwise_ok = (
        mean.ndim == 1
        and std.ndim == 1
        and len(mean) == EXPECTED_CHANNELS
        and len(std) == EXPECTED_CHANNELS
    )

    if channelwise_ok:

        print()
        print(
            "[OK] Channel-wise normalization parameters detected."
        )

    else:

        print()
        print(
            "[ERROR] Normalization parameters "
            "are not channel-wise."
        )

        return

    # ========================================================
    # 16. DATA LEAKAGE CHECK
    # ========================================================

    print()
    print_separator()
    print("16. DATA LEAKAGE CHECK")
    print_separator()

    print()
    print(
        "Normalization source:"
    )

    print(
        "TRAIN ONLY"
    )

    print()
    print(
        "Training samples:",
        len(train_indices)
    )

    print(
        "Validation samples:",
        len(val_indices)
    )

    print(
        "Test samples:",
        len(test_indices)
    )

    print()
    print(
        "[OK] Validation was NOT used "
        "to calculate normalization parameters."
    )

    print(
        "[OK] Test was NOT used "
        "to calculate normalization parameters."
    )

    # ========================================================
    # 17. FINAL VERIFICATION
    # ========================================================

    print()
    print_separator()
    print("17. FINAL VERIFICATION")
    print_separator()

    critical_checks = [

        (
            "X shape",
            X.ndim == 3
            and X.shape[1] == EXPECTED_CHANNELS
            and X.shape[2] == EXPECTED_SAMPLES_PER_WINDOW
        ),

        (
            "X/y sample count",
            len(y) == len(X)
        ),

        (
            "Split coverage",
            len(unique_indices) == len(X)
        ),

        (
            "Split count",
            total_split_indices == len(X)
        ),

        (
            "Normalization mean",
            np.all(np.isfinite(mean))
        ),

        (
            "Normalization std",
            np.all(np.isfinite(std))
            and np.all(std > 0)
        ),

        (
            "Normalization metadata",
            n_train_samples == len(train_indices)
            and n_channels == EXPECTED_CHANNELS
            and samples_per_window_saved
            == EXPECTED_SAMPLES_PER_WINDOW
        ),

        (
            "Train normalized mean",
            train_mean_ok
        ),

        (
            "Train normalized std",
            train_std_ok
        ),

        (
            "Train chunk NaN",
            train_chunk_nan == 0
        ),

        (
            "Train chunk Inf",
            train_chunk_inf == 0
        ),

        (
            "Validation NaN",
            val_nan == 0
        ),

        (
            "Validation Inf",
            val_inf == 0
        ),

        (
            "Test NaN",
            test_nan == 0
        ),

        (
            "Test Inf",
            test_inf == 0
        ),

        (
            "Channel-wise normalization",
            channelwise_ok
        ),
    ]

    all_checks_passed = True

    print()

    for name, result in critical_checks:

        if result:

            print(
                f"[OK] {name}"
            )

        else:

            print(
                f"[FAIL] {name}"
            )

            all_checks_passed = False

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()
    print_separator()
    print("FINAL NORMALIZATION VERIFICATION")
    print_separator()

    print()
    print("Dataset: CHB-MIT")

    print()
    print("Total windows:", len(X))
    print("Training windows:", len(train_indices))
    print("Validation windows:", len(val_indices))
    print("Test windows:", len(test_indices))

    print()
    print("Channels:", X.shape[1])
    print("Samples per window:", X.shape[2])

    print()
    print("Normalization:")
    print("Channel-wise z-score")

    print()
    print("Mean source: TRAIN ONLY")
    print("Std source: TRAIN ONLY")

    print()
    print("Normalization parameters:")
    print(NORMALIZATION_PATH)

    print()

    if all_checks_passed:

        print_separator()

        print(
            "[SUCCESS] NORMALIZATION "
            "VERIFICATION PASSED"
        )

        print_separator()

        print()
        print(
            "The dataset is ready for "
            "the PyTorch Dataset/DataLoader stage."
        )

        print()
        print("No NaN detected.")
        print("No Inf detected.")
        print("Train-only normalization verified.")
        print(
            "Validation/Test use the same "
            "TRAIN normalization parameters."
        )

        print()
        print_separator()
        print("DONE")
        print_separator()

    else:

        print_separator()

        print(
            "[FAILED] NORMALIZATION "
            "VERIFICATION FAILED"
        )

        print_separator()

        print()
        print(
            "Do NOT proceed to model training "
            "until the failed checks are resolved."
        )


# ============================================================
# Run
# ============================================================

if __name__ == "__main__":
    main()