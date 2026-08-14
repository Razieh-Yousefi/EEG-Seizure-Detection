import os
import gc
import numpy as np
import mne


# ============================================================
# CHB-MIT DATASET PREPARATION
# MEMORY-SAFE + CONTROLLED NON-SEIZURE SAMPLING
# ============================================================

CHB_ROOT = (
    r"C:\Users\rezay\Desktop"
    r"\chb-mit-scalp-eeg-database-1.0.0"
    r"\chb-mit-scalp-eeg-database-1.0.0"
)

OUTPUT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

WINDOW_SECONDS = 5

LOW_FREQ = 0.5
HIGH_FREQ = 40.0

# ------------------------------------------------------------
# Maximum number of non-seizure windows to keep
# ------------------------------------------------------------

MAX_NON_SEIZURE_WINDOWS = 20000

# Reproducible random sampling
RANDOM_SEED = 42

rng = np.random.default_rng(
    RANDOM_SEED
)


# ============================================================
# Standard 23-channel order
# ============================================================

TARGET_CHANNELS = [
    "FP1-F7",
    "F7-T7",
    "T7-P7",
    "P7-O1",
    "FP1-F3",
    "F3-C3",
    "C3-P3",
    "P3-O1",
    "FP2-F4",
    "F4-C4",
    "C4-P4",
    "P4-O2",
    "FP2-F8",
    "F8-T8",
    "T8-P8",
    "P8-O2",
    "FZ-CZ",
    "CZ-PZ",
    "P7-T7",
    "T7-FT9",
    "FT9-FT10",
    "FT10-T8",
    "T8-P8",
]


# ============================================================
# Normalize channel names
# ============================================================

def normalize_channel_name(name):

    name = str(name).strip().upper()

    # MNE may create names such as:
    #
    # T8-P8-0
    # T8-P8-1
    #
    # Convert both to T8-P8.
    if name.startswith("T8-P8-"):

        return "T8-P8"

    return name


# ============================================================
# Load seizure intervals
# ============================================================

def load_seizure_times(summary_file):

    seizure_times = {}

    current_file = None
    current_start = None

    with open(
        summary_file,
        "r",
        encoding="latin1"
    ) as file:

        for line in file:

            line = line.strip()

            # ------------------------------------------------
            # File name
            # ------------------------------------------------

            if line.startswith("File Name:"):

                current_file = (
                    line.split(
                        ":",
                        1
                    )[1]
                    .strip()
                )

                seizure_times[
                    current_file
                ] = []

                current_start = None

            # ------------------------------------------------
            # Seizure start
            # ------------------------------------------------

            elif line.startswith(
                "Seizure Start Time:"
            ):

                start = int(
                    line.split(
                        ":",
                        1
                    )[1]
                    .replace(
                        "seconds",
                        ""
                    )
                    .strip()
                )

                current_start = start

            # ------------------------------------------------
            # Seizure end
            # ------------------------------------------------

            elif line.startswith(
                "Seizure End Time:"
            ):

                end = int(
                    line.split(
                        ":",
                        1
                    )[1]
                    .replace(
                        "seconds",
                        ""
                    )
                    .strip()
                )

                if (
                    current_file is not None
                    and current_start is not None
                ):

                    seizure_times[
                        current_file
                    ].append(
                        (
                            current_start,
                            end
                        )
                    )

                    current_start = None

    return seizure_times


# ============================================================
# Check whether a window overlaps seizure
# ============================================================

def check_seizure_label(
    window_start,
    window_end,
    intervals
):

    for start, end in intervals:

        if (
            window_start < end
            and window_end > start
        ):

            return 1

    return 0


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 75)
    print("CHB-MIT MEMORY-SAFE DATASET PREPARATION")
    print("=" * 75)

    print()
    print("CHB ROOT:")
    print(CHB_ROOT)

    print()
    print(
        "Target channels:",
        len(TARGET_CHANNELS)
    )

    print(
        "Window:",
        WINDOW_SECONDS,
        "seconds"
    )

    print(
        "Filter:",
        LOW_FREQ,
        "-",
        HIGH_FREQ,
        "Hz"
    )

    print(
        "Maximum non-seizure windows:",
        MAX_NON_SEIZURE_WINDOWS
    )

    print(
        "Random seed:",
        RANDOM_SEED
    )

    # ========================================================
    # Check CHB root
    # ========================================================

    if not os.path.exists(CHB_ROOT):

        print()
        print("[FATAL] CHB_ROOT does not exist:")
        print(CHB_ROOT)

        return

    # ========================================================
    # Find patients
    # ========================================================

    patient_dirs = sorted(
        [
            d
            for d in os.listdir(CHB_ROOT)

            if os.path.isdir(
                os.path.join(
                    CHB_ROOT,
                    d
                )
            )

            and d.lower().startswith(
                "chb"
            )

            and d[3:].isdigit()
        ]
    )

    print()
    print(
        "Patients found:",
        len(patient_dirs)
    )

    print(
        patient_dirs
    )

    # ========================================================
    # Dataset containers
    #
    # IMPORTANT:
    #
    # We do NOT keep all 457k windows.
    #
    # We keep:
    #
    #   1. ALL seizure windows
    #   2. At most 20,000 non-seizure windows
    #
    # This keeps the dataset manageable.
    # ========================================================

    seizure_X = []
    seizure_y = []
    seizure_groups = []
    seizure_patients = []

    negative_X = []
    negative_y = []
    negative_groups = []
    negative_patients = []

    # ========================================================
    # Counters
    # ========================================================

    total_edfs = 0
    processed_edfs = 0
    skipped_edfs = 0

    total_windows_seen = 0
    total_seizure_windows_seen = 0
    total_non_seizure_windows_seen = 0

    missing_channel_counter = {}

    # ========================================================
    # Process patients
    # ========================================================

    for patient in patient_dirs:

        patient_dir = os.path.join(
            CHB_ROOT,
            patient
        )

        summary_file = os.path.join(
            patient_dir,
            f"{patient}-summary.txt"
        )

        if not os.path.exists(
            summary_file
        ):

            print()
            print(
                "[WARNING] Summary not found:",
                patient
            )

            continue

        # ----------------------------------------------------
        # Load seizure intervals
        # ----------------------------------------------------

        seizure_intervals = (
            load_seizure_times(
                summary_file
            )
        )

        # ----------------------------------------------------
        # Find EDF files
        # ----------------------------------------------------

        edf_files = sorted(
            [
                f
                for f in os.listdir(
                    patient_dir
                )

                if f.lower().endswith(
                    ".edf"
                )
            ]
        )

        print()
        print("-" * 75)
        print(
            f"{patient}: "
            f"{len(edf_files)} EDF files"
        )
        print("-" * 75)

        # ====================================================
        # Process EDFs
        # ====================================================

        for edf_name in edf_files:

            total_edfs += 1

            file_path = os.path.join(
                patient_dir,
                edf_name
            )

            print(
                f"[{total_edfs}] "
                f"{patient}/{edf_name}",
                end=" "
            )

            try:

                # ====================================================
                # Read EDF
                # ====================================================

                raw = mne.io.read_raw_edf(
                    file_path,
                    preload=True,
                    verbose=False
                )

                # ====================================================
                # Build normalized channel map
                # ====================================================

                channel_map = {}

                for (
                    source_index,
                    original_name
                ) in enumerate(
                    raw.ch_names
                ):

                    normalized_name = (
                        normalize_channel_name(
                            original_name
                        )
                    )

                    # Keep first occurrence
                    if (
                        normalized_name
                        not in channel_map
                    ):

                        channel_map[
                            normalized_name
                        ] = source_index

                # ====================================================
                # Missing channel report
                # ====================================================

                missing_channels = []

                for target in TARGET_CHANNELS:

                    if (
                        target
                        not in channel_map
                    ):

                        missing_channels.append(
                            target
                        )

                        missing_channel_counter[
                            target
                        ] = (
                            missing_channel_counter.get(
                                target,
                                0
                            )
                            + 1
                        )

                # ====================================================
                # Read data
                # ====================================================

                eeg_data = raw.get_data()

                sfreq = float(
                    raw.info["sfreq"]
                )

                window_size = int(
                    round(
                        WINDOW_SECONDS
                        * sfreq
                    )
                )

                number_of_windows = (
                    eeg_data.shape[1]
                    // window_size
                )

                if (
                    number_of_windows
                    <= 0
                ):

                    print(
                        "[SKIP] "
                        "File shorter than 5 seconds"
                    )

                    skipped_edfs += 1

                    del raw
                    del eeg_data

                    gc.collect()

                    continue

                # ====================================================
                # Construct standardized 23-channel data
                #
                # Use float64 for MNE filtering.
                # ====================================================

                standardized = np.zeros(
                    (
                        len(
                            TARGET_CHANNELS
                        ),
                        eeg_data.shape[1]
                    ),
                    dtype=np.float64
                )

                for (
                    target_index,
                    target
                ) in enumerate(
                    TARGET_CHANNELS
                ):

                    if (
                        target
                        in channel_map
                    ):

                        source_index = (
                            channel_map[
                                target
                            ]
                        )

                        standardized[
                            target_index
                        ] = (
                            eeg_data[
                                source_index
                            ]
                        )

                # ====================================================
                # Free original raw data before filtering
                # ====================================================

                del raw
                del eeg_data

                gc.collect()

                # ====================================================
                # Filter
                # ====================================================

                standardized = (
                    mne.filter.filter_data(
                        standardized,
                        sfreq=sfreq,
                        l_freq=LOW_FREQ,
                        h_freq=HIGH_FREQ,
                        verbose=False
                    )
                )

                # Convert to float32 after filtering
                standardized = (
                    standardized.astype(
                        np.float32,
                        copy=False
                    )
                )

                # ====================================================
                # Seizure intervals for this EDF
                # ====================================================

                intervals = (
                    seizure_intervals.get(
                        edf_name,
                        []
                    )
                )

                file_seizure_windows = 0
                file_non_seizure_windows = 0

                # ====================================================
                # Create windows
                # ====================================================

                for i in range(
                    number_of_windows
                ):

                    start_sample = (
                        i * window_size
                    )

                    end_sample = (
                        start_sample
                        + window_size
                    )

                    window = standardized[
                        :,
                        start_sample:end_sample
                    ]

                    if window.shape != (
                        len(
                            TARGET_CHANNELS
                        ),
                        window_size
                    ):

                        continue

                    window_start_time = (
                        start_sample
                        / sfreq
                    )

                    window_end_time = (
                        end_sample
                        / sfreq
                    )

                    label = (
                        check_seizure_label(
                            window_start_time,
                            window_end_time,
                            intervals
                        )
                    )

                    total_windows_seen += 1

                    # ====================================================
                    # SEIZURE
                    #
                    # Keep ALL seizure windows.
                    # ====================================================

                    if label == 1:

                        seizure_X.append(
                            window.copy()
                        )

                        seizure_y.append(
                            1
                        )

                        seizure_groups.append(
                            f"{patient}/{edf_name}"
                        )

                        seizure_patients.append(
                            patient
                        )

                        file_seizure_windows += 1

                        total_seizure_windows_seen += 1

                    # ====================================================
                    # NON-SEIZURE
                    #
                    # Reservoir sampling:
                    #
                    # Keep a random representative sample
                    # of at most MAX_NON_SEIZURE_WINDOWS.
                    # ====================================================

                    else:

                        total_non_seizure_windows_seen += 1

                        file_non_seizure_windows += 1

                        current_count = (
                            len(
                                negative_X
                            )
                        )

                        if (
                            current_count
                            < MAX_NON_SEIZURE_WINDOWS
                        ):

                            negative_X.append(
                                window.copy()
                            )

                            negative_y.append(
                                0
                            )

                            negative_groups.append(
                                f"{patient}/{edf_name}"
                            )

                            negative_patients.append(
                                patient
                            )

                        else:

                            # Random replacement
                            replacement_index = (
                                rng.integers(
                                    0,
                                    total_non_seizure_windows_seen
                                )
                            )

                            if (
                                replacement_index
                                < MAX_NON_SEIZURE_WINDOWS
                            ):

                                negative_X[
                                    replacement_index
                                ] = (
                                    window.copy()
                                )

                                negative_y[
                                    replacement_index
                                ] = 0

                                negative_groups[
                                    replacement_index
                                ] = (
                                    f"{patient}/{edf_name}"
                                )

                                negative_patients[
                                    replacement_index
                                ] = patient

                processed_edfs += 1

                # ====================================================
                # Report EDF
                # ====================================================

                print(
                    "[OK] "
                    f"windows={number_of_windows} "
                    f"seizure={file_seizure_windows} "
                    f"non-seizure={file_non_seizure_windows} "
                    f"kept-negatives={len(negative_X)}"
                )

                # ====================================================
                # Free file-level memory
                # ====================================================

                del standardized

                gc.collect()

            except Exception as e:

                print(
                    "[ERROR]",
                    type(e).__name__,
                    ":",
                    e
                )

                skipped_edfs += 1

                gc.collect()

    # ========================================================
    # BUILD FINAL DATASET
    # ========================================================

    print()
    print("=" * 75)
    print("BUILDING CONTROLLED DATASET")
    print("=" * 75)

    print()
    print(
        "Total windows seen:",
        total_windows_seen
    )

    print(
        "Total seizure windows seen:",
        total_seizure_windows_seen
    )

    print(
        "Total non-seizure windows seen:",
        total_non_seizure_windows_seen
    )

    print()
    print(
        "Seizure windows kept:",
        len(seizure_X)
    )

    print(
        "Non-seizure windows kept:",
        len(negative_X)
    )

    # ========================================================
    # Check
    # ========================================================

    if len(seizure_X) == 0:

        print()
        print(
            "[FATAL] "
            "No seizure windows were found."
        )

        return

    if len(negative_X) == 0:

        print()
        print(
            "[FATAL] "
            "No non-seizure windows were found."
        )

        return

    # ========================================================
    # Convert seizure data
    # ========================================================

    print()
    print(
        "Converting seizure windows..."
    )

    seizure_X = np.asarray(
        seizure_X,
        dtype=np.float32
    )

    seizure_y = np.asarray(
        seizure_y,
        dtype=np.int64
    )

    seizure_groups = np.asarray(
        seizure_groups
    )

    seizure_patients = np.asarray(
        seizure_patients
    )

    # ========================================================
    # Convert negative data
    # ========================================================

    print(
        "Converting non-seizure windows..."
    )

    negative_X = np.asarray(
        negative_X,
        dtype=np.float32
    )

    negative_y = np.asarray(
        negative_y,
        dtype=np.int64
    )

    negative_groups = np.asarray(
        negative_groups
    )

    negative_patients = np.asarray(
        negative_patients
    )

    # ========================================================
    # Combine
    # ========================================================

    print()
    print(
        "Combining seizure and non-seizure data..."
    )

    X = np.concatenate(
        [
            seizure_X,
            negative_X
        ],
        axis=0
    )

    y = np.concatenate(
        [
            seizure_y,
            negative_y
        ],
        axis=0
    )

    groups = np.concatenate(
        [
            seizure_groups,
            negative_groups
        ],
        axis=0
    )

    patients = np.concatenate(
        [
            seizure_patients,
            negative_patients
        ],
        axis=0
    )

    # ========================================================
    # Shuffle final dataset
    # ========================================================

    print(
        "Shuffling final dataset..."
    )

    shuffle_indices = (
        rng.permutation(
            len(X)
        )
    )

    X = X[
        shuffle_indices
    ]

    y = y[
        shuffle_indices
    ]

    groups = groups[
        shuffle_indices
    ]

    patients = patients[
        shuffle_indices
    ]

    # ========================================================
    # Free intermediate arrays
    # ========================================================

    del seizure_X
    del seizure_y
    del seizure_groups
    del seizure_patients

    del negative_X
    del negative_y
    del negative_groups
    del negative_patients

    del shuffle_indices

    gc.collect()

    # ========================================================
    # Final validation
    # ========================================================

    print()
    print("=" * 75)
    print("FINAL DATASET")
    print("=" * 75)

    print()
    print(
        "X shape:",
        X.shape
    )

    print(
        "X dtype:",
        X.dtype
    )

    print(
        "y shape:",
        y.shape
    )

    print(
        "y dtype:",
        y.dtype
    )

    print(
        "groups shape:",
        groups.shape
    )

    print(
        "patients shape:",
        patients.shape
    )

    print()
    print(
        "Labels:",
        np.unique(
            y,
            return_counts=True
        )
    )

    print()
    print(
        "Total EDFs:",
        total_edfs
    )

    print(
        "Processed EDFs:",
        processed_edfs
    )

    print(
        "Skipped EDFs:",
        skipped_edfs
    )

    print()
    print(
        "Total final windows:",
        len(X)
    )

    print(
        "Final seizure windows:",
        int(
            np.sum(y)
        )
    )

    print(
        "Final non-seizure windows:",
        int(
            len(y)
            - np.sum(y)
        )
    )

    # ========================================================
    # Verify shape
    # ========================================================

    expected_shape = (
        len(TARGET_CHANNELS),
        1280
    )

    if X.ndim != 3:

        print()
        print(
            "[FATAL] "
            "Unexpected X dimensions:",
            X.shape
        )

        return

    if X.shape[1:] != expected_shape:

        print()
        print(
            "[FATAL] "
            "Unexpected window shape:",
            X.shape[1:]
        )

        return

    # ========================================================
    # Missing channel report
    # ========================================================

    print()
    print("=" * 75)
    print("MISSING CHANNEL REPORT")
    print("=" * 75)

    if (
        len(
            missing_channel_counter
        )
        == 0
    ):

        print(
            "No missing target channels."
        )

    else:

        for (
            channel,
            count
        ) in sorted(
            missing_channel_counter.items()
        ):

            print(
                f"{channel}: "
                f"{count} EDF files"
            )

    # ========================================================
    # Output paths
    # ========================================================

    x_path = os.path.join(
        OUTPUT_DIR,
        "X_chbmit_full.npy"
    )

    y_path = os.path.join(
        OUTPUT_DIR,
        "y_chbmit_full.npy"
    )

    groups_path = os.path.join(
        OUTPUT_DIR,
        "groups_chbmit_full.npy"
    )

    patients_path = os.path.join(
        OUTPUT_DIR,
        "patients_chbmit_full.npy"
    )

    # ========================================================
    # Remove old outputs if they exist
    # ========================================================

    print()
    print("=" * 75)
    print("SAVING DATASET")
    print("=" * 75)

    old_files = [
        x_path,
        y_path,
        groups_path,
        patients_path
    ]

    for old_file in old_files:

        if os.path.exists(
            old_file
        ):

            print(
                "Removing old:",
                old_file
            )

            os.remove(
                old_file
            )

    # ========================================================
    # Save
    # ========================================================

    print()
    print(
        "Saving X..."
    )

    np.save(
        x_path,
        X
    )

    print(
        "Saving y..."
    )

    np.save(
        y_path,
        y
    )

    print(
        "Saving groups..."
    )

    np.save(
        groups_path,
        groups
    )

    print(
        "Saving patients..."
    )

    np.save(
        patients_path,
        patients
    )

    # ========================================================
    # Final file sizes
    # ========================================================

    print()
    print("=" * 75)
    print("DATASET SAVED SUCCESSFULLY")
    print("=" * 75)

    print()
    print(
        x_path
    )

    print(
        y_path
    )

    print(
        groups_path
    )

    print(
        patients_path
    )

    print()
    print(
        "FINAL SHAPE:",
        X.shape
    )

    print(
        "SEIZURE WINDOWS:",
        int(
            np.sum(y)
        )
    )

    print(
        "NON-SEIZURE WINDOWS:",
        int(
            len(y)
            - np.sum(y)
        )
    )

    print()
    print("=" * 75)
    print("DONE")
    print("=" * 75)

    # ========================================================
    # Cleanup
    # ========================================================

    del X
    del y
    del groups
    del patients

    gc.collect()


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()