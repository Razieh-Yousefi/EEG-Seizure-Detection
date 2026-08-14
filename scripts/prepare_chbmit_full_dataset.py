import os
import numpy as np
import mne

# ============================================================
# CHB-MIT FULL DATASET PREPARATION
# ============================================================

# مسیر دیتاست CHB-MIT
CHB_ROOT = r"C:\Users\rezay\Desktop\chb-mit-scalp-eeg-database-1.0.0\chb-mit-scalp-eeg-database-1.0.0"

# خروجی‌ها در ریشه پروژه
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

WINDOW_SECONDS = 5
LOW_FREQ = 0.5
HIGH_FREQ = 40.0

# کانال‌های مشترک مورد استفاده
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
    "T8-P8-0",
    "P8-O2",
    "FZ-CZ",
    "CZ-PZ",
    "P7-T7",
    "T7-FT9",
    "FT9-FT10",
    "FT10-T8",
    "T8-P8-1",
]


# ============================================================
# Load seizure intervals from official summary
# ============================================================

def load_seizure_times(summary_file):

    seizure_times = {}

    current_file = None

    with open(summary_file, "r", encoding="latin1") as file:

        for line in file:

            line = line.strip()

            if line.startswith("File Name:"):

                current_file = line.split(":", 1)[1].strip()
                seizure_times[current_file] = []

            elif line.startswith("Seizure Start Time:"):

                start = int(
                    line.split(":", 1)[1]
                    .replace("seconds", "")
                    .strip()
                )

                if current_file is not None:

                    if len(seizure_times[current_file]) == 0:
                        seizure_times[current_file].append([start, None])
                    else:
                        seizure_times[current_file][-1][0] = start

            elif line.startswith("Seizure End Time:"):

                end = int(
                    line.split(":", 1)[1]
                    .replace("seconds", "")
                    .strip()
                )

                if current_file is not None:

                    if (
                        len(seizure_times[current_file]) > 0
                        and seizure_times[current_file][-1][1] is None
                    ):
                        seizure_times[current_file][-1][1] = end

    # تبدیل به tuple
    for file_name in seizure_times:

        seizure_times[file_name] = [
            (start, end)
            for start, end in seizure_times[file_name]
            if end is not None
        ]

    return seizure_times


# ============================================================
# Check window overlap with seizure
# ============================================================

def check_seizure_label(window_start, window_end, intervals):

    for start, end in intervals:

        if window_start < end and window_end > start:
            return 1

    return 0


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print("CHB-MIT FULL DATASET PREPARATION")
    print("=" * 70)

    X = []
    y = []
    groups = []
    patients = []

    total_edfs = 0
    processed_edfs = 0
    skipped_edfs = 0

    patient_dirs = sorted(
        [
            d
            for d in os.listdir(CHB_ROOT)
            if os.path.isdir(os.path.join(CHB_ROOT, d))
            and d.lower().startswith("chb")
            and d[3:].isdigit()
        ]
    )

    print("Patients found:", len(patient_dirs))
    print(patient_dirs)

    # --------------------------------------------------------
    # Process patients
    # --------------------------------------------------------

    for patient in patient_dirs:

        patient_dir = os.path.join(CHB_ROOT, patient)

        summary_file = os.path.join(
            patient_dir,
            f"{patient}-summary.txt"
        )

        if not os.path.exists(summary_file):

            print(
                f"[WARNING] Summary not found for {patient}: "
                f"{summary_file}"
            )

            continue

        seizure_intervals = load_seizure_times(summary_file)

        edf_files = sorted(
            [
                f
                for f in os.listdir(patient_dir)
                if f.lower().endswith(".edf")
            ]
        )

        print()
        print("-" * 70)
        print(
            f"{patient}: {len(edf_files)} EDF files"
        )
        print("-" * 70)

        for edf_name in edf_files:

            total_edfs += 1

            file_path = os.path.join(
                patient_dir,
                edf_name
            )

            print(
                f"[{total_edfs}] {patient}/{edf_name}",
                end=" "
            )

            try:

                raw = mne.io.read_raw_edf(
                    file_path,
                    preload=True,
                    verbose=False
                )

                # ------------------------------------------------
                # Handle duplicate channel names
                # ------------------------------------------------

                raw.rename_channels(
                    lambda name: name.strip()
                )

                # MNE automatically numbers duplicates.
                # We use the first occurrence of T8-P8 if needed.
                available = raw.ch_names

                selected_channels = []

                for channel in TARGET_CHANNELS:

                    if channel in available:
                        selected_channels.append(channel)

                    elif channel == "T8-P8-0" and "T8-P8-0" in available:
                        selected_channels.append(channel)

                    elif channel == "T8-P8-1" and "T8-P8-1" in available:
                        selected_channels.append(channel)

                # ------------------------------------------------
                # Require all 23 channels
                # ------------------------------------------------

                if len(selected_channels) != len(TARGET_CHANNELS):

                    print(
                        f"[SKIP] channels "
                        f"{len(selected_channels)}/{len(TARGET_CHANNELS)}"
                    )

                    skipped_edfs += 1
                    continue

                raw.pick(selected_channels)

                # ------------------------------------------------
                # Filter
                # ------------------------------------------------

                raw.filter(
                    l_freq=LOW_FREQ,
                    h_freq=HIGH_FREQ,
                    verbose=False
                )

                eeg_data = raw.get_data()

                sfreq = float(raw.info["sfreq"])

                window_size = int(
                    WINDOW_SECONDS * sfreq
                )

                number_of_windows = (
                    eeg_data.shape[1] // window_size
                )

                intervals = seizure_intervals.get(
                    edf_name,
                    []
                )

                seizure_windows = 0

                # ------------------------------------------------
                # Create windows
                # ------------------------------------------------

                for i in range(number_of_windows):

                    start_sample = i * window_size
                    end_sample = start_sample + window_size

                    window = eeg_data[
                        :,
                        start_sample:end_sample
                    ]

                    # فقط windowهای دقیق 1280 نمونه‌ای
                    if window.shape != (
                        len(TARGET_CHANNELS),
                        window_size
                    ):
                        continue

                    window_start_time = (
                        start_sample / sfreq
                    )

                    window_end_time = (
                        end_sample / sfreq
                    )

                    label = check_seizure_label(
                        window_start_time,
                        window_end_time,
                        intervals
                    )

                    X.append(
                        window.astype(np.float32)
                    )

                    y.append(label)

                    groups.append(
                        f"{patient}/{edf_name}"
                    )

                    patients.append(
                        patient
                    )

                    if label == 1:
                        seizure_windows += 1

                processed_edfs += 1

                print(
                    f"[OK] windows={number_of_windows}, "
                    f"seizure_windows={seizure_windows}"
                )

            except Exception as e:

                print(
                    f"[ERROR] {type(e).__name__}: {e}"
                )

                skipped_edfs += 1

    # ============================================================
    # Convert to numpy
    # ============================================================

    print()
    print("=" * 70)
    print("CONVERTING DATASET")
    print("=" * 70)

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)
    groups = np.asarray(groups)
    patients = np.asarray(patients)

    print("X shape:", X.shape)
    print("X dtype:", X.dtype)

    print("y shape:", y.shape)
    print("y dtype:", y.dtype)

    print()
    print("Labels:")
    print(
        np.unique(
            y,
            return_counts=True
        )
    )

    print()
    print("Total EDFs:", total_edfs)
    print("Processed EDFs:", processed_edfs)
    print("Skipped EDFs:", skipped_edfs)

    # ============================================================
    # Save
    # ============================================================

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

    np.save(x_path, X)
    np.save(y_path, y)
    np.save(groups_path, groups)
    np.save(patients_path, patients)

    print()
    print("=" * 70)
    print("DATASET SAVED SUCCESSFULLY")
    print("=" * 70)

    print(x_path)
    print(y_path)
    print(groups_path)
    print(patients_path)

    print()
    print("FINAL SHAPE:", X.shape)
    print("SEIZURE WINDOWS:", int(np.sum(y)))
    print("NON-SEIZURE WINDOWS:", int(len(y) - np.sum(y)))


if __name__ == "__main__":
    main()