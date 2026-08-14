import os
import re
import numpy as np
import mne


# ============================================================
# SETTINGS
# ============================================================

# پیدا کردن پوشه‌ای که همین فایل Python داخل آن قرار دارد
SCRIPT_FOLDER = os.path.dirname(
    os.path.abspath(__file__)
)

# فایل‌های EDF و summary در همین پوشه هستند
DATA_FOLDER = SCRIPT_FOLDER

# طول هر پنجره EEG بر حسب ثانیه
WINDOW_SECONDS = 5

# تعداد کانال‌هایی که استفاده می‌کنیم
TARGET_CHANNELS = 23


# ============================================================
# Summary file
# ============================================================

summary_file = os.path.join(
    SCRIPT_FOLDER,
    "chb01-summary.txt"
)


# ============================================================
# Load seizure information
# ============================================================

def load_seizure_times(summary_file):

    seizure_intervals = {}

    with open(summary_file, "r") as file:
        lines = file.readlines()

    current_file = None
    current_start = None
    current_end = None

    for line in lines:

        line = line.strip()

        if line.startswith("File Name:"):

            current_file = line.split(
                ":",
                1
            )[1].strip()

            current_start = None
            current_end = None

        elif line.startswith("Seizure Start Time:"):

            current_start = int(
                re.findall(
                    r"\d+",
                    line
                )[0]
            )

        elif line.startswith("Seizure End Time:"):

            current_end = int(
                re.findall(
                    r"\d+",
                    line
                )[0]
            )

            if current_file is not None:

                if current_file not in seizure_intervals:

                    seizure_intervals[current_file] = []

                seizure_intervals[current_file].append(
                    (
                        current_start,
                        current_end
                    )
                )

    return seizure_intervals


# ============================================================
# Load seizure information
# ============================================================

seizure_intervals = load_seizure_times(
    summary_file
)


print("========================================")
print("SEIZURE INFORMATION")
print("========================================")

for filename, intervals in seizure_intervals.items():

    print(filename)

    for start, end in intervals:

        print(
            f"  {start} sec -> {end} sec"
        )


# ============================================================
# Find EDF files
# ============================================================

edf_files = sorted(
    [
        file
        for file in os.listdir(DATA_FOLDER)
        if file.lower().endswith(".edf")
    ]
)


print("\n========================================")
print("EDF FILES")
print("========================================")

print(
    f"Total EDF files: {len(edf_files)}"
)


# ============================================================
# Storage
# ============================================================

X_all = []
y_all = []


# ============================================================
# Process EDF files
# ============================================================

for edf_file in edf_files:

    print("\n========================================")
    print(
        f"Processing: {edf_file}"
    )
    print("========================================")

    file_path = os.path.join(
        DATA_FOLDER,
        edf_file
    )


    # --------------------------------------------------------
    # Read EDF
    # --------------------------------------------------------

    raw = mne.io.read_raw_edf(
        file_path,
        preload=True,
        verbose=True
    )


    # --------------------------------------------------------
    # Select EEG channels
    # --------------------------------------------------------

    available_channels = raw.ch_names

    print(
        f"Available channels: "
        f"{len(available_channels)}"
    )

    if len(available_channels) < TARGET_CHANNELS:

        print(
            f"Skipping {edf_file}: "
            f"only {len(available_channels)} "
            f"channels found."
        )

        continue


    selected_channels = available_channels[
        :TARGET_CHANNELS
    ]


    raw.pick(selected_channels)


    print(
        f"Selected channels: "
        f"{len(selected_channels)}"
    )


    # --------------------------------------------------------
    # Band-pass filter
    # --------------------------------------------------------

    print(
        "Applying band-pass filter "
        "(0.5 - 40 Hz)..."
    )

    raw.filter(
        l_freq=0.5,
        h_freq=40.0
    )


    # --------------------------------------------------------
    # Get EEG data
    # --------------------------------------------------------

    data = raw.get_data()

    sampling_rate = raw.info["sfreq"]


    print(
        f"Sampling rate: "
        f"{sampling_rate} Hz"
    )


    # --------------------------------------------------------
    # Calculate window size
    # --------------------------------------------------------

    samples_per_window = int(
        WINDOW_SECONDS *
        sampling_rate
    )

    total_samples = data.shape[1]

    number_of_windows = (
        total_samples //
        samples_per_window
    )


    print(
        f"Number of windows: "
        f"{number_of_windows}"
    )


    # --------------------------------------------------------
    # Get seizure intervals for this EDF file
    # --------------------------------------------------------

    intervals = seizure_intervals.get(
        edf_file,
        []
    )


    print(
        f"Seizure intervals: "
        f"{intervals}"
    )


    # --------------------------------------------------------
    # Counters
    # --------------------------------------------------------

    file_seizure_count = 0

    file_normal_count = 0


    # --------------------------------------------------------
    # Create 5-second windows
    # --------------------------------------------------------

    for window_index in range(
        number_of_windows
    ):

        start_sample = (
            window_index *
            samples_per_window
        )

        end_sample = (
            start_sample +
            samples_per_window
        )


        window = data[
            :,
            start_sample:end_sample
        ]


        # ----------------------------------------------------
        # Make sure window has correct size
        # ----------------------------------------------------

        if window.shape[1] != samples_per_window:

            continue


        # ----------------------------------------------------
        # Convert samples to seconds
        # ----------------------------------------------------

        window_start = (
            start_sample /
            sampling_rate
        )

        window_end = (
            end_sample /
            sampling_rate
        )


        # ----------------------------------------------------
        # Determine label
        # ----------------------------------------------------

        label = 0


        for seizure_start, seizure_end in intervals:

            overlap = (
                window_start < seizure_end
                and
                window_end > seizure_start
            )


            if overlap:

                label = 1

                break


        # ----------------------------------------------------
        # Store window
        # ----------------------------------------------------

        X_all.append(window)

        y_all.append(label)


        # ----------------------------------------------------
        # Update counters
        # ----------------------------------------------------

        if label == 1:

            file_seizure_count += 1

        else:

            file_normal_count += 1


    # --------------------------------------------------------
    # File statistics
    # --------------------------------------------------------

    print(
        f"Seizure windows in file: "
        f"{file_seizure_count}"
    )

    print(
        f"Normal windows in file: "
        f"{file_normal_count}"
    )


# ============================================================
# Convert to NumPy arrays
# ============================================================

print("\n========================================")
print("CREATING FINAL DATASET")
print("========================================")


X_all = np.array(
    X_all,
    dtype=np.float32
)


y_all = np.array(
    y_all,
    dtype=np.int8
)


print(
    "X shape:",
    X_all.shape
)


print(
    "y shape:",
    y_all.shape
)


# ============================================================
# Label distribution
# ============================================================

print("\nLabel distribution:")


unique_labels, label_counts = np.unique(
    y_all,
    return_counts=True
)


print(
    (unique_labels, label_counts)
)


# ============================================================
# Print final statistics
# ============================================================

print("\n========================================")
print("FINAL STATISTICS")
print("========================================")


seizure_count = np.sum(
    y_all == 1
)


normal_count = np.sum(
    y_all == 0
)


print(
    f"Total windows: {len(y_all)}"
)


print(
    f"Seizure windows: {seizure_count}"
)


print(
    f"Normal windows: {normal_count}"
)


# ============================================================
# Save dataset
# ============================================================

X_output = os.path.join(
    SCRIPT_FOLDER,
    "X_all_seizures.npy"
)


y_output = os.path.join(
    SCRIPT_FOLDER,
    "y_all_seizures.npy"
)


np.save(
    X_output,
    X_all
)


np.save(
    y_output,
    y_all
)


# ============================================================
# Finished
# ============================================================

print("\n========================================")
print("DATASET SAVED SUCCESSFULLY")
print("========================================")


print(
    "X file:"
)


print(
    X_output
)


print(
    "y file:"
)


print(
    y_output
)


print("\nDONE")