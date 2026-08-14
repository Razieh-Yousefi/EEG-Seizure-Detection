import os
import numpy as np
import mne


# Read seizure intervals from summary file

def load_seizure_times(summary_file):

    seizure_times = {}

    current_file = None

    with open(summary_file, "r") as file:

        for line in file:

            line = line.strip()

            if line.startswith("File Name:"):

                current_file = line.split(":")[1].strip()
                seizure_times[current_file] = []


            elif line.startswith("Seizure Start Time:"):

                start = int(
                    line.split(":")[1]
                    .replace("seconds", "")
                    .strip()
                )


            elif line.startswith("Seizure End Time:"):

                end = int(
                    line.split(":")[1]
                    .replace("seconds", "")
                    .strip()
                )

                seizure_times[current_file].append(
                    (start, end)
                )

    return seizure_times



# Check if window overlaps seizure interval

def check_seizure_label(window_start, window_end, intervals):

    for start, end in intervals:

        if window_start < end and window_end > start:

            return 1

    return 0



# Dataset folder

data_folder = "data/chb01"


summary_file = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "chb01-summary.txt"
)

# Load seizure information

seizure_intervals = load_seizure_times(summary_file)


print("Seizure information loaded:")
print(seizure_intervals["chb01_03.edf"])



# Window size

window_seconds = 5



X = []
y = []
groups = []



# Find EDF files

edf_files = [
    f for f in os.listdir(data_folder)
    if f.endswith(".edf")
]


print("\nEDF files found:")
print(edf_files)



# Process EDF files

for file in edf_files:


    print("\nProcessing:", file)


    file_path = os.path.join(
        data_folder,
        file
    )


    raw = mne.io.read_raw_edf(
        file_path,
        preload=True
    )


    raw.filter(
        l_freq=0.5,
        h_freq=40
    )


    eeg_data = raw.get_data()


    sfreq = int(raw.info["sfreq"])


    window_size = window_seconds * sfreq


    number_of_windows = eeg_data.shape[1] // window_size


    print("Number of windows:", number_of_windows)



    # Create windows

    for i in range(number_of_windows):


        start_sample = i * window_size

        end_sample = start_sample + window_size



        window = eeg_data[
            :,
            start_sample:end_sample
        ]



        window_start_time = start_sample / sfreq

        window_end_time = end_sample / sfreq



        label = check_seizure_label(
            window_start_time,
            window_end_time,
            seizure_intervals.get(file, [])
        )



        X.append(window)

        y.append(label)

        groups.append(file)



print("\nTotal windows:")

print(len(X))



# Convert to numpy arrays

X = np.array(X)

y = np.array(y)

groups = np.array(groups)



print("\nNumber of seizure windows:")

print(np.sum(y))


print("Number of non-seizure windows:")

print(len(y) - np.sum(y))



print("\nDataset shape:")

print(X.shape)

print(y.shape)



# Save dataset

np.save(
    "X_full.npy",
    X
)


np.save(
    "y_full.npy",
    y
)


np.save(
    "groups_full.npy",
    groups
)



print("\nDataset saved successfully!")