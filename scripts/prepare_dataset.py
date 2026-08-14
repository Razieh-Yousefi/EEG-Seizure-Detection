import mne
import numpy as np

# Path to the EEG file
file_path = "data/chb01/chb01_03.edf"

# Read EEG file
raw = mne.io.read_raw_edf(file_path, preload=True)

# Apply band-pass filter
raw.filter(l_freq=0.5, h_freq=40)

# Get EEG data
data = raw.get_data()

# Sampling frequency
sfreq = raw.info["sfreq"]

print("Sampling Frequency:")
print(sfreq)

print()

print("EEG Shape:")
print(data.shape)

# Window length (5 seconds)
window_size = int(sfreq * 5)

print()
print("Window Size (samples):")
print(window_size)

# Total number of windows
num_windows = data.shape[1] // window_size

print()
print("Number of Windows:")
print(num_windows)

# Create a list to store all windows
windows = []

# Extract all 5-second windows
for i in range(num_windows):

    start = i * window_size
    end = start + window_size

    window = data[:, start:end]

    windows.append(window) 

print()
print("Number of extracted windows:")
print(len(windows))

print()
print("Shape of first window:")
print(windows[0].shape)

print()
print("Shape of last window:")
print(windows[-1].shape)

# Seizure interval (seconds)
seizure_start = 2996
seizure_end = 3036

# Create labels
labels = []

for i in range(num_windows):

    window_start = i * 5
    window_end = window_start + 5

    if window_end > seizure_start and window_start < seizure_end:
        labels.append(1)
    else:
        labels.append(0)

print()
print("Number of labels:")
print(len(labels))

print()
print("Number of seizure windows:")
print(sum(labels))

print()
print("Number of non-seizure windows:")
print(len(labels) - sum(labels))

# Convert lists to NumPy arrays
X = np.array(windows)
y = np.array(labels)

print()
print("Final dataset shapes:")

print("X shape:")
print(X.shape)

print("y shape:")
print(y.shape)

# Save prepared dataset

np.save("X_eeg.npy", X)
np.save("y_labels.npy", y)

print("Dataset saved successfully!")

from sklearn.model_selection import train_test_split

# Split dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training data shape:")
print(X_train.shape)

print("Testing data shape:")
print(X_test.shape)

print("Training labels:")
print(np.unique(y_train, return_counts=True))

print("Testing labels:")
print(np.unique(y_test, return_counts=True))

# Save train and test datasets

np.save("X_train.npy", X_train)
np.save("X_test.npy", X_test)

np.save("y_train.npy", y_train)
np.save("y_test.npy", y_test)

print("Train and test datasets saved successfully!")

# Check label distribution

unique, counts = np.unique(y, return_counts=True)

print("Label distribution:")

for label, count in zip(unique, counts):
    print(label, ":", count)