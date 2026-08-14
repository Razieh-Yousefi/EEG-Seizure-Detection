import mne

# Path to the EEG file
file_path = "data/chb01/chb01_03.edf"

# Load the EDF file
raw = mne.io.read_raw_edf(file_path, preload=True)

# Apply band-pass filter
raw.filter(l_freq=0.5, h_freq=40)


# Seizure interval (from chb01-summary.txt)
# start = 2996
# end = 3036

# Crop the seizure segment
# raw.crop(tmin=start, tmax=end)

# Convert EEG signal to NumPy array
data = raw.get_data()

print("Shape of EEG data:")
print(data.shape)

# Extract the first 5-second window
window1 = data[:, 0:1280]

print("\nFirst window shape:")
print(window1.shape)

# Store all windows
windows = []

# Window size (5 seconds)
window_size = 5 * int(raw.info["sfreq"])

# Extract all non-overlapping windows
for start in range(0, data.shape[1] - window_size + 1, window_size):
    window = data[:, start:start + window_size]
    windows.append(window)

print("\nNumber of windows:")
print(len(windows))

# Display information about the first window
print("\nFirst window:")
print(windows[0])

print("\nShape of the first window:")
print(windows[0].shape)
# Check the shape of every window
print("\nShape of all windows:")

for i, window in enumerate(windows):
    print(f"Window {i+1}: {window.shape}")

# Create labels for all windows
labels = [1] * len(windows)

print("\nLabels:")
print(labels)

# Check data types
print("\nData types:")
print(type(windows))
print(type(labels))

import numpy as np

# Convert lists to NumPy arrays
X = np.array(windows)
y = np.array(labels)

print("\nShape of X:")
print(X.shape)

print("\nShape of y:")
print(y.shape)

print("\nData types after conversion:")
print(type(X))
print(type(y))
