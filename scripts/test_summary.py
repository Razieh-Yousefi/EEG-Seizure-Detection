import os


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
                start = int(line.split(":")[1].replace("seconds", "").strip())


            elif line.startswith("Seizure End Time:"):
                end = int(line.split(":")[1].replace("seconds", "").strip())

                seizure_times[current_file].append((start, end))

    return seizure_times


summary_file = "data/chb01/chb01-summary.txt"

seizure_intervals = load_seizure_times(summary_file)


print(seizure_intervals["chb01_03.edf"])