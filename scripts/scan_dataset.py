import os
import re

# ============================================================
# Dataset root
# ============================================================

DATA_ROOT = "data"

# ============================================================
# Find patient folders
# ============================================================

patient_folders = sorted(
    [
        folder
        for folder in os.listdir(DATA_ROOT)
        if os.path.isdir(os.path.join(DATA_ROOT, folder))
        and folder.startswith("chb")
    ]
)

print("========================================")
print("CHB-MIT DATASET SCAN")
print("========================================")

print("\nPatients found:")
print(patient_folders)

# ============================================================
# Scan each patient
# ============================================================

total_patients = 0
total_edf_files = 0
total_seizure_files = 0
total_seizures = 0

print("\n========================================")

for patient in patient_folders:

    patient_path = os.path.join(DATA_ROOT, patient)

    # --------------------------------------------------------
    # Find summary file
    # --------------------------------------------------------

    summary_file = os.path.join(
        patient_path,
        f"{patient}-summary.txt"
    )

    if not os.path.exists(summary_file):
        print(f"\n{patient}: summary file NOT FOUND")
        continue

    # --------------------------------------------------------
    # Read summary
    # --------------------------------------------------------

    with open(summary_file, "r") as file:
        text = file.read()

    # --------------------------------------------------------
    # Find EDF files
    # --------------------------------------------------------

    edf_files = sorted(
        [
            file
            for file in os.listdir(patient_path)
            if file.lower().endswith(".edf")
        ]
    )

    # --------------------------------------------------------
    # Find seizure annotations
    # --------------------------------------------------------

    seizure_files = set()

    for match in re.finditer(
        r"File Name:\s*(\S+\.edf)",
        text
    ):

        filename = match.group(1)

        start = match.start()
        next_match = re.search(
            r"\nFile Name:",
            text[start + 1:]
        )

        if next_match:
            section = text[
                start:
                start + 1 + next_match.start()
            ]
        else:
            section = text[start:]

        if "Seizure Start Time" in section:
            seizure_files.add(filename)

    # --------------------------------------------------------
    # Count seizures
    # --------------------------------------------------------

    seizure_count = len(
        re.findall(
            r"Seizure Start Time:",
            text
        )
    )

    # --------------------------------------------------------
    # Update totals
    # --------------------------------------------------------

    total_patients += 1
    total_edf_files += len(edf_files)
    total_seizure_files += len(seizure_files)
    total_seizures += seizure_count

    # --------------------------------------------------------
    # Print patient information
    # --------------------------------------------------------

    print(f"\nPatient: {patient}")

    print(
        f"EDF files      : {len(edf_files)}"
    )

    print(
        f"Seizure files   : {len(seizure_files)}"
    )

    print(
        f"Seizure events  : {seizure_count}"
    )

    if seizure_files:
        print("Files containing seizure:")

        for filename in sorted(seizure_files):
            print(
                f"  - {filename}"
            )

# ============================================================
# Final summary
# ============================================================

print("\n========================================")
print("FINAL DATASET SUMMARY")
print("========================================")

print(
    f"Patients found       : {total_patients}"
)

print(
    f"Total EDF files      : {total_edf_files}"
)

print(
    f"Files with seizure   : {total_seizure_files}"
)

print(
    f"Total seizure events : {total_seizures}"
)

print("\n========================================")
print("SCAN COMPLETE")
print("========================================")