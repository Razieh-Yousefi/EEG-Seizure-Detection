import os
import numpy as np


# ============================================================
# CHB-MIT DATASET AUDIT
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

X_PATH = os.path.join(
    BASE_DIR,
    "X_chbmit_full.npy"
)

Y_PATH = os.path.join(
    BASE_DIR,
    "y_chbmit_full.npy"
)

GROUPS_PATH = os.path.join(
    BASE_DIR,
    "groups_chbmit_full.npy"
)

PATIENTS_PATH = os.path.join(
    BASE_DIR,
    "patients_chbmit_full.npy"
)


# ============================================================
# Helper
# ============================================================

def print_section(title):

    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


# ============================================================
# Main
# ============================================================

def main():

    print_section(
        "CHB-MIT DATASET AUDIT"
    )

    # --------------------------------------------------------
    # Check files
    # --------------------------------------------------------

    print_section(
        "1. CHECKING DATASET FILES"
    )

    files = {
        "X": X_PATH,
        "y": Y_PATH,
        "groups": GROUPS_PATH,
        "patients": PATIENTS_PATH
    }

    all_files_exist = True

    for name, path in files.items():

        exists = os.path.exists(path)

        if exists:

            size_mb = (
                os.path.getsize(path)
                / (1024 ** 2)
            )

            print(
                f"[OK] {name}"
            )

            print(
                f"     Path: {path}"
            )

            print(
                f"     Size: {size_mb:.2f} MB"
            )

        else:

            print(
                f"[MISSING] {name}"
            )

            print(
                f"     Expected path: {path}"
            )

            all_files_exist = False

    if not all_files_exist:

        print()
        print(
            "[FATAL] One or more dataset files are missing."
        )

        return

    # --------------------------------------------------------
    # Load arrays
    # --------------------------------------------------------

    print_section(
        "2. LOADING DATASET"
    )

    try:

        X = np.load(
            X_PATH,
            mmap_mode="r"
        )

        y = np.load(
            Y_PATH,
            mmap_mode="r"
        )

        groups = np.load(
            GROUPS_PATH,
            mmap_mode="r"
        )

        patients = np.load(
            PATIENTS_PATH,
            mmap_mode="r"
        )

        print(
            "[OK] All dataset files loaded successfully."
        )

    except Exception as e:

        print(
            f"[FATAL] Could not load dataset: "
            f"{type(e).__name__}: {e}"
        )

        return

    # --------------------------------------------------------
    # Basic shapes
    # --------------------------------------------------------

    print_section(
        "3. BASIC SHAPE CHECK"
    )

    print(
        "X shape:",
        X.shape
    )

    print(
        "X dtype:",
        X.dtype
    )

    print()

    print(
        "y shape:",
        y.shape
    )

    print(
        "y dtype:",
        y.dtype
    )

    print()

    print(
        "groups shape:",
        groups.shape
    )

    print(
        "groups dtype:",
        groups.dtype
    )

    print()

    print(
        "patients shape:",
        patients.shape
    )

    print(
        "patients dtype:",
        patients.dtype
    )

    # --------------------------------------------------------
    # Check sample counts
    # --------------------------------------------------------

    print_section(
        "4. SAMPLE COUNT CONSISTENCY"
    )

    n_x = len(X)
    n_y = len(y)
    n_groups = len(groups)
    n_patients = len(patients)

    print(
        "X samples:",
        n_x
    )

    print(
        "y samples:",
        n_y
    )

    print(
        "groups samples:",
        n_groups
    )

    print(
        "patients samples:",
        n_patients
    )

    if (
        n_x == n_y
        and n_x == n_groups
        and n_x == n_patients
    ):

        print()
        print(
            "[OK] All arrays have the same number of samples."
        )

    else:

        print()
        print(
            "[ERROR] Array sample counts do not match!"
        )

    # --------------------------------------------------------
    # Check EEG dimensions
    # --------------------------------------------------------

    print_section(
        "5. EEG DIMENSION CHECK"
    )

    expected_channels = 23
    expected_samples = 1280

    if X.ndim != 3:

        print(
            f"[ERROR] Expected X to have 3 dimensions, "
            f"got {X.ndim}"
        )

    else:

        print(
            "Number of windows:",
            X.shape[0]
        )

        print(
            "Number of channels:",
            X.shape[1]
        )

        print(
            "Samples per window:",
            X.shape[2]
        )

        channels_ok = (
            X.shape[1]
            == expected_channels
        )

        samples_ok = (
            X.shape[2]
            == expected_samples
        )

        if channels_ok:

            print(
                f"[OK] Channels = "
                f"{expected_channels}"
            )

        else:

            print(
                f"[ERROR] Expected "
                f"{expected_channels} channels, "
                f"got {X.shape[1]}"
            )

        if samples_ok:

            print(
                f"[OK] Samples per window = "
                f"{expected_samples}"
            )

        else:

            print(
                f"[ERROR] Expected "
                f"{expected_samples} samples, "
                f"got {X.shape[2]}"
            )

    # --------------------------------------------------------
    # Check labels
    # --------------------------------------------------------

    print_section(
        "6. LABEL CHECK"
    )

    unique_labels, label_counts = np.unique(
        y,
        return_counts=True
    )

    print(
        "Unique labels:",
        unique_labels
    )

    print(
        "Label counts:",
        label_counts
    )

    valid_labels = set(
        unique_labels.tolist()
    ).issubset(
        {0, 1}
    )

    if valid_labels:

        print()
        print(
            "[OK] Labels contain only 0 and 1."
        )

    else:

        print()
        print(
            "[ERROR] Invalid labels detected!"
        )

    seizure_count = int(
        np.sum(y == 1)
    )

    non_seizure_count = int(
        np.sum(y == 0)
    )

    total_count = len(y)

    print()

    print(
        "Total samples:",
        total_count
    )

    print(
        "Seizure samples:",
        seizure_count
    )

    print(
        "Non-seizure samples:",
        non_seizure_count
    )

    if total_count > 0:

        seizure_percent = (
            seizure_count
            / total_count
            * 100
        )

        non_seizure_percent = (
            non_seizure_count
            / total_count
            * 100
        )

        print()

        print(
            f"Seizure percentage: "
            f"{seizure_percent:.4f}%"
        )

        print(
            f"Non-seizure percentage: "
            f"{non_seizure_percent:.4f}%"
        )

    # --------------------------------------------------------
    # Patient check
    # --------------------------------------------------------

    print_section(
        "7. PATIENT CHECK"
    )

    unique_patients, patient_counts = np.unique(
        patients,
        return_counts=True
    )

    print(
        "Unique patients:",
        len(unique_patients)
    )

    print()

    for patient, count in zip(
        unique_patients,
        patient_counts
    ):

        print(
            f"{patient}: {count} windows"
        )

    # --------------------------------------------------------
    # Group / EDF check
    # --------------------------------------------------------

    print_section(
        "8. EDF GROUP CHECK"
    )

    unique_groups, group_counts = np.unique(
        groups,
        return_counts=True
    )

    print(
        "Unique EDF groups:",
        len(unique_groups)
    )

    print()

    print(
        "First 10 EDF groups:"
    )

    for group, count in zip(
        unique_groups[:10],
        group_counts[:10]
    ):

        print(
            f"{group}: {count} windows"
        )

    print()

    print(
        "Last 10 EDF groups:"
    )

    for group, count in zip(
        unique_groups[-10:],
        group_counts[-10:]
    ):

        print(
            f"{group}: {count} windows"
        )

    # --------------------------------------------------------
    # Check group/patient relationship
    # --------------------------------------------------------

    print_section(
        "9. GROUP AND PATIENT CONSISTENCY"
    )

    mismatch_count = 0

    for index in range(len(groups)):

        expected_patient = (
            str(groups[index])
            .split("/")[0]
        )

        actual_patient = str(
            patients[index]
        )

        if expected_patient != actual_patient:

            mismatch_count += 1

            if mismatch_count <= 10:

                print(
                    "[MISMATCH]"
                )

                print(
                    "Index:",
                    index
                )

                print(
                    "Group:",
                    groups[index]
                )

                print(
                    "Patient:",
                    patients[index]
                )

    if mismatch_count == 0:

        print(
            "[OK] Every group matches its patient."
        )

    else:

        print()

        print(
            f"[ERROR] Found "
            f"{mismatch_count} mismatches."
        )

    # --------------------------------------------------------
    # Numerical integrity check
    # --------------------------------------------------------

    print_section(
        "10. NUMERICAL INTEGRITY CHECK"
    )

    print(
        "Checking dataset in chunks..."
    )

    chunk_size = 100

    nan_count = 0
    inf_count = 0

    for start in range(
        0,
        len(X),
        chunk_size
    ):

        end = min(
            start + chunk_size,
            len(X)
        )

        chunk = X[start:end]

        nan_count += int(
            np.isnan(chunk).sum()
        )

        inf_count += int(
            np.isinf(chunk).sum()
        )

        if (
            (start // chunk_size + 1)
            % 20
            == 0
            or end == len(X)
        ):

            print(
                f"Checked "
                f"{end}/{len(X)} samples"
            )

    print()

    print(
        "NaN values:",
        nan_count
    )

    print(
        "Inf values:",
        inf_count
    )

    if nan_count == 0:

        print(
            "[OK] No NaN values found."
        )

    else:

        print(
            "[ERROR] NaN values detected!"
        )

    if inf_count == 0:

        print(
            "[OK] No Inf values found."
        )

    else:

        print(
            "[ERROR] Inf values detected!"
        )

    # --------------------------------------------------------
    # Random sample inspection
    # --------------------------------------------------------

    print_section(
        "11. RANDOM SAMPLE INSPECTION"
    )

    rng = np.random.default_rng(
        seed=42
    )

    sample_count = min(
        10,
        len(X)
    )

    random_indices = rng.choice(
        len(X),
        size=sample_count,
        replace=False
    )

    for index in sorted(
        random_indices
    ):

        sample = X[index]

        print()

        print(
            f"Sample index: {index}"
        )

        print(
            f"Label: {y[index]}"
        )

        print(
            f"Group: {groups[index]}"
        )

        print(
            f"Patient: {patients[index]}"
        )

        print(
            f"Shape: {sample.shape}"
        )

        print(
            f"Min: {float(np.min(sample)):.8f}"
        )

        print(
            f"Max: {float(np.max(sample)):.8f}"
        )

        print(
            f"Mean: {float(np.mean(sample)):.8f}"
        )

        print(
            f"Std: {float(np.std(sample)):.8f}"
        )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print_section(
        "FINAL AUDIT SUMMARY"
    )

    print(
        "Dataset samples:",
        len(X)
    )

    print(
        "EEG shape per sample:",
        X.shape[1:]
    )

    print(
        "Seizure samples:",
        seizure_count
    )

    print(
        "Non-seizure samples:",
        non_seizure_count
    )

    print(
        "Unique patients:",
        len(unique_patients)
    )

    print(
        "Unique EDF groups:",
        len(unique_groups)
    )

    print(
        "NaN values:",
        nan_count
    )

    print(
        "Inf values:",
        inf_count
    )

    print()

    if (
        n_x == n_y
        and n_x == n_groups
        and n_x == n_patients
        and valid_labels
        and mismatch_count == 0
        and nan_count == 0
        and inf_count == 0
        and X.ndim == 3
        and X.shape[1] == expected_channels
        and X.shape[2] == expected_samples
    ):

        print(
            "[SUCCESS] DATASET AUDIT PASSED"
        )

    else:

        print(
            "[WARNING] DATASET AUDIT FOUND ISSUES"
        )

    print()
    print("=" * 70)
    print("AUDIT COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()