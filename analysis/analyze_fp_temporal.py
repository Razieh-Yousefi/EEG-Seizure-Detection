# ================================================================
# analyze_fp_temporal.py
#
# Temporal analysis of false-positive predictions.
#
# PURPOSE:
# - Identify consecutive / clustered false-positive windows.
# - Measure FP run lengths.
# - Compare isolated FPs vs clustered FPs.
# - Analyze each patient separately.
# - Use original dataset indices to reconstruct temporal order.
#
# IMPORTANT:
# - Does NOT modify model.
# - Does NOT modify X/y.
# - Does NOT modify threshold.
# - Does NOT perform threshold optimization.
# ================================================================

import os
import json
import numpy as np


# ================================================================
# 1. CONFIGURATION
# ================================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

Y_FILE = os.path.join(
    BASE_DIR,
    "y_chbmit_full.npy"
)

PATIENTS_FILE = os.path.join(
    BASE_DIR,
    "patients_chbmit_full.npy"
)

TEST_INDICES_FILE = os.path.join(
    BASE_DIR,
    "test_indices.npy"
)

THRESHOLD_FILE = os.path.join(
    BASE_DIR,
    "validation_threshold_results.json"
)

PROBABILITIES_FILE = os.path.join(
    BASE_DIR,
    "test_window_probabilities.npz"
)

OUTPUT_FILE = os.path.join(
    BASE_DIR,
    "fp_temporal_analysis_results.json"
)

WINDOW_SECONDS = 5.0


# ================================================================
# 2. HEADER
# ================================================================

print()
print("=" * 70)
print("TEMPORAL FALSE-POSITIVE ANALYSIS")
print("=" * 70)

print()
print("Base directory:")
print(BASE_DIR)


# ================================================================
# 3. CHECK FILES
# ================================================================

print()
print("=" * 70)
print("1. CHECKING INPUT FILES")
print("=" * 70)

required_files = [
    Y_FILE,
    PATIENTS_FILE,
    TEST_INDICES_FILE,
    THRESHOLD_FILE,
    PROBABILITIES_FILE,
]

for path in required_files:

    if os.path.exists(path):
        print("[OK]", path)

    else:
        raise FileNotFoundError(
            f"Required file not found:\n{path}"
        )


# ================================================================
# 4. LOAD DATA
# ================================================================

print()
print("=" * 70)
print("2. LOADING DATA")
print("=" * 70)

y = np.load(Y_FILE)

patients = np.load(
    PATIENTS_FILE,
    allow_pickle=True
)

test_indices = np.load(
    TEST_INDICES_FILE
)

print()
print("Full labels shape:")
print(y.shape)

print()
print("Patients shape:")
print(patients.shape)

print()
print("Test indices shape:")
print(test_indices.shape)


# ================================================================
# 5. LOAD VALIDATION THRESHOLD
# ================================================================

print()
print("=" * 70)
print("3. LOADING VALIDATION THRESHOLD")
print("=" * 70)

with open(
    THRESHOLD_FILE,
    "r",
    encoding="utf-8"
) as f:

    threshold_data = json.load(f)


def find_threshold(obj):

    if isinstance(obj, dict):

        preferred_keys = [
            "selected_threshold",
            "validation_threshold",
            "best_threshold",
            "threshold",
        ]

        for key in preferred_keys:

            if key in obj:

                value = obj[key]

                if isinstance(
                    value,
                    (int, float)
                ):

                    return float(value)

        for value in obj.values():

            result = find_threshold(value)

            if result is not None:
                return result

    elif isinstance(obj, list):

        for item in obj:

            result = find_threshold(item)

            if result is not None:
                return result

    return None


VALIDATION_THRESHOLD = find_threshold(
    threshold_data
)

if VALIDATION_THRESHOLD is None:

    raise RuntimeError(
        "Could not find validation threshold."
    )

print()
print(
    f"Validation threshold: "
    f"{VALIDATION_THRESHOLD:.4f}"
)


# ================================================================
# 6. LOAD INDIVIDUAL TEST PROBABILITIES
# ================================================================

print()
print("=" * 70)
print("4. LOADING INDIVIDUAL TEST PROBABILITIES")
print("=" * 70)

probability_data = np.load(
    PROBABILITIES_FILE
)

required_probability_keys = [
    "test_indices",
    "patients",
    "labels",
    "probabilities",
]

for key in required_probability_keys:

    if key not in probability_data:

        raise RuntimeError(
            f"Missing '{key}' in "
            f"test_window_probabilities.npz"
        )


saved_indices = np.asarray(
    probability_data["test_indices"]
)

saved_patients = np.asarray(
    probability_data["patients"]
)

saved_labels = np.asarray(
    probability_data["labels"]
)

probabilities = np.asarray(
    probability_data["probabilities"],
    dtype=np.float32
)


# ================================================================
# 7. VERIFY ALIGNMENT
# ================================================================

print()
print("=" * 70)
print("5. VERIFYING PROBABILITY ALIGNMENT")
print("=" * 70)

if len(saved_indices) != len(test_indices):

    raise ValueError(
        "Saved probability count does not match "
        "test_indices count."
    )

if len(probabilities) != len(test_indices):

    raise ValueError(
        "Probability count does not match "
        "test sample count."
    )

if not np.array_equal(
    saved_indices,
    test_indices
):

    raise ValueError(
        "Saved test_indices do not exactly match "
        "current test_indices.npy."
    )

expected_labels = y[test_indices]

expected_patients = patients[test_indices]

if not np.array_equal(
    saved_labels,
    expected_labels
):

    raise ValueError(
        "Saved labels do not match "
        "y[test_indices]."
    )

if not np.array_equal(
    saved_patients,
    expected_patients
):

    raise ValueError(
        "Saved patients do not match "
        "patients[test_indices]."
    )

print()
print("[OK] Probability alignment verified.")

print(
    "Number of probabilities:",
    len(probabilities)
)

print(
    "Probability min:",
    f"{probabilities.min():.6f}"
)

print(
    "Probability max:",
    f"{probabilities.max():.6f}"
)


# ================================================================
# 8. CREATE TEST ARRAYS
# ================================================================

print()
print("=" * 70)
print("6. PREPARING TEST WINDOWS")
print("=" * 70)

y_test = expected_labels
patients_test = expected_patients


# ================================================================
# 9. CREATE PREDICTIONS
# ================================================================

print()
print("=" * 70)
print("7. CREATING TEST PREDICTIONS")
print("=" * 70)

predictions = (
    probabilities >= VALIDATION_THRESHOLD
).astype(np.int64)

fp_mask = (
    (predictions == 1)
    & (y_test == 0)
)

tp_mask = (
    (predictions == 1)
    & (y_test == 1)
)

fn_mask = (
    (predictions == 0)
    & (y_test == 1)
)

tn_mask = (
    (predictions == 0)
    & (y_test == 0)
)

print()
print("TP:", int(tp_mask.sum()))
print("FP:", int(fp_mask.sum()))
print("FN:", int(fn_mask.sum()))
print("TN:", int(tn_mask.sum()))


# ================================================================
# 10. RUN-LENGTH ANALYSIS
# ================================================================

def analyze_runs(
    mask,
    probabilities,
    original_indices
):

    runs = []

    start = None

    for i, value in enumerate(mask):

        if value and start is None:

            start = i

        elif not value and start is not None:

            end = i - 1

            run_probs = probabilities[
                start:end + 1
            ]

            run_indices = original_indices[
                start:end + 1
            ]

            runs.append({

                "start_test_position":
                    int(start),

                "end_test_position":
                    int(end),

                "start_original_index":
                    int(run_indices[0]),

                "end_original_index":
                    int(run_indices[-1]),

                "length_windows":
                    int(end - start + 1),

                "duration_seconds":
                    float(
                        (end - start + 1)
                        * WINDOW_SECONDS
                    ),

                "probability_min":
                    float(run_probs.min()),

                "probability_mean":
                    float(run_probs.mean()),

                "probability_max":
                    float(run_probs.max()),
            })

            start = None

    if start is not None:

        end = len(mask) - 1

        run_probs = probabilities[
            start:end + 1
        ]

        run_indices = original_indices[
            start:end + 1
        ]

        runs.append({

            "start_test_position":
                int(start),

            "end_test_position":
                int(end),

            "start_original_index":
                int(run_indices[0]),

            "end_original_index":
                int(run_indices[-1]),

            "length_windows":
                int(end - start + 1),

            "duration_seconds":
                float(
                    (end - start + 1)
                    * WINDOW_SECONDS
                ),

            "probability_min":
                float(run_probs.min()),

            "probability_mean":
                float(run_probs.mean()),

            "probability_max":
                float(run_probs.max()),
        })

    return runs


# ================================================================
# 11. GLOBAL TEMPORAL ANALYSIS
# ================================================================

print()
print("=" * 70)
print("8. GLOBAL TEMPORAL ANALYSIS")
print("=" * 70)

global_runs = []

unique_patients = np.unique(
    patients_test
)

for patient in unique_patients:

    patient_mask = (
        patients_test == patient
    )

    patient_indices = (
        test_indices[patient_mask]
    )

    patient_probabilities = (
        probabilities[patient_mask]
    )

    patient_labels = (
        y_test[patient_mask]
    )

    patient_predictions = (
        predictions[patient_mask]
    )

    # ------------------------------------------------------------
    # IMPORTANT:
    # Sort by original dataset index.
    # ------------------------------------------------------------

    order = np.argsort(
        patient_indices
    )

    patient_indices = (
        patient_indices[order]
    )

    patient_probabilities = (
        patient_probabilities[order]
    )

    patient_labels = (
        patient_labels[order]
    )

    patient_predictions = (
        patient_predictions[order]
    )

    patient_fp_mask = (
        (patient_predictions == 1)
        & (patient_labels == 0)
    )

    runs = analyze_runs(
        patient_fp_mask,
        patient_probabilities,
        patient_indices
    )

    for run in runs:

        item = dict(run)

        item["patient"] = str(patient)

        global_runs.append(item)


print()
print(
    "Total FP runs:",
    len(global_runs)
)


# ================================================================
# 12. RUN LENGTH STATISTICS
# ================================================================

print()
print("=" * 70)
print("9. FP RUN-LENGTH STATISTICS")
print("=" * 70)

if len(global_runs) > 0:

    lengths = np.asarray([
        r["length_windows"]
        for r in global_runs
    ])

    print()
    print(
        "Shortest FP run:",
        int(lengths.min()),
        "window(s)"
    )

    print(
        "Longest FP run:",
        int(lengths.max()),
        "window(s)"
    )

    print(
        "Mean FP run:",
        f"{lengths.mean():.2f}",
        "window(s)"
    )

    print()
    print("Run-length distribution:")

    unique_lengths, length_counts = np.unique(
        lengths,
        return_counts=True
    )

    for length, count in zip(
        unique_lengths,
        length_counts
    ):

        print(
            f"  {int(length):3d} window(s)"
            f" -> {int(count):3d} run(s)"
        )


# ================================================================
# 13. ISOLATED VS CLUSTERED FP
# ================================================================

print()
print("=" * 70)
print("10. ISOLATED VS CLUSTERED FALSE POSITIVES")
print("=" * 70)

isolated_fp_count = 0
clustered_fp_count = 0
clustered_runs = 0

for run in global_runs:

    length = run["length_windows"]

    if length == 1:

        isolated_fp_count += 1

    else:

        clustered_fp_count += length
        clustered_runs += 1


total_fp = int(
    fp_mask.sum()
)

print()
print("Total FP windows:", total_fp)

print(
    "Isolated FP windows:",
    isolated_fp_count
)

print(
    "Clustered FP windows:",
    clustered_fp_count
)

print(
    "Clustered FP runs:",
    clustered_runs
)

if total_fp > 0:

    print()
    print(
        "Isolated FP percentage:",
        f"{100.0 * isolated_fp_count / total_fp:.2f}%"
    )

    print(
        "Clustered FP percentage:",
        f"{100.0 * clustered_fp_count / total_fp:.2f}%"
    )


# ================================================================
# 14. PATIENT-LEVEL TEMPORAL ANALYSIS
# ================================================================

print()
print("=" * 70)
print("11. PATIENT-LEVEL TEMPORAL ANALYSIS")
print("=" * 70)

patient_temporal_results = {}


for patient in unique_patients:

    patient_mask = (
        patients_test == patient
    )

    patient_indices = (
        test_indices[patient_mask]
    )

    p = probabilities[
        patient_mask
    ]

    y_p = y_test[
        patient_mask
    ]

    pred_p = predictions[
        patient_mask
    ]

    # ------------------------------------------------------------
    # Sort chronologically using original dataset indices.
    # ------------------------------------------------------------

    order = np.argsort(
        patient_indices
    )

    patient_indices = (
        patient_indices[order]
    )

    p = p[order]

    y_p = y_p[order]

    pred_p = pred_p[order]

    fp_p = (
        (pred_p == 1)
        & (y_p == 0)
    )

    runs = analyze_runs(
        fp_p,
        p,
        patient_indices
    )

    lengths = np.asarray([
        r["length_windows"]
        for r in runs
    ])

    isolated = int(
        np.sum(lengths == 1)
    ) if len(lengths) > 0 else 0

    clustered = int(
        np.sum(
            lengths[lengths > 1]
        )
    ) if len(lengths) > 0 else 0

    patient_name = str(patient)

    print()
    print("-" * 70)
    print(patient_name)
    print("-" * 70)

    print(
        "FP windows:",
        int(fp_p.sum())
    )

    print(
        "FP runs:",
        len(runs)
    )

    print(
        "Isolated FP windows:",
        isolated
    )

    print(
        "Clustered FP windows:",
        clustered
    )

    if len(lengths) > 0:

        print(
            "Longest run:",
            int(lengths.max()),
            "window(s)"
        )

        print(
            "Mean run:",
            f"{lengths.mean():.2f}",
            "window(s)"
        )

    else:

        print(
            "No false-positive runs."
        )

    patient_temporal_results[
        patient_name
    ] = {

        "fp_windows":
            int(fp_p.sum()),

        "fp_runs":
            int(len(runs)),

        "isolated_fp_windows":
            isolated,

        "clustered_fp_windows":
            clustered,

        "longest_run_windows":
            (
                int(lengths.max())
                if len(lengths) > 0
                else 0
            ),

        "mean_run_windows":
            (
                float(lengths.mean())
                if len(lengths) > 0
                else 0.0
            ),

        "runs":
            runs,
    }


# ================================================================
# 15. TOP FP RUNS
# ================================================================

print()
print("=" * 70)
print("12. LONGEST FALSE-POSITIVE RUNS")
print("=" * 70)

all_runs = []

for patient_name, data in (
    patient_temporal_results.items()
):

    for run in data["runs"]:

        item = dict(run)

        item["patient"] = patient_name

        all_runs.append(item)


all_runs.sort(
    key=lambda x: (
        x["length_windows"],
        x["probability_mean"]
    ),
    reverse=True
)

top_runs = all_runs[:20]

print()

if len(top_runs) == 0:

    print("No false-positive runs found.")

else:

    for i, run in enumerate(
        top_runs,
        start=1
    ):

        print(
            f"{i:02d}. "
            f"{run['patient']}"
            f" | length="
            f"{run['length_windows']}"
            f" windows"
            f" | duration="
            f"{run['duration_seconds']:.1f}s"
            f" | mean_prob="
            f"{run['probability_mean']:.4f}"
            f" | max_prob="
            f"{run['probability_max']:.4f}"
            f" | original_index="
            f"{run['start_original_index']}"
            f"-"
            f"{run['end_original_index']}"
        )


# ================================================================
# 16. SAVE RESULTS
# ================================================================

print()
print("=" * 70)
print("13. SAVING TEMPORAL ANALYSIS")
print("=" * 70)

results = {

    "validation_threshold":
        float(VALIDATION_THRESHOLD),

    "window_duration_seconds":
        float(WINDOW_SECONDS),

    "test_samples":
        int(len(test_indices)),

    "global": {

        "tp":
            int(tp_mask.sum()),

        "fp":
            int(fp_mask.sum()),

        "fn":
            int(fn_mask.sum()),

        "tn":
            int(tn_mask.sum()),

        "fp_runs":
            int(len(global_runs)),

        "isolated_fp_windows":
            int(isolated_fp_count),

        "clustered_fp_windows":
            int(clustered_fp_count),

        "clustered_fp_runs":
            int(clustered_runs),

        "isolated_fp_percentage":
            (
                float(
                    100.0
                    * isolated_fp_count
                    / total_fp
                )
                if total_fp > 0
                else 0.0
            ),

        "clustered_fp_percentage":
            (
                float(
                    100.0
                    * clustered_fp_count
                    / total_fp
                )
                if total_fp > 0
                else 0.0
            ),

        "fp_runs":
            global_runs,
    },

    "patient_results":
        patient_temporal_results,

    "top_20_longest_runs":
        top_runs,

    "note":
        (
            "Temporal diagnostic analysis only. "
            "Probabilities were loaded from "
            "test_window_probabilities.npz. "
            "Temporal order was reconstructed "
            "using original dataset indices. "
            "No model, dataset, or validation "
            "threshold was modified."
        ),
}


with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        results,
        f,
        indent=2,
        ensure_ascii=False
    )


print()
print(
    "[OK] Temporal analysis saved:"
)

print(
    OUTPUT_FILE
)


# ================================================================
# 17. FINAL
# ================================================================

print()
print("=" * 70)
print("TEMPORAL FALSE-POSITIVE ANALYSIS COMPLETED")
print("=" * 70)

print()
print("No model modification.")
print("No dataset modification.")
print("No threshold modification.")

print()
print("=" * 70)