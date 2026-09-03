# ================================================================
# evaluate_test_patient_seizure_events.py
#
# STRICT FINAL TEST PATIENT / SEIZURE-EVENT LEVEL EVALUATION
#
# IMPORTANT:
# - NO optimization on test set
# - NO threshold fitting on test set
# - NO feature fitting on test set
# - Artifact threshold/rule comes from VALIDATION only
# - Uses STRICT validation-frozen window-level test predictions
# ================================================================

import os
import json
import numpy as np


# ================================================================
# 1. PROJECT PATHS
# ================================================================

PROJECT_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATA_DIR = os.path.join(
    PROJECT_DIR,
    "data"
)

RESULTS_DIR = os.path.join(
    PROJECT_DIR,
    "results"
)


# ----------------------------------------------------------------
# IMPORTANT:
# We now use the STRICT validation-frozen result.
# ----------------------------------------------------------------

FINAL_TEST_FILE = os.path.join(
    RESULTS_DIR,
    "final_test_artifact_rejection_scores_strict.npz"
)

FINAL_TEST_JSON = os.path.join(
    RESULTS_DIR,
    "final_test_artifact_rejection_evaluation_strict.json"
)

TEST_INDICES_FILE = os.path.join(
    DATA_DIR,
    "test_indices.npy"
)

LABELS_FILE = os.path.join(
    DATA_DIR,
    "y_chbmit_full.npy"
)

PATIENTS_FILE = os.path.join(
    DATA_DIR,
    "patients_chbmit_full.npy"
)


# ----------------------------------------------------------------
# Strict outputs
# ----------------------------------------------------------------

OUTPUT_JSON = os.path.join(
    RESULTS_DIR,
    "final_test_patient_seizure_event_evaluation_strict.json"
)

OUTPUT_NPZ = os.path.join(
    RESULTS_DIR,
    "final_test_patient_seizure_event_results_strict.npz"
)


# ================================================================
# 2. SETTINGS
# ================================================================

BASELINE_THRESHOLD = 0.95

WINDOW_SECONDS = 5.0

# Same event-construction rule used in the original analysis.
# This is NOT optimized on test.
MAX_GAP_WINDOWS = 1


# ================================================================
# 3. HELPER FUNCTIONS
# ================================================================

def find_array(npz_data, possible_names, required=True):
    """
    Find an array in an NPZ file using a list of possible key names.
    """

    for name in possible_names:

        if name in npz_data.files:

            print(
                f"[OK] Using array '{name}'"
            )

            return np.asarray(
                npz_data[name]
            )

    if required:

        raise KeyError(
            "Could not find any of these arrays:\n"
            + "\n".join(
                f"  - {x}"
                for x in possible_names
            )
            + "\n\nAvailable arrays:\n"
            + "\n".join(
                f"  - {x}"
                for x in npz_data.files
            )
        )

    return None


def safe_divide(a, b):

    if b == 0:

        return 0.0

    return float(a / b)


def classification_counts(
    y_true,
    prediction
):

    y_true = np.asarray(
        y_true,
        dtype=np.int64
    )

    prediction = np.asarray(
        prediction,
        dtype=bool
    )

    tp = int(
        np.sum(
            (y_true == 1)
            &
            prediction
        )
    )

    fp = int(
        np.sum(
            (y_true == 0)
            &
            prediction
        )
    )

    tn = int(
        np.sum(
            (y_true == 0)
            &
            (~prediction)
        )
    )

    fn = int(
        np.sum(
            (y_true == 1)
            &
            (~prediction)
        )
    )

    return tp, fp, tn, fn


def metric_summary(
    tp,
    fp,
    tn,
    fn
):

    recall = safe_divide(
        tp,
        tp + fn
    )

    precision = safe_divide(
        tp,
        tp + fp
    )

    specificity = safe_divide(
        tn,
        tn + fp
    )

    accuracy = safe_divide(
        tp + tn,
        tp + tn + fp + fn
    )

    f1 = (
        2.0
        * precision
        * recall
        / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {

        "tp": int(tp),

        "fp": int(fp),

        "tn": int(tn),

        "fn": int(fn),

        "recall": float(
            recall
        ),

        "sensitivity": float(
            recall
        ),

        "specificity": float(
            specificity
        ),

        "precision": float(
            precision
        ),

        "accuracy": float(
            accuracy
        ),

        "f1": float(
            f1
        ),
    }


# ================================================================
# HEADER
# ================================================================

print()

print(
    "=" * 76
)

print(
    "STRICT FINAL TEST PATIENT / SEIZURE-EVENT LEVEL EVALUATION"
)

print(
    "=" * 76
)

print()

print(
    "Project:"
)

print(
    PROJECT_DIR
)

print()

print(
    "IMPORTANT:"
)

print(
    "- Uses STRICT validation-frozen test predictions."
)

print(
    "- NO optimization is performed."
)

print(
    "- NO test threshold search."
)

print(
    "- NO test feature fitting."
)

print(
    "- Artifact rule comes from validation only."
)


# ================================================================
# 4. CHECK FILES
# ================================================================

print()

print(
    "=" * 76
)

print(
    "1. CHECKING REQUIRED FILES"
)

print(
    "=" * 76
)


required_files = [

    FINAL_TEST_FILE,

    FINAL_TEST_JSON,

    TEST_INDICES_FILE,

    LABELS_FILE,

    PATIENTS_FILE,

]


for path in required_files:

    if os.path.exists(
        path
    ):

        print(
            "[OK]",
            path
        )

    else:

        print(
            "[MISSING]",
            path
        )

        raise FileNotFoundError(
            path
        )


# ================================================================
# 5. LOAD STRICT FINAL TEST RESULTS
# ================================================================

print()

print(
    "=" * 76
)

print(
    "2. LOADING STRICT FINAL TEST RESULTS"
)

print(
    "=" * 76
)


data = np.load(
    FINAL_TEST_FILE,
    allow_pickle=True
)


print()

print(
    "Available arrays:"
)


for key in data.files:

    try:

        shape = data[key].shape

    except Exception:

        shape = "unknown"

    print(
        f"{key:40s}: shape={shape}"
    )


# ================================================================
# 6. LOAD STRICT JSON
# ================================================================

print()

print(
    "=" * 76
)

print(
    "3. LOADING STRICT TEST METADATA"
)

print(
    "=" * 76
)


with open(
    FINAL_TEST_JSON,
    "r",
    encoding="utf-8"
) as f:

    strict_json = json.load(
        f
    )


print()

print(
    "[OK] Strict test evaluation JSON loaded."
)


# ================================================================
# 7. LOAD TEST INDICES
# ================================================================

print()

print(
    "=" * 76
)

print(
    "4. LOADING TEST INDICES"
)

print(
    "=" * 76
)


# Prefer the indices stored inside strict NPZ.
# If they are not stored there, use data/test_indices.npy.

test_indices_from_npz = find_array(

    data,

    [
        "test_indices",
        "indices",
        "dataset_indices",
    ],

    required=False
)


test_indices_from_file = np.asarray(
    np.load(
        TEST_INDICES_FILE
    ),
    dtype=np.int64
)


if test_indices_from_npz is not None:

    test_indices = np.asarray(
        test_indices_from_npz,
        dtype=np.int64
    )

    if not np.array_equal(
        test_indices,
        test_indices_from_file
    ):

        raise RuntimeError(
            "Strict NPZ test_indices do not match "
            "data/test_indices.npy."
        )

    print(
        "[OK] Strict NPZ indices match test_indices.npy."
    )

else:

    test_indices = test_indices_from_file

    print(
        "[OK] Using data/test_indices.npy."
    )


print(
    "Test indices:",
    len(test_indices)
)


# ================================================================
# 8. LOAD LABELS
# ================================================================

print()

print(
    "=" * 76
)

print(
    "5. LOADING TEST LABELS"
)

print(
    "=" * 76
)


labels = find_array(

    data,

    [
        "labels",
        "test_labels",
        "y_test",
        "y_true",
    ]
)


labels = np.asarray(
    labels,
    dtype=np.int64
).reshape(-1)


print(
    "Labels:",
    len(labels)
)


# ================================================================
# 9. LOAD TEST PROBABILITIES
# ================================================================

print()

print(
    "=" * 76
)

print(
    "6. LOADING TEST PROBABILITIES"
)

print(
    "=" * 76
)


probabilities = find_array(

    data,

    [
        "probabilities",
        "test_probabilities",
        "y_prob",
        "probs",
        "seizure_probabilities",
    ]
)


probabilities = np.asarray(
    probabilities,
    dtype=np.float64
).reshape(-1)


print(
    "Probabilities:",
    len(probabilities)
)


# ================================================================
# 10. LOAD STRICT ARTIFACT SCORE
# ================================================================

print()

print(
    "=" * 76
)

print(
    "7. LOADING STRICT ARTIFACT SCORE"
)

print(
    "=" * 76
)


artifact_score = find_array(

    data,

    [
        "artifact_score",
        "artifact_scores",
        "test_artifact_score",
        "strict_artifact_score",
    ]
)


artifact_score = np.asarray(
    artifact_score,
    dtype=np.float64
).reshape(-1)


print(
    "Artifact scores:",
    len(artifact_score)
)


# ================================================================
# 11. LOAD BASELINE PREDICTIONS
# ================================================================

print()

print(
    "=" * 76
)

print(
    "8. LOADING BASELINE PREDICTIONS"
)

print(
    "=" * 76
)


baseline_array = find_array(

    data,

    [
        "baseline_positive",
        "baseline_predictions",
        "baseline_pred",
        "predictions_baseline",
    ],

    required=False
)


if baseline_array is None:

    print()

    print(
        "[INFO] Baseline prediction array not found."
    )

    print(
        "Reconstructing baseline using probability >= 0.95."
    )

    baseline_positive = (
        probabilities
        >=
        BASELINE_THRESHOLD
    )

else:

    baseline_positive = np.asarray(
        baseline_array,
        dtype=bool
    ).reshape(-1)


# ================================================================
# 12. LOAD STRICT FINAL PREDICTIONS
# ================================================================

print()

print(
    "=" * 76
)

print(
    "9. LOADING STRICT FINAL PREDICTIONS"
)

print(
    "=" * 76
)


final_array = find_array(

    data,

    [
        "final_positive",
        "final_predictions",
        "strict_predictions",
        "artifact_rejection_predictions",
        "predictions_after_artifact_rejection",
    ],

    required=False
)


artifact_rejected_array = find_array(

    data,

    [
        "artifact_rejected",
        "rejection_mask",
        "artifact_rejection_mask",
        "rejected",
    ],

    required=False
)


if final_array is not None:

    final_positive = np.asarray(
        final_array,
        dtype=bool
    ).reshape(-1)

else:

    if artifact_rejected_array is None:

        raise RuntimeError(
            "Neither strict final predictions nor "
            "artifact rejection mask was found in the strict NPZ."
        )

    artifact_rejected_temp = np.asarray(
        artifact_rejected_array,
        dtype=bool
    ).reshape(-1)

    final_positive = (
        baseline_positive
        &
        (~artifact_rejected_temp)
    )


if artifact_rejected_array is not None:

    artifact_rejected = np.asarray(
        artifact_rejected_array,
        dtype=bool
    ).reshape(-1)

else:

    # Only baseline-positive windows can be removed.
    artifact_rejected = (
        baseline_positive
        &
        (~final_positive)
    )


# ================================================================
# 13. VERIFY FINAL-PREDICTION CONSISTENCY
# ================================================================

print()

print(
    "=" * 76
)

print(
    "10. VERIFYING STRICT PREDICTION CONSISTENCY"
)

print(
    "=" * 76
)


expected_final_positive = (
    baseline_positive
    &
    (~artifact_rejected)
)


if not np.array_equal(
    final_positive,
    expected_final_positive
):

    mismatch_count = int(
        np.sum(
            final_positive
            !=
            expected_final_positive
        )
    )

    raise RuntimeError(
        "Strict final predictions are inconsistent with "
        "baseline_positive & ~artifact_rejected. "
        f"Mismatched windows: {mismatch_count}"
    )


print()

print(
    "[OK] strict final predictions are internally consistent."
)


# ================================================================
# 14. ALIGNMENT CHECK
# ================================================================

print()

print(
    "=" * 76
)

print(
    "11. VERIFYING ARRAY ALIGNMENT"
)

print(
    "=" * 76
)


n = len(
    test_indices
)


arrays_to_check = {

    "labels":
        labels,

    "probabilities":
        probabilities,

    "artifact_score":
        artifact_score,

    "baseline_positive":
        baseline_positive,

    "artifact_rejected":
        artifact_rejected,

    "final_positive":
        final_positive,

}


for name, arr in arrays_to_check.items():

    if len(
        arr
    ) != n:

        raise RuntimeError(
            f"{name} length mismatch: "
            f"{len(arr)} != {n}"
        )


if not np.all(
    np.isfinite(
        probabilities
    )
):

    raise RuntimeError(
        "Probabilities contain NaN/Inf."
    )


if not np.all(
    np.isfinite(
        artifact_score
    )
):

    raise RuntimeError(
        "Artifact scores contain NaN/Inf."
    )


print()

print(
    "[OK] All strict window-level arrays are aligned."
)

print(
    "Test windows:",
    n
)


# ================================================================
# 15. LOAD ORIGINAL LABEL/PATIENT INFORMATION
# ================================================================

print()

print(
    "=" * 76
)

print(
    "12. LOADING ORIGINAL DATASET INFORMATION"
)

print(
    "=" * 76
)


all_labels = np.asarray(
    np.load(
        LABELS_FILE
    ),
    dtype=np.int64
).reshape(-1)


all_patients = np.load(
    PATIENTS_FILE,
    allow_pickle=True
)


if len(
    all_labels
) != len(
    all_patients
):

    raise RuntimeError(
        "Full labels/patients length mismatch."
    )


if np.min(
    test_indices
) < 0:

    raise RuntimeError(
        "Negative test index detected."
    )


if np.max(
    test_indices
) >= len(
    all_patients
):

    raise RuntimeError(
        "Test index exceeds full dataset length."
    )


patients = all_patients[
    test_indices
]


original_test_labels = all_labels[
    test_indices
]


if not np.array_equal(
    labels,
    original_test_labels
):

    mismatch_count = int(
        np.sum(
            labels
            !=
            original_test_labels
        )
    )

    raise RuntimeError(
        "Saved strict labels do not match original labels "
        f"at test_indices. Mismatches: {mismatch_count}"
    )


print()

print(
    "[OK] Strict labels match original dataset labels."
)

print(
    "Unique test patients:",
    len(
        np.unique(
            patients
        )
    )
)


# ================================================================
# 16. SORT BY PATIENT AND ORIGINAL DATASET INDEX
# ================================================================

print()

print(
    "=" * 76
)

print(
    "13. SORTING TEST WINDOWS"
)

print(
    "=" * 76
)


order = np.lexsort(
    (
        test_indices,
        patients.astype(
            str
        )
    )
)


sorted_indices = test_indices[
    order
]

sorted_labels = labels[
    order
]

sorted_probabilities = probabilities[
    order
]

sorted_artifact_score = artifact_score[
    order
]

sorted_baseline_positive = baseline_positive[
    order
]

sorted_final_positive = final_positive[
    order
]

sorted_artifact_rejected = artifact_rejected[
    order
]

sorted_patients = patients[
    order
]


print()

print(
    "[OK] Windows sorted by patient and original dataset index."
)


# ================================================================
# 17. STRICT WINDOW-LEVEL SUMMARY
# ================================================================

print()

print(
    "=" * 76
)

print(
    "14. STRICT WINDOW-LEVEL SANITY CHECK"
)

print(
    "=" * 76
)


(
    baseline_window_tp,
    baseline_window_fp,
    baseline_window_tn,
    baseline_window_fn

) = classification_counts(

    labels,

    baseline_positive
)


(
    final_window_tp,
    final_window_fp,
    final_window_tn,
    final_window_fn

) = classification_counts(

    labels,

    final_positive
)


baseline_window_metrics = metric_summary(

    baseline_window_tp,

    baseline_window_fp,

    baseline_window_tn,

    baseline_window_fn
)


final_window_metrics = metric_summary(

    final_window_tp,

    final_window_fp,

    final_window_tn,

    final_window_fn
)


print()

print(
    "BASELINE"
)

print(
    "-" * 76
)

print(
    f"TP={baseline_window_tp} "
    f"FP={baseline_window_fp} "
    f"TN={baseline_window_tn} "
    f"FN={baseline_window_fn}"
)

print(
    f"Recall      = "
    f"{baseline_window_metrics['recall']:.6f}"
)

print(
    f"Specificity = "
    f"{baseline_window_metrics['specificity']:.6f}"
)

print(
    f"Precision   = "
    f"{baseline_window_metrics['precision']:.6f}"
)

print(
    f"F1          = "
    f"{baseline_window_metrics['f1']:.6f}"
)

print(
    f"Accuracy    = "
    f"{baseline_window_metrics['accuracy']:.6f}"
)


print()

print(
    "STRICT VALIDATION-FROZEN ARTIFACT REJECTION"
)

print(
    "-" * 76
)

print(
    f"TP={final_window_tp} "
    f"FP={final_window_fp} "
    f"TN={final_window_tn} "
    f"FN={final_window_fn}"
)

print(
    f"Recall      = "
    f"{final_window_metrics['recall']:.6f}"
)

print(
    f"Specificity = "
    f"{final_window_metrics['specificity']:.6f}"
)

print(
    f"Precision   = "
    f"{final_window_metrics['precision']:.6f}"
)

print(
    f"F1          = "
    f"{final_window_metrics['f1']:.6f}"
)

print(
    f"Accuracy    = "
    f"{final_window_metrics['accuracy']:.6f}"
)


# ================================================================
# 18. EXPECTED STRICT WINDOW RESULTS SANITY CHECK
# ================================================================

print()

print(
    "=" * 76
)

print(
    "15. VERIFYING AGAINST STRICT WINDOW EVALUATION"
)

print(
    "=" * 76
)


expected_baseline = {

    "tp": 78,

    "fp": 56,

    "tn": 2965,

    "fn": 15,
}


expected_final = {

    "tp": 76,

    "fp": 42,

    "tn": 2979,

    "fn": 17,
}


actual_baseline = {

    "tp": baseline_window_tp,

    "fp": baseline_window_fp,

    "tn": baseline_window_tn,

    "fn": baseline_window_fn,
}


actual_final = {

    "tp": final_window_tp,

    "fp": final_window_fp,

    "tn": final_window_tn,

    "fn": final_window_fn,
}


if actual_baseline != expected_baseline:

    raise RuntimeError(
        "Baseline window metrics do not match the "
        "previous STRICT evaluation.\n"
        f"Expected: {expected_baseline}\n"
        f"Actual:   {actual_baseline}"
    )


if actual_final != expected_final:

    raise RuntimeError(
        "Strict final window metrics do not match the "
        "previous STRICT evaluation.\n"
        f"Expected: {expected_final}\n"
        f"Actual:   {actual_final}"
    )


print()

print(
    "[OK] Window-level counts exactly match the strict evaluation."
)


# ================================================================
# 19. PATIENT-LEVEL WINDOW DIAGNOSTICS
# ================================================================

print()

print(
    "=" * 76
)

print(
    "16. PATIENT-LEVEL WINDOW DIAGNOSTICS"
)

print(
    "=" * 76
)


patient_ids = np.unique(
    sorted_patients
)


patient_window_results = []


for patient in patient_ids:

    mask = (
        sorted_patients
        ==
        patient
    )


    p_labels = sorted_labels[
        mask
    ]


    p_baseline = sorted_baseline_positive[
        mask
    ]


    p_final = sorted_final_positive[
        mask
    ]


    p_artifact = sorted_artifact_rejected[
        mask
    ]


    (
        p_baseline_tp,
        p_baseline_fp,
        p_baseline_tn,
        p_baseline_fn

    ) = classification_counts(

        p_labels,

        p_baseline
    )


    (
        p_final_tp,
        p_final_fp,
        p_final_tn,
        p_final_fn

    ) = classification_counts(

        p_labels,

        p_final
    )


    patient_window_results.append({

        "patient":
            str(
                patient
            ),

        "windows":
            int(
                np.sum(
                    mask
                )
            ),

        "true_seizure_windows":
            int(
                np.sum(
                    p_labels == 1
                )
            ),

        "baseline_positive_windows":
            int(
                np.sum(
                    p_baseline
                )
            ),

        "final_positive_windows":
            int(
                np.sum(
                    p_final
                )
            ),

        "artifact_rejected_windows":
            int(
                np.sum(
                    p_artifact
                )
            ),

        "baseline_tp_windows":
            p_baseline_tp,

        "baseline_fp_windows":
            p_baseline_fp,

        "baseline_tn_windows":
            p_baseline_tn,

        "baseline_fn_windows":
            p_baseline_fn,

        "final_tp_windows":
            p_final_tp,

        "final_fp_windows":
            p_final_fp,

        "final_tn_windows":
            p_final_tn,

        "final_fn_windows":
            p_final_fn,
    })


print()

print(
    f"Patients analyzed: "
    f"{len(patient_window_results)}"
)


for row in patient_window_results:

    print()

    print(
        f"Patient: "
        f"{row['patient']}"
    )

    print(
        f"  Windows: "
        f"{row['windows']}"
    )

    print(
        f"  Baseline: "
        f"TP={row['baseline_tp_windows']} "
        f"FP={row['baseline_fp_windows']} "
        f"FN={row['baseline_fn_windows']}"
    )

    print(
        f"  Strict:   "
        f"TP={row['final_tp_windows']} "
        f"FP={row['final_fp_windows']} "
        f"FN={row['final_fn_windows']}"
    )


# ================================================================
# 20. EVENT CONSTRUCTION
# ================================================================

print()

print(
    "=" * 76
)

print(
    "17. CONSTRUCTING SEIZURE EVENTS"
)

print(
    "=" * 76
)


def make_event_record(
    patient,
    indices,
    labels_for_windows,
    probabilities_for_windows,
    positions
):

    event_indices = indices[
        positions
    ]

    event_labels = labels_for_windows[
        positions
    ]

    event_probabilities = probabilities_for_windows[
        positions
    ]


    start_index = int(
        np.min(
            event_indices
        )
    )

    end_index = int(
        np.max(
            event_indices
        )
    )


    return {

        "patient":
            str(
                patient
            ),

        "start_index":
            start_index,

        "end_index":
            end_index,

        "window_count":
            int(
                len(
                    event_indices
                )
            ),

        "duration_seconds":
            float(
                len(
                    event_indices
                )
                *
                WINDOW_SECONDS
            ),

        "max_probability":
            float(
                np.max(
                    event_probabilities
                )
            ),

        "mean_probability":
            float(
                np.mean(
                    event_probabilities
                )
            ),

        "contains_true_seizure_window":
            bool(
                np.any(
                    event_labels == 1
                )
            ),

        "true_seizure_window_count":
            int(
                np.sum(
                    event_labels == 1
                )
            ),

        "false_positive_window_count":
            int(
                np.sum(
                    event_labels == 0
                )
            ),
    }


def build_events(
    indices,
    predictions,
    labels_for_windows,
    probabilities_for_windows,
    patients_for_windows,
    max_gap_windows=1
):

    """
    Build predicted/ground-truth events separately for each patient.

    Positive windows are merged when their ORIGINAL DATASET indices
    are close enough according to MAX_GAP_WINDOWS.

    This follows the same deterministic event-building rule that was
    already used in the original project script.

    No test-set optimization occurs here.
    """

    events = []


    unique_patients = np.unique(
        patients_for_windows
    )


    for patient in unique_patients:

        patient_mask = (
            patients_for_windows
            ==
            patient
        )


        p_indices = indices[
            patient_mask
        ]


        p_predictions = predictions[
            patient_mask
        ]


        p_labels = labels_for_windows[
            patient_mask
        ]


        p_probabilities = probabilities_for_windows[
            patient_mask
        ]


        local_order = np.argsort(
            p_indices
        )


        p_indices = p_indices[
            local_order
        ]


        p_predictions = p_predictions[
            local_order
        ]


        p_labels = p_labels[
            local_order
        ]


        p_probabilities = p_probabilities[
            local_order
        ]


        positive_positions = np.where(
            p_predictions
        )[0]


        if len(
            positive_positions
        ) == 0:

            continue


        current_group = [
            positive_positions[0]
        ]


        for pos in positive_positions[1:]:

            previous = current_group[
                -1
            ]


            index_gap = (

                int(
                    p_indices[
                        pos
                    ]
                )

                -

                int(
                    p_indices[
                        previous
                    ]
                )
            )


            if index_gap <= (
                max_gap_windows
                +
                1
            ):

                current_group.append(
                    pos
                )

            else:

                events.append(

                    make_event_record(

                        patient,

                        p_indices,

                        p_labels,

                        p_probabilities,

                        current_group
                    )
                )


                current_group = [
                    pos
                ]


        events.append(

            make_event_record(

                patient,

                p_indices,

                p_labels,

                p_probabilities,

                current_group
            )
        )


    return events


# ================================================================
# 21. BUILD GROUND-TRUTH EVENTS
# ================================================================

print()

print(
    "Building ground-truth seizure events..."
)


true_events = build_events(

    sorted_indices,

    sorted_labels.astype(
        bool
    ),

    sorted_labels,

    sorted_probabilities,

    sorted_patients,

    max_gap_windows=
        MAX_GAP_WINDOWS
)


print(
    "Ground-truth seizure events:",
    len(
        true_events
    )
)


# ================================================================
# 22. BUILD BASELINE PREDICTED EVENTS
# ================================================================

print()

print(
    "Building baseline predicted events..."
)


baseline_events = build_events(

    sorted_indices,

    sorted_baseline_positive,

    sorted_labels,

    sorted_probabilities,

    sorted_patients,

    max_gap_windows=
        MAX_GAP_WINDOWS
)


print(
    "Baseline predicted events:",
    len(
        baseline_events
    )
)


# ================================================================
# 23. BUILD STRICT PREDICTED EVENTS
# ================================================================

print()

print(
    "Building STRICT predicted events..."
)


final_events = build_events(

    sorted_indices,

    sorted_final_positive,

    sorted_labels,

    sorted_probabilities,

    sorted_patients,

    max_gap_windows=
        MAX_GAP_WINDOWS
)


print(
    "Strict predicted events:",
    len(
        final_events
    )
)


# ================================================================
# 24. EVENT MATCHING
# ================================================================

def event_detection_summary(
    predicted_events,
    ground_truth_events
):

    """
    Match predicted events to ground-truth events.

    A predicted event counts as TP when it overlaps an unmatched
    ground-truth event belonging to the same patient.

    A ground-truth event can only be matched once.

    Unmatched predicted events = FP.
    Unmatched true events = FN.
    """

    matched_true = set()

    tp = 0

    fp = 0

    matches = []


    for pred_id, pred in enumerate(
        predicted_events
    ):

        matched_id = None


        for true_id, true in enumerate(
            ground_truth_events
        ):

            if true_id in matched_true:

                continue


            if pred[
                "patient"
            ] != true[
                "patient"
            ]:

                continue


            pred_start = pred[
                "start_index"
            ]

            pred_end = pred[
                "end_index"
            ]

            true_start = true[
                "start_index"
            ]

            true_end = true[
                "end_index"
            ]


            overlap = (

                max(
                    pred_start,
                    true_start
                )

                <=

                min(
                    pred_end,
                    true_end
                )
            )


            if overlap:

                matched_id = true_id

                break


        if matched_id is not None:

            tp += 1

            matched_true.add(
                matched_id
            )


            matches.append({

                "predicted_event":
                    int(
                        pred_id
                    ),

                "true_event":
                    int(
                        matched_id
                    ),

                "patient":
                    pred[
                        "patient"
                    ],
            })

        else:

            fp += 1


    fn = (

        len(
            ground_truth_events
        )

        -

        len(
            matched_true
        )
    )


    recall = safe_divide(
        tp,
        tp + fn
    )


    precision = safe_divide(
        tp,
        tp + fp
    )


    f1 = (

        2.0
        *
        precision
        *
        recall
        /
        (
            precision
            +
            recall
        )

        if (
            precision
            +
            recall
        ) > 0

        else 0.0
    )


    return {

        "tp":
            int(
                tp
            ),

        "fp":
            int(
                fp
            ),

        "fn":
            int(
                fn
            ),

        "recall":
            float(
                recall
            ),

        "precision":
            float(
                precision
            ),

        "f1":
            float(
                f1
            ),

        "matches":
            matches,
    }


# ================================================================
# 25. GLOBAL EVENT-LEVEL RESULTS
# ================================================================

print()

print(
    "=" * 76
)

print(
    "18. STRICT EVENT-LEVEL RESULTS"
)

print(
    "=" * 76
)


baseline_event_metrics = event_detection_summary(

    baseline_events,

    true_events
)


final_event_metrics = event_detection_summary(

    final_events,

    true_events
)


print()

print(
    "BASELINE EVENT DETECTION"
)

print(
    "-" * 76
)

print(
    "TP events:",
    baseline_event_metrics[
        "tp"
    ]
)

print(
    "FP events:",
    baseline_event_metrics[
        "fp"
    ]
)

print(
    "FN events:",
    baseline_event_metrics[
        "fn"
    ]
)

print(
    f"Recall: "
    f"{baseline_event_metrics['recall']:.6f}"
)

print(
    f"Precision: "
    f"{baseline_event_metrics['precision']:.6f}"
)

print(
    f"F1: "
    f"{baseline_event_metrics['f1']:.6f}"
)


print()

print(
    "STRICT VALIDATION-FROZEN EVENT DETECTION"
)

print(
    "-" * 76
)

print(
    "TP events:",
    final_event_metrics[
        "tp"
    ]
)

print(
    "FP events:",
    final_event_metrics[
        "fp"
    ]
)

print(
    "FN events:",
    final_event_metrics[
        "fn"
    ]
)

print(
    f"Recall: "
    f"{final_event_metrics['recall']:.6f}"
)

print(
    f"Precision: "
    f"{final_event_metrics['precision']:.6f}"
)

print(
    f"F1: "
    f"{final_event_metrics['f1']:.6f}"
)


# ================================================================
# 26. EVENT-LEVEL CHANGE
# ================================================================

event_recall_change = (

    final_event_metrics[
        "recall"
    ]

    -

    baseline_event_metrics[
        "recall"
    ]
)


event_precision_change = (

    final_event_metrics[
        "precision"
    ]

    -

    baseline_event_metrics[
        "precision"
    ]
)


event_f1_change = (

    final_event_metrics[
        "f1"
    ]

    -

    baseline_event_metrics[
        "f1"
    ]
)


event_fp_reduction = (

    safe_divide(

        baseline_event_metrics[
            "fp"
        ]

        -

        final_event_metrics[
            "fp"
        ],

        baseline_event_metrics[
            "fp"
        ]
    )
)


print()

print(
    "=" * 76
)

print(
    "19. EVENT-LEVEL CHANGE"
)

print(
    "=" * 76
)


print()

print(
    f"FP event reduction: "
    f"{event_fp_reduction * 100:.2f}%"
)

print(
    f"Recall change: "
    f"{event_recall_change * 100:+.2f} pp"
)

print(
    f"Precision change: "
    f"{event_precision_change * 100:+.2f} pp"
)

print(
    f"F1 change: "
    f"{event_f1_change * 100:+.2f} pp"
)


# ================================================================
# 27. PATIENT-LEVEL EVENT SUMMARY
# ================================================================

print()

print(
    "=" * 76
)

print(
    "20. PATIENT-LEVEL EVENT SUMMARY"
)

print(
    "=" * 76
)


patient_event_summary = []


for patient in patient_ids:

    patient_string = str(
        patient
    )


    p_true = [

        event

        for event in true_events

        if event[
            "patient"
        ] == patient_string
    ]


    p_baseline = [

        event

        for event in baseline_events

        if event[
            "patient"
        ] == patient_string
    ]


    p_final = [

        event

        for event in final_events

        if event[
            "patient"
        ] == patient_string
    ]


    p_baseline_metrics = event_detection_summary(

        p_baseline,

        p_true
    )


    p_final_metrics = event_detection_summary(

        p_final,

        p_true
    )


    patient_event_summary.append({

        "patient":
            patient_string,

        "true_events":
            int(
                len(
                    p_true
                )
            ),

        "baseline_predicted_events":
            int(
                len(
                    p_baseline
                )
            ),

        "strict_predicted_events":
            int(
                len(
                    p_final
                )
            ),

        "baseline":
            p_baseline_metrics,

        "strict":
            p_final_metrics,
    })


    print()

    print(
        f"Patient: "
        f"{patient_string}"
    )

    print(
        f"  True events: "
        f"{len(p_true)}"
    )

    print(
        f"  Baseline: "
        f"TP={p_baseline_metrics['tp']} "
        f"FP={p_baseline_metrics['fp']} "
        f"FN={p_baseline_metrics['fn']} "
        f"Recall={p_baseline_metrics['recall']:.4f} "
        f"Precision={p_baseline_metrics['precision']:.4f}"
    )

    print(
        f"  Strict:   "
        f"TP={p_final_metrics['tp']} "
        f"FP={p_final_metrics['fp']} "
        f"FN={p_final_metrics['fn']} "
        f"Recall={p_final_metrics['recall']:.4f} "
        f"Precision={p_final_metrics['precision']:.4f}"
    )


# ================================================================
# 28. WINDOW-LEVEL CHANGE
# ================================================================

window_fp_reduction = safe_divide(

    baseline_window_fp
    -
    final_window_fp,

    baseline_window_fp
)


window_recall_change = (

    final_window_metrics[
        "recall"
    ]

    -

    baseline_window_metrics[
        "recall"
    ]
)


window_precision_change = (

    final_window_metrics[
        "precision"
    ]

    -

    baseline_window_metrics[
        "precision"
    ]
)


window_specificity_change = (

    final_window_metrics[
        "specificity"
    ]

    -

    baseline_window_metrics[
        "specificity"
    ]
)


window_f1_change = (

    final_window_metrics[
        "f1"
    ]

    -

    baseline_window_metrics[
        "f1"
    ]
)


window_accuracy_change = (

    final_window_metrics[
        "accuracy"
    ]

    -

    baseline_window_metrics[
        "accuracy"
    ]
)


# ================================================================
# 29. SAVE NPZ
# ================================================================

print()

print(
    "=" * 76
)

print(
    "21. SAVING STRICT EVENT RESULTS"
)

print(
    "=" * 76
)


np.savez(

    OUTPUT_NPZ,

    test_indices=
        test_indices,

    labels=
        labels,

    probabilities=
        probabilities,

    artifact_score=
        artifact_score,

    artifact_rejected=
        artifact_rejected,

    baseline_positive=
        baseline_positive,

    final_positive=
        final_positive,

    patients=
        np.asarray(
            patients
        ),

    baseline_window_tp=
        np.asarray(
            baseline_window_tp
        ),

    baseline_window_fp=
        np.asarray(
            baseline_window_fp
        ),

    baseline_window_tn=
        np.asarray(
            baseline_window_tn
        ),

    baseline_window_fn=
        np.asarray(
            baseline_window_fn
        ),

    final_window_tp=
        np.asarray(
            final_window_tp
        ),

    final_window_fp=
        np.asarray(
            final_window_fp
        ),

    final_window_tn=
        np.asarray(
            final_window_tn
        ),

    final_window_fn=
        np.asarray(
            final_window_fn
        ),

    baseline_event_count=
        np.asarray(
            len(
                baseline_events
            )
        ),

    final_event_count=
        np.asarray(
            len(
                final_events
            )
        ),

    true_event_count=
        np.asarray(
            len(
                true_events
            )
        ),

    baseline_event_tp=
        np.asarray(
            baseline_event_metrics[
                "tp"
            ]
        ),

    baseline_event_fp=
        np.asarray(
            baseline_event_metrics[
                "fp"
            ]
        ),

    baseline_event_fn=
        np.asarray(
            baseline_event_metrics[
                "fn"
            ]
        ),

    final_event_tp=
        np.asarray(
            final_event_metrics[
                "tp"
            ]
        ),

    final_event_fp=
        np.asarray(
            final_event_metrics[
                "fp"
            ]
        ),

    final_event_fn=
        np.asarray(
            final_event_metrics[
                "fn"
            ]
        ),
)


print()

print(
    "[OK] NPZ saved:"
)

print(
    OUTPUT_NPZ
)


# ================================================================
# 30. SAVE JSON
# ================================================================

output = {

    "project":
        PROJECT_DIR,

    "evaluation_type":
        "strict_validation_frozen_patient_seizure_event",

    "important": {

        "test_used_for_optimization":
            False,

        "test_threshold_fitting":
            False,

        "test_feature_fitting":
            False,

        "artifact_rule_source":
            "validation",

        "strict_window_result_source":
            os.path.basename(
                FINAL_TEST_FILE
            ),
    },

    "settings": {

        "baseline_probability_threshold":
            float(
                BASELINE_THRESHOLD
            ),

        "window_seconds":
            float(
                WINDOW_SECONDS
            ),

        "max_gap_windows":
            int(
                MAX_GAP_WINDOWS
            ),
    },

    "dataset": {

        "test_windows":
            int(
                len(
                    labels
                )
            ),

        "unique_test_patients":
            int(
                len(
                    patient_ids
                )
            ),

        "ground_truth_event_count":
            int(
                len(
                    true_events
                )
            ),
    },

    "window_level": {

        "baseline":
            baseline_window_metrics,

        "strict_validation_frozen":
            final_window_metrics,

        "change": {

            "fp_reduction_percent":
                float(
                    window_fp_reduction
                    *
                    100
                ),

            "recall_change_percentage_points":
                float(
                    window_recall_change
                    *
                    100
                ),

            "precision_change_percentage_points":
                float(
                    window_precision_change
                    *
                    100
                ),

            "specificity_change_percentage_points":
                float(
                    window_specificity_change
                    *
                    100
                ),

            "f1_change_percentage_points":
                float(
                    window_f1_change
                    *
                    100
                ),

            "accuracy_change_percentage_points":
                float(
                    window_accuracy_change
                    *
                    100
                ),
        },
    },

    "event_level": {

        "true_event_count":
            int(
                len(
                    true_events
                )
            ),

        "baseline":
            baseline_event_metrics,

        "strict_validation_frozen":
            final_event_metrics,

        "change": {

            "fp_reduction_percent":
                float(
                    event_fp_reduction
                    *
                    100
                ),

            "recall_change_percentage_points":
                float(
                    event_recall_change
                    *
                    100
                ),

            "precision_change_percentage_points":
                float(
                    event_precision_change
                    *
                    100
                ),

            "f1_change_percentage_points":
                float(
                    event_f1_change
                    *
                    100
                ),
        },
    },

    "patient_window_results":
        patient_window_results,

    "patient_event_summary":
        patient_event_summary,

    "baseline_events":
        baseline_events,

    "strict_events":
        final_events,

    "true_events":
        true_events,
}


with open(
    OUTPUT_JSON,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        output,
        f,
        indent=2,
        ensure_ascii=False
    )


print()

print(
    "[OK] JSON saved:"
)

print(
    OUTPUT_JSON
)


# ================================================================
# 31. FINAL INTERPRETATION
# ================================================================

print()

print(
    "=" * 76
)

print(
    "22. FINAL STRICT SUMMARY"
)

print(
    "=" * 76
)


print()

print(
    "WINDOW LEVEL"
)

print(
    "-" * 76
)

print(
    f"Baseline:"
    f" TP={baseline_window_tp}"
    f" FP={baseline_window_fp}"
    f" TN={baseline_window_tn}"
    f" FN={baseline_window_fn}"
)

print(
    f"Strict:"
    f"   TP={final_window_tp}"
    f" FP={final_window_fp}"
    f" TN={final_window_tn}"
    f" FN={final_window_fn}"
)

print()

print(
    f"FP reduction: "
    f"{window_fp_reduction * 100:.2f}%"
)

print(
    f"Recall: "
    f"{baseline_window_metrics['recall']:.6f}"
    f" -> "
    f"{final_window_metrics['recall']:.6f}"
)

print(
    f"Precision: "
    f"{baseline_window_metrics['precision']:.6f}"
    f" -> "
    f"{final_window_metrics['precision']:.6f}"
)

print(
    f"F1: "
    f"{baseline_window_metrics['f1']:.6f}"
    f" -> "
    f"{final_window_metrics['f1']:.6f}"
)


print()

print(
    "EVENT LEVEL"
)

print(
    "-" * 76
)

print(
    f"Ground-truth events: "
    f"{len(true_events)}"
)

print(
    f"Baseline:"
    f" TP={baseline_event_metrics['tp']}"
    f" FP={baseline_event_metrics['fp']}"
    f" FN={baseline_event_metrics['fn']}"
)

print(
    f"Strict:"
    f"   TP={final_event_metrics['tp']}"
    f" FP={final_event_metrics['fp']}"
    f" FN={final_event_metrics['fn']}"
)

print()

print(
    f"Event FP reduction: "
    f"{event_fp_reduction * 100:.2f}%"
)

print(
    f"Event recall: "
    f"{baseline_event_metrics['recall']:.6f}"
    f" -> "
    f"{final_event_metrics['recall']:.6f}"
)

print(
    f"Event precision: "
    f"{baseline_event_metrics['precision']:.6f}"
    f" -> "
    f"{final_event_metrics['precision']:.6f}"
)

print(
    f"Event F1: "
    f"{baseline_event_metrics['f1']:.6f}"
    f" -> "
    f"{final_event_metrics['f1']:.6f}"
)


print()

print(
    "=" * 76
)

print(
    "STRICT PATIENT / SEIZURE-EVENT EVALUATION COMPLETED"
)

print(
    "=" * 76
)


print()

print(
    "No test optimization was performed."
)

print(
    "No threshold was fitted on the test set."
)

print(
    "No feature statistics were fitted on the test set."
)

print(
    "Strict validation-frozen predictions were used."
)


print()

print(
    "Outputs:"
)

print(
    OUTPUT_JSON
)

print(
    OUTPUT_NPZ
)


print()

print(
    "=" * 76
)

print(
    "DONE"
)

print(
    "=" * 76
)