
import json
from pathlib import Path

import numpy as np


# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = Path(
    r"C:\Users\rezay\Desktop\EEG_Seizure_Project"
)

RESULTS_DIR = PROJECT_DIR / "results"

NPZ_PATH = (
    RESULTS_DIR /
    "test_window_probabilities.npz"
)

THRESHOLD_PATH = (
    RESULTS_DIR /
    "validation_threshold_results.json"
)

PERSISTENCE_PATH = (
    RESULTS_DIR /
    "validation_persistence_rule.json"
)

OUTPUT_PATH = (
    RESULTS_DIR /
    "final_test_persistence_evaluation.json"
)


# ============================================================
# FROZEN VALIDATION RULE
# ============================================================
#
# IMPORTANT:
# These values are NOT used to search Test.
#
# They are loaded from the Validation result file.
#
# Expected frozen rule:
#
# window threshold       = 0.560
# positive fraction      >= 0.005
# minimum cluster size   >= 4
# maximum gap            <= 2
#
# ============================================================


# ============================================================
# HELPERS
# ============================================================

def metrics(y_true, y_pred):
    """
    Calculate patient-level classification metrics.
    """

    y_true = np.asarray(
        y_true
    ).astype(int)

    y_pred = np.asarray(
        y_pred
    ).astype(int)

    tp = int(
        np.sum(
            (y_true == 1)
            & (y_pred == 1)
        )
    )

    fp = int(
        np.sum(
            (y_true == 0)
            & (y_pred == 1)
        )
    )

    fn = int(
        np.sum(
            (y_true == 1)
            & (y_pred == 0)
        )
    )

    tn = int(
        np.sum(
            (y_true == 0)
            & (y_pred == 0)
        )
    )

    sensitivity = (
        tp / (tp + fn)
        if (tp + fn)
        else 0.0
    )

    specificity = (
        tn / (tn + fp)
        if (tn + fp)
        else 0.0
    )

    precision = (
        tp / (tp + fp)
        if (tp + fp)
        else 0.0
    )

    f1 = (
        2.0
        * precision
        * sensitivity
        / (precision + sensitivity)
        if (precision + sensitivity)
        else 0.0
    )

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "f1": f1,
    }


def find_threshold(obj):
    """
    Recursively find a threshold value inside
    validation_threshold_results.json.
    """

    if isinstance(obj, dict):

        for key, value in obj.items():

            key_lower = str(key).lower()

            if (
                "threshold" in key_lower
                and isinstance(
                    value,
                    (int, float)
                )
            ):

                value = float(value)

                if 0.0 < value < 1.0:
                    return value

            result = find_threshold(
                value
            )

            if result is not None:
                return result

    elif isinstance(obj, list):

        for item in obj:

            result = find_threshold(
                item
            )

            if result is not None:
                return result

    return None


def get_runs(binary):
    """
    Return contiguous positive runs.

    Example:

    [0,1,1,0,1]

    -> [(1,2), (4,4)]
    """

    binary = np.asarray(
        binary
    ).astype(int)

    runs = []

    start = None

    for i, value in enumerate(
        binary
    ):

        if value == 1:

            if start is None:
                start = i

        elif start is not None:

            runs.append(
                (
                    start,
                    i - 1
                )
            )

            start = None

    if start is not None:

        runs.append(
            (
                start,
                len(binary) - 1
            )
        )

    return runs


def temporal_clusters(
    binary,
    max_gap
):
    """
    Build temporal positive clusters.

    Positive windows separated by at most
    max_gap negative windows belong to the
    same temporal cluster.

    Example with max_gap=2:

    positive indices:
        10, 11, 14

    gaps:
        0 and 2

    -> one cluster of size 3.
    """

    positive_indices = np.where(
        binary
    )[0]

    if len(
        positive_indices
    ) == 0:

        return []

    clusters = []

    current = [
        int(
            positive_indices[0]
        )
    ]

    for idx in positive_indices[1:]:

        idx = int(idx)

        gap = (
            idx
            - current[-1]
            - 1
        )

        if gap <= max_gap:

            current.append(
                idx
            )

        else:

            clusters.append(
                current
            )

            current = [idx]

    clusters.append(
        current
    )

    return clusters


def patient_features(
    probabilities,
    threshold,
    max_gap
):
    """
    Calculate temporal features for one patient.
    """

    probabilities = np.asarray(
        probabilities
    ).astype(float)

    positive = (
        probabilities
        >= threshold
    )

    total_windows = int(
        len(probabilities)
    )

    positive_windows = int(
        np.sum(positive)
    )

    positive_fraction = (
        positive_windows
        / total_windows
        if total_windows
        else 0.0
    )

    runs = get_runs(
        positive
    )

    run_lengths = [
        end - start + 1
        for start, end in runs
    ]

    if run_lengths:

        max_run = int(
            max(run_lengths)
        )

        mean_run = float(
            np.mean(
                run_lengths
            )
        )

    else:

        max_run = 0
        mean_run = 0.0

    clusters = temporal_clusters(
        positive,
        max_gap
    )

    cluster_sizes = [
        len(cluster)
        for cluster in clusters
    ]

    cluster_max_size = (
        int(
            max(cluster_sizes)
        )
        if cluster_sizes
        else 0
    )

    cluster_count = int(
        len(clusters)
    )

    q95 = float(
        np.percentile(
            probabilities,
            95
        )
    )

    return {
        "windows": total_windows,
        "positive_windows": positive_windows,
        "positive_fraction": positive_fraction,
        "runs": len(runs),
        "max_run": max_run,
        "mean_run": mean_run,
        "cluster_count": cluster_count,
        "cluster_max_size": cluster_max_size,
        "q95": q95,
    }


def apply_frozen_rule(
    feature,
    fraction_threshold,
    min_runs,
    max_run,
    min_cluster_size,
):
    """
    Apply the already-frozen Validation rule.

    IMPORTANT:
    No optimization happens here.
    """

    if (
        feature["positive_fraction"]
        < fraction_threshold
    ):
        return 0

    if (
        feature["runs"]
        < min_runs
    ):
        return 0

    if (
        max_run is not None
        and feature["max_run"]
        > max_run
    ):
        return 0

    if (
        feature["cluster_max_size"]
        < min_cluster_size
    ):
        return 0

    return 1


# ============================================================
# HEADER
# ============================================================

print(
    "=" * 70
)

print(
    "FINAL TEST FROZEN TEMPORAL "
    "PERSISTENCE EVALUATION"
)

print(
    "=" * 70
)

print()
print(
    "IMPORTANT:"
)

print(
    "This is a FROZEN Test evaluation."
)

print(
    "No Test optimization is performed."
)

print(
    "No Test rule search is performed."
)

print(
    "No threshold is changed."
)

print(
    "No model or dataset is modified."
)


# ============================================================
# INPUT CHECK
# ============================================================

print()
print(
    "=" * 70
)

print(
    "1. CHECKING INPUT FILES"
)

print(
    "=" * 70
)

for path in [
    NPZ_PATH,
    THRESHOLD_PATH,
    PERSISTENCE_PATH,
]:

    if not path.exists():

        raise FileNotFoundError(
            f"Required file not found:\n{path}"
        )

    print(
        f"[OK] {path}"
    )


# ============================================================
# LOAD FROZEN VALIDATION THRESHOLD
# ============================================================

print()
print(
    "=" * 70
)

print(
    "2. LOADING FROZEN VALIDATION THRESHOLD"
)

print(
    "=" * 70
)

with open(
    THRESHOLD_PATH,
    "r",
    encoding="utf-8"
) as f:

    threshold_data = json.load(
        f
    )


WINDOW_THRESHOLD = find_threshold(
    threshold_data
)

if WINDOW_THRESHOLD is None:

    raise RuntimeError(
        "Could not find validation "
        "window threshold."
    )

print(
    f"Frozen Validation window threshold: "
    f"{WINDOW_THRESHOLD:.6f}"
)


# ============================================================
# LOAD FROZEN PERSISTENCE RULE
# ============================================================

print()
print(
    "=" * 70
)

print(
    "3. LOADING FROZEN VALIDATION "
    "PERSISTENCE RULE"
)

print(
    "=" * 70
)

with open(
    PERSISTENCE_PATH,
    "r",
    encoding="utf-8"
) as f:

    persistence_data = json.load(
        f
    )


best_rule = (
    persistence_data.get(
        "best_rule"
    )
)

if best_rule is None:

    raise RuntimeError(
        "No best persistence rule "
        "was found in validation result."
    )


fraction_threshold = float(
    best_rule[
        "fraction_threshold"
    ]
)

min_runs = int(
    best_rule[
        "min_runs"
    ]
)

max_run = best_rule[
    "max_run"
]

if max_run is not None:

    max_run = int(
        max_run
    )

min_cluster_size = int(
    best_rule[
        "min_cluster_size"
    ]
)

max_gap = int(
    best_rule[
        "max_gap"
    ]
)


print()
print(
    "Frozen rule:"
)

print(
    f"Minimum positive fraction : "
    f"{fraction_threshold:.6f}"
)

print(
    f"Minimum number of runs    : "
    f"{min_runs}"
)

print(
    "Maximum run length        : "
    + (
        "ANY"
        if max_run is None
        else str(max_run)
    )
)

print(
    f"Minimum cluster size      : "
    f"{min_cluster_size}"
)

print(
    f"Maximum gap               : "
    f"{max_gap}"
)


# ============================================================
# SANITY CHECK FROZEN RULE
# ============================================================

print()
print(
    "=" * 70
)

print(
    "4. VERIFYING FROZEN RULE"
)

print(
    "=" * 70
)

if not (
    0.0
    <= fraction_threshold
    <= 1.0
):

    raise RuntimeError(
        "Invalid frozen fraction threshold."
    )

if min_runs < 0:

    raise RuntimeError(
        "Invalid frozen min_runs."
    )

if (
    max_run is not None
    and max_run < 0
):

    raise RuntimeError(
        "Invalid frozen max_run."
    )

if min_cluster_size < 1:

    raise RuntimeError(
        "Invalid frozen cluster size."
    )

if max_gap < 0:

    raise RuntimeError(
        "Invalid frozen max_gap."
    )

print(
    "[OK] Frozen rule parameters valid."
)

print(
    "[OK] Rule will NOT be changed on Test."
)


# ============================================================
# LOAD TEST DATA
# ============================================================

print()
print(
    "=" * 70
)

print(
    "5. LOADING TEST DATA"
)

print(
    "=" * 70
)

data = np.load(
    NPZ_PATH,
    allow_pickle=True
)

print()
print(
    "Available NPZ arrays:"
)

for key in data.files:

    print(
        f"  {key:30s} "
        f"shape={data[key].shape}"
    )


required_keys = [
    "test_indices",
    "patients",
    "labels",
    "probabilities",
]

for key in required_keys:

    if key not in data.files:

        raise RuntimeError(
            f"Required array missing: {key}"
        )


test_indices = np.asarray(
    data["test_indices"]
)

patients = np.asarray(
    data["patients"]
).astype(str)

labels = np.asarray(
    data["labels"]
).astype(int)

probabilities = np.asarray(
    data["probabilities"]
).astype(float)


print()
print(
    f"Test samples: "
    f"{len(probabilities)}"
)

print(
    f"Probabilities shape: "
    f"{probabilities.shape}"
)

print(
    f"Labels shape       : "
    f"{labels.shape}"
)

print(
    f"Patients shape     : "
    f"{patients.shape}"
)

print(
    f"Indices shape      : "
    f"{test_indices.shape}"
)


# ============================================================
# VERIFY TEST DATA
# ============================================================

print()
print(
    "=" * 70
)

print(
    "6. VERIFYING TEST DATA"
)

print(
    "=" * 70
)

if not (
    len(test_indices)
    == len(patients)
    == len(labels)
    == len(probabilities)
):

    raise RuntimeError(
        "Test arrays are not aligned."
    )

print(
    "[OK] Arrays aligned."
)

if not np.all(
    np.isfinite(
        probabilities
    )
):

    raise RuntimeError(
        "Test probabilities contain "
        "non-finite values."
    )

print(
    "[OK] Probabilities finite."
)

unique_labels = np.unique(
    labels
)

print(
    f"[OK] Window labels present: "
    f"{unique_labels.tolist()}"
)


# ============================================================
# BUILD PATIENT FEATURES
# ============================================================

print()
print(
    "=" * 70
)

print(
    "7. BUILDING TEST PATIENT "
    "TEMPORAL FEATURES"
)

print(
    "=" * 70
)

patient_ids = sorted(
    np.unique(
        patients
    )
)

patient_results = {}

y_true = []
y_pred = []


for patient in patient_ids:

    mask = (
        patients == patient
    )

    p = probabilities[
        mask
    ]

    y = labels[
        mask
    ]

    # --------------------------------------------------------
    # IMPORTANT CORRECTION
    # --------------------------------------------------------
    #
    # Window labels are NOT required to be identical.
    #
    # A patient may contain:
    #
    #   seizure windows     -> label 1
    #   non-seizure windows -> label 0
    #
    # Therefore patient-level ground truth is:
    #
    #   patient = positive
    #   if at least one window has label 1.
    #
    # This is the appropriate patient-level conversion
    # for the seizure-patient evaluation.
    # --------------------------------------------------------

    true_label = int(
        np.any(
            y == 1
        )
    )

    f = patient_features(
        p,
        WINDOW_THRESHOLD,
        max_gap
    )

    prediction = apply_frozen_rule(
        f,
        fraction_threshold,
        min_runs,
        max_run,
        min_cluster_size
    )

    y_true.append(
        true_label
    )

    y_pred.append(
        prediction
    )

    patient_results[
        patient
    ] = {
        "true_label":
            true_label,

        "prediction":
            int(prediction),

        **f,
    }

    print(
        f"{patient:8s} "
        f"true={true_label} "
        f"pred={prediction} "
        f"windows={f['windows']:4d} "
        f"positive={f['positive_windows']:3d} "
        f"fraction={f['positive_fraction']:.4f} "
        f"runs={f['runs']:3d} "
        f"maxrun={f['max_run']:2d} "
        f"cluster_max={f['cluster_max_size']:2d} "
        f"Q95={f['q95']:.6f}"
    )


# ============================================================
# FINAL TEST METRICS
# ============================================================

print()
print(
    "=" * 70
)

print(
    "8. FINAL TEST PATIENT-LEVEL RESULTS"
)

print(
    "=" * 70
)

final_metrics = metrics(
    y_true,
    y_pred
)

print()

print(
    json.dumps(
        final_metrics,
        indent=2
    )
)


# ============================================================
# PATIENT CONFUSION MATRIX
# ============================================================

print()
print(
    "=" * 70
)

print(
    "9. PATIENT CONFUSION MATRIX"
)

print(
    "=" * 70
)

for patient in patient_ids:

    r = patient_results[
        patient
    ]

    true_label = r[
        "true_label"
    ]

    prediction = r[
        "prediction"
    ]

    if (
        true_label == 1
        and prediction == 1
    ):

        status = "TP"

    elif (
        true_label == 0
        and prediction == 1
    ):

        status = "FP"

    elif (
        true_label == 1
        and prediction == 0
    ):

        status = "FN"

    else:

        status = "TN"

    print(
        f"{patient:8s} "
        f"true={true_label} "
        f"pred={prediction} "
        f"{status}"
    )


# ============================================================
# FROZEN Q95 BASELINE
# ============================================================

print()
print(
    "=" * 70
)

print(
    "10. COMPARISON WITH FROZEN Q95 BASELINE"
)

print(
    "=" * 70
)

q95_true = []
q95_pred = []


for patient in patient_ids:

    r = patient_results[
        patient
    ]

    q95_true.append(
        r["true_label"]
    )

    q95_pred.append(
        int(
            r["q95"]
            >= 0.50
        )
    )


q95_metrics = metrics(
    q95_true,
    q95_pred
)


print()

print(
    f"{'Metric':20s}"
    f"{'Q95':>15s}"
    f"{'Persistence':>18s}"
)

print(
    "-" * 55
)

for key in [
    "sensitivity",
    "specificity",
    "precision",
    "f1",
]:

    print(
        f"{key:20s}"
        f"{q95_metrics[key]:15.6f}"
        f"{final_metrics[key]:18.6f}"
    )


# ============================================================
# FALSE POSITIVES
# ============================================================

print()
print(
    "=" * 70
)

print(
    "11. FINAL TEST FALSE POSITIVES"
)

print(
    "=" * 70
)

false_positive_patients = []

for patient in patient_ids:

    r = patient_results[
        patient
    ]

    if (
        r["true_label"] == 0
        and r["prediction"] == 1
    ):

        false_positive_patients.append(
            patient
        )


if not false_positive_patients:

    print()
    print(
        "[NONE] No patient-level "
        "false positives."
    )

else:

    for patient in (
        false_positive_patients
    ):

        r = patient_results[
            patient
        ]

        print(
            f"{patient:8s} "
            f"Q95={r['q95']:.6f} "
            f"fraction="
            f"{r['positive_fraction']:.6f} "
            f"positive_windows="
            f"{r['positive_windows']} "
            f"runs={r['runs']} "
            f"cluster_max="
            f"{r['cluster_max_size']}"
        )


# ============================================================
# FALSE NEGATIVES
# ============================================================

print()
print(
    "=" * 70
)

print(
    "12. FINAL TEST FALSE NEGATIVES"
)

print(
    "=" * 70
)

false_negative_patients = []

for patient in patient_ids:

    r = patient_results[
        patient
    ]

    if (
        r["true_label"] == 1
        and r["prediction"] == 0
    ):

        false_negative_patients.append(
            patient
        )


if not false_negative_patients:

    print()
    print(
        "[NONE] No patient-level "
        "false negatives."
    )

else:

    for patient in (
        false_negative_patients
    ):

        r = patient_results[
            patient
        ]

        print(
            f"{patient:8s} "
            f"Q95={r['q95']:.6f} "
            f"fraction="
            f"{r['positive_fraction']:.6f} "
            f"positive_windows="
            f"{r['positive_windows']} "
            f"runs={r['runs']} "
            f"cluster_max="
            f"{r['cluster_max_size']}"
        )


# ============================================================
# FINAL INTERPRETATION
# ============================================================

print()
print(
    "=" * 70
)

print(
    "13. FINAL INTERPRETATION"
)

print(
    "=" * 70
)

print()

print(
    "The temporal persistence rule was "
    "selected exclusively on Validation."
)

print(
    "The Test set was used only for "
    "one frozen evaluation."
)

print(
    "No Test threshold optimization "
    "was performed."
)

print(
    "No Test persistence-rule search "
    "was performed."
)

print(
    "No model was modified."
)

print(
    "No dataset was modified."
)

print(
    "The Validation threshold was "
    "not modified."
)

print()

print(
    f"Final Test sensitivity: "
    f"{final_metrics['sensitivity']:.6f}"
)

print(
    f"Final Test specificity: "
    f"{final_metrics['specificity']:.6f}"
)

print(
    f"Final Test precision: "
    f"{final_metrics['precision']:.6f}"
)

print(
    f"Final Test F1: "
    f"{final_metrics['f1']:.6f}"
)

print()

print(
    f"Test TP: {final_metrics['tp']}"
)

print(
    f"Test FP: {final_metrics['fp']}"
)

print(
    f"Test FN: {final_metrics['fn']}"
)

print(
    f"Test TN: {final_metrics['tn']}"
)


# ============================================================
# SAVE FINAL RESULT
# ============================================================

print()
print(
    "=" * 70
)

print(
    "14. SAVING FINAL RESULTS"
)

print(
    "=" * 70
)


result = {

    "analysis":
        "final_test_frozen_persistence_evaluation",

    "project_directory":
        str(PROJECT_DIR),

    "results_directory":
        str(RESULTS_DIR),

    "window_threshold":
        WINDOW_THRESHOLD,

    "frozen_validation_rule": {

        "fraction_threshold":
            fraction_threshold,

        "min_runs":
            min_runs,

        "max_run":
            max_run,

        "min_cluster_size":
            min_cluster_size,

        "max_gap":
            max_gap,
    },

    "test_metrics":
        final_metrics,

    "q95_baseline_metrics":
        q95_metrics,

    "patients":
        patient_results,

    "false_positive_patients":
        false_positive_patients,

    "false_negative_patients":
        false_negative_patients,

    "test_used_for_optimization":
        False,

    "test_rule_search":
        False,

    "test_threshold_search":
        False,

    "model_modified":
        False,

    "dataset_modified":
        False,

    "validation_threshold_modified":
        False,

    "validation_rule_modified":
        False,
}


with open(
    OUTPUT_PATH,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        result,
        f,
        indent=2,
        ensure_ascii=False
    )


print()

print(
    "[OK] Results saved:"
)

print(
    OUTPUT_PATH
)

print()

print(
    "=" * 70
)

print(
    "FINAL TEST FROZEN TEMPORAL "
    "PERSISTENCE EVALUATION COMPLETED"
)

print(
    "=" * 70
)

print()

print(
    "No model was modified."
)

print(
    "No dataset was modified."
)

print(
    "No Test optimization was performed."
)

print(
    "No Test rule search was performed."
)

print(
    "No Test threshold search was performed."
)

print(
    "Validation rule remained frozen."
)