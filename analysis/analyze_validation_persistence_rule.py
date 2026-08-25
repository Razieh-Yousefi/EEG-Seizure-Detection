import json
from pathlib import Path

import numpy as np


# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = Path(r"C:\Users\rezay\Desktop\EEG_Seizure_Project")
RESULTS_DIR = PROJECT_DIR / "results"

NPZ_PATH = RESULTS_DIR / "validation_window_probabilities.npz"
THRESHOLD_PATH = RESULTS_DIR / "validation_threshold_results.json"

OUTPUT_PATH = RESULTS_DIR / "validation_persistence_rule.json"


# ============================================================
# CONFIGURATION
# ============================================================

REQUIRED_SENSITIVITY = 0.90

# Minimum number of positive windows required inside a temporal
# neighborhood / block.
MIN_POSITIVE_WINDOWS = [1, 2, 3, 4, 5]

# Maximum allowed gap between positive windows.
MAX_GAPS = [0, 1, 2, 3, 4, 5]

# Minimum fraction of positive windows.
FRACTION_THRESHOLDS = [
    0.005,
    0.010,
    0.015,
    0.020,
    0.030,
    0.050,
    0.075,
    0.100,
]

# Maximum run length required for a patient to be accepted.
# None means no restriction.
MAX_RUN_THRESHOLDS = [
    None,
    1,
    2,
    3,
    4,
    5,
]

# Minimum number of positive runs.
MIN_RUN_THRESHOLDS = [
    0,
    1,
    2,
    3,
    5,
    10,
]


# ============================================================
# HELPERS
# ============================================================

def metrics(y_true, y_pred):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))

    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0

    f1 = (
        2 * precision * sensitivity / (precision + sensitivity)
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


def get_runs(binary):
    binary = np.asarray(binary).astype(int)

    runs = []

    start = None

    for i, value in enumerate(binary):
        if value == 1 and start is None:
            start = i

        elif value == 0 and start is not None:
            runs.append((start, i - 1))
            start = None

    if start is not None:
        runs.append((start, len(binary) - 1))

    return runs


def patient_features(probabilities, threshold):
    positive = probabilities >= threshold

    positive_count = int(np.sum(positive))
    total = len(probabilities)

    fraction = positive_count / total if total else 0.0

    runs = get_runs(positive)

    run_lengths = [
        end - start + 1
        for start, end in runs
    ]

    if run_lengths:
        max_run = int(max(run_lengths))
        mean_run = float(np.mean(run_lengths))
    else:
        max_run = 0
        mean_run = 0.0

    # Count "near-persistent" groups where positive windows are
    # separated by at most MAX_GAPS.
    #
    # Example:
    # positives at 10,11,13 with max_gap=1
    # are treated as one temporal cluster because the gap is 1.
    #
    def temporal_clusters(max_gap):
        pos_indices = np.where(positive)[0]

        if len(pos_indices) == 0:
            return []

        clusters = []
        current = [int(pos_indices[0])]

        for idx in pos_indices[1:]:
            idx = int(idx)

            gap = idx - current[-1] - 1

            if gap <= max_gap:
                current.append(idx)
            else:
                clusters.append(current)
                current = [idx]

        clusters.append(current)

        return clusters

    cluster_info = {}

    for max_gap in MAX_GAPS:
        clusters = temporal_clusters(max_gap)

        lengths = [len(c) for c in clusters]

        cluster_info[max_gap] = {
            "number": len(clusters),
            "max_size": int(max(lengths)) if lengths else 0,
            "mean_size": float(np.mean(lengths)) if lengths else 0.0,
        }

    return {
        "windows": total,
        "positive_windows": positive_count,
        "positive_fraction": fraction,
        "runs": len(runs),
        "max_run": max_run,
        "mean_run": mean_run,
        "cluster_info": cluster_info,
    }


def rule_prediction(
    feature,
    fraction_threshold,
    min_runs,
    max_run,
    min_cluster_size,
    max_gap,
):
    if feature["positive_fraction"] < fraction_threshold:
        return 0

    if feature["runs"] < min_runs:
        return 0

    if max_run is not None and feature["max_run"] > max_run:
        return 0

    cluster = feature["cluster_info"][max_gap]

    if cluster["max_size"] < min_cluster_size:
        return 0

    return 1


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("VALIDATION TEMPORAL PERSISTENCE RULE SEARCH")
print("=" * 70)

print()
print("Project directory:")
print(PROJECT_DIR)

print()
print("Results directory:")
print(RESULTS_DIR)


# ============================================================
# INPUT CHECK
# ============================================================

print()
print("=" * 70)
print("1. CHECKING INPUT FILES")
print("=" * 70)

if not NPZ_PATH.exists():
    raise FileNotFoundError(NPZ_PATH)

if not THRESHOLD_PATH.exists():
    raise FileNotFoundError(THRESHOLD_PATH)

print(f"[OK] {NPZ_PATH}")
print(f"[OK] {THRESHOLD_PATH}")


# ============================================================
# LOAD DATA
# ============================================================

print()
print("=" * 70)
print("2. LOADING VALIDATION DATA")
print("=" * 70)

data = np.load(NPZ_PATH, allow_pickle=True)

print()
print("Available NPZ arrays:")

for key in data.files:
    print(f"  {key:30s} shape={data[key].shape}")

validation_indices = np.asarray(data["validation_indices"])
patients = np.asarray(data["patients"]).astype(str)
labels = np.asarray(data["labels"]).astype(int)
probabilities = np.asarray(data["probabilities"]).astype(float)

print()
print(f"Validation samples: {len(probabilities)}")
print(f"Probabilities shape: {probabilities.shape}")
print(f"Labels shape       : {labels.shape}")
print(f"Patients shape     : {patients.shape}")
print(f"Indices shape      : {validation_indices.shape}")


# ============================================================
# VERIFY
# ============================================================

print()
print("=" * 70)
print("3. VERIFYING VALIDATION DATA")
print("=" * 70)

if not (
    len(probabilities)
    == len(labels)
    == len(patients)
    == len(validation_indices)
):
    raise RuntimeError("Validation arrays are not aligned.")

print("[OK] Arrays aligned.")

if not np.all(np.isfinite(probabilities)):
    raise RuntimeError("Probabilities contain non-finite values.")

print("[OK] Probabilities finite.")


# ============================================================
# THRESHOLD
# ============================================================

print()
print("=" * 70)
print("4. LOADING FROZEN VALIDATION THRESHOLD")
print("=" * 70)

with open(THRESHOLD_PATH, "r", encoding="utf-8") as f:
    threshold_data = json.load(f)


def find_threshold(obj):
    if isinstance(obj, dict):
        for key, value in obj.items():

            key_lower = str(key).lower()

            if (
                "threshold" in key_lower
                and isinstance(value, (int, float))
            ):
                if 0.0 < float(value) < 1.0:
                    return float(value)

            result = find_threshold(value)

            if result is not None:
                return result

    elif isinstance(obj, list):
        for item in obj:
            result = find_threshold(item)

            if result is not None:
                return result

    return None


WINDOW_THRESHOLD = find_threshold(threshold_data)

if WINDOW_THRESHOLD is None:
    raise RuntimeError(
        "Could not find validation threshold."
    )

print(
    f"Window threshold: "
    f"{WINDOW_THRESHOLD:.6f}"
)


# ============================================================
# PATIENT FEATURES
# ============================================================

print()
print("=" * 70)
print("5. BUILDING PATIENT TEMPORAL FEATURES")
print("=" * 70)

patient_ids = sorted(np.unique(patients))

features = {}

for patient in patient_ids:

    mask = patients == patient

    p = probabilities[mask]

    y = labels[mask]

    # IMPORTANT:
    # labels are WINDOW-level labels, not patient-level labels.
    #
    # A patient is considered positive if at least one validation
    # window is labeled positive.
    #
    # Therefore mixed labels such as [0, 1] within one patient
    # are expected and must NOT be treated as an error.

    unique_labels = np.unique(y)

    if not np.all(np.isin(unique_labels, [0, 1])):
        raise RuntimeError(
            f"Unexpected labels for patient {patient}: "
            f"{unique_labels}"
        )

    true_label = int(np.any(y == 1))

    f = patient_features(
        p,
        WINDOW_THRESHOLD
    )

    features[patient] = {
        "true_label": true_label,
        **f,
    }

    print(
        f"{patient:8s} "
        f"true={true_label} "
        f"windows={f['windows']:4d} "
        f"positive={f['positive_windows']:3d} "
        f"fraction={f['positive_fraction']:.4f} "
        f"runs={f['runs']:3d} "
        f"max_run={f['max_run']:2d}"
    )


# ============================================================
# BASELINE
# ============================================================

print()
print("=" * 70)
print("6. BASELINE Q95 RULE")
print("=" * 70)

baseline_pred = []

baseline_true = []

for patient in patient_ids:

    mask = patients == patient

    p = probabilities[mask]

    q95 = float(np.percentile(p, 95))

    pred = int(q95 >= 0.50)

    baseline_pred.append(pred)
    baseline_true.append(features[patient]["true_label"])

baseline_metrics = metrics(
    baseline_true,
    baseline_pred
)

print(json.dumps(
    baseline_metrics,
    indent=2
))


# ============================================================
# SEARCH
# ============================================================

print()
print("=" * 70)
print("7. TEMPORAL PERSISTENCE RULE SEARCH")
print("=" * 70)

print()
print("IMPORTANT:")
print("This search is performed ONLY on Validation.")
print("Test data is NOT used.")
print()
print(
    "The search tests whether requiring temporal persistence "
    "can remove the Validation false positive while preserving "
    f"sensitivity >= {REQUIRED_SENSITIVITY:.2f}."
)


candidates = []

candidate_id = 0

for fraction_threshold in FRACTION_THRESHOLDS:

    for min_runs in MIN_RUN_THRESHOLDS:

        for max_run in MAX_RUN_THRESHOLDS:

            for min_cluster_size in MIN_POSITIVE_WINDOWS:

                for max_gap in MAX_GAPS:

                    candidate_id += 1

                    y_true = []
                    y_pred = []

                    for patient in patient_ids:

                        f = features[patient]

                        pred = rule_prediction(
                            f,
                            fraction_threshold,
                            min_runs,
                            max_run,
                            min_cluster_size,
                            max_gap,
                        )

                        y_true.append(
                            f["true_label"]
                        )

                        y_pred.append(pred)

                    m = metrics(
                        y_true,
                        y_pred
                    )

                    if m["sensitivity"] >= REQUIRED_SENSITIVITY:

                        candidates.append({
                            "candidate_id": candidate_id,
                            "fraction_threshold": fraction_threshold,
                            "min_runs": min_runs,
                            "max_run": max_run,
                            "min_cluster_size": min_cluster_size,
                            "max_gap": max_gap,
                            "metrics": m,
                        })


print()
print("=" * 70)
print("8. SEARCH SUMMARY")
print("=" * 70)

total_candidates = candidate_id

print(
    f"Total candidates: "
    f"{total_candidates}"
)

print(
    f"Safe candidates: "
    f"{len(candidates)}"
)


# ============================================================
# ZERO FP
# ============================================================

zero_fp = [
    c
    for c in candidates
    if c["metrics"]["fp"] == 0
]


print()
print("=" * 70)
print("9. ZERO-FP CHECK")
print("=" * 70)

print(
    f"Zero-FP safe candidates: "
    f"{len(zero_fp)}"
)


# ============================================================
# SORT
# ============================================================

# Prefer:
# 1. zero FP
# 2. sensitivity
# 3. specificity
# 4. F1
# 5. precision
#
# If zero-FP is impossible, preserve sensitivity and maximize
# specificity/F1.

ranked = sorted(
    candidates,
    key=lambda c: (
        c["metrics"]["fp"] == 0,
        c["metrics"]["specificity"],
        c["metrics"]["f1"],
        c["metrics"]["precision"],
        c["metrics"]["sensitivity"],
    ),
    reverse=True,
)


# ============================================================
# PRINT TOP
# ============================================================

print()
print("=" * 70)
print("10. TOP VALIDATION PERSISTENCE RULES")
print("=" * 70)

print()

header = (
    "RULE PARAMETERS".ljust(75)
    + "TP FP FN  SENS   SPEC   PREC    F1"
)

print(header)
print("-" * len(header))

for c in ranked[:40]:

    m = c["metrics"]

    max_run_text = (
        "ANY"
        if c["max_run"] is None
        else str(c["max_run"])
    )

    rule_text = (
        f"fraction>={c['fraction_threshold']:.3f} "
        f"runs>={c['min_runs']} "
        f"maxrun<={max_run_text} "
        f"cluster>={c['min_cluster_size']} "
        f"gap<={c['max_gap']}"
    )

    print(
        f"{rule_text.ljust(75)}"
        f"{m['tp']:2d} "
        f"{m['fp']:2d} "
        f"{m['fn']:2d} "
        f"{m['sensitivity']:.4f} "
        f"{m['specificity']:.4f} "
        f"{m['precision']:.4f} "
        f"{m['f1']:.4f}"
    )


# ============================================================
# BEST RULE
# ============================================================

print()
print("=" * 70)
print("11. BEST VALIDATION PERSISTENCE RULE")
print("=" * 70)


if not ranked:

    print()
    print(
        "[WARNING] No candidate maintained "
        f"sensitivity >= {REQUIRED_SENSITIVITY:.2f}."
    )

    best_rule = None

else:

    best_rule = ranked[0]

    print()
    print("Best rule parameters:")

    print(
        f"Minimum positive fraction : "
        f"{best_rule['fraction_threshold']:.3f}"
    )

    print(
        f"Minimum number of runs    : "
        f"{best_rule['min_runs']}"
    )

    print(
        "Maximum run length        : "
        + (
            "ANY"
            if best_rule["max_run"] is None
            else str(best_rule["max_run"])
        )
    )

    print(
        f"Minimum cluster size      : "
        f"{best_rule['min_cluster_size']}"
    )

    print(
        f"Maximum gap               : "
        f"{best_rule['max_gap']}"
    )

    print()
    print("Metrics:")

    print(
        json.dumps(
            best_rule["metrics"],
            indent=2
        )
    )


# ============================================================
# PATIENT-BY-PATIENT
# ============================================================

print()
print("=" * 70)
print("12. PATIENT-BY-PATIENT ANALYSIS")
print("=" * 70)

patient_predictions = {}

if best_rule is not None:

    for patient in patient_ids:

        f = features[patient]

        pred = rule_prediction(
            f,
            best_rule["fraction_threshold"],
            best_rule["min_runs"],
            best_rule["max_run"],
            best_rule["min_cluster_size"],
            best_rule["max_gap"],
        )

        patient_predictions[patient] = {
            "true_label": f["true_label"],
            "prediction": pred,
            "positive_fraction": f["positive_fraction"],
            "positive_windows": f["positive_windows"],
            "runs": f["runs"],
            "max_run": f["max_run"],
            "mean_run": f["mean_run"],
            "cluster_max_size": (
                f["cluster_info"]
                [best_rule["max_gap"]]
                ["max_size"]
            ),
        }

        print(
            f"{patient:8s} "
            f"true={f['true_label']} "
            f"pred={pred} "
            f"fraction={f['positive_fraction']:.4f} "
            f"runs={f['runs']} "
            f"max_run={f['max_run']} "
            f"cluster_max="
            f"{patient_predictions[patient]['cluster_max_size']}"
        )


# ============================================================
# ZERO FP INTERPRETATION
# ============================================================

print()
print("=" * 70)
print("13. INTERPRETATION")
print("=" * 70)

if zero_fp:

    print()
    print(
        "[SUCCESS]"
    )

    print(
        "At least one Validation rule achieved:"
    )

    print(
        f"  sensitivity >= {REQUIRED_SENSITIVITY:.2f}"
    )

    print(
        "  FP = 0"
    )

    print()
    print(
        "This does NOT mean the rule should automatically "
        "be applied to Test."
    )

    print(
        "The next step is to freeze the selected Validation "
        "rule and evaluate it once on Test."
    )

else:

    print()
    print(
        "[INFO]"
    )

    print(
        "No zero-FP persistence rule was found on Validation "
        "while maintaining the required sensitivity."
    )

    print()
    print(
        "Therefore we should NOT invent a stronger rule "
        "using Test data."
    )


# ============================================================
# SAVE
# ============================================================

print()
print("=" * 70)
print("14. SAVING RESULTS")
print("=" * 70)


result = {
    "analysis": "validation_persistence_rule_search",
    "project_directory": str(PROJECT_DIR),
    "window_threshold": WINDOW_THRESHOLD,
    "required_sensitivity": REQUIRED_SENSITIVITY,

    "search_space": {
        "fraction_thresholds": FRACTION_THRESHOLDS,
        "min_runs": MIN_RUN_THRESHOLDS,
        "max_run": MAX_RUN_THRESHOLDS,
        "min_cluster_size": MIN_POSITIVE_WINDOWS,
        "max_gap": MAX_GAPS,
    },

    "baseline": baseline_metrics,

    "total_candidates": total_candidates,

    "safe_candidates": len(candidates),

    "zero_fp_candidates": len(zero_fp),

    "best_rule": best_rule,

    "patient_predictions": patient_predictions,

    "patient_features": features,

    "test_used": False,

    "test_optimization": False,

    "model_modified": False,

    "dataset_modified": False,

    "validation_threshold_modified": False,
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
print("[OK] Results saved:")
print(OUTPUT_PATH)

print()
print("=" * 70)
print("VALIDATION TEMPORAL PERSISTENCE SEARCH COMPLETED")
print("=" * 70)

print()
print("No model was modified.")
print("No dataset was modified.")
print("Validation threshold was NOT modified.")
print("Test data was NOT used.")
print("No Test optimization was performed.")

if zero_fp:
    print()
    print(
        "IMPORTANT:"
    )
    print(
        "A zero-FP Validation persistence rule exists."
    )
    print(
        "Do NOT tune it further using Test."
    )
    print(
        "The next step is frozen Test evaluation."
    )
else:
    print()
    print(
        "IMPORTANT:"
    )
    print(
        "No zero-FP Validation persistence rule exists "
        "under the sensitivity constraint."
    )