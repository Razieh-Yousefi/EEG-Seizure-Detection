# ================================================================
# analyze_validation_feature_filters.py
# ================================================================

import os
import json
import numpy as np


# ================================================================
# PATHS
# ================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")

FEATURE_FILE = os.path.join(
    RESULTS_DIR,
    "validation_fp_feature_arrays.npz"
)

PROB_FILE = os.path.join(
    RESULTS_DIR,
    "validation_window_probabilities.npz"
)

THRESHOLD_FILE = os.path.join(
    RESULTS_DIR,
    "validation_threshold_results.json"
)

OUTPUT_FILE = os.path.join(
    RESULTS_DIR,
    "validation_feature_filter_analysis.json"
)


# ================================================================
# HEADER
# ================================================================

print()
print("=" * 70)
print("VALIDATION FEATURE-BASED FP FILTER ANALYSIS")
print("=" * 70)

print()
print("Project directory:")
print(PROJECT_DIR)

print()
print("Results directory:")
print(RESULTS_DIR)


# ================================================================
# CHECK FILES
# ================================================================

print()
print("=" * 70)
print("1. CHECKING INPUT FILES")
print("=" * 70)

required_files = [
    FEATURE_FILE,
    PROB_FILE,
    THRESHOLD_FILE,
]

for path in required_files:

    if os.path.exists(path):
        print("[OK]", path)

    else:
        raise FileNotFoundError(path)


# ================================================================
# LOAD FEATURE ARRAYS
# ================================================================

print()
print("=" * 70)
print("2. LOADING FEATURE ARRAYS")
print("=" * 70)

features = np.load(
    FEATURE_FILE,
    allow_pickle=True
)

print()
print("Available arrays:")

for key in features.files:

    print(
        f"{key:30s}: "
        f"shape={features[key].shape}"
    )


# ================================================================
# LOAD VALIDATION PROBABILITIES
# ================================================================

print()
print("=" * 70)
print("3. LOADING VALIDATION PROBABILITIES")
print("=" * 70)

prob_data = np.load(
    PROB_FILE,
    allow_pickle=True
)

probabilities = np.asarray(
    prob_data["probabilities"],
    dtype=np.float64
)

labels = np.asarray(
    prob_data["labels"],
    dtype=np.int64
)

indices = np.asarray(
    prob_data["validation_indices"],
    dtype=np.int64
)

patients = np.asarray(
    prob_data["patients"]
)

print()
print("Validation samples:", len(probabilities))
print("Probability shape:", probabilities.shape)
print("Labels shape:", labels.shape)
print("Indices shape:", indices.shape)
print("Patients shape:", patients.shape)


# ================================================================
# LOAD THRESHOLD
# ================================================================

print()
print("=" * 70)
print("4. LOADING VALIDATION THRESHOLD")
print("=" * 70)

with open(
    THRESHOLD_FILE,
    "r",
    encoding="utf-8"
) as f:

    threshold_data = json.load(f)


def find_threshold(obj):

    if isinstance(obj, dict):

        for key, value in obj.items():

            if (
                "threshold" in str(key).lower()
                and isinstance(value, (int, float))
            ):

                return float(value)

        for value in obj.values():

            result = find_threshold(value)

            if result is not None:
                return result

    return None


threshold = find_threshold(
    threshold_data
)

if threshold is None:

    raise RuntimeError(
        "Could not find validation threshold."
    )

print()
print("Validation threshold:")
print(threshold)


# ================================================================
# VERIFY DATA
# ================================================================

print()
print("=" * 70)
print("5. VERIFYING VALIDATION DATA")
print("=" * 70)

n = len(probabilities)

if len(labels) != n:
    raise RuntimeError("Labels are not aligned.")

if len(indices) != n:
    raise RuntimeError("Indices are not aligned.")

if len(patients) != n:
    raise RuntimeError("Patients are not aligned.")

if not np.all(np.isfinite(probabilities)):
    raise RuntimeError(
        "Validation probabilities contain NaN/Inf."
    )

print()
print("[OK] Validation arrays aligned.")
print("[OK] Probabilities are finite.")


# ================================================================
# IDENTIFY TP / FP
# ================================================================

print()
print("=" * 70)
print("6. IDENTIFYING VALIDATION TP / FP")
print("=" * 70)

predictions = (
    probabilities >= threshold
).astype(np.int64)

tp_mask = (
    (labels == 1)
    & (predictions == 1)
)

fp_mask = (
    (labels == 0)
    & (predictions == 1)
)

tp_positions = np.where(tp_mask)[0]
fp_positions = np.where(fp_mask)[0]

print()
print("Validation TP:", len(tp_positions))
print("Validation FP:", len(fp_positions))


# ================================================================
# FEATURE MAPPING
# ================================================================

print()
print("=" * 70)
print("7. DETECTING FEATURE ARRAYS")
print("=" * 70)

FEATURES = [
    (
        "global_rms",
        "tp_global_rms",
        "fp_global_rms"
    ),
    (
        "global_ptp",
        "tp_global_ptp",
        "fp_global_ptp"
    ),
    (
        "global_std",
        "tp_global_std",
        "fp_global_std"
    ),
    (
        "global_variance",
        "tp_global_variance",
        "fp_global_variance"
    ),
    (
        "line_length",
        "tp_line_length",
        "fp_line_length"
    ),
    (
        "zero_crossing_rate",
        "tp_zero_crossing_rate",
        "fp_zero_crossing_rate"
    ),
]


available_features = []

for feature_name, tp_name, fp_name in FEATURES:

    if (
        tp_name in features.files
        and fp_name in features.files
    ):

        tp_values = np.asarray(
            features[tp_name],
            dtype=np.float64
        )

        fp_values = np.asarray(
            features[fp_name],
            dtype=np.float64
        )

        if (
            len(tp_values) == len(tp_positions)
            and len(fp_values) == len(fp_positions)
        ):

            available_features.append(
                (
                    feature_name,
                    tp_values,
                    fp_values
                )
            )

            print(
                f"[OK] {feature_name}"
            )

        else:

            print(
                f"[SKIP] {feature_name} "
                f"(length mismatch)"
            )

    else:

        print(
            f"[MISSING] {feature_name}"
        )


if not available_features:

    raise RuntimeError(
        "No compatible global feature arrays found."
    )


# ================================================================
# METRICS
# ================================================================

def calculate_metrics(tp, fp, fn, tn):

    sensitivity = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0.0
    )

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0.0
    )

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0.0
    )

    f1 = (
        2 * precision * sensitivity
        / (precision + sensitivity)
        if (precision + sensitivity) > 0
        else 0.0
    )

    return {
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "precision": float(precision),
        "f1": float(f1),
    }


# ================================================================
# BASELINE
# ================================================================

print()
print("=" * 70)
print("8. BASELINE")
print("=" * 70)

tp = int(np.sum(
    (labels == 1)
    & (predictions == 1)
))

fp = int(np.sum(
    (labels == 0)
    & (predictions == 1)
))

fn = int(np.sum(
    (labels == 1)
    & (predictions == 0)
))

tn = int(np.sum(
    (labels == 0)
    & (predictions == 0)
))

baseline = calculate_metrics(
    tp,
    fp,
    fn,
    tn
)

print()

for key, value in baseline.items():

    if isinstance(value, float):
        print(
            f"{key}: {value:.6f}"
        )

    else:
        print(
            f"{key}: {value}"
        )


# ================================================================
# FEATURE STATISTICS
# ================================================================

print()
print("=" * 70)
print("9. FEATURE COMPARISON")
print("=" * 70)

feature_statistics = {}


for feature_name, tp_values, fp_values in available_features:

    tp_mean = float(np.mean(tp_values))
    fp_mean = float(np.mean(fp_values))

    tp_median = float(np.median(tp_values))
    fp_median = float(np.median(fp_values))

    tp_std = float(np.std(tp_values))
    fp_std = float(np.std(fp_values))

    ratio = (
        fp_mean / tp_mean
        if tp_mean != 0
        else np.nan
    )

    print()
    print(feature_name)

    print(
        f"  TP mean     : {tp_mean:.10f}"
    )

    print(
        f"  FP mean     : {fp_mean:.10f}"
    )

    print(
        f"  TP median   : {tp_median:.10f}"
    )

    print(
        f"  FP median   : {fp_median:.10f}"
    )

    print(
        f"  FP/TP ratio : {ratio:.6f}"
    )

    feature_statistics[feature_name] = {
        "tp_mean": tp_mean,
        "fp_mean": fp_mean,
        "tp_median": tp_median,
        "fp_median": fp_median,
        "tp_std": tp_std,
        "fp_std": fp_std,
        "fp_tp_mean_ratio": (
            float(ratio)
            if np.isfinite(ratio)
            else None
        ),
    }


# ================================================================
# BUILD FULL FEATURE ARRAYS
# ================================================================

print()
print("=" * 70)
print("10. BUILDING FULL VALIDATION FEATURE ARRAYS")
print("=" * 70)

full_features = {}

for feature_name, tp_values, fp_values in available_features:

    full_feature = np.full(
        n,
        np.nan,
        dtype=np.float64
    )

    full_feature[
        tp_positions
    ] = tp_values

    full_feature[
        fp_positions
    ] = fp_values

    full_features[
        feature_name
    ] = full_feature

    print(
        "[OK]",
        feature_name
    )


# ================================================================
# FEATURE FILTER SEARCH
# ================================================================

print()
print("=" * 70)
print("11. FEATURE FILTER SEARCH")
print("=" * 70)

results = []

for feature_name, tp_values, fp_values in available_features:

    full_feature = full_features[
        feature_name
    ]

    combined = np.concatenate(
        [
            tp_values,
            fp_values
        ]
    )

    combined = combined[
        np.isfinite(combined)
    ]

    if len(combined) == 0:
        continue

    thresholds = np.unique(
        np.quantile(
            combined,
            np.linspace(
                0.02,
                0.98,
                49
            )
        )
    )

    for feature_threshold in thresholds:

        for direction in [
            "reject_low",
            "reject_high"
        ]:

            filtered_predictions = (
                predictions.copy()
            )

            positive_positions = np.where(
                predictions == 1
            )[0]

            positive_features = (
                full_feature[
                    positive_positions
                ]
            )

            if direction == "reject_low":

                reject_mask = (
                    positive_features
                    <= feature_threshold
                )

            else:

                reject_mask = (
                    positive_features
                    >= feature_threshold
                )

            valid_reject = (
                reject_mask
                & np.isfinite(
                    positive_features
                )
            )

            filtered_predictions[
                positive_positions[
                    valid_reject
                ]
            ] = 0

            new_tp = int(np.sum(
                (labels == 1)
                & (filtered_predictions == 1)
            ))

            new_fp = int(np.sum(
                (labels == 0)
                & (filtered_predictions == 1)
            ))

            new_fn = int(np.sum(
                (labels == 1)
                & (filtered_predictions == 0)
            ))

            new_tn = int(np.sum(
                (labels == 0)
                & (filtered_predictions == 0)
            ))

            metrics = calculate_metrics(
                new_tp,
                new_fp,
                new_fn,
                new_tn
            )

            fp_reduction = (
                (fp - new_fp)
                / fp
                * 100
                if fp > 0
                else 0.0
            )

            results.append({
                "feature": feature_name,
                "direction": direction,
                "threshold": float(
                    feature_threshold
                ),
                "tp": new_tp,
                "fp": new_fp,
                "fn": new_fn,
                "tn": new_tn,
                "sensitivity": metrics[
                    "sensitivity"
                ],
                "specificity": metrics[
                    "specificity"
                ],
                "precision": metrics[
                    "precision"
                ],
                "f1": metrics[
                    "f1"
                ],
                "fp_reduction": float(
                    fp_reduction
                ),
            })


# ================================================================
# SAFE CANDIDATES
# ================================================================

print()
print("=" * 70)
print("12. SAFE CANDIDATES")
print("=" * 70)

required_sensitivity = 0.90

safe_candidates = [
    result
    for result in results
    if result["sensitivity"]
    >= required_sensitivity
]

safe_candidates.sort(
    key=lambda result: (
        result["f1"],
        result["fp_reduction"]
    ),
    reverse=True
)

print()
print(
    "Required sensitivity:",
    required_sensitivity
)

print(
    "Safe candidates:",
    len(safe_candidates)
)

print()
print(
    "FEATURE | DIRECTION | THRESHOLD | "
    "TP | FP | FN | SENS | PRECISION | F1 | FP REDUCTION"
)

print("-" * 105)

for candidate in safe_candidates[:20]:

    print(
        f"{candidate['feature']:20s} | "
        f"{candidate['direction']:11s} | "
        f"{candidate['threshold']:.8f} | "
        f"{candidate['tp']:3d} | "
        f"{candidate['fp']:3d} | "
        f"{candidate['fn']:3d} | "
        f"{candidate['sensitivity']:.4f} | "
        f"{candidate['precision']:.4f} | "
        f"{candidate['f1']:.4f} | "
        f"{candidate['fp_reduction']:.2f}%"
    )


# ================================================================
# BEST CANDIDATE
# ================================================================

print()
print("=" * 70)
print("13. BEST VALIDATION FEATURE FILTER")
print("=" * 70)

if safe_candidates:

    best = safe_candidates[0]

    print()
    print("Feature:", best["feature"])
    print("Direction:", best["direction"])
    print(
        "Threshold:",
        best["threshold"]
    )

    print()
    print("TP:", best["tp"])
    print("FP:", best["fp"])
    print("FN:", best["fn"])
    print("TN:", best["tn"])

    print()
    print(
        f"Sensitivity: "
        f"{best['sensitivity']:.6f}"
    )

    print(
        f"Specificity: "
        f"{best['specificity']:.6f}"
    )

    print(
        f"Precision: "
        f"{best['precision']:.6f}"
    )

    print(
        f"F1: "
        f"{best['f1']:.6f}"
    )

    print(
        f"FP reduction: "
        f"{best['fp_reduction']:.2f}%"
    )

else:

    best = None

    print()
    print(
        "No feature filter reached "
        "the required sensitivity."
    )


# ================================================================
# SAVE RESULTS
# ================================================================

print()
print("=" * 70)
print("14. SAVING RESULTS")
print("=" * 70)

output = {
    "baseline": baseline,
    "required_sensitivity": required_sensitivity,
    "feature_statistics": feature_statistics,
    "safe_candidates": safe_candidates[:100],
    "best_candidate": best,
}

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        output,
        f,
        indent=4
    )


print()
print("[OK] Results saved:")
print(OUTPUT_FILE)

print()
print("No model or dataset was modified.")
print("Test data was NOT used.")
print("This analysis is VALIDATION-ONLY.")

print()
print("=" * 70)
print("VALIDATION FEATURE FILTER ANALYSIS COMPLETED")
print("=" * 70)