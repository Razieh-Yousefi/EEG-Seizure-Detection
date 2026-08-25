import json
from pathlib import Path

import numpy as np


# ======================================================================
# PATHS
# ======================================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_DIR / "results"

PROB_FILE = RESULTS_DIR / "validation_window_probabilities.npz"
FEATURE_FILE = RESULTS_DIR / "validation_fp_feature_arrays.npz"
THRESHOLD_FILE = RESULTS_DIR / "validation_threshold_results.json"

OUTPUT_FILE = RESULTS_DIR / "validation_multifeature_filter_analysis.json"


# ======================================================================
# CONFIGURATION
# ======================================================================

BASE_THRESHOLD = 0.56
REQUIRED_SENSITIVITY = 0.90

FEATURES = [
    "global_rms",
    "global_ptp",
    "global_std",
    "global_variance",
    "line_length",
    "zero_crossing_rate",
]


# ======================================================================
# UTILITY FUNCTIONS
# ======================================================================

def calculate_metrics(y_true, y_pred):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

    if precision + sensitivity > 0:
        f1 = 2 * precision * sensitivity / (precision + sensitivity)
    else:
        f1 = 0.0

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


def print_separator():
    print("=" * 70)


# ======================================================================
# START
# ======================================================================

print()
print_separator()
print("VALIDATION-ONLY MULTI-FEATURE FP FILTER ANALYSIS")
print_separator()

print()
print("Project directory:")
print(PROJECT_DIR)

print()
print("Results directory:")
print(RESULTS_DIR)


# ======================================================================
# 1. CHECK FILES
# ======================================================================

print()
print_separator()
print("1. CHECKING INPUT FILES")
print_separator()

for path in [PROB_FILE, FEATURE_FILE, THRESHOLD_FILE]:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found:\n{path}")
    print(f"[OK] {path}")


# ======================================================================
# 2. LOAD VALIDATION PROBABILITIES
# ======================================================================

print()
print_separator()
print("2. LOADING VALIDATION PROBABILITIES")
print_separator()

prob_data = np.load(PROB_FILE)

probabilities = np.asarray(prob_data["probabilities"]).reshape(-1)
labels = np.asarray(prob_data["labels"]).reshape(-1)
validation_indices = np.asarray(
    prob_data["validation_indices"]
).reshape(-1)

if "patients" in prob_data:
    patients = np.asarray(prob_data["patients"]).reshape(-1)
else:
    patients = None

print()
print(f"Validation samples: {len(probabilities)}")
print(f"Probability shape: {probabilities.shape}")
print(f"Labels shape: {labels.shape}")
print(f"Indices shape: {validation_indices.shape}")

if patients is not None:
    print(f"Patients shape: {patients.shape}")


# ======================================================================
# 3. VERIFY VALIDATION DATA
# ======================================================================

print()
print_separator()
print("3. VERIFYING VALIDATION DATA")
print_separator()

if not (
    len(probabilities)
    == len(labels)
    == len(validation_indices)
):
    raise RuntimeError(
        "Validation probabilities, labels and indices are not aligned."
    )

if patients is not None and len(patients) != len(probabilities):
    raise RuntimeError("Patients array is not aligned.")

if not np.all(np.isfinite(probabilities)):
    raise RuntimeError("Validation probabilities contain non-finite values.")

print("[OK] Validation arrays aligned.")
print("[OK] Probabilities are finite.")


# ======================================================================
# 4. LOAD THRESHOLD
# ======================================================================

print()
print_separator()
print("4. LOADING VALIDATION THRESHOLD")
print_separator()

with open(THRESHOLD_FILE, "r", encoding="utf-8") as f:
    threshold_data = json.load(f)

stored_threshold = None

possible_keys = [
    "validation_threshold",
    "best_threshold",
    "threshold",
]

for key in possible_keys:
    if key in threshold_data:
        stored_threshold = float(threshold_data[key])
        break

if stored_threshold is None:
    raise RuntimeError(
        "Could not find validation threshold in JSON file."
    )

BASE_THRESHOLD = stored_threshold

print()
print("Validation threshold:")
print(BASE_THRESHOLD)


# ======================================================================
# 5. BASELINE
# ======================================================================

print()
print_separator()
print("5. BASELINE")
print_separator()

baseline_predictions = (
    probabilities >= BASE_THRESHOLD
).astype(int)

baseline_metrics = calculate_metrics(
    labels,
    baseline_predictions,
)

for key, value in baseline_metrics.items():
    if isinstance(value, float):
        print(f"{key}: {value:.6f}")
    else:
        print(f"{key}: {value}")


# ======================================================================
# 6. LOAD FEATURE ARRAYS
# ======================================================================

print()
print_separator()
print("6. LOADING FEATURE ARRAYS")
print_separator()

feature_data = np.load(FEATURE_FILE)

print()
print("Available arrays:")

for name in feature_data.files:
    print(
        f"{name:30s}: "
        f"shape={feature_data[name].shape}"
    )


# ======================================================================
# 7. IDENTIFY TP / FP INDICES
# ======================================================================

print()
print_separator()
print("7. IDENTIFYING VALIDATION TP / FP")
print_separator()

tp_mask = (
    (labels == 1)
    & (probabilities >= BASE_THRESHOLD)
)

fp_mask = (
    (labels == 0)
    & (probabilities >= BASE_THRESHOLD)
)

tp_positions = np.where(tp_mask)[0]
fp_positions = np.where(fp_mask)[0]

tp_global_indices = validation_indices[tp_positions]
fp_global_indices = validation_indices[fp_positions]

print()
print(f"Validation TP: {len(tp_positions)}")
print(f"Validation FP: {len(fp_positions)}")


# ======================================================================
# 8. MAP FEATURE ARRAYS CORRECTLY
# ======================================================================
#
# IMPORTANT:
#
# validation_fp_feature_arrays.npz contains TP/FP features only.
#
# Its tp_indices/fp_indices are GLOBAL dataset indices.
#
# The validation probability file also contains GLOBAL indices.
#
# We therefore construct a mapping:
#
# global_index -> validation_position
#
# and then place TP/FP features into arrays of length 3070.
#
# This fixes the previous IndexError where an index such as 3102
# was incorrectly used directly inside an array of length 3070.
# ======================================================================

print()
print_separator()
print("8. BUILDING FULL VALIDATION FEATURE ARRAYS")
print_separator()

feature_arrays = {}

validation_index_to_position = {
    int(global_idx): int(position)
    for position, global_idx in enumerate(validation_indices)
}


def build_full_feature(
    feature_name,
    tp_key,
    fp_key,
):
    full_array = np.full(
        len(validation_indices),
        np.nan,
        dtype=np.float64,
    )

    if tp_key not in feature_data.files:
        return None

    if fp_key not in feature_data.files:
        return None

    tp_values = np.asarray(
        feature_data[tp_key]
    ).reshape(-1)

    fp_values = np.asarray(
        feature_data[fp_key]
    ).reshape(-1)

    tp_indices_key = "tp_indices"
    fp_indices_key = "fp_indices"

    if tp_indices_key not in feature_data.files:
        raise RuntimeError(
            "tp_indices is missing from feature file."
        )

    if fp_indices_key not in feature_data.files:
        raise RuntimeError(
            "fp_indices is missing from feature file."
        )

    stored_tp_global = np.asarray(
        feature_data[tp_indices_key]
    ).reshape(-1)

    stored_fp_global = np.asarray(
        feature_data[fp_indices_key]
    ).reshape(-1)

    if len(stored_tp_global) != len(tp_values):
        raise RuntimeError(
            f"TP feature length mismatch for {feature_name}: "
            f"{len(stored_tp_global)} indices vs "
            f"{len(tp_values)} values."
        )

    if len(stored_fp_global) != len(fp_values):
        raise RuntimeError(
            f"FP feature length mismatch for {feature_name}: "
            f"{len(stored_fp_global)} indices vs "
            f"{len(fp_values)} values."
        )

    for global_idx, value in zip(
        stored_tp_global,
        tp_values,
    ):
        position = validation_index_to_position.get(
            int(global_idx)
        )

        if position is not None:
            full_array[position] = float(value)

    for global_idx, value in zip(
        stored_fp_global,
        fp_values,
    ):
        position = validation_index_to_position.get(
            int(global_idx)
        )

        if position is not None:
            full_array[position] = float(value)

    valid_count = np.sum(np.isfinite(full_array))

    if valid_count == 0:
        return None

    print(
        f"[OK] {feature_name:<20s} "
        f"mapped={valid_count}/{len(full_array)}"
    )

    return full_array


for feature in FEATURES:
    full = build_full_feature(
        feature,
        f"tp_{feature}",
        f"fp_{feature}",
    )

    if full is not None:
        feature_arrays[feature] = full


if len(feature_arrays) == 0:
    raise RuntimeError(
        "No compatible feature arrays could be mapped."
    )


# ======================================================================
# 9. VERIFY FEATURE COVERAGE
# ======================================================================

print()
print_separator()
print("9. FEATURE COVERAGE")
print_separator()

for feature, values in feature_arrays.items():
    finite_mask = np.isfinite(values)

    print()
    print(feature)
    print(
        f"  available: "
        f"{np.sum(finite_mask)}/{len(values)}"
    )

    tp_available = np.sum(
        finite_mask & tp_mask
    )

    fp_available = np.sum(
        finite_mask & fp_mask
    )

    print(
        f"  TP feature values: {tp_available}/{np.sum(tp_mask)}"
    )

    print(
        f"  FP feature values: {fp_available}/{np.sum(fp_mask)}"
    )


# ======================================================================
# 10. MULTI-FEATURE SEARCH
# ======================================================================

print()
print_separator()
print("10. MULTI-FEATURE FILTER SEARCH")
print_separator()

#
# Strategy:
#
# We search filters of the form:
#
#   Keep prediction if:
#
#       probability >= base threshold
#       AND
#       feature satisfies condition
#
# For each feature we test:
#
#   feature >= percentile
#   feature <= percentile
#
# We then test combinations of TWO features.
#
# The filter is evaluated ONLY on validation data.
#

candidate_percentiles = [
    1,
    2,
    5,
    10,
    15,
    20,
    25,
    30,
    40,
    50,
    60,
    70,
    75,
    80,
    85,
    90,
    95,
    98,
    99,
]


def evaluate_prediction_mask(pred_mask):
    return calculate_metrics(
        labels,
        pred_mask.astype(int),
    )


candidates = []


# ----------------------------------------------------------------------
# Single-feature candidates
# ----------------------------------------------------------------------

for feature_name, values in feature_arrays.items():

    finite = np.isfinite(values)

    if np.sum(finite) == 0:
        continue

    finite_values = values[finite]

    for percentile in candidate_percentiles:

        threshold = np.percentile(
            finite_values,
            percentile,
        )

        # --------------------------------------------------------------
        # Feature >= threshold
        # --------------------------------------------------------------

        keep_mask = (
            probabilities >= BASE_THRESHOLD
        )

        filter_mask = (
            (~finite)
            | (values >= threshold)
        )

        prediction_mask = (
            keep_mask
            & filter_mask
        )

        metrics = evaluate_prediction_mask(
            prediction_mask
        )

        fp_reduction = (
            (baseline_metrics["fp"] - metrics["fp"])
            / baseline_metrics["fp"]
            * 100
            if baseline_metrics["fp"] > 0
            else 0
        )

        candidate = {
            "type": "single_feature",
            "feature_1": feature_name,
            "direction_1": ">=",
            "threshold_1": float(threshold),
            "percentile_1": percentile,
            **metrics,
            "fp_reduction": fp_reduction,
        }

        candidates.append(candidate)

        # --------------------------------------------------------------
        # Feature <= threshold
        # --------------------------------------------------------------

        filter_mask = (
            (~finite)
            | (values <= threshold)
        )

        prediction_mask = (
            keep_mask
            & filter_mask
        )

        metrics = evaluate_prediction_mask(
            prediction_mask
        )

        fp_reduction = (
            (baseline_metrics["fp"] - metrics["fp"])
            / baseline_metrics["fp"]
            * 100
            if baseline_metrics["fp"] > 0
            else 0
        )

        candidate = {
            "type": "single_feature",
            "feature_1": feature_name,
            "direction_1": "<=",
            "threshold_1": float(threshold),
            "percentile_1": percentile,
            **metrics,
            "fp_reduction": fp_reduction,
        }

        candidates.append(candidate)


# ----------------------------------------------------------------------
# Two-feature combinations
# ----------------------------------------------------------------------

feature_names = list(feature_arrays.keys())

for i in range(len(feature_names)):

    for j in range(i + 1, len(feature_names)):

        feature_1 = feature_names[i]
        feature_2 = feature_names[j]

        values_1 = feature_arrays[feature_1]
        values_2 = feature_arrays[feature_2]

        finite_1 = np.isfinite(values_1)
        finite_2 = np.isfinite(values_2)

        if (
            np.sum(finite_1) == 0
            or np.sum(finite_2) == 0
        ):
            continue

        percentiles_1 = {
            p: np.percentile(
                values_1[finite_1],
                p,
            )
            for p in candidate_percentiles
        }

        percentiles_2 = {
            p: np.percentile(
                values_2[finite_2],
                p,
            )
            for p in candidate_percentiles
        }

        for p1 in candidate_percentiles:

            for p2 in candidate_percentiles:

                threshold_1 = percentiles_1[p1]
                threshold_2 = percentiles_2[p2]

                for direction_1 in [">=", "<="]:
                    for direction_2 in [">=", "<="]:

                        if direction_1 == ">=":
                            condition_1 = (
                                (~finite_1)
                                | (values_1 >= threshold_1)
                            )
                        else:
                            condition_1 = (
                                (~finite_1)
                                | (values_1 <= threshold_1)
                            )

                        if direction_2 == ">=":
                            condition_2 = (
                                (~finite_2)
                                | (values_2 >= threshold_2)
                            )
                        else:
                            condition_2 = (
                                (~finite_2)
                                | (values_2 <= threshold_2)
                            )

                        prediction_mask = (
                            (probabilities >= BASE_THRESHOLD)
                            & condition_1
                            & condition_2
                        )

                        metrics = evaluate_prediction_mask(
                            prediction_mask
                        )

                        fp_reduction = (
                            (
                                baseline_metrics["fp"]
                                - metrics["fp"]
                            )
                            / baseline_metrics["fp"]
                            * 100
                            if baseline_metrics["fp"] > 0
                            else 0
                        )

                        candidates.append({
                            "type": "two_feature",
                            "feature_1": feature_1,
                            "direction_1": direction_1,
                            "threshold_1": float(threshold_1),
                            "percentile_1": p1,
                            "feature_2": feature_2,
                            "direction_2": direction_2,
                            "threshold_2": float(threshold_2),
                            "percentile_2": p2,
                            **metrics,
                            "fp_reduction": fp_reduction,
                        })


# ======================================================================
# 11. SAFE CANDIDATES
# ======================================================================

print()
print_separator()
print("11. SAFE VALIDATION CANDIDATES")
print_separator()

safe_candidates = [
    c
    for c in candidates
    if c["sensitivity"] >= REQUIRED_SENSITIVITY
]


safe_candidates.sort(
    key=lambda x: (
        x["f1"],
        x["fp_reduction"],
    ),
    reverse=True,
)

print()
print(f"Required sensitivity: {REQUIRED_SENSITIVITY}")
print(f"Total candidates: {len(candidates)}")
print(f"Safe candidates: {len(safe_candidates)}")


print()
print(
    "TYPE | FEATURES | TP | FP | FN | "
    "SENS | PRECISION | F1 | FP REDUCTION"
)
print("-" * 120)


for candidate in safe_candidates[:20]:

    if candidate["type"] == "single_feature":

        feature_description = (
            f"{candidate['feature_1']} "
            f"{candidate['direction_1']} "
            f"{candidate['threshold_1']:.6g}"
        )

    else:

        feature_description = (
            f"{candidate['feature_1']} "
            f"{candidate['direction_1']} "
            f"{candidate['threshold_1']:.4g}"
            " AND "
            f"{candidate['feature_2']} "
            f"{candidate['direction_2']} "
            f"{candidate['threshold_2']:.4g}"
        )

    print(
        f"{candidate['type']:<14} | "
        f"{feature_description:<55} | "
        f"{candidate['tp']:3d} | "
        f"{candidate['fp']:3d} | "
        f"{candidate['fn']:3d} | "
        f"{candidate['sensitivity']:.4f} | "
        f"{candidate['precision']:.4f} | "
        f"{candidate['f1']:.4f} | "
        f"{candidate['fp_reduction']:.2f}%"
    )


# ======================================================================
# 12. BEST CANDIDATE
# ======================================================================

print()
print_separator()
print("12. BEST VALIDATION MULTI-FEATURE FILTER")
print_separator()

best_candidate = None

if safe_candidates:
    best_candidate = safe_candidates[0]

    print()
    print("BEST CANDIDATE FOUND:")

    for key, value in best_candidate.items():
        print(f"{key}: {value}")

else:
    print()
    print(
        "No multi-feature filter reached the required "
        f"sensitivity of {REQUIRED_SENSITIVITY}."
    )

    #
    # For research purposes, also report the best candidates
    # regardless of the sensitivity constraint.
    #

    ranked_all = sorted(
        candidates,
        key=lambda x: (
            x["f1"],
            x["sensitivity"],
            x["fp_reduction"],
        ),
        reverse=True,
    )

    print()
    print("TOP OVERALL CANDIDATES WITHOUT SAFETY CONSTRAINT:")

    for candidate in ranked_all[:10]:

        if candidate["type"] == "single_feature":
            description = (
                f"{candidate['feature_1']} "
                f"{candidate['direction_1']} "
                f"{candidate['threshold_1']:.6g}"
            )
        else:
            description = (
                f"{candidate['feature_1']} "
                f"{candidate['direction_1']} "
                f"{candidate['threshold_1']:.4g}"
                " AND "
                f"{candidate['feature_2']} "
                f"{candidate['direction_2']} "
                f"{candidate['threshold_2']:.4g}"
            )

        print(
            f"{candidate['type']:<14} "
            f"{description:<60} "
            f"Sens={candidate['sensitivity']:.4f} "
            f"FP={candidate['fp']} "
            f"F1={candidate['f1']:.4f}"
        )


# ======================================================================
# 13. SAVE RESULTS
# ======================================================================

print()
print_separator()
print("13. SAVING RESULTS")
print_separator()

output = {
    "analysis": "validation_only_multi_feature_fp_filter",
    "project_directory": str(PROJECT_DIR),
    "base_threshold": float(BASE_THRESHOLD),
    "required_sensitivity": float(REQUIRED_SENSITIVITY),

    "validation_samples": int(len(probabilities)),

    "baseline": baseline_metrics,

    "feature_names": list(feature_arrays.keys()),

    "total_candidates": len(candidates),

    "safe_candidate_count": len(safe_candidates),

    "safe_candidates": safe_candidates[:100],

    "best_candidate": best_candidate,

    "note": (
        "This analysis used validation data only. "
        "No model, dataset, validation threshold, "
        "or test predictions were modified."
    ),
}


with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        output,
        f,
        indent=2,
    )


print()
print("[OK] Results saved:")
print(OUTPUT_FILE)

print()
print("No model or dataset was modified.")
print("Test data was NOT used.")
print("This analysis is VALIDATION-ONLY.")

print()
print_separator()
print("VALIDATION MULTI-FEATURE FILTER ANALYSIS COMPLETED")
print_separator()