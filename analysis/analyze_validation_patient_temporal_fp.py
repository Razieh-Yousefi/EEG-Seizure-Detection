import json
from pathlib import Path

import numpy as np


# ============================================================
# CONFIG
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_DIR / "results"

PROB_FILE = RESULTS_DIR / "validation_window_probabilities.npz"
THRESHOLD_FILE = RESULTS_DIR / "validation_threshold_results.json"

OUTPUT_FILE = RESULTS_DIR / "validation_patient_temporal_fp_analysis.json"

REQUIRED_SENSITIVITY = 0.90


# ============================================================
# HELPERS
# ============================================================

def safe_float(x):
    return float(x) if np.isfinite(x) else None


def metrics(tp, fp, fn, tn):
    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = (
        2 * precision * sensitivity / (precision + sensitivity)
        if (precision + sensitivity)
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


def evaluate_patient_rule(patient_stats, rule_name, predicate):
    tp = fp = fn = tn = 0

    predictions = []

    for patient, stats in patient_stats.items():
        pred = bool(predicate(stats))
        true = bool(stats["true_label"])

        if pred and true:
            tp += 1
        elif pred and not true:
            fp += 1
        elif not pred and true:
            fn += 1
        else:
            tn += 1

        predictions.append({
            "patient": patient,
            "true_label": int(true),
            "prediction": int(pred),
        })

    result = metrics(tp, fp, fn, tn)
    result["rule"] = rule_name
    result["predictions"] = predictions

    return result


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("VALIDATION PATIENT TEMPORAL FP ANALYSIS")
print("=" * 70)

print()
print("Project directory:")
print(PROJECT_DIR)

print()
print("Results directory:")
print(RESULTS_DIR)


# ============================================================
# 1. CHECK INPUTS
# ============================================================

print()
print("=" * 70)
print("1. CHECKING INPUT FILES")
print("=" * 70)

if not PROB_FILE.exists():
    raise FileNotFoundError(PROB_FILE)

if not THRESHOLD_FILE.exists():
    raise FileNotFoundError(THRESHOLD_FILE)

print("[OK]", PROB_FILE)
print("[OK]", THRESHOLD_FILE)


# ============================================================
# 2. LOAD DATA
# ============================================================

print()
print("=" * 70)
print("2. LOADING VALIDATION DATA")
print("=" * 70)

data = np.load(PROB_FILE, allow_pickle=True)

print("Available NPZ arrays:")
for key in data.files:
    print(f"  {key:30s} shape={data[key].shape}")

def get_first_existing(data, names):
    for name in names:
        if name in data.files:
            return data[name]
    return None


probabilities = get_first_existing(
    data,
    ["probabilities", "probs", "window_probabilities", "predictions"]
)

labels = get_first_existing(
    data,
    ["labels", "y", "window_labels", "targets"]
)

patients = get_first_existing(
    data,
    ["patients", "patient_ids", "patient", "patient_names"]
)

indices = get_first_existing(
    data,
    ["indices", "window_indices", "sample_indices", "original_indices"]
)


if probabilities is None:
    raise RuntimeError(
        "Could not find probability array in validation NPZ. "
        f"Available keys: {data.files}"
    )

if labels is None:
    raise RuntimeError(
        "Could not find label array in validation NPZ. "
        f"Available keys: {data.files}"
    )

if patients is None:
    raise RuntimeError(
        "Could not find patient array in validation NPZ. "
        f"Available keys: {data.files}"
    )


probabilities = np.asarray(probabilities, dtype=float)
labels = np.asarray(labels, dtype=int)
patients = np.asarray(patients)

if indices is None:
    print()
    print("[INFO] No indices array found.")
    print("[INFO] Using sequential sample indices.")

    indices = np.arange(len(probabilities), dtype=int)
else:
    indices = np.asarray(indices, dtype=int)

print()
print("Resolved arrays:")
print("  probabilities:", probabilities.shape)
print("  labels       :", labels.shape)
print("  patients     :", patients.shape)
print("  indices      :", indices.shape)

print("Validation samples:", len(probabilities))
print("Probabilities shape:", probabilities.shape)
print("Labels shape:", labels.shape)
print("Indices shape:", indices.shape)
print("Patients shape:", patients.shape)


# ============================================================
# 3. VERIFY
# ============================================================

print()
print("=" * 70)
print("3. VERIFYING DATA")
print("=" * 70)

n = len(probabilities)

if not (
    len(labels) == n
    and len(indices) == n
    and len(patients) == n
):
    raise RuntimeError("Validation arrays are not aligned.")

if not np.all(np.isfinite(probabilities)):
    raise RuntimeError("Probabilities contain non-finite values.")

print("[OK] Arrays aligned.")
print("[OK] Probabilities finite.")


# ============================================================
# 4. LOAD WINDOW THRESHOLD
# ============================================================

print()
print("=" * 70)
print("4. LOADING VALIDATION THRESHOLD")
print("=" * 70)

with open(THRESHOLD_FILE, "r", encoding="utf-8") as f:
    threshold_results = json.load(f)

threshold = float(
    threshold_results.get(
        "best_threshold",
        threshold_results.get(
            "validation_threshold",
            threshold_results.get("threshold")
        )
    )
)

print(f"Window threshold: {threshold:.6f}")


# ============================================================
# 5. WINDOW PREDICTIONS
# ============================================================

window_pred = probabilities >= threshold


# ============================================================
# 6. BUILD PATIENT STATISTICS
# ============================================================

print()
print("=" * 70)
print("5. BUILDING PATIENT STATISTICS")
print("=" * 70)

patient_stats = {}

for patient in np.unique(patients):

    mask = patients == patient

    p = probabilities[mask]
    y = labels[mask]
    pred = window_pred[mask]

    true_label = int(np.any(y == 1))

    positive_windows = int(np.sum(pred))
    total_windows = int(len(p))

    q50 = float(np.quantile(p, 0.50))
    q75 = float(np.quantile(p, 0.75))
    q80 = float(np.quantile(p, 0.80))
    q85 = float(np.quantile(p, 0.85))
    q90 = float(np.quantile(p, 0.90))
    q95 = float(np.quantile(p, 0.95))
    q99 = float(np.quantile(p, 0.99))

    # Longest consecutive run of positive windows
    max_run = 0
    current_run = 0

    for value in pred:
        if value:
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            current_run = 0

    # Fraction of windows predicted positive
    positive_fraction = (
        positive_windows / total_windows
        if total_windows
        else 0.0
    )

    # Number of positive runs
    positive_runs = 0
    previous = False

    for value in pred:
        if value and not previous:
            positive_runs += 1
        previous = bool(value)

    patient_stats[str(patient)] = {
        "total_windows": total_windows,
        "positive_windows": positive_windows,
        "positive_fraction": float(positive_fraction),
        "positive_runs": int(positive_runs),
        "max_positive_run": int(max_run),
        "true_label": true_label,

        "q50": q50,
        "q75": q75,
        "q80": q80,
        "q85": q85,
        "q90": q90,
        "q95": q95,
        "q99": q99,

        "max_probability": float(np.max(p)),
        "mean_probability": float(np.mean(p)),
        "median_probability": float(np.median(p)),
    }

    print(
        f"{str(patient):8s} "
        f"windows={total_windows:4d} "
        f"positive={positive_windows:3d} "
        f"fraction={positive_fraction:.4f} "
        f"runs={positive_runs:3d} "
        f"max_run={max_run:3d} "
        f"Q95={q95:.6f} "
        f"true={true_label}"
    )


# ============================================================
# 7. BASELINE Q95
# ============================================================

print()
print("=" * 70)
print("6. BASELINE Q95")
print("=" * 70)

baseline = evaluate_patient_rule(
    patient_stats,
    "Q95 >= 0.50",
    lambda s: s["q95"] >= 0.50,
)

print(json.dumps(baseline, indent=2))


# ============================================================
# 8. TEMPORAL RULE SEARCH
# ============================================================

print()
print("=" * 70)
print("7. TEMPORAL RULE SEARCH")
print("=" * 70)

candidates = []


# ------------------------------------------------------------
# A. Q95 + minimum positive fraction
# ------------------------------------------------------------

q95_thresholds = [
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
]

fraction_thresholds = [
    0.005,
    0.010,
    0.015,
    0.020,
    0.030,
    0.050,
    0.075,
    0.100,
    0.150,
]

for q95_t in q95_thresholds:

    for frac_t in fraction_thresholds:

        name = (
            f"Q95 >= {q95_t:.2f} AND "
            f"positive_fraction >= {frac_t:.3f}"
        )

        result = evaluate_patient_rule(
            patient_stats,
            name,
            lambda s, q=q95_t, f=frac_t:
                s["q95"] >= q and
                s["positive_fraction"] >= f,
        )

        result["type"] = "q95_positive_fraction"
        result["q95_threshold"] = q95_t
        result["positive_fraction_threshold"] = frac_t

        candidates.append(result)


# ------------------------------------------------------------
# B. Q95 + minimum positive windows
# ------------------------------------------------------------

minimum_windows_values = [
    1,
    3,
    5,
    10,
    15,
    20,
    30,
    50,
]

for q95_t in q95_thresholds:

    for min_windows in minimum_windows_values:

        name = (
            f"Q95 >= {q95_t:.2f} AND "
            f"positive_windows >= {min_windows}"
        )

        result = evaluate_patient_rule(
            patient_stats,
            name,
            lambda s, q=q95_t, m=min_windows:
                s["q95"] >= q and
                s["positive_windows"] >= m,
        )

        result["type"] = "q95_min_positive_windows"
        result["q95_threshold"] = q95_t
        result["minimum_positive_windows"] = min_windows

        candidates.append(result)


# ------------------------------------------------------------
# C. Q95 + maximum consecutive run
# ------------------------------------------------------------

max_run_values = [
    1,
    2,
    3,
    5,
    10,
    15,
    20,
]

for q95_t in q95_thresholds:

    for max_run_t in max_run_values:

        name = (
            f"Q95 >= {q95_t:.2f} AND "
            f"max_positive_run >= {max_run_t}"
        )

        result = evaluate_patient_rule(
            patient_stats,
            name,
            lambda s, q=q95_t, m=max_run_t:
                s["q95"] >= q and
                s["max_positive_run"] >= m,
        )

        result["type"] = "q95_max_positive_run"
        result["q95_threshold"] = q95_t
        result["minimum_max_positive_run"] = max_run_t

        candidates.append(result)


# ------------------------------------------------------------
# D. Q95 + number of positive runs
# ------------------------------------------------------------

run_values = [
    1,
    2,
    3,
    5,
    10,
    20,
]

for q95_t in q95_thresholds:

    for runs_t in run_values:

        name = (
            f"Q95 >= {q95_t:.2f} AND "
            f"positive_runs >= {runs_t}"
        )

        result = evaluate_patient_rule(
            patient_stats,
            name,
            lambda s, q=q95_t, r=runs_t:
                s["q95"] >= q and
                s["positive_runs"] >= r,
        )

        result["type"] = "q95_positive_runs"
        result["q95_threshold"] = q95_t
        result["minimum_positive_runs"] = runs_t

        candidates.append(result)


# ============================================================
# 9. SAFE CANDIDATES
# ============================================================

print()
print("=" * 70)
print("8. SAFE VALIDATION CANDIDATES")
print("=" * 70)

safe = [
    x for x in candidates
    if x["sensitivity"] >= REQUIRED_SENSITIVITY
]

print("Required sensitivity:", REQUIRED_SENSITIVITY)
print("Total candidates:", len(candidates))
print("Safe candidates:", len(safe))


safe_sorted = sorted(
    safe,
    key=lambda x: (
        x["f1"],
        x["precision"],
        -x["fp"],
    ),
    reverse=True,
)


print()
print(
    "TYPE | RULE | TP | FP | FN | SENS | PREC | F1"
)
print("-" * 120)

for result in safe_sorted[:20]:

    print(
        f"{result['type']:28s} | "
        f"{result['rule'][:55]:55s} | "
        f"{result['tp']:2d} | "
        f"{result['fp']:2d} | "
        f"{result['fn']:2d} | "
        f"{result['sensitivity']:.4f} | "
        f"{result['precision']:.4f} | "
        f"{result['f1']:.4f}"
    )


# ============================================================
# 10. BEST RULE
# ============================================================

print()
print("=" * 70)
print("9. BEST VALIDATION TEMPORAL RULE")
print("=" * 70)

if safe_sorted:

    best = safe_sorted[0]

    print("Best rule:")
    print("type:", best["type"])
    print("rule:", best["rule"])
    print("TP:", best["tp"])
    print("FP:", best["fp"])
    print("FN:", best["fn"])
    print("TN:", best["tn"])
    print("Sensitivity:", best["sensitivity"])
    print("Specificity:", best["specificity"])
    print("Precision:", best["precision"])
    print("F1:", best["f1"])

else:

    best = None

    print(
        "No temporal rule reached the required "
        f"sensitivity of {REQUIRED_SENSITIVITY}."
    )


# ============================================================
# 11. PATIENT-BY-PATIENT ANALYSIS
# ============================================================

print()
print("=" * 70)
print("10. PATIENT-BY-PATIENT TEMPORAL STATISTICS")
print("=" * 70)

for patient, stats in patient_stats.items():

    print()
    print(patient)

    print("  true_label:", stats["true_label"])
    print("  q95:", f"{stats['q95']:.6f}")
    print(
        "  positive_fraction:",
        f"{stats['positive_fraction']:.6f}"
    )
    print(
        "  positive_windows:",
        stats["positive_windows"]
    )
    print(
        "  positive_runs:",
        stats["positive_runs"]
    )
    print(
        "  max_positive_run:",
        stats["max_positive_run"]
    )


# ============================================================
# 12. SAVE
# ============================================================

print()
print("=" * 70)
print("11. SAVING RESULTS")
print("=" * 70)

output = {
    "analysis": "validation_patient_temporal_fp_analysis",

    "validation_threshold": threshold,

    "required_sensitivity": REQUIRED_SENSITIVITY,

    "baseline_q95": baseline,

    "patient_statistics": patient_stats,

    "total_candidates": len(candidates),

    "safe_candidates": safe_sorted,

    "best_candidate": best,

    "protocol": {
        "model_modified": False,
        "dataset_modified": False,
        "threshold_modified": False,
        "test_data_used": False,
        "test_optimization_performed": False,
    },
}

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(
        output,
        f,
        indent=2,
        ensure_ascii=False,
    )

print("[OK] Results saved:")
print(OUTPUT_FILE)

print()
print("=" * 70)
print("VALIDATION PATIENT TEMPORAL FP ANALYSIS COMPLETED")
print("=" * 70)

print()
print("No model was modified.")
print("No dataset was modified.")
print("Validation threshold was NOT modified.")
print("Test data was NOT used.")
print("No Test optimization was performed.")