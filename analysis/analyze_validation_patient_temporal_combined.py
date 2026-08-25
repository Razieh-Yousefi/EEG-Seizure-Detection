import json
from pathlib import Path

import numpy as np


# ============================================================
# VALIDATION PATIENT TEMPORAL COMBINED RULE SEARCH
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_DIR / "results"

PROB_FILE = RESULTS_DIR / "validation_window_probabilities.npz"
THRESHOLD_FILE = RESULTS_DIR / "validation_threshold_results.json"

OUTPUT_FILE = RESULTS_DIR / "validation_patient_temporal_combined.json"


print("=" * 70)
print("VALIDATION PATIENT TEMPORAL COMBINED RULE SEARCH")
print("=" * 70)

print()
print("Project directory:")
print(PROJECT_DIR)

print()
print("Results directory:")
print(RESULTS_DIR)


# ============================================================
# 1. CHECK INPUT FILES
# ============================================================

print()
print("=" * 70)
print("1. CHECKING INPUT FILES")
print("=" * 70)

if not PROB_FILE.exists():
    raise FileNotFoundError(PROB_FILE)

if not THRESHOLD_FILE.exists():
    raise FileNotFoundError(THRESHOLD_FILE)

print(f"[OK] {PROB_FILE}")
print(f"[OK] {THRESHOLD_FILE}")


# ============================================================
# 2. LOAD VALIDATION DATA
# ============================================================

print()
print("=" * 70)
print("2. LOADING VALIDATION DATA")
print("=" * 70)

data = np.load(PROB_FILE, allow_pickle=True)

print()
print("Available NPZ arrays:")
for key in data.files:
    print(f"  {key:30s} shape={data[key].shape}")


def get_array(names):
    for name in names:
        if name in data.files:
            return data[name]
    return None


probabilities = get_array([
    "probabilities",
    "probs",
    "window_probabilities",
])

labels = get_array([
    "labels",
    "y",
    "window_labels",
    "targets",
])

patients = get_array([
    "patients",
    "patient_ids",
    "patient",
    "patient_names",
])

indices = get_array([
    "indices",
    "validation_indices",
    "window_indices",
    "sample_indices",
    "original_indices",
])


if probabilities is None:
    raise RuntimeError(
        f"Probability array not found. Available keys: {data.files}"
    )

if labels is None:
    raise RuntimeError(
        f"Label array not found. Available keys: {data.files}"
    )

if patients is None:
    raise RuntimeError(
        f"Patient array not found. Available keys: {data.files}"
    )


probabilities = np.asarray(probabilities, dtype=float)
labels = np.asarray(labels, dtype=int)
patients = np.asarray(patients)

if indices is None:
    print()
    print("[INFO] No index array found.")
    print("[INFO] Using sequential validation indices.")
    indices = np.arange(len(probabilities), dtype=int)
else:
    indices = np.asarray(indices, dtype=int)


print()
print("Validation samples:", len(probabilities))
print("Probabilities shape:", probabilities.shape)
print("Labels shape:", labels.shape)
print("Indices shape:", indices.shape)
print("Patients shape:", patients.shape)


# ============================================================
# 3. VERIFY DATA
# ============================================================

print()
print("=" * 70)
print("3. VERIFYING VALIDATION DATA")
print("=" * 70)

n = len(probabilities)

if len(labels) != n or len(patients) != n or len(indices) != n:
    raise RuntimeError("Validation arrays are not aligned.")

if not np.all(np.isfinite(probabilities)):
    raise RuntimeError("Probabilities contain non-finite values.")

print("[OK] Arrays aligned.")
print("[OK] Probabilities finite.")


# ============================================================
# 4. LOAD VALIDATION THRESHOLD
# ============================================================

print()
print("=" * 70)
print("4. LOADING VALIDATION THRESHOLD")
print("=" * 70)

with open(THRESHOLD_FILE, "r", encoding="utf-8") as f:
    threshold_data = json.load(f)


def find_threshold(obj):
    if isinstance(obj, dict):
        for key in [
            "threshold",
            "validation_threshold",
            "best_threshold",
        ]:
            if key in obj:
                try:
                    return float(obj[key])
                except Exception:
                    pass

        for value in obj.values():
            result = find_threshold(value)
            if result is not None:
                return result

    return None


WINDOW_THRESHOLD = find_threshold(threshold_data)

if WINDOW_THRESHOLD is None:
    raise RuntimeError("Could not find validation threshold.")

print(f"Window threshold: {WINDOW_THRESHOLD:.6f}")


# ============================================================
# 5. HELPER FUNCTIONS
# ============================================================

def count_positive_runs(binary):
    binary = np.asarray(binary, dtype=int)

    if len(binary) == 0:
        return 0

    starts = np.sum(
        (binary == 1)
        & (
            np.concatenate(([0], binary[:-1]))
            == 0
        )
    )

    return int(starts)


def maximum_positive_run(binary):
    binary = np.asarray(binary, dtype=int)

    max_run = 0
    current = 0

    for value in binary:
        if value == 1:
            current += 1
            max_run = max(max_run, current)
        else:
            current = 0

    return int(max_run)


def calculate_metrics(predictions, true_labels):
    predictions = np.asarray(predictions, dtype=int)
    true_labels = np.asarray(true_labels, dtype=int)

    tp = int(np.sum((predictions == 1) & (true_labels == 1)))
    fp = int(np.sum((predictions == 1) & (true_labels == 0)))
    fn = int(np.sum((predictions == 0) & (true_labels == 1)))
    tn = int(np.sum((predictions == 0) & (true_labels == 0)))

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
        2 * precision * sensitivity / (precision + sensitivity)
        if (precision + sensitivity) > 0
        else 0.0
    )

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "precision": float(precision),
        "f1": float(f1),
    }


# ============================================================
# 6. BUILD PATIENT STATISTICS
# ============================================================

print()
print("=" * 70)
print("6. BUILDING PATIENT STATISTICS")
print("=" * 70)

window_predictions = (
    probabilities >= WINDOW_THRESHOLD
).astype(int)

unique_patients = np.unique(patients)

patient_stats = []

for patient in unique_patients:

    mask = patients == patient

    p = probabilities[mask]
    y = labels[mask]
    pred = window_predictions[mask]

    patient_name = str(patient)

    true_label = int(np.any(y == 1))

    positive_windows = int(np.sum(pred == 1))
    total_windows = int(len(pred))

    positive_fraction = (
        positive_windows / total_windows
        if total_windows > 0
        else 0.0
    )

    q95 = float(np.quantile(p, 0.95))

    q90 = float(np.quantile(p, 0.90))
    q99 = float(np.quantile(p, 0.99))

    positive_runs = count_positive_runs(pred)
    max_positive_run = maximum_positive_run(pred)

    mean_run_length = (
        positive_windows / positive_runs
        if positive_runs > 0
        else 0.0
    )

    # --------------------------------------------------------
    # Long-run fraction
    # --------------------------------------------------------

    long_run_2 = int(max_positive_run >= 2)
    long_run_3 = int(max_positive_run >= 3)
    long_run_4 = int(max_positive_run >= 4)
    long_run_5 = int(max_positive_run >= 5)

    # --------------------------------------------------------
    # Temporal density
    # --------------------------------------------------------

    if total_windows > 1:
        transitions = np.sum(
            pred[1:] != pred[:-1]
        )
    else:
        transitions = 0

    transition_rate = (
        transitions / (total_windows - 1)
        if total_windows > 1
        else 0.0
    )

    patient_stats.append({
        "patient": patient_name,
        "total_windows": total_windows,
        "positive_windows": positive_windows,
        "positive_fraction": float(positive_fraction),
        "positive_runs": int(positive_runs),
        "max_positive_run": int(max_positive_run),
        "mean_positive_run_length": float(mean_run_length),
        "q90": q90,
        "q95": q95,
        "q99": q99,
        "max_probability": float(np.max(p)),
        "mean_probability": float(np.mean(p)),
        "median_probability": float(np.median(p)),
        "transition_rate": float(transition_rate),
        "long_run_2": long_run_2,
        "long_run_3": long_run_3,
        "long_run_4": long_run_4,
        "long_run_5": long_run_5,
        "true_label": true_label,
    })


for s in patient_stats:
    print(
        f"{s['patient']:8s} "
        f"windows={s['total_windows']:4d} "
        f"positive={s['positive_windows']:3d} "
        f"fraction={s['positive_fraction']:.4f} "
        f"runs={s['positive_runs']:3d} "
        f"max_run={s['max_positive_run']:2d} "
        f"mean_run={s['mean_positive_run_length']:.2f} "
        f"Q95={s['q95']:.6f} "
        f"true={s['true_label']}"
    )


# ============================================================
# 7. BASELINE Q95
# ============================================================

print()
print("=" * 70)
print("7. BASELINE Q95")
print("=" * 70)

baseline_predictions = np.array([
    int(s["q95"] >= 0.50)
    for s in patient_stats
])

true_patient_labels = np.array([
    s["true_label"]
    for s in patient_stats
])

baseline_metrics = calculate_metrics(
    baseline_predictions,
    true_patient_labels,
)

print(json.dumps(
    baseline_metrics,
    indent=2
))

print("Rule: Q95 >= 0.50")


# ============================================================
# 8. COMBINED TEMPORAL RULE SEARCH
# ============================================================

print()
print("=" * 70)
print("8. COMBINED TEMPORAL RULE SEARCH")
print("=" * 70)

required_sensitivity = 0.90

q95_thresholds = [
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
    0.95,
]

positive_fraction_thresholds = [
    0.005,
    0.010,
    0.015,
    0.020,
    0.030,
    0.050,
    0.075,
    0.100,
    0.125,
    0.150,
    0.200,
]

positive_run_thresholds = [
    1,
    2,
    3,
    4,
    5,
    6,
]

positive_runs_thresholds = [
    1,
    2,
    3,
    5,
    10,
    15,
    20,
    30,
]

candidates = []


# ============================================================
# RULE FAMILY A
# Q95 + positive fraction
# ============================================================

for q95_t in q95_thresholds:

    for fraction_t in positive_fraction_thresholds:

        predictions = np.array([
            int(
                s["q95"] >= q95_t
                and s["positive_fraction"] >= fraction_t
            )
            for s in patient_stats
        ])

        metrics = calculate_metrics(
            predictions,
            true_patient_labels,
        )

        candidates.append({
            "type": "q95_positive_fraction",
            "q95_threshold": q95_t,
            "positive_fraction_threshold": fraction_t,
            "metrics": metrics,
        })


# ============================================================
# RULE FAMILY B
# Q95 + max positive run
# ============================================================

for q95_t in q95_thresholds:

    for run_t in positive_run_thresholds:

        predictions = np.array([
            int(
                s["q95"] >= q95_t
                and s["max_positive_run"] >= run_t
            )
            for s in patient_stats
        ])

        metrics = calculate_metrics(
            predictions,
            true_patient_labels,
        )

        candidates.append({
            "type": "q95_max_positive_run",
            "q95_threshold": q95_t,
            "max_positive_run_threshold": run_t,
            "metrics": metrics,
        })


# ============================================================
# RULE FAMILY C
# Q95 + number of positive runs
# ============================================================

for q95_t in q95_thresholds:

    for runs_t in positive_runs_thresholds:

        predictions = np.array([
            int(
                s["q95"] >= q95_t
                and s["positive_runs"] >= runs_t
            )
            for s in patient_stats
        ])

        metrics = calculate_metrics(
            predictions,
            true_patient_labels,
        )

        candidates.append({
            "type": "q95_positive_runs",
            "q95_threshold": q95_t,
            "positive_runs_threshold": runs_t,
            "metrics": metrics,
        })


# ============================================================
# RULE FAMILY D
# Q95 + fraction + max run
# ============================================================

for q95_t in q95_thresholds:

    for fraction_t in positive_fraction_thresholds:

        for run_t in positive_run_thresholds:

            predictions = np.array([
                int(
                    s["q95"] >= q95_t
                    and s["positive_fraction"] >= fraction_t
                    and s["max_positive_run"] >= run_t
                )
                for s in patient_stats
            ])

            metrics = calculate_metrics(
                predictions,
                true_patient_labels,
            )

            candidates.append({
                "type": "q95_fraction_maxrun",
                "q95_threshold": q95_t,
                "positive_fraction_threshold": fraction_t,
                "max_positive_run_threshold": run_t,
                "metrics": metrics,
            })


# ============================================================
# RULE FAMILY E
# Q95 + fraction + number of runs
# ============================================================

for q95_t in q95_thresholds:

    for fraction_t in positive_fraction_thresholds:

        for runs_t in positive_runs_thresholds:

            predictions = np.array([
                int(
                    s["q95"] >= q95_t
                    and s["positive_fraction"] >= fraction_t
                    and s["positive_runs"] >= runs_t
                )
                for s in patient_stats
            ])

            metrics = calculate_metrics(
                predictions,
                true_patient_labels,
            )

            candidates.append({
                "type": "q95_fraction_runs",
                "q95_threshold": q95_t,
                "positive_fraction_threshold": fraction_t,
                "positive_runs_threshold": runs_t,
                "metrics": metrics,
            })


# ============================================================
# RULE FAMILY F
# Q95 + max run + number of runs
# ============================================================

for q95_t in q95_thresholds:

    for run_t in positive_run_thresholds:

        for runs_t in positive_runs_thresholds:

            predictions = np.array([
                int(
                    s["q95"] >= q95_t
                    and s["max_positive_run"] >= run_t
                    and s["positive_runs"] >= runs_t
                )
                for s in patient_stats
            ])

            metrics = calculate_metrics(
                predictions,
                true_patient_labels,
            )

            candidates.append({
                "type": "q95_maxrun_runs",
                "q95_threshold": q95_t,
                "max_positive_run_threshold": run_t,
                "positive_runs_threshold": runs_t,
                "metrics": metrics,
            })


print()
print("Total candidates:", len(candidates))


# ============================================================
# 9. SAFE CANDIDATES
# ============================================================

safe_candidates = [
    c
    for c in candidates
    if c["metrics"]["sensitivity"] >= required_sensitivity
]


print()
print("=" * 70)
print("9. SAFE VALIDATION CANDIDATES")
print("=" * 70)

print(
    f"Required sensitivity: {required_sensitivity:.2f}"
)

print(
    f"Safe candidates: {len(safe_candidates)}"
)


# ============================================================
# SORT SAFE CANDIDATES
# ============================================================

safe_candidates_sorted = sorted(
    safe_candidates,
    key=lambda c: (
        c["metrics"]["f1"],
        c["metrics"]["precision"],
        -c["metrics"]["fp"],
    ),
    reverse=True,
)


print()
print(
    "TYPE | PARAMETERS | TP | FP | FN | "
    "SENS | PREC | F1"
)

print("-" * 120)


def parameter_string(c):

    parts = []

    if "q95_threshold" in c:
        parts.append(
            f"Q95 >= {c['q95_threshold']:.2f}"
        )

    if "positive_fraction_threshold" in c:
        parts.append(
            f"fraction >= "
            f"{c['positive_fraction_threshold']:.3f}"
        )

    if "max_positive_run_threshold" in c:
        parts.append(
            f"max_run >= "
            f"{c['max_positive_run_threshold']}"
        )

    if "positive_runs_threshold" in c:
        parts.append(
            f"positive_runs >= "
            f"{c['positive_runs_threshold']}"
        )

    return " AND ".join(parts)


for c in safe_candidates_sorted[:30]:

    m = c["metrics"]

    print(
        f"{c['type']:27s} | "
        f"{parameter_string(c):60s} | "
        f"{m['tp']:2d} | "
        f"{m['fp']:2d} | "
        f"{m['fn']:2d} | "
        f"{m['sensitivity']:.4f} | "
        f"{m['precision']:.4f} | "
        f"{m['f1']:.4f}"
    )


# ============================================================
# 10. BEST RULE
# ============================================================

print()
print("=" * 70)
print("10. BEST VALIDATION TEMPORAL COMBINED RULE")
print("=" * 70)


if len(safe_candidates_sorted) == 0:

    print()
    print(
        "No combined temporal rule reached "
        "the required sensitivity."
    )

    best_candidate = None

else:

    best_candidate = safe_candidates_sorted[0]

    print("Best rule:")
    print(
        "type:",
        best_candidate["type"]
    )

    print(
        "parameters:",
        parameter_string(best_candidate)
    )

    print(
        "metrics:"
    )

    print(
        json.dumps(
            best_candidate["metrics"],
            indent=2
        )
    )


# ============================================================
# 11. PATIENT-BY-PATIENT ANALYSIS
# ============================================================

print()
print("=" * 70)
print("11. PATIENT-BY-PATIENT ANALYSIS")
print("=" * 70)


if best_candidate is not None:

    for s in patient_stats:

        prediction = int(
            (
                s["q95"]
                >= best_candidate["q95_threshold"]
            )
        )

        if (
            best_candidate["type"]
            == "q95_positive_fraction"
        ):
            prediction = int(
                s["q95"]
                >= best_candidate["q95_threshold"]
                and
                s["positive_fraction"]
                >= best_candidate[
                    "positive_fraction_threshold"
                ]
            )

        elif (
            best_candidate["type"]
            == "q95_max_positive_run"
        ):
            prediction = int(
                s["q95"]
                >= best_candidate["q95_threshold"]
                and
                s["max_positive_run"]
                >= best_candidate[
                    "max_positive_run_threshold"
                ]
            )

        elif (
            best_candidate["type"]
            == "q95_positive_runs"
        ):
            prediction = int(
                s["q95"]
                >= best_candidate["q95_threshold"]
                and
                s["positive_runs"]
                >= best_candidate[
                    "positive_runs_threshold"
                ]
            )

        elif (
            best_candidate["type"]
            == "q95_fraction_maxrun"
        ):
            prediction = int(
                s["q95"]
                >= best_candidate["q95_threshold"]
                and
                s["positive_fraction"]
                >= best_candidate[
                    "positive_fraction_threshold"
                ]
                and
                s["max_positive_run"]
                >= best_candidate[
                    "max_positive_run_threshold"
                ]
            )

        elif (
            best_candidate["type"]
            == "q95_fraction_runs"
        ):
            prediction = int(
                s["q95"]
                >= best_candidate["q95_threshold"]
                and
                s["positive_fraction"]
                >= best_candidate[
                    "positive_fraction_threshold"
                ]
                and
                s["positive_runs"]
                >= best_candidate[
                    "positive_runs_threshold"
                ]
            )

        elif (
            best_candidate["type"]
            == "q95_maxrun_runs"
        ):
            prediction = int(
                s["q95"]
                >= best_candidate["q95_threshold"]
                and
                s["max_positive_run"]
                >= best_candidate[
                    "max_positive_run_threshold"
                ]
                and
                s["positive_runs"]
                >= best_candidate[
                    "positive_runs_threshold"
                ]
            )

        print()
        print(s["patient"])
        print(
            f"  true_label: {s['true_label']}"
        )
        print(
            f"  prediction: {prediction}"
        )
        print(
            f"  q95: {s['q95']:.6f}"
        )
        print(
            f"  positive_fraction: "
            f"{s['positive_fraction']:.6f}"
        )
        print(
            f"  positive_windows: "
            f"{s['positive_windows']}"
        )
        print(
            f"  positive_runs: "
            f"{s['positive_runs']}"
        )
        print(
            f"  max_positive_run: "
            f"{s['max_positive_run']}"
        )
        print(
            f"  transition_rate: "
            f"{s['transition_rate']:.6f}"
        )


# ============================================================
# 12. SAVE RESULTS
# ============================================================

print()
print("=" * 70)
print("12. SAVING RESULTS")
print("=" * 70)


output = {
    "analysis": "validation_patient_temporal_combined",
    "validation_threshold": float(WINDOW_THRESHOLD),
    "required_sensitivity": required_sensitivity,
    "patient_statistics": patient_stats,
    "baseline_q95": {
        "rule": "Q95 >= 0.50",
        "metrics": baseline_metrics,
    },
    "total_candidates": len(candidates),
    "safe_candidates": len(safe_candidates),
    "best_candidate": best_candidate,
}


with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        output,
        f,
        indent=2,
        ensure_ascii=False,
    )


print(f"[OK] Results saved:")
print(OUTPUT_FILE)

print()
print("No model was modified.")
print("No dataset was modified.")
print("Validation threshold was NOT modified.")
print("Test data was NOT used.")
print("No Test optimization was performed.")

print()
print("=" * 70)
print("VALIDATION PATIENT TEMPORAL COMBINED ANALYSIS COMPLETED")
print("=" * 70)