import json
from pathlib import Path

import numpy as np


# ======================================================================
# CONFIG
# ======================================================================

PROJECT_DIR = Path(r"C:\Users\rezay\Desktop\EEG_Seizure_Project")
RESULTS_DIR = PROJECT_DIR / "results"

TEST_NPZ = RESULTS_DIR / "test_window_probabilities.npz"
THRESHOLD_JSON = RESULTS_DIR / "validation_threshold_results.json"
FINAL_REPORT_JSON = RESULTS_DIR / "final_test_patient_level_report.json"

OUTPUT_JSON = RESULTS_DIR / "test_fp_window_localization.json"


# ======================================================================
# HELPERS
# ======================================================================

def print_header(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def compute_runs(binary_array):
    """
    Return lengths of consecutive positive runs.
    """
    x = np.asarray(binary_array, dtype=bool)

    if len(x) == 0:
        return []

    runs = []
    current = 0

    for value in x:
        if value:
            current += 1
        else:
            if current > 0:
                runs.append(current)
                current = 0

    if current > 0:
        runs.append(current)

    return runs


def safe_float(x):
    return float(x) if np.isfinite(x) else None


# ======================================================================
# START
# ======================================================================

print_header("TEST FALSE-POSITIVE WINDOW LOCALIZATION")

print("\nProject directory:")
print(PROJECT_DIR)

print("\nResults directory:")
print(RESULTS_DIR)


# ======================================================================
# 1. CHECK INPUTS
# ======================================================================

print_header("1. CHECKING INPUT FILES")

for path in [TEST_NPZ, THRESHOLD_JSON, FINAL_REPORT_JSON]:
    if path.exists():
        print(f"[OK] {path}")
    else:
        print(f"[ERROR] Missing: {path}")
        raise FileNotFoundError(path)


# ======================================================================
# 2. LOAD TEST DATA
# ======================================================================

print_header("2. LOADING TEST DATA")

data = np.load(TEST_NPZ, allow_pickle=True)

print("\nAvailable NPZ arrays:")
for key in data.files:
    print(f"  {key:30s} shape={data[key].shape}")

required = ["probabilities", "patients", "labels"]

for key in required:
    if key not in data.files:
        raise KeyError(f"Missing required array: {key}")

probabilities = np.asarray(data["probabilities"], dtype=float)
patients = np.asarray(data["patients"]).astype(str)
labels = np.asarray(data["labels"], dtype=int)

if "test_indices" in data.files:
    indices = np.asarray(data["test_indices"], dtype=int)
else:
    print("[INFO] No test_indices array found.")
    print("[INFO] Using sequential sample indices.")
    indices = np.arange(len(probabilities), dtype=int)

print("\nTest samples:", len(probabilities))
print("Probabilities shape:", probabilities.shape)
print("Patients shape     :", patients.shape)
print("Labels shape       :", labels.shape)
print("Indices shape      :", indices.shape)


# ======================================================================
# 3. VERIFY DATA
# ======================================================================

print_header("3. VERIFYING DATA")

if not (
    len(probabilities)
    == len(patients)
    == len(labels)
    == len(indices)
):
    raise ValueError("Arrays are not aligned.")

print("[OK] Arrays aligned.")

if not np.all(np.isfinite(probabilities)):
    raise ValueError("Probabilities contain non-finite values.")

print("[OK] Probabilities finite.")


# ======================================================================
# 4. LOAD FROZEN VALIDATION THRESHOLD
# ======================================================================

print_header("4. LOADING FROZEN VALIDATION THRESHOLD")

with open(THRESHOLD_JSON, "r", encoding="utf-8") as f:
    threshold_data = json.load(f)


def find_threshold(obj):
    """
    Recursively search common threshold keys.
    """
    if isinstance(obj, dict):
        for key in [
            "window_threshold",
            "threshold",
            "best_threshold",
            "optimal_threshold",
        ]:
            if key in obj:
                value = obj[key]
                if isinstance(value, (int, float)):
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


WINDOW_THRESHOLD = find_threshold(threshold_data)

if WINDOW_THRESHOLD is None:
    raise ValueError("Could not find validation threshold.")

print(f"Frozen validation threshold: {WINDOW_THRESHOLD:.6f}")

print("\nIMPORTANT:")
print("This threshold comes from Validation.")
print("No Test threshold optimization is performed.")


# ======================================================================
# 5. IDENTIFY TEST FALSE-POSITIVE PATIENTS
# ======================================================================

print_header("5. IDENTIFYING TEST FALSE-POSITIVE PATIENTS")

unique_patients = np.unique(patients)

patient_info = {}

for patient in unique_patients:

    mask = patients == patient

    p = probabilities[mask]
    y = labels[mask]
    idx = indices[mask]

    # Sort by sample/test index
    order = np.argsort(idx)

    p = p[order]
    y = y[order]
    idx = idx[order]

    true_label = int(np.max(y))

    positive_mask = p >= WINDOW_THRESHOLD

    positive_count = int(np.sum(positive_mask))
    total_count = len(p)

    fraction = positive_count / total_count if total_count else 0.0

    q95 = float(np.percentile(p, 95))

    prediction = int(q95 >= 0.50)

    if true_label == 0 and prediction == 1:
        patient_info[patient] = {
            "patient": patient,
            "true_label": true_label,
            "probabilities": p,
            "labels": y,
            "indices": idx,
            "positive_mask": positive_mask,
            "positive_count": positive_count,
            "total_count": total_count,
            "fraction": fraction,
            "q95": q95,
        }

        print(
            f"[FALSE POSITIVE] {patient} "
            f"windows={total_count} "
            f"positive={positive_count} "
            f"fraction={fraction:.4f} "
            f"Q95={q95:.6f}"
        )

fp_patients = sorted(patient_info.keys())

print("\nTotal Test false-positive patients:", len(fp_patients))

if not fp_patients:
    print("[INFO] No false-positive patients found.")
    raise SystemExit(0)


# ======================================================================
# 6. LOCALIZE POSITIVE WINDOWS
# ======================================================================

print_header("6. LOCALIZING FALSE-POSITIVE WINDOWS")

all_fp_windows = []

patient_results = {}

for patient in fp_patients:

    info = patient_info[patient]

    p = info["probabilities"]
    idx = info["indices"]
    positive_mask = info["positive_mask"]

    positive_positions = np.where(positive_mask)[0]

    positive_probs = p[positive_mask]
    positive_indices = idx[positive_mask]

    runs = compute_runs(positive_mask)

    run_details = []

    start = None
    length = 0

    for i, flag in enumerate(positive_mask):

        if flag and start is None:
            start = i
            length = 1

        elif flag and start is not None:
            length += 1

        elif not flag and start is not None:

            end = i - 1

            run_probs = p[start:end + 1]
            run_indices = idx[start:end + 1]

            run_details.append({
                "start_position": int(start),
                "end_position": int(end),
                "length": int(length),
                "start_index": int(run_indices[0]),
                "end_index": int(run_indices[-1]),
                "max_probability": float(np.max(run_probs)),
                "mean_probability": float(np.mean(run_probs)),
            })

            start = None
            length = 0

    if start is not None:

        end = len(positive_mask) - 1

        run_probs = p[start:end + 1]
        run_indices = idx[start:end + 1]

        run_details.append({
            "start_position": int(start),
            "end_position": int(end),
            "length": int(end - start + 1),
            "start_index": int(run_indices[0]),
            "end_index": int(run_indices[-1]),
            "max_probability": float(np.max(run_probs)),
            "mean_probability": float(np.mean(run_probs)),
        })

    patient_windows = []

    for pos, sample_idx, prob in zip(
        positive_positions,
        positive_indices,
        positive_probs,
    ):

        record = {
            "patient": patient,
            "position": int(pos),
            "index": int(sample_idx),
            "probability": float(prob),
        }

        patient_windows.append(record)
        all_fp_windows.append(record)

    patient_results[patient] = {
        "patient": patient,
        "true_label": int(info["true_label"]),
        "total_windows": int(info["total_count"]),
        "positive_windows": int(info["positive_count"]),
        "positive_fraction": float(info["fraction"]),
        "q95": float(info["q95"]),
        "positive_window_details": patient_windows,
        "positive_runs": run_details,
    }


# ======================================================================
# 7. PRINT WINDOW DETAILS
# ======================================================================

print_header("7. FALSE-POSITIVE WINDOW DETAILS")

for patient in fp_patients:

    result = patient_results[patient]

    print(f"\n{patient}")
    print("-" * 70)

    print(
        f"Q95                 : "
        f"{result['q95']:.6f}"
    )

    print(
        f"Positive fraction   : "
        f"{result['positive_fraction']:.6f}"
    )

    print(
        f"Positive windows    : "
        f"{result['positive_windows']}"
    )

    print(
        f"Positive runs       : "
        f"{len(result['positive_runs'])}"
    )

    print("\nPositive runs:")

    for j, run in enumerate(result["positive_runs"], start=1):

        print(
            f"  Run {j:3d}: "
            f"index {run['start_index']} -> "
            f"{run['end_index']} | "
            f"length={run['length']} | "
            f"max={run['max_probability']:.6f} | "
            f"mean={run['mean_probability']:.6f}"
        )


# ======================================================================
# 8. TOP FALSE-POSITIVE WINDOWS
# ======================================================================

print_header("8. TOP FALSE-POSITIVE WINDOWS")

sorted_fp_windows = sorted(
    all_fp_windows,
    key=lambda x: x["probability"],
    reverse=True,
)

print("\nTop 30 positive windows across all FP patients:")

print(
    f"{'RANK':>4} | "
    f"{'PATIENT':>8} | "
    f"{'INDEX':>8} | "
    f"{'PROBABILITY':>12}"
)

print("-" * 45)

for rank, record in enumerate(sorted_fp_windows[:30], start=1):

    print(
        f"{rank:4d} | "
        f"{record['patient']:>8} | "
        f"{record['index']:8d} | "
        f"{record['probability']:12.6f}"
    )


# ======================================================================
# 9. PROBABILITY DISTRIBUTION OF FP WINDOWS
# ======================================================================

print_header("9. FALSE-POSITIVE PROBABILITY DISTRIBUTION")

fp_probs = np.asarray(
    [x["probability"] for x in all_fp_windows],
    dtype=float,
)

distribution = {
    "count": int(len(fp_probs)),
    "mean": float(np.mean(fp_probs)),
    "median": float(np.median(fp_probs)),
    "std": float(np.std(fp_probs)),
    "min": float(np.min(fp_probs)),
    "max": float(np.max(fp_probs)),
    "q25": float(np.percentile(fp_probs, 25)),
    "q75": float(np.percentile(fp_probs, 75)),
    "q90": float(np.percentile(fp_probs, 90)),
    "q95": float(np.percentile(fp_probs, 95)),
    "q99": float(np.percentile(fp_probs, 99)),
}

for key, value in distribution.items():
    print(f"{key:>8}: {value}")


# ======================================================================
# 10. RUN LENGTH SUMMARY
# ======================================================================

print_header("10. FALSE-POSITIVE RUN-LENGTH SUMMARY")

all_runs = []

for patient in fp_patients:
    for run in patient_results[patient]["positive_runs"]:
        all_runs.append(run["length"])

all_runs = np.asarray(all_runs, dtype=int)

run_summary = {
    "number_of_runs": int(len(all_runs)),
    "mean_run_length": float(np.mean(all_runs)),
    "median_run_length": float(np.median(all_runs)),
    "max_run_length": int(np.max(all_runs)),
    "single_window_runs": int(np.sum(all_runs == 1)),
    "two_window_runs": int(np.sum(all_runs == 2)),
    "three_or_more_window_runs": int(np.sum(all_runs >= 3)),
}

for key, value in run_summary.items():
    print(f"{key:>28}: {value}")


# ======================================================================
# 11. PATIENT COMPARISON
# ======================================================================

print_header("11. FALSE-POSITIVE PATIENT COMPARISON")

print(
    f"{'PATIENT':>8} | "
    f"{'Q95':>8} | "
    f"{'FRACTION':>10} | "
    f"{'POS':>5} | "
    f"{'RUNS':>5} | "
    f"{'MAXRUN':>6} | "
    f"{'MEAN_POS':>10}"
)

print("-" * 75)

for patient in fp_patients:

    result = patient_results[patient]

    runs = result["positive_runs"]

    run_lengths = [x["length"] for x in runs]

    max_run = max(run_lengths) if run_lengths else 0

    mean_pos = float(
        np.mean(
            [
                x["probability"]
                for x in result["positive_window_details"]
            ]
        )
    )

    print(
        f"{patient:>8} | "
        f"{result['q95']:8.4f} | "
        f"{result['positive_fraction']:10.4f} | "
        f"{result['positive_windows']:5d} | "
        f"{len(runs):5d} | "
        f"{max_run:6d} | "
        f"{mean_pos:10.4f}"
    )


# ======================================================================
# 12. DIAGNOSTIC INTERPRETATION
# ======================================================================

print_header("12. DIAGNOSTIC INTERPRETATION")

print(
    """
This analysis is diagnostic only.

No threshold is changed.
No patient-level rule is changed.
No Test optimization is performed.

The purpose is to determine whether Test false positives
are concentrated in short isolated positive windows or
whether they contain longer temporal clusters.
"""
)

for patient in fp_patients:

    result = patient_results[patient]

    run_lengths = [
        x["length"]
        for x in result["positive_runs"]
    ]

    single_fraction = (
        sum(x == 1 for x in run_lengths) / len(run_lengths)
        if run_lengths
        else 0.0
    )

    print(
        f"{patient}: "
        f"{single_fraction * 100:.1f}% "
        f"of positive runs are single-window runs."
    )


# ======================================================================
# 13. SAVE RESULTS
# ======================================================================

print_header("13. SAVING RESULTS")

output = {
    "analysis": "test_false_positive_window_localization",
    "window_threshold": float(WINDOW_THRESHOLD),
    "test_optimization_performed": False,
    "rule_modified": False,
    "false_positive_patients": fp_patients,
    "false_positive_window_count": int(len(all_fp_windows)),
    "probability_distribution": distribution,
    "run_length_summary": run_summary,
    "patients": patient_results,
}

with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(
        output,
        f,
        indent=2,
        ensure_ascii=False,
    )

print(f"[OK] Results saved:")
print(OUTPUT_JSON)


# ======================================================================
# FINAL
# ======================================================================

print_header("TEST FALSE-POSITIVE WINDOW LOCALIZATION COMPLETED")

print("\nNo model was modified.")
print("No dataset was modified.")
print("Validation threshold was NOT modified.")
print("No Test optimization was performed.")
print("No Test rule search was performed.")

print("\nNext step:")
print("Use the saved localization results to determine whether")
print("false positives have a consistent temporal pattern.")