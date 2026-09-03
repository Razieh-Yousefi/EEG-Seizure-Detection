# ================================================================
# plot_final_test_report.py
#
# FINAL TEST-SET PERFORMANCE VISUALIZATION
#
# IMPORTANT:
# - No optimization is performed.
# - No threshold is fitted.
# - Test set is used only for visualization/reporting.
# - Artifact rule was selected on validation only.
# ================================================================

import os
import json

import numpy as np
import matplotlib.pyplot as plt


# ================================================================
# 1. PROJECT PATHS
# ================================================================

PROJECT_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

RESULTS_DIR = os.path.join(
    PROJECT_DIR,
    "results"
)

REPORT_JSON = os.path.join(
    RESULTS_DIR,
    "FINAL_TEST_REPORT.json"
)

ARTIFACT_JSON = os.path.join(
    RESULTS_DIR,
    "final_test_artifact_rejection_evaluation.json"
)

EVENT_JSON = os.path.join(
    RESULTS_DIR,
    "final_test_patient_seizure_event_evaluation.json"
)

ARTIFACT_NPZ = os.path.join(
    RESULTS_DIR,
    "final_test_artifact_rejection_scores.npz"
)

OUTPUT_DIR = os.path.join(
    RESULTS_DIR,
    "final_test_figures"
)


# ================================================================
# 2. HEADER
# ================================================================

print()
print("=" * 72)
print("FINAL TEST-SET PERFORMANCE VISUALIZATION")
print("=" * 72)

print()
print("Project:")
print(PROJECT_DIR)

print()
print("Output directory:")
print(OUTPUT_DIR)


# ================================================================
# 3. CHECK REQUIRED FILES
# ================================================================

print()
print("=" * 72)
print("1. CHECKING REQUIRED FILES")
print("=" * 72)

required_files = [
    REPORT_JSON,
    ARTIFACT_JSON,
    EVENT_JSON,
    ARTIFACT_NPZ,
]

for path in required_files:

    if not os.path.exists(path):

        print(
            "[MISSING]",
            path
        )

        raise FileNotFoundError(
            path
        )

    print(
        "[OK]",
        path
    )


os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ================================================================
# 4. LOAD FINAL RESULTS
# ================================================================

print()
print("=" * 72)
print("2. LOADING FINAL RESULTS")
print("=" * 72)

with open(
    REPORT_JSON,
    "r",
    encoding="utf-8"
) as f:

    report = json.load(
        f
    )


with open(
    ARTIFACT_JSON,
    "r",
    encoding="utf-8"
) as f:

    artifact_report = json.load(
        f
    )


with open(
    EVENT_JSON,
    "r",
    encoding="utf-8"
) as f:

    event_report = json.load(
        f
    )


data = np.load(
    ARTIFACT_NPZ,
    allow_pickle=True
)


print()
print(
    "[OK] Reports loaded."
)


# ================================================================
# 5. HELPER
# ================================================================

def get_metric(
    dictionary,
    possible_names,
    required=True,
    default=None
):

    for name in possible_names:

        if name in dictionary:

            return dictionary[
                name
            ]

    if required:

        raise KeyError(
            "Could not find any of these keys: "
            + ", ".join(
                possible_names
            )
        )

    return default


# ================================================================
# 6. EXTRACT WINDOW-LEVEL RESULTS
# ================================================================

print()
print("=" * 72)
print("3. EXTRACTING WINDOW-LEVEL METRICS")
print("=" * 72)


window_section = report.get(
    "window_level",
    {}
)


baseline_window_raw = window_section.get(
    "baseline",
    {}
)


final_window_raw = window_section.get(
    "artifact_rejection",
    window_section.get(
        "final",
        {}
    )
)


if not baseline_window_raw:

    raise RuntimeError(
        "Baseline window-level results "
        "were not found in FINAL_TEST_REPORT.json"
    )


if not final_window_raw:

    raise RuntimeError(
        "Artifact-rejection window-level results "
        "were not found in FINAL_TEST_REPORT.json"
    )


baseline_window = {

    "TP":
        int(
            get_metric(
                baseline_window_raw,
                [
                    "TP",
                    "tp"
                ]
            )
        ),

    "FP":
        int(
            get_metric(
                baseline_window_raw,
                [
                    "FP",
                    "fp"
                ]
            )
        ),

    "TN":
        int(
            get_metric(
                baseline_window_raw,
                [
                    "TN",
                    "tn"
                ]
            )
        ),

    "FN":
        int(
            get_metric(
                baseline_window_raw,
                [
                    "FN",
                    "fn"
                ]
            )
        ),

    "Recall":
        float(
            get_metric(
                baseline_window_raw,
                [
                    "recall",
                    "Recall",
                    "sensitivity",
                    "Sensitivity"
                ]
            )
        ),

    "Specificity":
        float(
            get_metric(
                baseline_window_raw,
                [
                    "specificity",
                    "Specificity"
                ]
            )
        ),

    "Precision":
        float(
            get_metric(
                baseline_window_raw,
                [
                    "precision",
                    "Precision"
                ]
            )
        ),

    "F1":
        float(
            get_metric(
                baseline_window_raw,
                [
                    "F1",
                    "f1",
                    "f1_score"
                ]
            )
        ),
}


final_window = {

    "TP":
        int(
            get_metric(
                final_window_raw,
                [
                    "TP",
                    "tp"
                ]
            )
        ),

    "FP":
        int(
            get_metric(
                final_window_raw,
                [
                    "FP",
                    "fp"
                ]
            )
        ),

    "TN":
        int(
            get_metric(
                final_window_raw,
                [
                    "TN",
                    "tn"
                ]
            )
        ),

    "FN":
        int(
            get_metric(
                final_window_raw,
                [
                    "FN",
                    "fn"
                ]
            )
        ),

    "Recall":
        float(
            get_metric(
                final_window_raw,
                [
                    "recall",
                    "Recall",
                    "sensitivity",
                    "Sensitivity"
                ]
            )
        ),

    "Specificity":
        float(
            get_metric(
                final_window_raw,
                [
                    "specificity",
                    "Specificity"
                ]
            )
        ),

    "Precision":
        float(
            get_metric(
                final_window_raw,
                [
                    "precision",
                    "Precision"
                ]
            )
        ),

    "F1":
        float(
            get_metric(
                final_window_raw,
                [
                    "F1",
                    "f1",
                    "f1_score"
                ]
            )
        ),
}


# ================================================================
# 7. EXTRACT EVENT-LEVEL RESULTS
# ================================================================

print()
print("=" * 72)
print("4. EXTRACTING EVENT-LEVEL METRICS")
print("=" * 72)


event_section = event_report.get(
    "event_level",
    {}
)


baseline_event_raw = event_section.get(
    "baseline",
    {}
)


final_event_raw = event_section.get(
    "final",
    event_section.get(
        "artifact_rejection",
        {}
    )
)


if not baseline_event_raw:

    raise RuntimeError(
        "Baseline event-level results "
        "were not found."
    )


if not final_event_raw:

    raise RuntimeError(
        "Final event-level results "
        "were not found."
    )


baseline_event = {

    "TP":
        int(
            get_metric(
                baseline_event_raw,
                [
                    "tp",
                    "TP",
                    "TP_events"
                ]
            )
        ),

    "FP":
        int(
            get_metric(
                baseline_event_raw,
                [
                    "fp",
                    "FP",
                    "FP_events"
                ]
            )
        ),

    "FN":
        int(
            get_metric(
                baseline_event_raw,
                [
                    "fn",
                    "FN",
                    "FN_events"
                ]
            )
        ),

    "Recall":
        float(
            get_metric(
                baseline_event_raw,
                [
                    "recall",
                    "Recall"
                ]
            )
        ),

    "Precision":
        float(
            get_metric(
                baseline_event_raw,
                [
                    "precision",
                    "Precision"
                ]
            )
        ),

    "F1":
        float(
            get_metric(
                baseline_event_raw,
                [
                    "f1",
                    "F1"
                ]
            )
        ),
}


final_event = {

    "TP":
        int(
            get_metric(
                final_event_raw,
                [
                    "tp",
                    "TP",
                    "TP_events"
                ]
            )
        ),

    "FP":
        int(
            get_metric(
                final_event_raw,
                [
                    "fp",
                    "FP",
                    "FP_events"
                ]
            )
        ),

    "FN":
        int(
            get_metric(
                final_event_raw,
                [
                    "fn",
                    "FN",
                    "FN_events"
                ]
            )
        ),

    "Recall":
        float(
            get_metric(
                final_event_raw,
                [
                    "recall",
                    "Recall"
                ]
            )
        ),

    "Precision":
        float(
            get_metric(
                final_event_raw,
                [
                    "precision",
                    "Precision"
                ]
            )
        ),

    "F1":
        float(
            get_metric(
                final_event_raw,
                [
                    "f1",
                    "F1"
                ]
            )
        ),
}


# ================================================================
# 8. ARTIFACT THRESHOLD
# ================================================================

artifact_threshold = float(
    artifact_report.get(
        "artifact_score_threshold",
        artifact_report.get(
            "validation_selected_artifact_threshold",
            0.525596
        )
    )
)


# ================================================================
# 9. PRINT FINAL SUMMARY
# ================================================================

print()
print("=" * 72)
print("5. FINAL TEST SUMMARY")
print("=" * 72)

print()
print("WINDOW LEVEL")

print(
    "Baseline:",
    f"Recall={baseline_window['Recall']:.6f}",
    f"Precision={baseline_window['Precision']:.6f}",
    f"F1={baseline_window['F1']:.6f}",
    f"Specificity={baseline_window['Specificity']:.6f}"
)

print(
    "Final:",
    f"Recall={final_window['Recall']:.6f}",
    f"Precision={final_window['Precision']:.6f}",
    f"F1={final_window['F1']:.6f}",
    f"Specificity={final_window['Specificity']:.6f}"
)


print()
print("EVENT LEVEL")

print(
    "Baseline:",
    f"Recall={baseline_event['Recall']:.6f}",
    f"Precision={baseline_event['Precision']:.6f}",
    f"F1={baseline_event['F1']:.6f}"
)

print(
    "Final:",
    f"Recall={final_event['Recall']:.6f}",
    f"Precision={final_event['Precision']:.6f}",
    f"F1={final_event['F1']:.6f}"
)


print()
print(
    "Validation-selected artifact threshold:",
    f"{artifact_threshold:.6f}"
)


# ================================================================
# 10. FIGURE 1
# WINDOW-LEVEL METRICS
# ================================================================

print()
print("=" * 72)
print("6. CREATING WINDOW-LEVEL METRIC FIGURE")
print("=" * 72)


metrics = [
    "Recall",
    "Precision",
    "F1",
    "Specificity",
]


baseline_values = [
    baseline_window[
        "Recall"
    ],
    baseline_window[
        "Precision"
    ],
    baseline_window[
        "F1"
    ],
    baseline_window[
        "Specificity"
    ],
]


final_values = [
    final_window[
        "Recall"
    ],
    final_window[
        "Precision"
    ],
    final_window[
        "F1"
    ],
    final_window[
        "Specificity"
    ],
]


x = np.arange(
    len(metrics)
)

width = 0.35


fig, ax = plt.subplots(
    figsize=(10, 6)
)


baseline_bars = ax.bar(
    x - width / 2,
    baseline_values,
    width,
    label="Baseline"
)


final_bars = ax.bar(
    x + width / 2,
    final_values,
    width,
    label="Artifact Rejection"
)


ax.set_ylabel(
    "Score"
)

ax.set_title(
    "Test-Set Window-Level Performance"
)

ax.set_xticks(
    x
)

ax.set_xticklabels(
    metrics
)

ax.set_ylim(
    0,
    1.08
)

ax.legend()

ax.grid(
    axis="y",
    alpha=0.25
)


for bars in [
    baseline_bars,
    final_bars,
]:

    for bar in bars:

        height = bar.get_height()

        ax.text(
            bar.get_x()
            + bar.get_width() / 2,
            height + 0.012,
            f"{height:.3f}",
            ha="center",
            va="bottom",
            fontsize=9
        )


fig.tight_layout()


path = os.path.join(
    OUTPUT_DIR,
    "01_window_level_metrics.png"
)


fig.savefig(
    path,
    dpi=200,
    bbox_inches="tight"
)


plt.close(
    fig
)


print(
    "[OK]",
    path
)


# ================================================================
# 11. FIGURE 2
# EVENT-LEVEL METRICS
# ================================================================

print()
print("=" * 72)
print("7. CREATING EVENT-LEVEL METRIC FIGURE")
print("=" * 72)


metrics = [
    "Recall",
    "Precision",
    "F1",
]


baseline_values = [
    baseline_event[
        "Recall"
    ],
    baseline_event[
        "Precision"
    ],
    baseline_event[
        "F1"
    ],
]


final_values = [
    final_event[
        "Recall"
    ],
    final_event[
        "Precision"
    ],
    final_event[
        "F1"
    ],
]


x = np.arange(
    len(metrics)
)


fig, ax = plt.subplots(
    figsize=(9, 6)
)


baseline_bars = ax.bar(
    x - width / 2,
    baseline_values,
    width,
    label="Baseline"
)


final_bars = ax.bar(
    x + width / 2,
    final_values,
    width,
    label="Artifact Rejection"
)


ax.set_ylabel(
    "Score"
)

ax.set_title(
    "Test-Set Event-Level Performance"
)

ax.set_xticks(
    x
)

ax.set_xticklabels(
    metrics
)

ax.set_ylim(
    0,
    1.08
)

ax.legend()

ax.grid(
    axis="y",
    alpha=0.25
)


for bars in [
    baseline_bars,
    final_bars,
]:

    for bar in bars:

        height = bar.get_height()

        ax.text(
            bar.get_x()
            + bar.get_width() / 2,
            height + 0.012,
            f"{height:.3f}",
            ha="center",
            va="bottom",
            fontsize=9
        )


fig.tight_layout()


path = os.path.join(
    OUTPUT_DIR,
    "02_event_level_metrics.png"
)


fig.savefig(
    path,
    dpi=200,
    bbox_inches="tight"
)


plt.close(
    fig
)


print(
    "[OK]",
    path
)


# ================================================================
# 12. FIGURE 3
# CLASSIFICATION COUNTS
# ================================================================

print()
print("=" * 72)
print("8. CREATING CLASSIFICATION COUNT FIGURE")
print("=" * 72)


categories = [
    "TP",
    "FP",
    "TN",
    "FN",
]


baseline_counts = [
    baseline_window[
        "TP"
    ],
    baseline_window[
        "FP"
    ],
    baseline_window[
        "TN"
    ],
    baseline_window[
        "FN"
    ],
]


final_counts = [
    final_window[
        "TP"
    ],
    final_window[
        "FP"
    ],
    final_window[
        "TN"
    ],
    final_window[
        "FN"
    ],
]


x = np.arange(
    len(categories)
)


fig, ax = plt.subplots(
    figsize=(10, 6)
)


baseline_bars = ax.bar(
    x - width / 2,
    baseline_counts,
    width,
    label="Baseline"
)


final_bars = ax.bar(
    x + width / 2,
    final_counts,
    width,
    label="Artifact Rejection"
)


ax.set_ylabel(
    "Number of Windows"
)

ax.set_title(
    "Test-Set Classification Counts"
)

ax.set_xticks(
    x
)

ax.set_xticklabels(
    categories
)

ax.legend()

ax.grid(
    axis="y",
    alpha=0.25
)


maximum_count = max(
    baseline_counts
    + final_counts
)


for bars in [
    baseline_bars,
    final_bars,
]:

    for bar in bars:

        height = bar.get_height()

        ax.text(
            bar.get_x()
            + bar.get_width() / 2,
            height
            + maximum_count * 0.008,
            f"{int(height)}",
            ha="center",
            va="bottom",
            fontsize=9
        )


fig.tight_layout()


path = os.path.join(
    OUTPUT_DIR,
    "03_confusion_matrix_counts.png"
)


fig.savefig(
    path,
    dpi=200,
    bbox_inches="tight"
)


plt.close(
    fig
)


print(
    "[OK]",
    path
)


# ================================================================
# 13. FIGURE 4
# FALSE POSITIVE REDUCTION
# ================================================================

print()
print("=" * 72)
print("9. CREATING FALSE-POSITIVE REDUCTION FIGURE")
print("=" * 72)


baseline_fp = baseline_window[
    "FP"
]

final_fp = final_window[
    "FP"
]


fp_reduction_percent = (
    (
        baseline_fp
        -
        final_fp
    )
    /
    baseline_fp
    *
    100.0
    if baseline_fp > 0
    else 0.0
)


fig, ax = plt.subplots(
    figsize=(8, 6)
)


bars = ax.bar(
    [
        "Baseline",
        "Artifact Rejection"
    ],
    [
        baseline_fp,
        final_fp
    ]
)


ax.set_ylabel(
    "False-Positive Windows"
)

ax.set_title(
    "False-Positive Reduction "
    f"({fp_reduction_percent:.2f}%)"
)

ax.grid(
    axis="y",
    alpha=0.25
)


for bar in bars:

    height = bar.get_height()

    ax.text(
        bar.get_x()
        + bar.get_width() / 2,
        height + 1,
        f"{int(height)}",
        ha="center",
        va="bottom"
    )


fig.tight_layout()


path = os.path.join(
    OUTPUT_DIR,
    "04_false_positive_reduction.png"
)


fig.savefig(
    path,
    dpi=200,
    bbox_inches="tight"
)


plt.close(
    fig
)


print(
    "[OK]",
    path
)


# ================================================================
# 14. FIGURE 5
# PRECISION-RECALL TRADE-OFF
# ================================================================

print()
print("=" * 72)
print("10. CREATING PRECISION-RECALL FIGURE")
print("=" * 72)


fig, ax = plt.subplots(
    figsize=(8, 7)
)


ax.scatter(
    baseline_window[
        "Recall"
    ],
    baseline_window[
        "Precision"
    ],
    s=140,
    label="Baseline"
)


ax.scatter(
    final_window[
        "Recall"
    ],
    final_window[
        "Precision"
    ],
    s=140,
    label="Artifact Rejection"
)


ax.annotate(
    "Baseline",
    (
        baseline_window[
            "Recall"
        ],
        baseline_window[
            "Precision"
        ]
    ),
    xytext=(
        8,
        8
    ),
    textcoords="offset points"
)


ax.annotate(
    "Artifact Rejection",
    (
        final_window[
            "Recall"
        ],
        final_window[
            "Precision"
        ]
    ),
    xytext=(
        8,
        8
    ),
    textcoords="offset points"
)


ax.set_xlabel(
    "Recall"
)

ax.set_ylabel(
    "Precision"
)

ax.set_title(
    "Test-Set Precision-Recall Operating Point"
)

ax.set_xlim(
    0.75,
    0.90
)

ax.set_ylim(
    0.50,
    0.76
)

ax.grid(
    alpha=0.25
)

ax.legend()


fig.tight_layout()


path = os.path.join(
    OUTPUT_DIR,
    "05_recall_precision_tradeoff.png"
)


fig.savefig(
    path,
    dpi=200,
    bbox_inches="tight"
)


plt.close(
    fig
)


print(
    "[OK]",
    path
)


# ================================================================
# 15. FIGURE 6
# ARTIFACT SCORE DISTRIBUTION
# ================================================================

print()
print("=" * 72)
print("11. CREATING ARTIFACT SCORE DISTRIBUTION")
print("=" * 72)


artifact_scores = np.asarray(
    data[
        "artifact_score"
    ],
    dtype=np.float64
)


labels = np.asarray(
    data[
        "labels"
    ],
    dtype=np.int64
)


baseline_positive = np.asarray(
    data[
        "baseline_positive"
    ],
    dtype=bool
)


tp_scores = artifact_scores[
    baseline_positive
    &
    (
        labels == 1
    )
]


fp_scores = artifact_scores[
    baseline_positive
    &
    (
        labels == 0
    )
]


fig, ax = plt.subplots(
    figsize=(10, 6)
)


bins = np.linspace(
    0,
    1,
    31
)


if len(tp_scores) > 0:

    ax.hist(
        tp_scores,
        bins=bins,
        alpha=0.60,
        label="True Positive"
    )


if len(fp_scores) > 0:

    ax.hist(
        fp_scores,
        bins=bins,
        alpha=0.60,
        label="False Positive"
    )


ax.axvline(
    artifact_threshold,
    linestyle="--",
    linewidth=2,
    label=(
        "Validation Threshold "
        f"{artifact_threshold:.6f}"
    )
)


ax.set_xlabel(
    "Artifact Score"
)

ax.set_ylabel(
    "Number of Windows"
)

ax.set_title(
    "Artifact Score Distribution for Model-Positive Test Windows"
)

ax.set_xlim(
    0,
    1
)

ax.legend()

ax.grid(
    axis="y",
    alpha=0.25
)


fig.tight_layout()


path = os.path.join(
    OUTPUT_DIR,
    "06_artifact_score_distribution.png"
)


fig.savefig(
    path,
    dpi=200,
    bbox_inches="tight"
)


plt.close(
    fig
)


print(
    "[OK]",
    path
)


# ================================================================
# 16. FIGURE 7
# PERFORMANCE CHANGES
# ================================================================

print()
print("=" * 72)
print("12. CREATING PERFORMANCE CHANGE FIGURE")
print("=" * 72)


change_names = [
    "Recall",
    "Precision",
    "F1",
]


window_changes = [

    final_window[
        "Recall"
    ]
    -
    baseline_window[
        "Recall"
    ],

    final_window[
        "Precision"
    ]
    -
    baseline_window[
        "Precision"
    ],

    final_window[
        "F1"
    ]
    -
    baseline_window[
        "F1"
    ],
]


event_changes = [

    final_event[
        "Recall"
    ]
    -
    baseline_event[
        "Recall"
    ],

    final_event[
        "Precision"
    ]
    -
    baseline_event[
        "Precision"
    ],

    final_event[
        "F1"
    ]
    -
    baseline_event[
        "F1"
    ],
]


x = np.arange(
    len(change_names)
)


fig, ax = plt.subplots(
    figsize=(10, 6)
)


window_bars = ax.bar(
    x - width / 2,
    window_changes,
    width,
    label="Window Level"
)


event_bars = ax.bar(
    x + width / 2,
    event_changes,
    width,
    label="Event Level"
)


ax.axhline(
    0,
    linewidth=1
)


ax.set_ylabel(
    "Absolute Change"
)

ax.set_title(
    "Performance Change After Artifact Rejection"
)

ax.set_xticks(
    x
)

ax.set_xticklabels(
    change_names
)

ax.legend()

ax.grid(
    axis="y",
    alpha=0.25
)


for bars in [
    window_bars,
    event_bars,
]:

    for bar in bars:

        height = bar.get_height()

        ax.text(
            bar.get_x()
            + bar.get_width() / 2,
            height
            + (
                0.003
                if height >= 0
                else -0.003
            ),
            f"{height:+.3f}",
            ha="center",
            va=(
                "bottom"
                if height >= 0
                else "top"
            ),
            fontsize=9
        )


fig.tight_layout()


path = os.path.join(
    OUTPUT_DIR,
    "07_performance_change.png"
)


fig.savefig(
    path,
    dpi=200,
    bbox_inches="tight"
)


plt.close(
    fig
)


print(
    "[OK]",
    path
)


# ================================================================
# 17. FINAL
# ================================================================

print()
print("=" * 72)
print("FINAL VISUALIZATION COMPLETED")
print("=" * 72)

print()
print(
    "Figures saved to:"
)

print(
    OUTPUT_DIR
)

print()
print(
    "Generated files:"
)


for filename in sorted(
    os.listdir(
        OUTPUT_DIR
    )
):

    print(
        " -",
        filename
    )


print()
print(
    "No optimization was performed."
)

print(
    "No threshold was fitted on test data."
)

print()
print("=" * 72)
print("DONE")
print("=" * 72)