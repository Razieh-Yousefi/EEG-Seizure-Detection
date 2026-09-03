# ================================================================
# evaluate_test_artifact_rejection.py
#
# STRICT FINAL TEST-SET ARTIFACT REJECTION EVALUATION
#
# This version reproduces EXACTLY the artifact-score construction
# selected on validation.
#
# IMPORTANT:
# - NO optimization on test.
# - NO parameter fitting on test.
# - NO test-label use for rule selection.
# - Baseline threshold comes from validation pipeline.
# - Artifact threshold comes from validation JSON.
# - Feature weights come from validation JSON.
# - q05/q95 feature references come from validation JSON.
# - Training normalization is reused.
# - Same feature definitions as validation optimizer are used.
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


TEST_PROBABILITY_FILE = os.path.join(
    DATA_DIR,
    "test_window_probabilities.npz"
)

TEST_INDEX_FILE = os.path.join(
    DATA_DIR,
    "test_indices.npy"
)

X_FILE = os.path.join(
    DATA_DIR,
    "X_chbmit_full.npy"
)

NORMALIZATION_FILE = os.path.join(
    DATA_DIR,
    "normalization_params.npz"
)

VALIDATION_RESULT_FILE = os.path.join(
    RESULTS_DIR,
    "validation_artifact_rejection_optimization_v2.json"
)

VALIDATION_SCORE_FILE = os.path.join(
    RESULTS_DIR,
    "validation_artifact_scores_v2.npz"
)


OUTPUT_JSON = os.path.join(
    RESULTS_DIR,
    "final_test_artifact_rejection_evaluation_strict.json"
)

OUTPUT_NPZ = os.path.join(
    RESULTS_DIR,
    "final_test_artifact_rejection_scores_strict.npz"
)


# ================================================================
# 2. FIXED SETTINGS
# ================================================================

FS = 256.0

EPS = 1e-12


# ================================================================
# 3. HEADER
# ================================================================

print()
print("=" * 76)
print("STRICT VALIDATION-FROZEN TEST ARTIFACT EVALUATION")
print("=" * 76)

print()
print("Project:")
print(PROJECT_DIR)

print()
print("IMPORTANT:")
print("- NO test-set optimization.")
print("- NO test-set feature fitting.")
print("- NO test threshold search.")
print("- Validation rule is reproduced exactly.")


# ================================================================
# 4. CHECK FILES
# ================================================================

print()
print("=" * 76)
print("1. CHECKING REQUIRED FILES")
print("=" * 76)

required_files = [

    TEST_PROBABILITY_FILE,

    TEST_INDEX_FILE,

    X_FILE,

    NORMALIZATION_FILE,

    VALIDATION_RESULT_FILE,

    VALIDATION_SCORE_FILE,
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


# ================================================================
# 5. LOAD VALIDATION RULE
# ================================================================

print()
print("=" * 76)
print("2. LOADING FROZEN VALIDATION RULE")
print("=" * 76)


with open(
    VALIDATION_RESULT_FILE,
    "r",
    encoding="utf-8"
) as f:

    validation_result = json.load(
        f
    )


if "baseline_threshold" not in validation_result:

    raise KeyError(
        "baseline_threshold missing from validation result."
    )


BASELINE_THRESHOLD = float(
    validation_result[
        "baseline_threshold"
    ]
)


if "best_candidate" not in validation_result:

    raise KeyError(
        "best_candidate missing from validation result."
    )


best_candidate = validation_result[
    "best_candidate"
]


if best_candidate is None:

    raise RuntimeError(
        "Validation did not produce a best candidate."
    )


if "score_threshold" not in best_candidate:

    raise KeyError(
        "score_threshold missing from best_candidate."
    )


ARTIFACT_THRESHOLD = float(
    best_candidate[
        "score_threshold"
    ]
)


if "feature_weights" not in validation_result:

    raise KeyError(
        "feature_weights missing from validation result."
    )


FEATURE_WEIGHTS = validation_result[
    "feature_weights"
]


if "feature_reference_statistics" not in validation_result:

    raise KeyError(
        "feature_reference_statistics missing."
    )


FEATURE_REFERENCE = validation_result[
    "feature_reference_statistics"
]


FEATURE_NAMES = [

    "mean_high_frequency_ratio",

    "mean_beta_relative_power",

    "mean_gamma_relative_power",

    "mean_zero_crossing_rate",
]


# ================================================================
# 6. VALIDATE RULE CONTENT
# ================================================================

for feature_name in FEATURE_NAMES:

    if feature_name not in FEATURE_WEIGHTS:

        raise KeyError(
            f"Missing validation weight: {feature_name}"
        )

    if feature_name not in FEATURE_REFERENCE:

        raise KeyError(
            f"Missing validation reference: {feature_name}"
        )

    reference = FEATURE_REFERENCE[
        feature_name
    ]

    for required_key in [
        "q05_nonseizure",
        "q95_nonseizure",
        "scale",
    ]:

        if required_key not in reference:

            raise KeyError(
                f"{feature_name} missing {required_key}"
            )


print()
print(
    "Baseline threshold:",
    BASELINE_THRESHOLD
)

print(
    "Artifact threshold:",
    ARTIFACT_THRESHOLD
)

print()
print(
    "Frozen feature weights:"
)

for feature_name in FEATURE_NAMES:

    print(
        f"{feature_name:35s} "
        f"{float(FEATURE_WEIGHTS[feature_name]):.6f}"
    )


print()
print(
    "Frozen validation feature references:"
)

for feature_name in FEATURE_NAMES:

    ref = FEATURE_REFERENCE[
        feature_name
    ]

    print(
        f"{feature_name:35s} "
        f"q05={float(ref['q05_nonseizure']):.8f} "
        f"q95={float(ref['q95_nonseizure']):.8f}"
    )


# ================================================================
# 7. CROSS-CHECK SAVED VALIDATION THRESHOLD
# ================================================================

validation_npz = np.load(
    VALIDATION_SCORE_FILE,
    allow_pickle=True
)


if "best_threshold" in validation_npz.files:

    npz_threshold = float(
        np.asarray(
            validation_npz[
                "best_threshold"
            ]
        ).reshape(-1)[0]
    )

    print()
    print(
        "Validation NPZ threshold:",
        npz_threshold
    )

    if not np.isclose(
        npz_threshold,
        ARTIFACT_THRESHOLD,
        atol=1e-8
    ):

        raise RuntimeError(
            "JSON and NPZ validation thresholds disagree."
        )


print()
print(
    "[OK] Frozen validation rule verified."
)


# ================================================================
# 8. LOAD TEST PROBABILITIES
# ================================================================

print()
print("=" * 76)
print("3. LOADING TEST PROBABILITIES")
print("=" * 76)


test_data = np.load(
    TEST_PROBABILITY_FILE,
    allow_pickle=True
)


print()
print(
    "Available test probability arrays:"
)

for key in test_data.files:

    print(
        f"{key:25s}: "
        f"shape={test_data[key].shape}"
    )


if "probabilities" not in test_data.files:

    raise KeyError(
        "probabilities missing from test probability file."
    )


if "labels" not in test_data.files:

    raise KeyError(
        "labels missing from test probability file."
    )


probabilities = np.asarray(
    test_data[
        "probabilities"
    ],
    dtype=np.float32
).reshape(-1)


labels = np.asarray(
    test_data[
        "labels"
    ],
    dtype=np.int64
).reshape(-1)


if "test_indices" in test_data.files:

    test_indices = np.asarray(
        test_data[
            "test_indices"
        ],
        dtype=np.int64
    ).reshape(-1)

else:

    test_indices = np.asarray(
        np.load(
            TEST_INDEX_FILE
        ),
        dtype=np.int64
    ).reshape(-1)


print()
print(
    "Probabilities:",
    len(probabilities)
)

print(
    "Labels:",
    len(labels)
)

print(
    "Test indices:",
    len(test_indices)
)


if not (
    len(probabilities)
    ==
    len(labels)
    ==
    len(test_indices)
):

    raise RuntimeError(
        "Test arrays are not aligned."
    )


if not np.all(
    np.isfinite(
        probabilities
    )
):

    raise RuntimeError(
        "Test probabilities contain NaN/Inf."
    )


# ================================================================
# 9. LOAD EEG
# ================================================================

print()
print("=" * 76)
print("4. LOADING EEG DATA")
print("=" * 76)


X = np.load(
    X_FILE,
    mmap_mode="r"
)


if X.ndim != 3:

    raise ValueError(
        "X must have shape "
        "(windows, channels, samples)."
    )


N_CHANNELS = X.shape[1]

N_SAMPLES = X.shape[2]

WINDOW_SECONDS = (
    N_SAMPLES / FS
)


print()
print(
    "X shape:",
    X.shape
)

print(
    "Channels:",
    N_CHANNELS
)

print(
    "Samples/window:",
    N_SAMPLES
)

print(
    "Window duration:",
    f"{WINDOW_SECONDS:.3f} sec"
)


if np.min(
    test_indices
) < 0:

    raise RuntimeError(
        "Negative test index detected."
    )


if np.max(
    test_indices
) >= len(X):

    raise RuntimeError(
        "Test index exceeds X length."
    )


# ================================================================
# 10. LOAD TRAINING NORMALIZATION
# ================================================================

print()
print("=" * 76)
print("5. LOADING TRAINING NORMALIZATION")
print("=" * 76)


norm_data = np.load(
    NORMALIZATION_FILE
)


channel_mean = np.asarray(
    norm_data[
        "channel_mean"
    ],
    dtype=np.float32
)


channel_std = np.asarray(
    norm_data[
        "channel_std"
    ],
    dtype=np.float32
)


if channel_mean.shape != (
    N_CHANNELS,
):

    raise ValueError(
        "channel_mean shape mismatch."
    )


if channel_std.shape != (
    N_CHANNELS,
):

    raise ValueError(
        "channel_std shape mismatch."
    )


if not np.all(
    np.isfinite(
        channel_mean
    )
):

    raise ValueError(
        "channel_mean contains NaN/Inf."
    )


if not np.all(
    np.isfinite(
        channel_std
    )
):

    raise ValueError(
        "channel_std contains NaN/Inf."
    )


if np.any(
    channel_std <= 0
):

    raise ValueError(
        "channel_std contains zero/negative values."
    )


print()
print(
    "[OK] Training normalization loaded."
)


# ================================================================
# 11. EXACT VALIDATION FEATURE FUNCTIONS
# ================================================================

def compute_zero_crossing_rate(
    signal
):

    if len(signal) < 2:

        return 0.0

    signs = np.signbit(
        signal
    )

    crossings = np.count_nonzero(
        signs[1:]
        !=
        signs[:-1]
    )

    return float(
        crossings
        /
        max(
            len(signal) - 1,
            1
        )
    )


def compute_window_features(
    window
):

    # ============================================================
    # EXACT SAME LOGIC AS VALIDATION OPTIMIZER
    # ============================================================

    signal = np.asarray(
        window,
        dtype=np.float32
    )


    # Remove per-window channel offsets.
    signal = (
        signal
        -
        np.mean(
            signal,
            axis=1,
            keepdims=True
        )
    )


    n_channels = signal.shape[0]

    n_samples = signal.shape[1]


    freqs = np.fft.rfftfreq(
        n_samples,
        d=1.0 / FS
    )


    taper = np.hanning(
        n_samples
    ).astype(
        np.float32
    )


    tapered = (
        signal
        *
        taper[
            None,
            :
        ]
    )


    fft_values = np.fft.rfft(
        tapered,
        axis=1
    )


    psd = (
        np.abs(
            fft_values
        )
        ** 2
    )


    # ------------------------------------------------------------
    # Total power = 0.5-45 Hz
    # ------------------------------------------------------------

    total_mask = (
        (freqs >= 0.5)
        &
        (freqs < 45.0)
    )


    if np.any(
        total_mask
    ):

        total_power = np.trapezoid(
            psd[
                :,
                total_mask
            ],
            freqs[
                total_mask
            ],
            axis=1
        )

    else:

        total_power = np.ones(
            n_channels,
            dtype=np.float64
        )


    total_power = np.maximum(
        total_power,
        EPS
    )


    # ------------------------------------------------------------
    # HF = 30-45 Hz
    # ------------------------------------------------------------

    high_mask = (
        (freqs >= 30.0)
        &
        (freqs < 45.0)
    )


    if np.any(
        high_mask
    ):

        high_power = np.trapezoid(
            psd[
                :,
                high_mask
            ],
            freqs[
                high_mask
            ],
            axis=1
        )

    else:

        high_power = np.zeros(
            n_channels,
            dtype=np.float64
        )


    high_frequency_ratio = (
        high_power
        /
        total_power
    )


    # ------------------------------------------------------------
    # Beta = 13-30 Hz
    # ------------------------------------------------------------

    beta_mask = (
        (freqs >= 13.0)
        &
        (freqs < 30.0)
    )


    if np.any(
        beta_mask
    ):

        beta_power = np.trapezoid(
            psd[
                :,
                beta_mask
            ],
            freqs[
                beta_mask
            ],
            axis=1
        )

    else:

        beta_power = np.zeros(
            n_channels,
            dtype=np.float64
        )


    beta_relative_power = (
        beta_power
        /
        total_power
    )


    # ------------------------------------------------------------
    # Gamma = 30-45 Hz
    # ------------------------------------------------------------

    gamma_mask = (
        (freqs >= 30.0)
        &
        (freqs < 45.0)
    )


    if np.any(
        gamma_mask
    ):

        gamma_power = np.trapezoid(
            psd[
                :,
                gamma_mask
            ],
            freqs[
                gamma_mask
            ],
            axis=1
        )

    else:

        gamma_power = np.zeros(
            n_channels,
            dtype=np.float64
        )


    gamma_relative_power = (
        gamma_power
        /
        total_power
    )


    # ------------------------------------------------------------
    # ZCR
    # ------------------------------------------------------------

    zcr = np.zeros(
        n_channels,
        dtype=np.float64
    )


    for ch in range(
        n_channels
    ):

        zcr[ch] = (
            compute_zero_crossing_rate(
                signal[
                    ch
                ]
            )
        )


    return {

        "mean_high_frequency_ratio":
            float(
                np.mean(
                    high_frequency_ratio
                )
            ),

        "mean_beta_relative_power":
            float(
                np.mean(
                    beta_relative_power
                )
            ),

        "mean_gamma_relative_power":
            float(
                np.mean(
                    gamma_relative_power
                )
            ),

        "mean_zero_crossing_rate":
            float(
                np.mean(
                    zcr
                )
            ),
    }


# ================================================================
# 12. EXTRACT TEST FEATURES
# ================================================================

print()
print("=" * 76)
print("6. EXTRACTING TEST FEATURES WITH VALIDATION LOGIC")
print("=" * 76)


feature_matrix = np.zeros(
    (
        len(
            test_indices
        ),
        len(
            FEATURE_NAMES
        )
    ),
    dtype=np.float64
)


for i, dataset_index in enumerate(
    test_indices
):

    raw_window = np.asarray(
        X[
            int(
                dataset_index
            )
        ],
        dtype=np.float32
    )


    # ============================================================
    # EXACT TRAINING-TIME NORMALIZATION USED IN VALIDATION
    # ============================================================

    normalized_window = (
        raw_window
        -
        channel_mean[
            :,
            None
        ]
    ) / channel_std[
        :,
        None
    ]


    features = compute_window_features(
        normalized_window
    )


    for j, feature_name in enumerate(
        FEATURE_NAMES
    ):

        feature_matrix[
            i,
            j
        ] = features[
            feature_name
        ]


    if (
        i == 0
        or
        (i + 1) % 100 == 0
        or
        (i + 1)
        ==
        len(
            test_indices
        )
    ):

        print(
            f"Processed "
            f"{i + 1}/"
            f"{len(test_indices)}"
        )


if not np.all(
    np.isfinite(
        feature_matrix
    )
):

    raise RuntimeError(
        "Feature matrix contains NaN/Inf."
    )


# ================================================================
# 13. FEATURE DISTRIBUTIONS
# ================================================================

print()
print("=" * 76)
print("7. TEST FEATURE DISTRIBUTIONS")
print("=" * 76)


for j, feature_name in enumerate(
    FEATURE_NAMES
):

    values = feature_matrix[
        :,
        j
    ]

    print()
    print(
        feature_name
    )

    print(
        "min =",
        f"{np.min(values):.8f}"
    )

    print(
        "max =",
        f"{np.max(values):.8f}"
    )

    print(
        "mean =",
        f"{np.mean(values):.8f}"
    )


# ================================================================
# 14. BUILD ARTIFACT SCORE USING VALIDATION REFERENCES
# ================================================================

print()
print("=" * 76)
print("8. RECONSTRUCTING FROZEN VALIDATION ARTIFACT SCORE")
print("=" * 76)


artifact_score = np.zeros(
    len(
        test_indices
    ),
    dtype=np.float64
)


normalized_feature_matrix = np.zeros_like(
    feature_matrix
)


for j, feature_name in enumerate(
    FEATURE_NAMES
):

    values = feature_matrix[
        :,
        j
    ]


    reference = FEATURE_REFERENCE[
        feature_name
    ]


    q05 = float(
        reference[
            "q05_nonseizure"
        ]
    )


    q95 = float(
        reference[
            "q95_nonseizure"
        ]
    )


    scale = (
        q95 - q05
    )


    # Use exactly the stored scale when valid.
    stored_scale = float(
        reference.get(
            "scale",
            scale
        )
    )


    if stored_scale > EPS:

        scale = stored_scale


    if scale <= EPS:

        raise RuntimeError(
            f"Invalid validation scale for "
            f"{feature_name}"
        )


    normalized = (
        values
        -
        q05
    ) / scale


    normalized = np.clip(
        normalized,
        0.0,
        1.0
    )


    normalized_feature_matrix[
        :,
        j
    ] = normalized


    weight = float(
        FEATURE_WEIGHTS[
            feature_name
        ]
    )


    artifact_score += (
        weight
        *
        normalized
    )


artifact_score = np.clip(
    artifact_score,
    0.0,
    1.0
)


if not np.all(
    np.isfinite(
        artifact_score
    )
):

    raise RuntimeError(
        "Artifact score contains NaN/Inf."
    )


print()
print(
    "Artifact score min:",
    f"{np.min(artifact_score):.6f}"
)

print(
    "Artifact score max:",
    f"{np.max(artifact_score):.6f}"
)

print(
    "Artifact score mean:",
    f"{np.mean(artifact_score):.6f}"
)

print(
    "Artifact score median:",
    f"{np.median(artifact_score):.6f}"
)


# ================================================================
# 15. METRIC FUNCTION
# ================================================================

def calculate_metrics(
    labels,
    predictions
):

    labels = np.asarray(
        labels,
        dtype=np.int64
    )


    predictions = np.asarray(
        predictions,
        dtype=bool
    )


    TP = int(
        np.sum(
            predictions
            &
            (labels == 1)
        )
    )


    FP = int(
        np.sum(
            predictions
            &
            (labels == 0)
        )
    )


    TN = int(
        np.sum(
            (~predictions)
            &
            (labels == 0)
        )
    )


    FN = int(
        np.sum(
            (~predictions)
            &
            (labels == 1)
        )
    )


    recall = (
        TP
        /
        max(
            TP + FN,
            1
        )
    )


    specificity = (
        TN
        /
        max(
            TN + FP,
            1
        )
    )


    precision = (
        TP
        /
        max(
            TP + FP,
            1
        )
    )


    accuracy = (
        TP + TN
    ) / max(
        TP + FP + TN + FN,
        1
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

        "TP":
            TP,

        "FP":
            FP,

        "TN":
            TN,

        "FN":
            FN,

        "recall":
            float(
                recall
            ),

        "sensitivity":
            float(
                recall
            ),

        "specificity":
            float(
                specificity
            ),

        "precision":
            float(
                precision
            ),

        "accuracy":
            float(
                accuracy
            ),

        "f1":
            float(
                f1
            ),
    }


# ================================================================
# 16. BASELINE TEST PERFORMANCE
# ================================================================

print()
print("=" * 76)
print("9. BASELINE TEST PERFORMANCE")
print("=" * 76)


baseline_positive = (
    probabilities
    >=
    BASELINE_THRESHOLD
)


baseline_metrics = calculate_metrics(
    labels,
    baseline_positive
)


for key, value in baseline_metrics.items():

    if isinstance(
        value,
        float
    ):

        print(
            f"{key:15s}: "
            f"{value:.6f}"
        )

    else:

        print(
            f"{key:15s}: "
            f"{value}"
        )


# ================================================================
# 17. APPLY FROZEN VALIDATION RULE
# ================================================================

print()
print("=" * 76)
print("10. APPLYING FROZEN VALIDATION ARTIFACT RULE")
print("=" * 76)


# IMPORTANT:
# Validation optimization used >= threshold.
# We reproduce that exact operator here.

artifact_rejected = (
    artifact_score
    >=
    ARTIFACT_THRESHOLD
)


final_positive = (
    baseline_positive
    &
    (~artifact_rejected)
)


final_metrics = calculate_metrics(
    labels,
    final_positive
)


for key, value in final_metrics.items():

    if isinstance(
        value,
        float
    ):

        print(
            f"{key:15s}: "
            f"{value:.6f}"
        )

    else:

        print(
            f"{key:15s}: "
            f"{value}"
        )


# ================================================================
# 18. CHANGE ANALYSIS
# ================================================================

print()
print("=" * 76)
print("11. TEST PERFORMANCE CHANGE")
print("=" * 76)


rejected_baseline_fp = int(
    np.sum(
        baseline_positive
        &
        (labels == 0)
        &
        artifact_rejected
    )
)


rejected_baseline_tp = int(
    np.sum(
        baseline_positive
        &
        (labels == 1)
        &
        artifact_rejected
    )
)


total_rejected_positive = int(
    np.sum(
        baseline_positive
        &
        artifact_rejected
    )
)


baseline_fp = baseline_metrics[
    "FP"
]

final_fp = final_metrics[
    "FP"
]


fp_reduction = (
    (
        baseline_fp
        -
        final_fp
    )
    /
    baseline_fp
    if baseline_fp > 0
    else 0.0
)


recall_change = (
    final_metrics[
        "recall"
    ]
    -
    baseline_metrics[
        "recall"
    ]
)


precision_change = (
    final_metrics[
        "precision"
    ]
    -
    baseline_metrics[
        "precision"
    ]
)


specificity_change = (
    final_metrics[
        "specificity"
    ]
    -
    baseline_metrics[
        "specificity"
    ]
)


f1_change = (
    final_metrics[
        "f1"
    ]
    -
    baseline_metrics[
        "f1"
    ]
)


accuracy_change = (
    final_metrics[
        "accuracy"
    ]
    -
    baseline_metrics[
        "accuracy"
    ]
)


print()
print(
    "Rejected baseline FP:",
    rejected_baseline_fp
)

print(
    "Rejected baseline TP:",
    rejected_baseline_tp
)

print(
    "Total rejected model-positive:",
    total_rejected_positive
)

print()
print(
    "FP reduction:",
    f"{fp_reduction * 100:.2f}%"
)

print(
    "Recall change:",
    f"{recall_change * 100:+.2f} percentage points"
)

print(
    "Precision change:",
    f"{precision_change * 100:+.2f} percentage points"
)

print(
    "Specificity change:",
    f"{specificity_change * 100:+.2f} percentage points"
)

print(
    "F1 change:",
    f"{f1_change * 100:+.2f} percentage points"
)

print(
    "Accuracy change:",
    f"{accuracy_change * 100:+.2f} percentage points"
)


# ================================================================
# 19. SANITY CHECK AGAINST VALIDATION SCORE FILE
# ================================================================

print()
print("=" * 76)
print("12. VALIDATION REPRODUCTION SANITY CHECK")
print("=" * 76)


# This does NOT touch test labels for fitting.
# It only checks that the formula implemented in this file can
# reproduce the already-saved validation scores from the saved
# validation feature arrays.

validation_feature_matrix = np.column_stack(
    [
        np.asarray(
            validation_npz[
                feature_name
            ],
            dtype=np.float64
        ).reshape(-1)

        for feature_name
        in FEATURE_NAMES
    ]
)


validation_score_rebuilt = np.zeros(
    len(
        validation_feature_matrix
    ),
    dtype=np.float64
)


for j, feature_name in enumerate(
    FEATURE_NAMES
):

    ref = FEATURE_REFERENCE[
        feature_name
    ]

    q05 = float(
        ref[
            "q05_nonseizure"
        ]
    )

    scale = float(
        ref[
            "scale"
        ]
    )


    normalized = (
        validation_feature_matrix[
            :,
            j
        ]
        -
        q05
    ) / scale


    normalized = np.clip(
        normalized,
        0.0,
        1.0
    )


    validation_score_rebuilt += (
        float(
            FEATURE_WEIGHTS[
                feature_name
            ]
        )
        *
        normalized
    )


saved_validation_score = np.asarray(
    validation_npz[
        "artifact_scores"
    ],
    dtype=np.float64
).reshape(-1)


max_abs_difference = float(
    np.max(
        np.abs(
            validation_score_rebuilt
            -
            saved_validation_score
        )
    )
)


mean_abs_difference = float(
    np.mean(
        np.abs(
            validation_score_rebuilt
            -
            saved_validation_score
        )
    )
)


print()
print(
    "Validation score max abs difference:",
    f"{max_abs_difference:.12f}"
)

print(
    "Validation score mean abs difference:",
    f"{mean_abs_difference:.12f}"
)


if max_abs_difference > 1e-6:

    raise RuntimeError(
        "STRICT REPRODUCTION FAILED: "
        "the implemented score does not reproduce "
        "saved validation artifact scores."
    )


print()
print(
    "[OK] Validation artifact score reproduced exactly."
)


# ================================================================
# 20. SAVE STRICT TEST NPZ
# ================================================================

print()
print("=" * 76)
print("13. SAVING STRICT TEST WINDOW RESULTS")
print("=" * 76)


save_dict = {

    "test_indices":
        test_indices,

    "labels":
        labels,

    "probabilities":
        probabilities,

    "baseline_positive":
        baseline_positive,

    "artifact_score":
        artifact_score.astype(
            np.float32
        ),

    "artifact_rejected":
        artifact_rejected,

    "final_positive":
        final_positive,

    "feature_names":
        np.asarray(
            FEATURE_NAMES
        ),

    "feature_matrix":
        feature_matrix.astype(
            np.float32
        ),

    "normalized_feature_matrix":
        normalized_feature_matrix.astype(
            np.float32
        ),

    "baseline_threshold":
        np.asarray(
            BASELINE_THRESHOLD,
            dtype=np.float64
        ),

    "artifact_threshold":
        np.asarray(
            ARTIFACT_THRESHOLD,
            dtype=np.float64
        ),
}


for j, feature_name in enumerate(
    FEATURE_NAMES
):

    save_dict[
        feature_name
    ] = feature_matrix[
        :,
        j
    ].astype(
        np.float32
    )


np.savez(
    OUTPUT_NPZ,
    **save_dict
)


print()
print(
    "[OK] Saved:"
)

print(
    OUTPUT_NPZ
)


# ================================================================
# 21. SAVE STRICT JSON REPORT
# ================================================================

report = {

    "evaluation_type":
        "strict_validation_frozen_test_evaluation",

    "project_directory":
        PROJECT_DIR,

    "sampling_rate_hz":
        FS,

    "window_samples":
        int(
            N_SAMPLES
        ),

    "window_seconds":
        float(
            WINDOW_SECONDS
        ),

    "test_windows":
        int(
            len(
                labels
            )
        ),

    "test_set_used_for_optimization":
        False,

    "test_labels_used_for_parameter_fitting":
        False,

    "model_modified":
        False,

    "dataset_modified":
        False,

    "baseline_probability_threshold":
        float(
            BASELINE_THRESHOLD
        ),

    "artifact_score_threshold":
        float(
            ARTIFACT_THRESHOLD
        ),

    "artifact_threshold_source":
        "validation_artifact_rejection_optimization_v2.json",

    "feature_weights":
        {
            key:
                float(
                    value
                )

            for key, value
            in FEATURE_WEIGHTS.items()
        },

    "feature_reference_statistics":
        FEATURE_REFERENCE,

    "validation_score_reproduction": {

        "max_absolute_difference":
            max_abs_difference,

        "mean_absolute_difference":
            mean_abs_difference,

        "passed":
            bool(
                max_abs_difference
                <=
                1e-6
            ),
    },

    "baseline":
        baseline_metrics,

    "artifact_rejection":
        final_metrics,

    "change": {

        "rejected_baseline_fp":
            rejected_baseline_fp,

        "rejected_baseline_tp":
            rejected_baseline_tp,

        "total_rejected_model_positive":
            total_rejected_positive,

        "fp_reduction":
            float(
                fp_reduction
            ),

        "recall_change":
            float(
                recall_change
            ),

        "precision_change":
            float(
                precision_change
            ),

        "specificity_change":
            float(
                specificity_change
            ),

        "f1_change":
            float(
                f1_change
            ),

        "accuracy_change":
            float(
                accuracy_change
            ),
    },
}


with open(
    OUTPUT_JSON,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        report,
        f,
        indent=4,
        ensure_ascii=False
    )


print()
print(
    "[OK] Saved:"
)

print(
    OUTPUT_JSON
)


# ================================================================
# 22. FINAL REPORT
# ================================================================

print()
print("=" * 76)
print("14. STRICT FINAL TEST RESULTS")
print("=" * 76)


print()
print("BASELINE")
print("-" * 76)

print(
    f"TP = {baseline_metrics['TP']}"
)

print(
    f"FP = {baseline_metrics['FP']}"
)

print(
    f"TN = {baseline_metrics['TN']}"
)

print(
    f"FN = {baseline_metrics['FN']}"
)

print(
    f"Recall = "
    f"{baseline_metrics['recall']:.6f}"
)

print(
    f"Specificity = "
    f"{baseline_metrics['specificity']:.6f}"
)

print(
    f"Precision = "
    f"{baseline_metrics['precision']:.6f}"
)

print(
    f"F1 = "
    f"{baseline_metrics['f1']:.6f}"
)

print(
    f"Accuracy = "
    f"{baseline_metrics['accuracy']:.6f}"
)


print()
print("STRICT VALIDATION-FROZEN ARTIFACT REJECTION")
print("-" * 76)

print(
    f"TP = {final_metrics['TP']}"
)

print(
    f"FP = {final_metrics['FP']}"
)

print(
    f"TN = {final_metrics['TN']}"
)

print(
    f"FN = {final_metrics['FN']}"
)

print(
    f"Recall = "
    f"{final_metrics['recall']:.6f}"
)

print(
    f"Specificity = "
    f"{final_metrics['specificity']:.6f}"
)

print(
    f"Precision = "
    f"{final_metrics['precision']:.6f}"
)

print(
    f"F1 = "
    f"{final_metrics['f1']:.6f}"
)

print(
    f"Accuracy = "
    f"{final_metrics['accuracy']:.6f}"
)


print()
print("CHANGE")
print("-" * 76)

print(
    f"FP reduction = "
    f"{fp_reduction * 100:.2f}%"
)

print(
    f"Rejected FP = "
    f"{rejected_baseline_fp}"
)

print(
    f"Rejected TP = "
    f"{rejected_baseline_tp}"
)

print(
    f"Recall change = "
    f"{recall_change * 100:+.2f} pp"
)

print(
    f"Precision change = "
    f"{precision_change * 100:+.2f} pp"
)

print(
    f"F1 change = "
    f"{f1_change * 100:+.2f} pp"
)


# ================================================================
# 23. INTERPRETATION
# ================================================================

print()
print("=" * 76)
print("15. INTERPRETATION")
print("=" * 76)


if (
    final_metrics[
        "f1"
    ]
    >
    baseline_metrics[
        "f1"
    ]
    and
    final_metrics[
        "precision"
    ]
    >
    baseline_metrics[
        "precision"
    ]
    and
    final_metrics[
        "recall"
    ]
    >=
    baseline_metrics[
        "recall"
    ]
    - 0.03
):

    interpretation = (
        "PROMISING: the strictly frozen validation rule "
        "improves precision/F1 with limited recall loss."
    )

elif (
    final_metrics[
        "f1"
    ]
    >
    baseline_metrics[
        "f1"
    ]
):

    interpretation = (
        "MIXED-POSITIVE: F1 improved, but recall loss "
        "requires review."
    )

else:

    interpretation = (
        "NO CONFIRMED BENEFIT: after strict validation-rule "
        "reproduction, the artifact rule does not improve F1."
    )


print()
print(
    interpretation
)


print()
print("=" * 76)
print("STRICT TEST EVALUATION COMPLETED")
print("=" * 76)

print()
print(
    "No test optimization was performed."
)

print(
    "Validation score reproduction passed."
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
print("=" * 76)
print("DONE")
print("=" * 76)