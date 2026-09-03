# EEG Seizure Detection Using Deep Learning

## Project Overview

This project focuses on automatic epileptic seizure detection from EEG (Electroencephalography) signals using deep learning.

The main objective is to develop a machine learning-based system capable of distinguishing seizure and non-seizure EEG segments using the **CHB-MIT Scalp EEG Database**.

The project combines:

- Biomedical signal processing
- EEG signal analysis
- Digital signal processing
- Machine learning
- Deep learning
- Temporal post-processing
- False-positive analysis
- Patient-level and seizure-event evaluation

The project is implemented primarily in **Python and PyTorch** and uses a 1D Convolutional Neural Network (1D-CNN) for EEG window classification.

The current final evaluation also includes a **validation-frozen artifact rejection post-processing layer** designed to reduce false-positive detections without modifying or retraining the trained CNN.

---

# Dataset

The project uses the **CHB-MIT Scalp EEG Database**, a publicly available dataset containing long-term multichannel scalp EEG recordings from pediatric patients with epilepsy.

The original EEG recordings are treated as the raw signal source for preprocessing, segmentation, normalization, model inference, and evaluation.

The dataset itself is **not included in this repository**.

The final evaluation uses fixed-length EEG windows extracted from the original recordings.

Each window contains:

```text
23 EEG channels
1280 samples per channel
256 Hz sampling frequency
5 seconds duration
```

---

# Project Pipeline

The overall processing pipeline is:

```text
CHB-MIT EEG Dataset
        |
        v
EEG Data Loading
        |
        v
Window Segmentation
        |
        v
Signal Normalization
        |
        v
1D-CNN Classification
        |
        v
Seizure Probability
        |
        v
CNN Decision Threshold
        |
        v
Window-Level Evaluation
        |
        v
False Positive Analysis
        |
        v
Validation-Based Artifact Analysis
        |
        v
Validation-Frozen Artifact Rejection
        |
        v
Strict Independent Test Evaluation
        |
        v
Seizure-Event Evaluation
```

The artifact rejection stage is applied only as a post-processing step after CNN inference.

The trained CNN itself is not modified during this stage.

---

# Data Processing

## 1. Window Segmentation

Long EEG recordings are divided into fixed-length temporal windows.

Each window contains multichannel EEG activity and is assigned a seizure or non-seizure label according to the dataset annotations.

The final evaluation uses 5-second windows with 23 EEG channels sampled at 256 Hz.

## 2. Normalization

EEG signals are normalized before being provided to the neural network.

Normalization reduces amplitude differences between channels and helps improve training stability.

The same frozen normalization procedure is used consistently during validation and final test evaluation.

## 3. Dataset Splitting

The data is separated into:

- Training set
- Validation set
- Test set

The test set remains isolated from model training and optimization.

The final evaluation uses the independent test set.

No test-set labels are used to fit the artifact rejection threshold or its feature parameters.

---

# Deep Learning Model

A one-dimensional Convolutional Neural Network (1D-CNN) is used for EEG classification.

The model processes multichannel temporal EEG signals and learns temporal patterns associated with seizure activity.

The main architecture is:

```text
Input EEG Channels
        |
        v
1D Convolution
        |
        v
Batch Normalization
        |
        v
ReLU
        |
        v
Max Pooling
        |
        v
1D Convolution
        |
        v
Batch Normalization
        |
        v
ReLU
        |
        v
Max Pooling
        |
        v
1D Convolution
        |
        v
Batch Normalization
        |
        v
ReLU
        |
        v
Adaptive Average Pooling
        |
        v
Fully Connected Layers
        |
        v
Seizure Probability
```

The final artifact-rejection analysis does not retrain or modify this CNN.

---

# Final Evaluation Methodology

The final test evaluation uses two stages:

```text
CNN probability
      |
      v
Threshold = 0.95
      |
      v
Initial seizure / non-seizure decision
      |
      v
Validation-frozen artifact score
      |
      v
Artifact rejection threshold = 0.525596
      |
      v
Final strict test prediction
```

The CNN decision threshold of **0.95** was fixed before the final test evaluation.

A separate artifact-rejection threshold of **0.525596** was selected using the validation set and then frozen before application to the independent test set.

The artifact-rejection rule uses the following signal-derived features:

- Mean high-frequency ratio
- Mean beta relative power
- Mean gamma relative power
- Mean zero-crossing rate

The validation-frozen feature weights are:

```text
High-frequency ratio       = 0.30
Beta relative power        = 0.25
Gamma relative power       = 0.25
Zero-crossing rate         = 0.20
```

The artifact threshold is:

```text
Artifact threshold = 0.525596
```

The feature transformations, reference statistics, weights, and threshold are frozen before final test evaluation.

---

# Final Window-Level Evaluation

The final strict evaluation was performed on the independent test set.

The test set contains:

- **3114 total windows**
- **3021 non-seizure windows**
- **93 seizure windows**

Two operating conditions are reported:

1. Baseline CNN prediction using threshold 0.95
2. Validation-frozen artifact rejection applied to the baseline CNN predictions

## Baseline Window-Level Performance

The baseline CNN at threshold 0.95 produced:

| Metric | Result |
| --- | ---: |
| Accuracy | **97.72%** |
| Sensitivity / Recall | **83.87%** |
| Specificity | **98.15%** |
| Precision | **58.21%** |
| F1-score | **68.72%** |

The baseline confusion matrix is:

```text
TN = 2965
FP = 56
FN = 15
TP = 78
```

## Final Validation-Frozen Artifact-Rejection Performance

After applying the validation-frozen artifact rejection rule:

| Metric | Result |
| --- | ---: |
| Accuracy | **98.11%** |
| Sensitivity / Recall | **81.72%** |
| Specificity | **98.61%** |
| Precision | **64.41%** |
| F1-score | **72.04%** |

The final strict confusion matrix is:

```text
TN = 2979
FP = 42
FN = 17
TP = 76
```

## Change Relative to Baseline

The validation-frozen artifact rejection produced the following changes:

| Measure | Change |
| --- | ---: |
| False positives | **56 → 42** |
| False-positive reduction | **25.00%** |
| True positives | **78 → 76** |
| Recall | **-2.15 percentage points** |
| Precision | **+6.20 percentage points** |
| Specificity | **+0.46 percentage points** |
| Accuracy | **+0.39 percentage points** |
| F1-score | **+3.32 percentage points** |

The artifact rejection rule therefore reduced false positives by 25% while producing a relatively small decrease in recall.

---

# Confusion Matrix

## Baseline CNN

The baseline window-level confusion matrix at threshold 0.95 is:

| True / Predicted | Non-Seizure | Seizure |
| --- | ---: | ---: |
| **Non-Seizure** | **2965** | **56** |
| **Seizure** | **15** | **78** |

The corresponding counts are:

```text
TN = 2965
FP = 56
FN = 15
TP = 78
```

## Final Strict Evaluation

The final validation-frozen artifact-rejection confusion matrix is:

| True / Predicted | Non-Seizure | Seizure |
| --- | ---: | ---: |
| **Non-Seizure** | **2979** | **42** |
| **Seizure** | **17** | **76** |

The corresponding counts are:

```text
TN = 2979
FP = 42
FN = 17
TP = 76
```

The strict evaluation is based on the independent test set and uses the artifact rejection rule frozen using validation data only.

---

# Threshold Analysis

A range of CNN decision thresholds was evaluated during the development and analysis stages.

Increasing the threshold generally reduces false positives but also increases false negatives.

Selected baseline operating points include:

| Threshold | Accuracy | Sensitivity | Specificity | Precision | F1 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.70 | 94.89% | 94.62% | 94.90% | 36.36% | 52.54% |
| 0.80 | 95.95% | 92.47% | 96.06% | 41.95% | 57.72% |
| 0.90 | 97.14% | 89.25% | 97.38% | 51.23% | 65.10% |
| **0.95** | **97.72%** | **83.87%** | **98.15%** | **58.21%** | **68.72%** |
| 0.99 | 98.30% | 65.59% | 99.30% | 74.39% | 69.71% |

The CNN threshold of **0.95** was retained as the fixed baseline operating point for the final evaluation.

Importantly, the final artifact-rejection threshold was **not selected using the test set**.

It was optimized on the validation set under predefined performance constraints and subsequently frozen.

---

# ROC and Precision-Recall Analysis

The trained CNN demonstrates strong discrimination between seizure and non-seizure EEG windows.

The previously obtained ROC-AUC is approximately:

```text
ROC-AUC = 0.9891
```

ROC-AUC is threshold-independent and describes the discriminative ability of the underlying CNN probability scores.

The repository contains evaluation plots and tables generated during model development and analysis.

Typical outputs include:

```text
results/

├── roc_curve.png
├── precision_recall_curve.png
├── confusion_matrix.png
├── threshold_vs_metrics.png
├── threshold_comparison_table.csv
├── real_threshold_metrics.png
└── real_threshold_table.csv
```

Final strict evaluation reports are generated separately so that exploratory and legacy outputs are not confused with the validation-frozen final evaluation.

---

# False Positive Analysis

A substantial part of the project focuses on understanding and reducing false-positive seizure detections.

Several analysis scripts were developed to investigate the characteristics of false-positive windows.

The analysis includes:

- False-positive signal inspection
- Channel-level analysis
- Frequency-domain analysis
- EEG morphology analysis
- Artifact analysis
- False-positive versus true-positive comparison
- Window localization
- Validation-set false-positive analysis
- Feature-based filtering experiments
- Probability smoothing experiments
- Temporal persistence analysis
- Validation-frozen artifact rejection

The purpose of these experiments is to determine whether false-positive detections exhibit temporal or signal-level characteristics that can be exploited without modifying the trained CNN.

The final artifact-rejection method was selected using validation data and then evaluated once on the independent test set.

---

# Validation-Frozen Artifact Rejection

The final artifact-rejection method is a post-processing layer applied after the CNN prediction.

The trained CNN is kept fixed.

The method uses four signal-level features:

```text
1. Mean high-frequency ratio
2. Mean beta relative power
3. Mean gamma relative power
4. Mean zero-crossing rate
```

The corresponding frozen weights are:

```text
High-frequency ratio       = 0.30
Beta relative power        = 0.25
Gamma relative power       = 0.25
Zero-crossing rate         = 0.20
```

The resulting artifact score is compared with:

```text
Artifact threshold = 0.525596
```

A model-positive window whose artifact score exceeds the frozen rejection threshold is rejected as a seizure prediction.

The threshold and feature transformations were determined from validation data only.

The final test set is used only for the final performance evaluation.

---

# Seizure-Event Level Evaluation

Window-level predictions were additionally evaluated at the seizure-event level.

The independent test set contains:

```text
Ground-truth seizure events = 92
```

The event-level evaluation compares the baseline CNN predictions with the validation-frozen artifact-rejection predictions.

## Baseline Event-Level Performance

```text
TP = 77
FP = 56
FN = 15
```

Resulting metrics:

| Metric | Result |
| --- | ---: |
| Recall | **83.70%** |
| Precision | **57.89%** |
| F1-score | **68.44%** |

## Final Strict Event-Level Performance

```text
TP = 75
FP = 42
FN = 17
```

Resulting metrics:

| Metric | Result |
| --- | ---: |
| Recall | **81.52%** |
| Precision | **64.10%** |
| F1-score | **71.77%** |

## Event-Level Change

| Measure | Change |
| --- | ---: |
| False positives | **56 → 42** |
| False-positive reduction | **25.00%** |
| Recall | **-2.17 percentage points** |
| Precision | **+6.21 percentage points** |
| F1-score | **+3.33 percentage points** |

The event-level results are consistent with the window-level conclusion: the validation-frozen artifact rejection substantially reduces false positives while causing a relatively small reduction in recall.

---

# Patient-Level Evaluation

Patient-level analysis was also investigated because EEG seizure detection is ultimately intended to operate over continuous patient recordings rather than isolated windows.

The strict test set contains:

```text
5 patients
```

Among these patients:

```text
1 patient contains ground-truth seizure events
4 patients contain no ground-truth seizure events
```

The seizure-positive patient contributes the 92 ground-truth seizure events used in the event-level evaluation.

Patient-level results should be interpreted cautiously because the evaluated patient set is small.

The main final quantitative evaluation therefore focuses on:

- Window-level performance
- Seizure-event-level performance

rather than treating the five-patient experiment as a definitive estimate of patient-independent clinical performance.

## Patient-Level Event Summary

The strict event-level evaluation produced the following patient-specific behavior:

```text
chb01
Ground-truth seizure events = 92

Baseline:
TP = 77
FP = 24
FN = 15

Validation-frozen artifact rejection:
TP = 75
FP = 21
FN = 17
```

For the seizure-negative patients, the strict artifact rejection reduced some false-positive events while leaving others unchanged.

The limited number of patients prevents strong conclusions about patient-independent generalization.

---

# Exploratory Temporal Persistence Analysis

Window-level classification treats each EEG window independently.

However, real seizure events are temporal phenomena. A single isolated high-probability window may therefore represent noise, artifact, or another transient EEG pattern rather than a true seizure.

Temporal persistence rules were investigated during an earlier exploratory stage of the project.

The exploratory frozen temporal rule used:

```text
Fraction threshold = 0.005
Minimum runs = 0
Minimum cluster size = 4
Maximum gap = 2
```

The temporal rule operates as a **post-processing layer** on top of the CNN predictions.

The trained model itself is not modified.

An earlier five-patient exploratory temporal experiment produced:

```text
TP = 1
FP = 0
FN = 0
TN = 4
```

which corresponds to:

```text
Sensitivity = 100%
Specificity = 100%
Precision   = 100%
F1-score    = 1.00
```

However, this result is based on only five patients and represents a different patient-level temporal aggregation experiment.

It should therefore **not** be interpreted as the primary final performance result of the current validation-frozen artifact-rejection evaluation.

The current final results are the strict window-level and seizure-event-level evaluations reported above.

---

# Experimental Safeguards

The final evaluation includes explicit safeguards against test-set leakage.

The final evaluation procedure follows these principles:

```text
Test set used for optimization: False

Test labels used to fit artifact features: False

Model modified during final evaluation: False

Dataset modified during final evaluation: False

Artifact-rejection parameters fitted on test set: False
```

The artifact-rejection threshold and feature reference statistics were frozen using the validation set before final test evaluation.

The final test set is used only for independent performance measurement.

A validation-reproduction sanity check confirmed that the frozen artifact-score implementation reproduced the validation artifact scores exactly:

```text
Maximum absolute difference = 0
Mean absolute difference    = 0
Validation reproduction     = Passed
```

This check helps ensure that the same feature transformation and scoring procedure is applied consistently between validation and final test evaluation.

---

# Main Findings

The current results demonstrate several important findings.

## 1. Strong CNN Discrimination

The trained CNN achieves:

```text
ROC-AUC = 0.9891
```

indicating strong separation between seizure and non-seizure EEG windows.

## 2. Strong Baseline Test Performance

At the fixed CNN threshold of 0.95, the independent test set produces:

```text
Accuracy    = 97.72%
Sensitivity = 83.87%
Specificity = 98.15%
Precision   = 58.21%
F1-score    = 68.72%
```

The CNN therefore achieves high specificity and sensitivity, while precision remains limited by false-positive detections.

## 3. Validation-Frozen Artifact Rejection Reduces False Positives

The validation-frozen artifact rejection changes the final window-level results to:

```text
Accuracy    = 98.11%
Sensitivity = 81.72%
Specificity = 98.61%
Precision   = 64.41%
F1-score    = 72.04%
```

False positives decrease from:

```text
56 → 42
```

corresponding to:

```text
25.00% false-positive reduction
```

## 4. Precision and F1 Improve

The artifact rejection increases:

```text
Precision:
58.21% → 64.41%

F1-score:
68.72% → 72.04%
```

while recall decreases modestly:

```text
83.87% → 81.72%
```

This demonstrates a measurable false-positive suppression benefit without retraining the CNN.

## 5. Recall-Precision Trade-off Remains

Although the artifact-rejection layer improves precision and F1-score, precision and F1-score remain below 80%.

Therefore, the current post-processing approach should not be considered a complete solution to the false-positive problem.

Further improvement will likely require improvements to the underlying model, richer temporal context, better artifact-aware representations, or larger patient-independent validation.

## 6. Event-Level Results Confirm the Window-Level Trend

At the seizure-event level, the final strict evaluation produces:

```text
Recall    = 81.52%
Precision = 64.10%
F1-score  = 71.77%
```

The event-level results again show a reduction in false positives at the cost of a relatively small reduction in recall.

---

# Limitations

The current project has several limitations.

- The dataset is relatively imbalanced between seizure and non-seizure windows.
- Window-level false positives remain significant.
- Precision and F1-score remain below 80% after artifact rejection.
- The final patient/event evaluation contains only a limited number of patients.
- Patient-level metrics therefore have high statistical uncertainty.
- The artifact-rejection method is a post-processing method rather than a learned temporal model.
- EEG artifacts and patient-specific signal characteristics can affect model predictions.
- The current evaluation is based on the CHB-MIT dataset and therefore does not establish external clinical generalization.
- Additional external validation is required before drawing conclusions about clinical deployment.
- The current artifact-rejection method is not intended to replace a dedicated signal-quality or artifact-detection model.
- The seizure-event distribution in the current test set is concentrated in a small number of patients, limiting conclusions about cross-patient event detection.

---

# Future Work

Future development will focus on improving both the underlying seizure classifier and the false-positive suppression stage.

## Signal Processing

- EEG filtering
- Frequency-domain analysis
- Time-frequency analysis
- EEG frequency-band characterization
- Artifact characterization
- Signal-quality assessment

## False Positive Reduction

- Improved temporal post-processing
- Patient-independent false-positive suppression
- Artifact-aware classification
- Multi-feature false-positive filtering
- Better calibration of seizure probabilities
- False-positive characterization across additional patients

## Temporal Modeling

Future experiments may investigate models that explicitly represent temporal context, such as:

- Temporal CNNs
- Recurrent neural networks
- LSTM/GRU architectures
- Transformer-based temporal models
- CNN + recurrent hybrid architectures

Temporal context may help distinguish isolated artifact-like events from persistent seizure activity.

## Model Improvement

Because the current artifact-rejection approach reaches a practical precision/recall trade-off, further improvement should also investigate the underlying CNN.

Potential directions include:

- Improved CNN architecture
- Multi-scale temporal convolutions
- Residual convolutional blocks
- Attention mechanisms
- Channel-aware feature extraction
- Better class-imbalance handling
- Patient-independent training strategies
- Probability calibration

## Patient-Level Generalization

A larger patient-level evaluation set will be required to determine whether the observed false-positive suppression generalizes across patients.

Future evaluation should include:

- More seizure-positive patients
- More seizure-negative patients
- Patient-independent test splits
- External datasets where possible
- Event-level sensitivity
- False alarms per hour
- Detection latency
- Robustness to different EEG recording conditions

---

# Project Structure

```text
EEG-Seizure-Detection/

│
├── data/                              # Dataset and generated data files
│
├── models/                            # Trained model weights
│
├── src/                               # Main source code
│   ├── train.py
│   ├── evaluate.py
│   ├── read_eeg.py
│   ├── chbmit_pytorch_dataset.py
│   ├── threshold_analysis.py
│   ├── real_threshold_analysis.py
│   ├── patient_rule_optimization.py
│   ├── final_frozen_patient_rule_test.py
│   ├── generate_evaluation_plots.py
│   └── generate_final_paper_results.py
│
├── analysis/                          # Error analysis and experiments
│   ├── analyse_false_positive.py
│   ├── analyze_fp_artifacts.py
│   ├── analyze_fp_channel_importance.py
│   ├── analyze_fp_filter_candidates.py
│   ├── analyze_fp_frequency.py
│   ├── analyze_fp_morphology.py
│   ├── analyze_fp_suppression.py
│   ├── analyze_patient_level_aggregation.py
│   ├── analyze_probability_smoothing.py
│   ├── analyze_temporal_persistence.py
│   ├── analyze_test_false_positive_characteristics.py
│   ├── analyze_test_fp_window_localization.py
│   ├── analyze_validation_feature_filters.py
│   ├── analyze_validation_fp_characteristics.py
│   ├── analyze_validation_fp_refinement.py
│   ├── analyze_validation_multifeature_filter.py
│   ├── analyze_validation_patient_level_aggregation.py
│   ├── analyze_validation_patient_q95_rules.py
│   ├── analyze_validation_patient_temporal_combined.py
│   ├── analyze_validation_patient_temporal_fp.py
│   ├── analyze_validation_persistence_rule.py
│   ├── analyze_validation_temporal_discriminator.py
│   ├── compare_fp_tp_signals.py
│   ├── compare_validation_test_distribution.py
│   ├── evaluate_test_patient_level_q95.py
│   ├── evaluate_test_patient_temporal_rule.py
│   ├── final_test_patient_level_report.py
│   ├── final_test_persistence_evaluation.py
│   ├── optimize_fp_filter_on_validation.py
│   ├── save_validation_probabilities.py
│   ├── validate_fp_filter_on_validation.py
│   ├── validate_fp_suppression.py
│   ├── optimize_artifact_rejection_validation.py
│   ├── evaluate_test_artifact_rejection.py
│   ├── evaluate_test_patient_seizure_events.py
│   ├── generate_final_test_report.py
│   ├── plot_final_test_report.py
│   └── save_test_probabilities.py
│
├── results/                           # Evaluation results and figures
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── precision_recall_curve.png
│   ├── threshold_vs_metrics.png
│   ├── threshold_comparison_table.csv
│   ├── real_threshold_metrics.png
│   ├── real_threshold_table.csv
│   ├── final_strict_test_report.csv
│   ├── final_strict_test_report.txt
│   └── ...
│
├── README.md
│
└── .gitignore
```

The `results/` directory may also contain intermediate, exploratory, or legacy analysis outputs generated during development.

The **validation-frozen strict test report** should be treated as the authoritative final evaluation record.

---

# Technologies Used

- Python
- PyTorch
- NumPy
- SciPy
- scikit-learn
- Matplotlib
- EEG signal processing
- Digital signal processing
- Deep learning
- Statistical evaluation

---

# Reproducibility

The repository contains the source code used for model evaluation, threshold analysis, false-positive investigation, validation-based artifact rejection, seizure-event evaluation, and final result generation.

The original CHB-MIT EEG dataset is not included because of its size and dataset distribution requirements.

The trained model weights and local dataset files should be placed in the appropriate directories before running the complete pipeline.

The final artifact-rejection evaluation is reproducible using the validation-frozen parameters saved during the validation optimization stage.

The strict final evaluation verifies that the validation-frozen artifact scores can be reproduced exactly before applying the rule to the independent test set.

Important final result files include:

```text
results/final_strict_test_report.csv
results/final_strict_test_report.txt

results/final_test_artifact_rejection_evaluation_strict.json
results/final_test_artifact_rejection_scores_strict.npz

results/final_test_patient_seizure_event_evaluation_strict.json
results/final_test_patient_seizure_event_results_strict.npz

results/validation_artifact_rejection_optimization_v2.json
results/validation_artifact_scores_v2.npz
```

---

# Experimental Integrity

The final evaluation follows a strict separation between development and final testing.

The artifact-rejection method was developed using validation data.

The test set was not used to:

- Optimize the artifact threshold
- Fit feature weights
- Estimate artifact-score reference statistics
- Modify the CNN
- Select the final artifact-rejection parameters

The independent test set was used only after all relevant parameters had been frozen.

This separation is intended to reduce the risk of test-set leakage and overly optimistic performance estimates.

---

# Project Status

**Current status: Final strict evaluation completed; model improvement remains the next major stage.**

The current version includes:

- 1D-CNN EEG classification
- Independent test-set evaluation
- ROC and precision-recall analysis
- Decision-threshold analysis
- Confusion matrix evaluation
- False-positive characterization
- Validation-based false-positive experiments
- Validation-frozen artifact rejection
- Strict independent test evaluation
- Seizure-event-level evaluation
- Patient-level exploratory analysis
- Temporal persistence analysis
- Final paper-oriented result generation
- Test-set leakage safeguards

The current final strict evaluation demonstrates that the validation-frozen artifact rejection can reduce false positives by **25%** while maintaining **81.72% window-level recall** and improving precision from **58.21% to 64.41%**.

The main remaining challenge is improving the underlying seizure detector and achieving stronger precision/F1 performance while preserving sensitivity.

The next major development stage is therefore focused on **model improvement, richer temporal modeling, larger patient-independent evaluation, and improved generalization of false-positive suppression**.