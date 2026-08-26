# EEG Seizure Detection Using Deep Learning

## Project Overview

This project focuses on automatic epileptic seizure detection from EEG (Electroencephalography) signals using deep learning.

The main objective is to develop a machine learning-based system capable of distinguishing seizure and non-seizure EEG segments using the **CHB-MIT Scalp EEG Database**.

The project combines:

* Biomedical signal processing
* EEG signal analysis
* Digital signal processing
* Machine learning
* Deep learning
* Temporal post-processing
* False-positive analysis
* Patient-level evaluation

The project is implemented primarily in **Python and PyTorch** and uses a 1D Convolutional Neural Network (1D-CNN) for EEG window classification.

---

# Dataset

The project uses the **CHB-MIT Scalp EEG Database**, a publicly available dataset containing long-term multichannel scalp EEG recordings from pediatric patients with epilepsy.

The original EEG recordings are treated as the raw signal source for preprocessing, segmentation, normalization, model inference, and evaluation.

The dataset itself is **not included in this repository**.

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
Threshold Selection
        |
        v
Window-Level Evaluation
        |
        v
False Positive Analysis
        |
        v
Temporal Persistence Analysis
        |
        v
Patient-Level Evaluation
```

---

# Data Processing

## 1. Window Segmentation

Long EEG recordings are divided into fixed-length temporal windows.

Each window contains multichannel EEG activity and is assigned a seizure or non-seizure label according to the dataset annotations.

## 2. Normalization

EEG signals are normalized before being provided to the neural network.

Normalization reduces amplitude differences between channels and helps improve training stability.

## 3. Dataset Splitting

The data is separated into:

* Training set
* Validation set
* Test set

The test set remains isolated from model training and optimization.

The final evaluation uses the independent test set.

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

---

# Final Window-Level Evaluation

The final model was evaluated on the independent test set using a decision threshold of:

```text
Threshold = 0.95
```

The test set contains:

* **3114 total windows**
* **3021 non-seizure windows**
* **93 seizure windows**

## Final Performance

| Metric               |     Result |
| -------------------- | ---------: |
| ROC-AUC              | **0.9891** |
| Accuracy             | **97.72%** |
| Sensitivity / Recall | **83.87%** |
| Specificity          | **98.15%** |
| Precision            | **58.21%** |
| F1-score             | **68.72%** |

The ROC-AUC of approximately **0.989** indicates strong discrimination between seizure and non-seizure EEG windows.

The relatively lower precision compared with specificity reflects the remaining false-positive problem, which is a major focus of the subsequent error-analysis stage.

---

# Confusion Matrix

The final window-level confusion matrix at threshold 0.95 is:

| True / Predicted |       Non-Seizure |         Seizure |
| ---------------- | ----------------: | --------------: |
| **Non-Seizure**  | **2965 (98.15%)** |  **56 (1.85%)** |
| **Seizure**      |   **15 (16.13%)** | **78 (83.87%)** |

The percentages are **row-normalized percentages**, meaning that each row is normalized by the total number of samples belonging to the corresponding true class.

The corresponding counts are:

```text
TN = 2965
FP = 56
FN = 15
TP = 78
```

The generated visualization is available in:

```text
results/confusion_matrix.png
```

---

# Threshold Analysis

A range of decision thresholds was evaluated on the test predictions.

Increasing the threshold generally reduces false positives but also increases false negatives.

Selected operating points include:

| Threshold |   Accuracy | Sensitivity | Specificity |  Precision |         F1 |
| --------: | ---------: | ----------: | ----------: | ---------: | ---------: |
|      0.70 |     94.89% |      94.62% |      94.90% |     36.36% |     52.54% |
|      0.80 |     95.95% |      92.47% |      96.06% |     41.95% |     57.72% |
|      0.90 |     97.14% |      89.25% |      97.38% |     51.23% |     65.10% |
|  **0.95** | **97.72%** |  **83.87%** |  **98.15%** | **58.21%** | **68.72%** |
|      0.99 |     98.30% |      65.59% |      99.30% |     74.39% |     69.71% |

The threshold of **0.95** was selected as the final operating point for the reported window-level evaluation.

---

# ROC and Precision-Recall Analysis

The repository contains the following evaluation plots:

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

The ROC-AUC of the final model is approximately:

```text
0.9891
```

---

# False Positive Analysis

A substantial part of the project focuses on understanding and reducing false-positive seizure detections.

Several analysis scripts were developed to investigate the characteristics of false-positive windows.

The analysis includes:

* False-positive signal inspection
* Channel-level analysis
* Frequency-domain analysis
* EEG morphology analysis
* Artifact analysis
* False-positive versus true-positive comparison
* Window localization
* Validation-set false-positive analysis
* Feature-based filtering experiments
* Probability smoothing experiments
* Temporal persistence analysis

The purpose of these experiments is to determine whether false-positive detections exhibit temporal or signal-level characteristics that can be exploited without modifying the trained CNN.

---

# Temporal Persistence Analysis

Window-level classification treats each EEG window independently.

However, real seizure events are temporal phenomena. A single isolated high-probability window may therefore represent noise, artifact, or another transient EEG pattern rather than a true seizure.

To investigate this issue, temporal persistence rules were evaluated.

The final frozen rule uses:

```text
Fraction threshold = 0.005
Minimum runs = 0
Minimum cluster size = 4
Maximum gap = 2
```

The temporal rule operates as a **post-processing layer** on top of the CNN predictions.

The trained model itself is not modified.

---

# Patient-Level Evaluation

Patient-level aggregation was evaluated using the frozen temporal decision rule.

The evaluated patient-level set contains:

```text
5 patients
1 seizure-positive patient
4 seizure-negative patients
```

## Frozen Temporal Rule

| Metric      |   Result |
| ----------- | -------: |
| Sensitivity | **100%** |
| Specificity | **100%** |
| Precision   | **100%** |
| F1-score    | **100%** |

The corresponding patient-level confusion matrix is:

```text
TP = 1
FP = 0
FN = 0
TN = 4
```

## Comparison With Baseline q95 Rule

A baseline patient-level q95 aggregation rule produced:

| Metric      | Baseline q95 | Frozen Temporal Rule |
| ----------- | -----------: | -------------------: |
| Sensitivity |         100% |             **100%** |
| Specificity |          25% |             **100%** |
| Precision   |          25% |             **100%** |
| F1-score    |         0.40 |             **1.00** |

The temporal rule therefore eliminated the false-positive patient decisions observed with the baseline rule in this small evaluation set.

### Important Interpretation

The patient-level results should be interpreted cautiously because the evaluated patient-level sample contains only **five patients**.

Therefore, the 100% patient-level metrics demonstrate the behavior of the frozen rule on this evaluation set, but should **not** be interpreted as a definitive estimate of clinical generalization performance.

---

# Experimental Safeguards

The final evaluation includes explicit safeguards against test-set leakage.

According to the final evaluation record:

```text
Test set used for optimization: False
Model modified during final evaluation: False
Dataset modified during final evaluation: False
```

The temporal rule was frozen before the final patient-level evaluation.

The final test evaluation therefore does not involve retraining or modifying the CNN based on test-set outcomes.

---

# Main Findings

The current results demonstrate several important findings.

### 1. Strong Window-Level Discrimination

The CNN achieves:

```text
ROC-AUC = 0.9891
```

indicating strong separation between seizure and non-seizure EEG windows.

### 2. High Specificity at the Final Operating Point

At threshold 0.95:

```text
Specificity = 98.15%
```

while maintaining:

```text
Sensitivity = 83.87%
```

### 3. False Positives Remain an Important Challenge

Despite strong overall accuracy, the seizure-class precision is:

```text
58.21%
```

indicating that false-positive detections remain an important limitation at the window level.

### 4. Temporal Persistence Can Reduce False Patient-Level Alarms

The frozen temporal rule improved patient-level specificity from:

```text
25% → 100%
```

on the evaluated five-patient set while maintaining 100% sensitivity.

This result motivates further investigation of temporal post-processing for EEG seizure detection.

---

# Limitations

The current project has several limitations.

* The dataset is relatively imbalanced between seizure and non-seizure windows.
* Window-level false positives remain significant.
* Patient-level evaluation currently involves only five patients.
* The patient-level 100% metrics therefore have high statistical uncertainty.
* The current temporal rule is a post-processing method rather than a learned temporal model.
* Additional external validation is required before drawing conclusions about clinical deployment.
* EEG artifacts and patient-specific signal characteristics can affect model predictions.

---

# Future Work

Future development will focus on:

## Signal Processing

* EEG filtering
* Frequency-domain analysis
* Time-frequency analysis
* EEG frequency-band characterization
* Artifact characterization

## False Positive Reduction

* Improved temporal post-processing
* Patient-independent false-positive suppression
* Signal-quality assessment
* Artifact-aware classification
* Multi-feature false-positive filtering

## Temporal Modeling

Future experiments may investigate models that explicitly represent temporal context, such as:

* Temporal CNNs
* Recurrent neural networks
* LSTM/GRU architectures
* Transformer-based temporal models

## Patient-Level Generalization

A larger patient-level evaluation set will be required to determine whether the observed temporal-persistence improvement generalizes across patients.

---

# Project Structure

```text
EEG-Seizure-Detection/
│
├── data/                       # Dataset files (not included)
│
├── models/                     # Trained model weights
│
├── src/                        # Main source code
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
├── analysis/                   # Error analysis and experiments
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
│   └── validate_fp_suppression.py
│
├── results/                    # Evaluation results and figures
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── precision_recall_curve.png
│   ├── threshold_vs_metrics.png
│   ├── threshold_comparison_table.csv
│   ├── real_threshold_metrics.png
│   ├── real_threshold_table.csv
│   ├── patient_level_results_table.csv
│   └── final_summary.txt
│
├── README.md
└── .gitignore
```

---

# Technologies Used

* Python
* PyTorch
* NumPy
* SciPy
* scikit-learn
* Matplotlib
* EEG signal processing
* Digital signal processing
* Deep learning
* Statistical evaluation

---

# Reproducibility

The repository contains the source code used for model evaluation, threshold analysis, false-positive investigation, temporal persistence analysis, and final result generation.

The original CHB-MIT EEG dataset is not included because of its size and dataset distribution requirements.

The trained model weights and local dataset files should be placed in the appropriate directories before running the complete pipeline.

---

# Project Status

**Current status: Final evaluation and error-analysis stage**

The current version includes:

* 1D-CNN EEG classification
* Independent test-set evaluation
* ROC and precision-recall analysis
* Decision-threshold analysis
* Confusion matrix visualization
* False-positive characterization
* Validation-based false-positive experiments
* Patient-level aggregation
* Temporal persistence analysis
* Frozen patient-level temporal evaluation
* Final paper-oriented result generation

The next major step is broader patient-level validation and improved generalization of false-positive suppression.
