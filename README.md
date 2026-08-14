# EEG Seizure Detection Using Deep Learning

## Project Overview

This project focuses on automatic epileptic seizure detection from EEG (Electroencephalography) signals using deep learning techniques.

The main goal is to develop a machine learning-based system capable of distinguishing seizure and non-seizure EEG segments using the CHB-MIT Scalp EEG Database.

The project combines concepts from:

- Biomedical signal processing
- Digital signal processing
- Machine learning
- Deep learning
- EEG analysis


---

# Dataset

The project uses the CHB-MIT Scalp EEG Database, which contains long-term EEG recordings from pediatric patients with epilepsy.

The dataset consists of multichannel EEG signals recorded using clinical EEG systems.

The original EEG recordings have already undergone acquisition, sampling, digitization, and storage processes before being provided as a public dataset.

In this project, the dataset is treated as the input signal source for further signal processing and classification.


---

# Project Pipeline

The implemented processing pipeline is:
EEG Dataset
|
↓
Data Loading
|
↓
Signal Windowing
|
↓
Normalization
|
↓
CNN-based Classification Model
|
↓
Probability Estimation
|
↓
Threshold Optimization
|
↓
Seizure / Non-Seizure Decision


---

# Data Processing

The EEG signals are processed using the following steps:

## 1. Window Segmentation

Long EEG recordings are divided into fixed-length temporal windows.

Each window represents a short segment of multichannel EEG activity.


## 2. Normalization

Signal normalization is applied to reduce amplitude variations between EEG channels and improve neural network training stability.


## 3. Dataset Splitting

The dataset is divided into:

- Training set
- Validation set
- Test set

The test data remains completely separated from training to provide unbiased performance evaluation.


---

# Deep Learning Model

A one-dimensional Convolutional Neural Network (1D-CNN) architecture is used for EEG classification.

The model receives multichannel EEG windows and learns temporal patterns associated with seizure activity.

Model structure:
Input EEG Channels
|
↓
1D Convolution Layers
|
↓
Batch Normalization
|
↓
Activation Functions
|
↓
Pooling Layers
|
↓
Fully Connected Classifier
|
↓
Seizure Probability


---

# Current Evaluation Results

The current trained model was evaluated on the independent test set.

Test dataset:

- Total test samples: 3114
- Seizure samples: 93
- Non-seizure samples: 3021


Performance:

| Metric | Value |
|--------|-------|
| Accuracy | 93.42% |
| Sensitivity (Recall) | 96.77% |
| Specificity | 93.31% |
| ROC-AUC | 0.989 |
| F1-score | 0.467 |


The model achieves high seizure detection sensitivity and strong discrimination ability.

However, the main current challenge is reducing False Positive detections caused by non-seizure EEG patterns being classified as seizures.


---

# Error Analysis and Current Challenges

Detailed analysis was performed to investigate false positive cases.

Current limitations:

- High number of false alarms
- Class imbalance between seizure and non-seizure samples
- Lack of temporal post-processing
- Limited utilization of EEG signal characteristics


Future improvements will focus on reducing false positives while maintaining high seizure detection sensitivity.


---

# Future Development Plan

The next stages of the project include:

## Signal Processing Improvements

- Digital filtering of EEG signals
- Frequency domain analysis using FFT
- Time-frequency analysis
- Extraction of EEG frequency band features


## False Positive Reduction

- Temporal smoothing
- Detection continuity analysis
- Patient-specific threshold optimization
- Post-processing decision algorithms


## Communication Systems Perspective

To strengthen the electrical engineering aspect of the project, future development will investigate:

- EEG signal acquisition chain
- Sampling and quantization effects
- Noise and artifact analysis
- Channel characteristics
- Signal transmission and measurement limitations


---
## Project Structure

```text
EEG_Seizure_Project/
│
├── data/                  # Dataset files (ignored)
├── raw_data/              # Raw EEG files (ignored)
├── models/                # Trained model weights
├── results/               # Evaluation results
├── scripts/               # Data preparation and experiment scripts
│
├── src/                   # Main source code
│   ├── train.py
│   ├── evaluate.py
│   ├── read_eeg.py
│   └── chbmit_pytorch_dataset.py
│
├── analysis/              # Error analysis and visualization scripts
│
├── README.md
└── .gitignore
```

---

# Technologies Used

- Python
- PyTorch
- NumPy
- Signal Processing Techniques
- Deep Learning
