# ================================================================
# save_validation_probabilities.py
#
# Generate individual validation-window seizure probabilities.
#
# IMPORTANT:
# - Model is NOT modified.
# - Dataset is NOT modified.
# - Test set is NOT used.
# - Training-time normalization is reused exactly.
# ================================================================

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


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

MODEL_DIR = os.path.join(
    PROJECT_DIR,
    "models"
)

RESULTS_DIR = os.path.join(
    PROJECT_DIR,
    "results"
)

OUTPUT_FILE = os.path.join(
    RESULTS_DIR,
    "validation_window_probabilities.npz"
)


# ================================================================
# 2. INPUT FILES
# ================================================================

X_FILE = os.path.join(
    DATA_DIR,
    "X_chbmit_full.npy"
)

Y_FILE = os.path.join(
    DATA_DIR,
    "y_chbmit_full.npy"
)

PATIENTS_FILE = os.path.join(
    DATA_DIR,
    "patients_chbmit_full.npy"
)

NORMALIZATION_FILE = os.path.join(
    DATA_DIR,
    "normalization_params.npz"
)

MODEL_FILE = os.path.join(
    MODEL_DIR,
    "best_chbmit_model.pt"
)


# ================================================================
# 3. VALIDATION INDICES
# ================================================================

VALIDATION_INDEX_CANDIDATES = [

    os.path.join(
        DATA_DIR,
        "validation_indices.npy"
    ),

    os.path.join(
        DATA_DIR,
        "val_indices.npy"
    ),

    os.path.join(
        DATA_DIR,
        "valid_indices.npy"
    )
]


# ================================================================
# 4. DEVICE
# ================================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

BATCH_SIZE = 32


# ================================================================
# HEADER
# ================================================================

print()
print("=" * 70)
print("SAVE VALIDATION WINDOW PROBABILITIES")
print("=" * 70)

print()
print("Project directory:")
print(PROJECT_DIR)

print()
print("Data directory:")
print(DATA_DIR)

print()
print("Model directory:")
print(MODEL_DIR)

print()
print("Results directory:")
print(RESULTS_DIR)

print()
print("Device:")
print(DEVICE)


# ================================================================
# 5. FIND VALIDATION INDICES
# ================================================================

print()
print("=" * 70)
print("1. LOCATING VALIDATION INDICES")
print("=" * 70)

VALIDATION_INDEX_FILE = None

for candidate in VALIDATION_INDEX_CANDIDATES:

    if os.path.exists(candidate):

        VALIDATION_INDEX_FILE = candidate

        break


if VALIDATION_INDEX_FILE is None:

    print()
    print(
        "Could not find validation indices."
    )

    print()
    print(
        "Searched:"
    )

    for candidate in VALIDATION_INDEX_CANDIDATES:

        print(
            " -",
            candidate
        )

    print()
    print(
        "Please locate the validation index file "
        "before continuing."
    )

    raise FileNotFoundError(
        "Validation indices file not found."
    )


print()
print(
    "[OK] Validation indices found:"
)

print(
    VALIDATION_INDEX_FILE
)


# ================================================================
# 6. CHECK INPUT FILES
# ================================================================

print()
print("=" * 70)
print("2. CHECKING INPUT FILES")
print("=" * 70)

required_files = [

    X_FILE,
    Y_FILE,
    PATIENTS_FILE,
    NORMALIZATION_FILE,
    MODEL_FILE,
    VALIDATION_INDEX_FILE
]


for path in required_files:

    if os.path.exists(path):

        print(
            "[OK]",
            path
        )

    else:

        print(
            "[MISSING]",
            path
        )

        raise FileNotFoundError(
            path
        )


# ================================================================
# 7. LOAD DATA
# ================================================================

print()
print("=" * 70)
print("3. LOADING DATA")
print("=" * 70)

X = np.load(
    X_FILE
)

y = np.load(
    Y_FILE
)

patients = np.load(
    PATIENTS_FILE,
    allow_pickle=True
)

validation_indices = np.load(
    VALIDATION_INDEX_FILE
)


print()
print(
    "X shape:",
    X.shape
)

print(
    "y shape:",
    y.shape
)

print(
    "patients shape:",
    patients.shape
)

print(
    "validation indices shape:",
    validation_indices.shape
)


# ================================================================
# 8. VALIDATE INDICES
# ================================================================

print()
print("=" * 70)
print("4. VERIFYING VALIDATION INDICES")
print("=" * 70)

if len(validation_indices) == 0:

    raise RuntimeError(
        "Validation index array is empty."
    )


if np.min(validation_indices) < 0:

    raise RuntimeError(
        "Validation indices contain negative values."
    )


if np.max(validation_indices) >= len(X):

    raise RuntimeError(
        "Validation index exceeds dataset size."
    )


print()
print(
    "[OK] Validation index range:"
)

print(
    "min =",
    int(np.min(validation_indices))
)

print(
    "max =",
    int(np.max(validation_indices))
)


# ================================================================
# 9. LOAD NORMALIZATION
# ================================================================

print()
print("=" * 70)
print("5. LOADING NORMALIZATION PARAMETERS")
print("=" * 70)

norm_data = np.load(
    NORMALIZATION_FILE
)

channel_mean = np.asarray(
    norm_data["channel_mean"],
    dtype=np.float32
)

channel_std = np.asarray(
    norm_data["channel_std"],
    dtype=np.float32
)


if channel_mean.shape != (X.shape[1],):

    raise ValueError(
        "channel_mean shape mismatch."
    )


if channel_std.shape != (X.shape[1],):

    raise ValueError(
        "channel_std shape mismatch."
    )


if not np.all(
    np.isfinite(channel_mean)
):

    raise ValueError(
        "channel_mean contains NaN/Inf."
    )


if not np.all(
    np.isfinite(channel_std)
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
    "[OK] Normalization parameters loaded."
)


# ================================================================
# 10. PREPARE VALIDATION DATA
# ================================================================

print()
print("=" * 70)
print("6. PREPARING VALIDATION DATA")
print("=" * 70)

X_validation = X[
    validation_indices
]

y_validation = y[
    validation_indices
]

patients_validation = patients[
    validation_indices
]


print()
print(
    "Raw validation samples:",
    len(X_validation)
)

print(
    "Raw X_validation shape:",
    X_validation.shape
)


# ================================================================
# 11. APPLY EXACT TRAINING NORMALIZATION
# ================================================================

X_validation = (

    X_validation.astype(
        np.float32
    )

    - channel_mean[
        None,
        :,
        None
    ]

) / channel_std[
    None,
    :,
    None
]


X_validation = np.ascontiguousarray(
    X_validation,
    dtype=np.float32
)


print()
print(
    "[OK] Exact training-time normalization applied."
)


# ================================================================
# 12. DATASET
# ================================================================

class EEGDataset(Dataset):

    def __init__(
        self,
        X_data
    ):

        self.X = X_data


    def __len__(
        self
    ):

        return len(
            self.X
        )


    def __getitem__(
        self,
        idx
    ):

        return torch.tensor(
            self.X[idx],
            dtype=torch.float32
        )


validation_dataset = EEGDataset(
    X_validation
)


validation_loader = DataLoader(
    validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ================================================================
# 13. MODEL DEFINITION
# ================================================================

print()
print("=" * 70)
print("7. CREATING MODEL")
print("=" * 70)


class EEGCNN(nn.Module):

    def __init__(
        self,
        n_channels=23
    ):

        super().__init__()


        self.features = nn.Sequential(

            nn.Conv1d(
                n_channels,
                32,
                kernel_size=7,
                padding=3
            ),

            nn.BatchNorm1d(
                32
            ),

            nn.ReLU(),

            nn.MaxPool1d(
                4
            ),

            nn.Conv1d(
                32,
                64,
                kernel_size=7,
                padding=3
            ),

            nn.BatchNorm1d(
                64
            ),

            nn.ReLU(),

            nn.MaxPool1d(
                4
            ),

            nn.Conv1d(
                64,
                128,
                kernel_size=5,
                padding=2
            ),

            nn.BatchNorm1d(
                128
            ),

            nn.ReLU(),

            nn.AdaptiveAvgPool1d(
                1
            )
        )


        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(
                128,
                64
            ),

            nn.ReLU(),

            nn.Dropout(
                0.4
            ),

            nn.Linear(
                64,
                2
            )
        )


    def forward(
        self,
        x
    ):

        x = self.features(
            x
        )

        x = self.classifier(
            x
        )

        return x


model = EEGCNN(
    n_channels=X_validation.shape[1]
)

model = model.to(
    DEVICE
)


# ================================================================
# 14. LOAD MODEL
# ================================================================

print()
print("=" * 70)
print("8. LOADING BEST MODEL")
print("=" * 70)

checkpoint = torch.load(
    MODEL_FILE,
    map_location=DEVICE
)


if isinstance(
    checkpoint,
    dict
):

    if "model_state_dict" in checkpoint:

        state_dict = checkpoint[
            "model_state_dict"
        ]

    elif "state_dict" in checkpoint:

        state_dict = checkpoint[
            "state_dict"
        ]

    else:

        state_dict = checkpoint

else:

    state_dict = checkpoint


model.load_state_dict(
    state_dict
)

model.eval()


print()
print(
    "[OK] Best model loaded."
)


# ================================================================
# 15. RUN VALIDATION INFERENCE
# ================================================================

print()
print("=" * 70)
print("9. RUNNING VALIDATION INFERENCE")
print("=" * 70)

probabilities = []


with torch.no_grad():

    total_batches = len(
        validation_loader
    )


    for batch_idx, inputs in enumerate(
        validation_loader,
        start=1
    ):

        inputs = inputs.to(
            DEVICE
        )


        logits = model(
            inputs
        )


        batch_probabilities = (
            torch.softmax(
                logits,
                dim=1
            )[:, 1]
        )


        probabilities.extend(
            batch_probabilities
            .cpu()
            .numpy()
            .tolist()
        )


        if (

            batch_idx == 1

            or batch_idx % 20 == 0

            or batch_idx == total_batches

        ):

            print(
                f"Processed "
                f"{batch_idx}/{total_batches} batches"
            )


probabilities = np.asarray(
    probabilities,
    dtype=np.float32
)


# ================================================================
# 16. VERIFY ALIGNMENT
# ================================================================

print()
print("=" * 70)
print("10. VERIFYING OUTPUT ALIGNMENT")
print("=" * 70)


if len(probabilities) != len(
    validation_indices
):

    raise RuntimeError(
        "Probability count does not match "
        "validation index count."
    )


if len(probabilities) != len(
    y_validation
):

    raise RuntimeError(
        "Probability count does not match labels."
    )


if len(probabilities) != len(
    patients_validation
):

    raise RuntimeError(
        "Probability count does not match patients."
    )


if not np.all(
    np.isfinite(probabilities)
):

    raise RuntimeError(
        "Validation probabilities contain NaN/Inf."
    )


print()
print(
    "[OK] Number of probabilities:",
    len(probabilities)
)

print(
    "[OK] Number of validation indices:",
    len(validation_indices)
)

print(
    "[OK] Number of labels:",
    len(y_validation)
)

print(
    "[OK] Number of patients:",
    len(patients_validation)
)


# ================================================================
# 17. SAVE
# ================================================================

print()
print("=" * 70)
print("11. SAVING VALIDATION PROBABILITIES")
print("=" * 70)


np.savez(
    OUTPUT_FILE,

    validation_indices=np.asarray(
        validation_indices,
        dtype=np.int64
    ),

    patients=np.asarray(
        patients_validation
    ),

    labels=np.asarray(
        y_validation,
        dtype=np.int64
    ),

    probabilities=np.asarray(
        probabilities,
        dtype=np.float32
    )
)


print()
print(
    "[OK] Saved:"
)

print(
    OUTPUT_FILE
)


# ================================================================
# 18. QUICK VERIFICATION
# ================================================================

print()
print("=" * 70)
print("12. QUICK VERIFICATION")
print("=" * 70)


saved = np.load(
    OUTPUT_FILE,
    allow_pickle=True
)


print()
print(
    "Saved arrays:"
)


for key in saved.files:

    print(
        f"{key:20s}: "
        f"shape={saved[key].shape}"
    )


print()
print(
    "Probability min:"
)

print(
    f"{saved['probabilities'].min():.6f}"
)


print()
print(
    "Probability max:"
)

print(
    f"{saved['probabilities'].max():.6f}"
)


print()
print(
    "Probability mean:"
)

print(
    f"{saved['probabilities'].mean():.6f}"
)


# ================================================================
# 19. CLASS DISTRIBUTION
# ================================================================

print()
print(
    "Validation class distribution:"
)

unique, counts = np.unique(
    saved["labels"],
    return_counts=True
)

for label, count in zip(
    unique,
    counts
):

    print(
        f"Class {int(label)}: {int(count)}"
    )


# ================================================================
# 20. FINAL
# ================================================================

print()
print("=" * 70)
print("VALIDATION PROBABILITY SAVING COMPLETED")
print("=" * 70)

print()
print(
    "Model was NOT modified."
)

print(
    "Dataset was NOT modified."
)

print(
    "Test set was NOT used."
)

print()
print(
    "Output:"
)

print(
    OUTPUT_FILE
)

print()
print("=" * 70)