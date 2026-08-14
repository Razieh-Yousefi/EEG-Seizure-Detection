import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader


# ============================================================
# CHB-MIT PYTORCH TRAINING PIPELINE
# STEP 1: DATA PIPELINE CHECK
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

print("=" * 70)
print("CHB-MIT PYTORCH TRAINING PIPELINE")
print("STEP 1: DATA PIPELINE CHECK")
print("=" * 70)

print()
print("Base directory:")
print(BASE_DIR)


# ============================================================
# IMPORT DATASET
# ============================================================

DATASET_FILE = os.path.join(
    BASE_DIR,
    "chbmit_pytorch_dataset.py"
)

if not os.path.exists(DATASET_FILE):

    raise FileNotFoundError(
        "\nDataset file not found:\n"
        + DATASET_FILE
    )

print()
print("[OK] Dataset file found:")
print(DATASET_FILE)


# Add current directory to Python path
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)


try:

    from chbmit_pytorch_dataset import (
        CHBMITDataset
    )

except ImportError as e:

    raise ImportError(
        "\nCould not import CHBMITDataset.\n"
        "Check chbmit_pytorch_dataset.py\n\n"
        f"Original error: {e}"
    )


print()
print("[OK] CHBMITDataset imported successfully.")


# ============================================================
# DEVICE
# ============================================================

print()
print("=" * 70)
print("CHECKING PYTORCH")
print("=" * 70)

print()
print("PyTorch version:")
print(torch.__version__)

print()
print("CUDA available:")
print(torch.cuda.is_available())

if torch.cuda.is_available():

    DEVICE = torch.device("cuda")

    print()
    print("[OK] CUDA GPU detected.")

else:

    DEVICE = torch.device("cpu")

    print()
    print("[INFO] CUDA is not available.")
    print("[INFO] Training will use CPU.")


print()
print("Selected device:")
print(DEVICE)


# ============================================================
# DATASET CREATION
# ============================================================

print()
print("=" * 70)
print("CREATING DATASETS")
print("=" * 70)

print()
print("Creating TRAIN dataset...")

train_dataset = CHBMITDataset(
    split="train"
)

print(
    "Train samples:",
    len(train_dataset)
)


print()
print("Creating VALIDATION dataset...")

val_dataset = CHBMITDataset(
    split="val"
)

print(
    "Validation samples:",
    len(val_dataset)
)


print()
print("Creating TEST dataset...")

test_dataset = CHBMITDataset(
    split="test"
)

print(
    "Test samples:",
    len(test_dataset)
)


# ============================================================
# DATASET SIZE CHECK
# ============================================================

print()
print("=" * 70)
print("CHECKING DATASET SIZES")
print("=" * 70)

EXPECTED_TRAIN = 14296
EXPECTED_VAL = 3070
EXPECTED_TEST = 3114

sizes_ok = True


if len(train_dataset) != EXPECTED_TRAIN:

    print(
        "[ERROR] Unexpected TRAIN dataset size:"
    )

    print(
        "Expected:",
        EXPECTED_TRAIN
    )

    print(
        "Actual:",
        len(train_dataset)
    )

    sizes_ok = False

else:

    print(
        "[OK] TRAIN:",
        len(train_dataset)
    )


if len(val_dataset) != EXPECTED_VAL:

    print(
        "[ERROR] Unexpected VALIDATION dataset size:"
    )

    print(
        "Expected:",
        EXPECTED_VAL
    )

    print(
        "Actual:",
        len(val_dataset)
    )

    sizes_ok = False

else:

    print(
        "[OK] VALIDATION:",
        len(val_dataset)
    )


if len(test_dataset) != EXPECTED_TEST:

    print(
        "[ERROR] Unexpected TEST dataset size:"
    )

    print(
        "Expected:",
        EXPECTED_TEST
    )

    print(
        "Actual:",
        len(test_dataset)
    )

    sizes_ok = False

else:

    print(
        "[OK] TEST:",
        len(test_dataset)
    )


if not sizes_ok:

    raise RuntimeError(
        "Dataset size check failed."
    )


# ============================================================
# LABEL DISTRIBUTION
# ============================================================

print()
print("=" * 70)
print("CHECKING LABEL DISTRIBUTION")
print("=" * 70)


Y_PATH = os.path.join(
    BASE_DIR,
    "y_chbmit_full.npy"
)

TRAIN_INDICES_PATH = os.path.join(
    BASE_DIR,
    "train_indices.npy"
)

VAL_INDICES_PATH = os.path.join(
    BASE_DIR,
    "val_indices.npy"
)

TEST_INDICES_PATH = os.path.join(
    BASE_DIR,
    "test_indices.npy"
)


y = np.load(
    Y_PATH
)

train_indices = np.load(
    TRAIN_INDICES_PATH
)

val_indices = np.load(
    VAL_INDICES_PATH
)

test_indices = np.load(
    TEST_INDICES_PATH
)


train_labels = y[
    train_indices
]

val_labels = y[
    val_indices
]

test_labels = y[
    test_indices
]


def print_label_statistics(
    name,
    labels
):

    seizure = int(
        np.sum(labels == 1)
    )

    non_seizure = int(
        np.sum(labels == 0)
    )

    total = len(labels)

    seizure_percentage = (
        seizure / total * 100.0
    )

    non_seizure_percentage = (
        non_seizure / total * 100.0
    )

    print()
    print(name)
    print("-" * 50)

    print(
        "Total:",
        total
    )

    print(
        "Seizure:",
        seizure,
        f"({seizure_percentage:.3f}%)"
    )

    print(
        "Non-seizure:",
        non_seizure,
        f"({non_seizure_percentage:.3f}%)"
    )


print_label_statistics(
    "TRAIN",
    train_labels
)

print_label_statistics(
    "VALIDATION",
    val_labels
)

print_label_statistics(
    "TEST",
    test_labels
)


# ============================================================
# CREATE DATALOADERS
# ============================================================

print()
print("=" * 70)
print("CREATING DATALOADERS")
print("=" * 70)

BATCH_SIZE = 32

print()
print("Batch size:")
print(BATCH_SIZE)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=False
)


val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=False
)


test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=False
)


print()
print("[OK] TRAIN DataLoader created.")

print(
    "Number of TRAIN batches:",
    len(train_loader)
)

print()
print("[OK] VALIDATION DataLoader created.")

print(
    "Number of VALIDATION batches:",
    len(val_loader)
)

print()
print("[OK] TEST DataLoader created.")

print(
    "Number of TEST batches:",
    len(test_loader)
)


# ============================================================
# TEST ONE SAMPLE
# ============================================================

print()
print("=" * 70)
print("TESTING INDIVIDUAL DATASET SAMPLES")
print("=" * 70)


for split_name, dataset in [
    ("TRAIN", train_dataset),
    ("VALIDATION", val_dataset),
    ("TEST", test_dataset)
]:

    print()
    print(split_name)

    X_sample, y_sample = dataset[0]

    print(
        "X shape:",
        tuple(X_sample.shape)
    )

    print(
        "X dtype:",
        X_sample.dtype
    )

    print(
        "y:",
        y_sample.item()
    )

    print(
        "y dtype:",
        y_sample.dtype
    )

    print(
        "X min:",
        float(torch.min(X_sample))
    )

    print(
        "X max:",
        float(torch.max(X_sample))
    )

    print(
        "X mean:",
        float(torch.mean(X_sample))
    )

    print(
        "X std:",
        float(torch.std(X_sample))
    )

    if X_sample.shape != (
        23,
        1280
    ):

        raise RuntimeError(
            f"{split_name} sample has "
            f"incorrect shape: "
            f"{tuple(X_sample.shape)}"
        )

    if X_sample.dtype != torch.float32:

        raise RuntimeError(
            f"{split_name} sample "
            f"is not float32."
        )

    if y_sample.dtype != torch.int64:

        raise RuntimeError(
            f"{split_name} label "
            f"is not int64."
        )

    if not torch.isfinite(
        X_sample
    ).all():

        raise RuntimeError(
            f"{split_name} sample "
            f"contains NaN or Inf."
        )

    print(
        "[OK] Sample is valid."
    )


# ============================================================
# TEST TRAIN BATCH
# ============================================================

print()
print("=" * 70)
print("TESTING TRAIN BATCH")
print("=" * 70)

train_batch_X, train_batch_y = next(
    iter(train_loader)
)


print()
print("TRAIN BATCH")

print(
    "X shape:",
    tuple(train_batch_X.shape)
)

print(
    "X dtype:",
    train_batch_X.dtype
)

print(
    "y shape:",
    tuple(train_batch_y.shape)
)

print(
    "y dtype:",
    train_batch_y.dtype
)

print(
    "X min:",
    float(torch.min(train_batch_X))
)

print(
    "X max:",
    float(torch.max(train_batch_X))
)

print(
    "X mean:",
    float(torch.mean(train_batch_X))
)

print(
    "X std:",
    float(torch.std(train_batch_X))
)

print(
    "Labels in batch:",
    torch.unique(
        train_batch_y
    ).tolist()
)


# ============================================================
# BATCH VALIDATION
# ============================================================

if train_batch_X.shape != (
    BATCH_SIZE,
    23,
    1280
):

    raise RuntimeError(
        "TRAIN batch has unexpected shape."
    )


if train_batch_X.dtype != torch.float32:

    raise RuntimeError(
        "TRAIN batch is not float32."
    )


if train_batch_y.dtype != torch.int64:

    raise RuntimeError(
        "TRAIN labels are not int64."
    )


if not torch.isfinite(
    train_batch_X
).all():

    raise RuntimeError(
        "TRAIN batch contains NaN or Inf."
    )


if not torch.all(
    (train_batch_y == 0)
    | (train_batch_y == 1)
):

    raise RuntimeError(
        "TRAIN batch contains "
        "invalid labels."
    )


print()
print(
    "[OK] TRAIN batch is valid."
)


# ============================================================
# TEST VALIDATION BATCH
# ============================================================

print()
print("=" * 70)
print("TESTING VALIDATION BATCH")
print("=" * 70)

val_batch_X, val_batch_y = next(
    iter(val_loader)
)

print()
print(
    "Validation X shape:",
    tuple(val_batch_X.shape)
)

print(
    "Validation y shape:",
    tuple(val_batch_y.shape)
)

print(
    "Validation X dtype:",
    val_batch_X.dtype
)

print(
    "Validation X min:",
    float(torch.min(val_batch_X))
)

print(
    "Validation X max:",
    float(torch.max(val_batch_X))
)

print(
    "Validation X mean:",
    float(torch.mean(val_batch_X))
)

print(
    "Validation X std:",
    float(torch.std(val_batch_X))
)

if not torch.isfinite(
    val_batch_X
).all():

    raise RuntimeError(
        "Validation batch contains NaN or Inf."
    )

print()
print(
    "[OK] Validation batch is valid."
)


# ============================================================
# TEST TEST BATCH
# ============================================================

print()
print("=" * 70)
print("TESTING TEST BATCH")
print("=" * 70)

test_batch_X, test_batch_y = next(
    iter(test_loader)
)

print()
print(
    "Test X shape:",
    tuple(test_batch_X.shape)
)

print(
    "Test y shape:",
    tuple(test_batch_y.shape)
)

print(
    "Test X dtype:",
    test_batch_X.dtype
)

print(
    "Test X min:",
    float(torch.min(test_batch_X))
)

print(
    "Test X max:",
    float(torch.max(test_batch_X))
)

print(
    "Test X mean:",
    float(torch.mean(test_batch_X))
)

print(
    "Test X std:",
    float(torch.std(test_batch_X))
)

if not torch.isfinite(
    test_batch_X
).all():

    raise RuntimeError(
        "Test batch contains NaN or Inf."
    )

print()
print(
    "[OK] Test batch is valid."
)


# ============================================================
# FINAL
# ============================================================

print()
print("=" * 70)
print("FINAL DATA PIPELINE CHECK")
print("=" * 70)

print()
print("[SUCCESS] PYTORCH TRAINING DATA PIPELINE IS READY")

print()
print("Train samples:", len(train_dataset))
print("Validation samples:", len(val_dataset))
print("Test samples:", len(test_dataset))

print()
print("Input shape:")
print("(23 channels, 1280 samples)")

print()
print("Batch size:")
print(BATCH_SIZE)

print()
print("Device:")
print(DEVICE)

print()
print("Normalization:")
print("TRAIN-ONLY channel-wise z-score")

print()
print("Data loading:")
print("ON-THE-FLY")

print()
print("Original X file:")
print("UNCHANGED")

print()
print("Next step:")
print("CREATE AND TRAIN THE PYTORCH CNN")

print()
print("=" * 70)
print("DONE")
print("=" * 70)

import sys
sys.path.append("..")