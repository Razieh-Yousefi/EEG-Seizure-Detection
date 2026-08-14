import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# ============================================================
# CHB-MIT PYTORCH DATASET
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

X_PATH = os.path.join(
    BASE_DIR,
    "X_chbmit_full.npy"
)

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

NORMALIZATION_PATH = os.path.join(
    BASE_DIR,
    "normalization_params.npz"
)


# ============================================================
# EXPECTED SHAPE
# ============================================================

EXPECTED_CHANNELS = 23
EXPECTED_SAMPLES_PER_WINDOW = 1280


# ============================================================
# DATASET CLASS
# ============================================================

class CHBMITDataset(Dataset):

    def __init__(
        self,
        X_path,
        y_path,
        indices_path,
        normalization_path
    ):

        self.X_path = X_path
        self.y_path = y_path
        self.indices_path = indices_path
        self.normalization_path = normalization_path

        # ----------------------------------------------------
        # Load X as memory-mapped array
        # ----------------------------------------------------

        self.X = np.load(
            self.X_path,
            mmap_mode="r"
        )

        # ----------------------------------------------------
        # Load labels
        # ----------------------------------------------------

        self.y = np.load(
            self.y_path
        )

        # ----------------------------------------------------
        # Load split indices
        # ----------------------------------------------------

        self.indices = np.load(
            self.indices_path
        )

        # ----------------------------------------------------
        # Load normalization parameters
        # ----------------------------------------------------

        normalization = np.load(
            self.normalization_path
        )

        self.mean = np.asarray(
            normalization["channel_mean"],
            dtype=np.float32
        )

        self.std = np.asarray(
            normalization["channel_std"],
            dtype=np.float32
        )

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        self._validate()

    def _validate(self):

        if self.X.ndim != 3:

            raise ValueError(
                f"X must be 3D, got {self.X.ndim}D"
            )

        if self.X.shape[1] != EXPECTED_CHANNELS:

            raise ValueError(
                f"Expected {EXPECTED_CHANNELS} channels, "
                f"got {self.X.shape[1]}"
            )

        if self.X.shape[2] != EXPECTED_SAMPLES_PER_WINDOW:

            raise ValueError(
                f"Expected {EXPECTED_SAMPLES_PER_WINDOW} "
                f"samples per window, "
                f"got {self.X.shape[2]}"
            )

        if len(self.y) != len(self.X):

            raise ValueError(
                "X and y sample counts do not match."
            )

        if len(self.indices) == 0:

            raise ValueError(
                "Dataset split contains zero samples."
            )

        if np.any(self.indices < 0):

            raise ValueError(
                "Negative sample index detected."
            )

        if np.any(
            self.indices >= len(self.X)
        ):

            raise ValueError(
                "Sample index exceeds X size."
            )

        if self.mean.shape != (
            EXPECTED_CHANNELS,
        ):

            raise ValueError(
                "Invalid normalization mean shape."
            )

        if self.std.shape != (
            EXPECTED_CHANNELS,
        ):

            raise ValueError(
                "Invalid normalization std shape."
            )

        if np.any(
            ~np.isfinite(self.mean)
        ):

            raise ValueError(
                "Normalization mean contains NaN/Inf."
            )

        if np.any(
            ~np.isfinite(self.std)
        ):

            raise ValueError(
                "Normalization std contains NaN/Inf."
            )

        if np.any(
            self.std <= 0
        ):

            raise ValueError(
                "Normalization std contains "
                "zero or negative values."
            )

    def __len__(self):

        return len(self.indices)

    def __getitem__(self, index):

        # ----------------------------------------------------
        # Original sample index
        # ----------------------------------------------------

        sample_index = int(
            self.indices[index]
        )

        # ----------------------------------------------------
        # Read one EEG window
        # Shape:
        # (23, 1280)
        # ----------------------------------------------------

        sample = np.asarray(
            self.X[sample_index],
            dtype=np.float32
        )

        # ----------------------------------------------------
        # Channel-wise normalization
        # ----------------------------------------------------

        sample = (
            sample
            - self.mean[:, None]
        ) / self.std[:, None]

        # ----------------------------------------------------
        # Convert to torch tensor
        # ----------------------------------------------------

        x = torch.from_numpy(
            sample.copy()
        )

        # ----------------------------------------------------
        # Label
        # ----------------------------------------------------

        y = torch.tensor(
            int(self.y[sample_index]),
            dtype=torch.long
        )

        return x, y


# ============================================================
# CREATE DATASETS
# ============================================================

def create_datasets():

    train_dataset = CHBMITDataset(
        X_PATH,
        Y_PATH,
        TRAIN_INDICES_PATH,
        NORMALIZATION_PATH
    )

    val_dataset = CHBMITDataset(
        X_PATH,
        Y_PATH,
        VAL_INDICES_PATH,
        NORMALIZATION_PATH
    )

    test_dataset = CHBMITDataset(
        X_PATH,
        Y_PATH,
        TEST_INDICES_PATH,
        NORMALIZATION_PATH
    )

    return (
        train_dataset,
        val_dataset,
        test_dataset
    )


# ============================================================
# CREATE DATALOADERS
# ============================================================

def create_dataloaders(
    batch_size=32,
    num_workers=0
):

    (
        train_dataset,
        val_dataset,
        test_dataset
    ) = create_datasets()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    return (
        train_loader,
        val_loader,
        test_loader
    )


# ============================================================
# BASIC TEST
# ============================================================

def main():

    print("=" * 70)
    print("CHB-MIT PYTORCH DATASET TEST")
    print("=" * 70)

    print()
    print("Base directory:")
    print(BASE_DIR)

    # --------------------------------------------------------
    # Create datasets
    # --------------------------------------------------------

    (
        train_dataset,
        val_dataset,
        test_dataset
    ) = create_datasets()

    print()
    print("Dataset sizes:")
    print(
        "Train:",
        len(train_dataset)
    )
    print(
        "Validation:",
        len(val_dataset)
    )
    print(
        "Test:",
        len(test_dataset)
    )

    # --------------------------------------------------------
    # Check one sample from each split
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("CHECKING INDIVIDUAL SAMPLES")
    print("=" * 70)

    for name, dataset in [
        ("TRAIN", train_dataset),
        ("VALIDATION", val_dataset),
        ("TEST", test_dataset)
    ]:

        x, y = dataset[0]

        print()
        print(name)

        print(
            "X shape:",
            tuple(x.shape)
        )

        print(
            "X dtype:",
            x.dtype
        )

        print(
            "y:",
            y.item()
        )

        print(
            "X min:",
            float(torch.min(x))
        )

        print(
            "X max:",
            float(torch.max(x))
        )

        print(
            "X mean:",
            float(torch.mean(x))
        )

        print(
            "X std:",
            float(torch.std(x))
        )

        if x.shape != (
            EXPECTED_CHANNELS,
            EXPECTED_SAMPLES_PER_WINDOW
        ):

            raise RuntimeError(
                f"{name} shape is incorrect."
            )

        if y.item() not in [0, 1]:

            raise RuntimeError(
                f"{name} label is invalid."
            )

        if not torch.isfinite(x).all():

            raise RuntimeError(
                f"{name} contains NaN or Inf."
            )

        print(
            "[OK] Sample is valid."
        )

    # --------------------------------------------------------
    # Create DataLoaders
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("CREATING DATALOADERS")
    print("=" * 70)

    (
        train_loader,
        val_loader,
        test_loader
    ) = create_dataloaders(
        batch_size=32,
        num_workers=0
    )

    # --------------------------------------------------------
    # Check one batch
    # --------------------------------------------------------

    print()
    print("Loading one TRAIN batch...")

    train_x, train_y = next(
        iter(train_loader)
    )

    print()
    print("TRAIN BATCH")

    print(
        "X shape:",
        tuple(train_x.shape)
    )

    print(
        "X dtype:",
        train_x.dtype
    )

    print(
        "y shape:",
        tuple(train_y.shape)
    )

    print(
        "y dtype:",
        train_y.dtype
    )

    print(
        "X min:",
        float(torch.min(train_x))
    )

    print(
        "X max:",
        float(torch.max(train_x))
    )

    print(
        "X mean:",
        float(torch.mean(train_x))
    )

    print(
        "X std:",
        float(torch.std(train_x))
    )

    print(
        "Labels in batch:",
        torch.unique(train_y).tolist()
    )

    # --------------------------------------------------------
    # Final checks
    # --------------------------------------------------------

    if train_x.shape != (
        32,
        EXPECTED_CHANNELS,
        EXPECTED_SAMPLES_PER_WINDOW
    ):

        raise RuntimeError(
            "TRAIN batch shape is incorrect."
        )

    if train_y.shape != (32,):

        raise RuntimeError(
            "TRAIN label batch shape is incorrect."
        )

    if not torch.isfinite(train_x).all():

        raise RuntimeError(
            "TRAIN batch contains NaN or Inf."
        )

    print()
    print("=" * 70)
    print("[SUCCESS] PYTORCH DATASET/DATALOADER TEST PASSED")
    print("=" * 70)

    print()
    print(
        "Dataset is ready for model training."
    )

    print(
        "Normalization is applied on-the-fly."
    )

    print(
        "Original X file remains unchanged."
    )

    print(
        "No normalized copy of X was created."
    )

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()


import sys
sys.path.append("..")