# ================================================================
# save_test_probabilities.py
#
# Generate model probabilities for test windows.
#
# PURPOSE:
# - Load trained CHB-MIT model
# - Run inference on test set
# - Save seizure probabilities
#
# NO MODEL MODIFICATION
# NO DATA MODIFICATION
# ================================================================


import os
import sys
import numpy as np
import torch

from torch.utils.data import DataLoader



# ================================================================
# PROJECT PATHS
# ================================================================

PROJECT_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)


SRC_DIR = os.path.join(
    PROJECT_DIR,
    "src"
)


sys.path.append(
    SRC_DIR
)



# ================================================================
# IMPORT PROJECT MODULES
# ================================================================

from chbmit_pytorch_dataset import CHBMITDataset

from generate_evaluation_plots import EEGCNN



# ================================================================
# DIRECTORIES
# ================================================================

DATA_DIR = os.path.join(
    PROJECT_DIR,
    "data"
)


MODEL_DIR = os.path.join(
    PROJECT_DIR,
    "models"
)



# ================================================================
# FILE PATHS
# ================================================================

X_PATH = os.path.join(
    DATA_DIR,
    "X_chbmit_full.npy"
)


Y_PATH = os.path.join(
    DATA_DIR,
    "y_chbmit_full.npy"
)


TEST_INDICES_PATH = os.path.join(
    DATA_DIR,
    "test_indices.npy"
)


NORMALIZATION_PATH = os.path.join(
    DATA_DIR,
    "normalization_params.npz"
)


MODEL_PATH = os.path.join(
    MODEL_DIR,
    "best_chbmit_model.pt"
)


OUT_FILE = os.path.join(
    DATA_DIR,
    "test_window_probabilities.npz"
)



# ================================================================
# DEVICE
# ================================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)



print("=" * 70)
print("SAVING TEST WINDOW PROBABILITIES")
print("=" * 70)


print()

print("Project:")
print(PROJECT_DIR)

print()

print("Device:")
print(DEVICE)



# ================================================================
# LOAD MODEL
# ================================================================


print()
print("=" * 70)
print("1. LOADING MODEL")
print("=" * 70)


model = EEGCNN()


checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)



if "model_state_dict" in checkpoint:

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

else:

    model.load_state_dict(
        checkpoint
    )



model.to(
    DEVICE
)


model.eval()


print("[OK] Model loaded")



# ================================================================
# LOAD DATASET
# ================================================================


print()
print("=" * 70)
print("2. LOADING TEST DATA")
print("=" * 70)



dataset = CHBMITDataset(

    X_PATH,

    Y_PATH,

    TEST_INDICES_PATH,

    NORMALIZATION_PATH

)



loader = DataLoader(

    dataset,

    batch_size=16,

    shuffle=False

)



print()

print(
    "Test samples:",
    len(dataset)
)



# ================================================================
# INFERENCE
# ================================================================


print()
print("=" * 70)
print("3. RUNNING INFERENCE")
print("=" * 70)



all_probs = []

all_labels = []



with torch.no_grad():


    for x, y in loader:


        x = x.to(
            DEVICE
        )


        output = model(
            x
        )


        prob = torch.softmax(
            output,
            dim=1
        )[:, 1]



        all_probs.extend(
            prob.cpu().numpy()
        )


        all_labels.extend(
            y.numpy()
        )



all_probs = np.array(
    all_probs
)


all_labels = np.array(
    all_labels
)



print()

print(
    "Probability samples:",
    len(all_probs)
)



# ================================================================
# SAVE
# ================================================================


print()
print("=" * 70)
print("4. SAVING RESULTS")
print("=" * 70)



np.savez(

    OUT_FILE,

    probabilities=all_probs,

    labels=all_labels

)



print()

print("[OK] Saved:")

print(
    OUT_FILE
)



print()

print("=" * 70)
print("COMPLETED")
print("=" * 70)