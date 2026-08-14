import os
import json
import numpy as np
import torch
import torch.nn as nn


# ============================================================
# CONFIG
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

TEST_INDICES_PATH = os.path.join(
    BASE_DIR,
    "test_indices.npy"
)

PROB_PATH = os.path.join(
    BASE_DIR,
    "test_window_probabilities.npz"
)

THRESHOLD_PATH = os.path.join(
    BASE_DIR,
    "validation_threshold_results.json"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "best_chbmit_model.pt"
)

OUTPUT_PATH = os.path.join(
    BASE_DIR,
    "fp_saliency_full_results.json"
)


CHANNELS = 23

DEVICE = torch.device("cpu")



# ============================================================
# MODEL
# ============================================================

class EEGCNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.features = nn.Sequential(

            nn.Conv1d(
                CHANNELS,
                32,
                kernel_size=7,
                padding=3
            ),

            nn.BatchNorm1d(32),

            nn.ReLU(),

            nn.MaxPool1d(4),


            nn.Conv1d(
                32,
                64,
                kernel_size=7,
                padding=3
            ),

            nn.BatchNorm1d(64),

            nn.ReLU(),

            nn.MaxPool1d(4),


            nn.Conv1d(
                64,
                128,
                kernel_size=5,
                padding=2
            ),

            nn.BatchNorm1d(128),

            nn.ReLU(),

            nn.AdaptiveAvgPool1d(1)
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


    def forward(self,x):

        x = self.features(x)

        x = self.classifier(x)

        return x



# ============================================================
# HEADER
# ============================================================

print("="*70)
print("FALSE-POSITIVE FULL SALIENCY ANALYSIS")
print("="*70)


print()
print("Device:")
print(DEVICE)



# ============================================================
# LOAD DATA
# ============================================================

X = np.load(
    X_PATH,
    mmap_mode="r"
)

y = np.load(
    Y_PATH
)


test_indices = np.load(
    TEST_INDICES_PATH
)


probs = np.load(
    PROB_PATH
)["probabilities"]



with open(
    THRESHOLD_PATH,
    "r"
) as f:

    threshold = json.load(f)["best_threshold"]



print()
print("X:", X.shape)

print(
    "Test:",
    len(test_indices)
)

print(
    "Threshold:",
    threshold
)



# ============================================================
# FIND FP / TP
# ============================================================

preds = (
    probs >= threshold
).astype(int)


labels = y[test_indices]


fp_positions = test_indices[
    (preds == 1) &
    (labels == 0)
]


tp_positions = test_indices[
    (preds == 1) &
    (labels == 1)
]


print()

print(
    "FP:",
    len(fp_positions)
)

print(
    "TP:",
    len(tp_positions)
)



# ============================================================
# LOAD MODEL
# ============================================================

print()
print("Loading model...")


model = EEGCNN()


checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)


model.load_state_dict(
    checkpoint["model_state_dict"]
)


model.eval()


print("[OK] Model loaded")



# ============================================================
# SALIENCY
# ============================================================


def calculate_saliency(signal):

    x = torch.tensor(
        signal,
        dtype=torch.float32,
        device=DEVICE
    )

    # add batch dimension
    x = x.unsqueeze(0)

    # make leaf tensor
    x.requires_grad_(True)

    output = model(x)

    score = output[0, 1]

    model.zero_grad()

    score.backward()

    saliency = (
        x.grad
        .abs()
        .detach()
        .cpu()
        .numpy()[0]
    )

    channel_saliency = (
        saliency.mean(axis=1)
    )

    return channel_saliency



# ============================================================
# ANALYSIS
# ============================================================

results = {

    "false_positive": [],

    "true_positive": []

}



print()
print("Analyzing FP...")


for idx in fp_positions[:50]:

    signal = np.asarray(
        X[idx],
        dtype=np.float32
    )


    sal = calculate_saliency(
        signal
    )


    results["false_positive"].append({

        "index":
            int(idx),

        "channel_saliency":
            sal.tolist(),

        "max_channel":
            int(np.argmax(sal)),

        "max_saliency":
            float(np.max(sal))

    })



print()
print("Analyzing TP...")


for idx in tp_positions[:50]:

    signal = np.asarray(
        X[idx],
        dtype=np.float32
    )


    sal = calculate_saliency(
        signal
    )


    results["true_positive"].append({

        "index":
            int(idx),

        "channel_saliency":
            sal.tolist(),

        "max_channel":
            int(np.argmax(sal)),

        "max_saliency":
            float(np.max(sal))

    })



# ============================================================
# SAVE
# ============================================================

with open(
    OUTPUT_PATH,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        results,
        f,
        indent=2
    )



print()
print("[OK] Saved:")
print(OUTPUT_PATH)


print()
print("="*70)
print("FULL SALIENCY ANALYSIS COMPLETED")
print("="*70)