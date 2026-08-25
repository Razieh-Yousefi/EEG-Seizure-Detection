import os
import json
import numpy as np
import torch
import torch.nn as nn

import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_curve,
    auc,
    precision_recall_curve,
    confusion_matrix,
    classification_report
)

from torch.utils.data import DataLoader

from chbmit_pytorch_dataset import CHBMITDataset



# ============================================================
# PATHS
# ============================================================

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


RESULT_DIR = os.path.join(
    PROJECT_DIR,
    "results"
)


os.makedirs(
    RESULT_DIR,
    exist_ok=True
)



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



JSON_PATH = os.path.join(
    PROJECT_DIR,
    "src",
    "evaluation_results_v2.json"
)



DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)



BATCH_SIZE = 16

CHANNELS = 23



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
                7,
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
                7,
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
                5,
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


    def forward(self,x):

        x = self.features(x)

        x = self.classifier(x)

        return x



# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    model = EEGCNN()

    model.to(
        DEVICE
    )


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


    model.eval()

    return model



# ============================================================
# INFERENCE
# ============================================================

def inference(
    model,
    loader
):

    probabilities=[]

    labels=[]


    with torch.no_grad():

        for x,y in loader:


            x=x.to(
                DEVICE
            )


            output=model(x)


            prob=torch.softmax(
                output,
                dim=1
            )[:,1]


            probabilities.extend(
                prob.cpu().numpy()
            )


            labels.extend(
                y.numpy()
            )


    return (

        np.array(labels),

        np.array(probabilities)

    )



# ============================================================
# MAIN
# ============================================================

def main():

    print("="*70)

    print(
        "GENERATING EVALUATION PLOTS"
    )

    print("="*70)



    with open(
        JSON_PATH,
        "r"
    ) as f:

        results=json.load(f)



    threshold = results[
        "selected_threshold"
    ]



    print(
        "Using threshold:",
        threshold
    )



    test_dataset = CHBMITDataset(

        X_PATH,

        Y_PATH,

        TEST_INDICES_PATH,

        NORMALIZATION_PATH

    )



    test_loader=DataLoader(

        test_dataset,

        batch_size=BATCH_SIZE,

        shuffle=False

    )



    model=load_model()



    y_true, y_prob = inference(

        model,

        test_loader

    )



    y_pred=(

        y_prob >= threshold

    ).astype(int)



    # ========================================================
    # CONFUSION MATRIX
    # ========================================================


    cm=confusion_matrix(

        y_true,

        y_pred

    )


    plt.figure(
        figsize=(6,5)
    )


    plt.imshow(
        cm
    )


    plt.title(
        "Confusion Matrix"
    )


    plt.xlabel(
        "Predicted"
    )


    plt.ylabel(
        "True"
    )


    for i in range(2):

        for j in range(2):

            plt.text(

                j,

                i,

                cm[i,j],

                ha="center",

                va="center"

            )


    plt.colorbar()


    plt.savefig(

        os.path.join(

            RESULT_DIR,

            "confusion_matrix.png"

        ),

        dpi=300,

        bbox_inches="tight"

    )


    plt.close()



    # ========================================================
    # ROC CURVE
    # ========================================================


    fpr,tpr,_=roc_curve(

        y_true,

        y_prob

    )


    roc_auc=auc(

        fpr,

        tpr

    )


    plt.figure(
        figsize=(6,5)
    )


    plt.plot(

        fpr,

        tpr,

        label=f"AUC={roc_auc:.4f}"

    )


    plt.plot(

        [0,1],

        [0,1],

        "--"

    )


    plt.xlabel(
        "False Positive Rate"
    )


    plt.ylabel(
        "True Positive Rate"
    )


    plt.title(
        "ROC Curve"
    )


    plt.legend()



    plt.savefig(

        os.path.join(

            RESULT_DIR,

            "roc_curve.png"

        ),

        dpi=300,

        bbox_inches="tight"

    )


    plt.close()



    # ========================================================
    # PRECISION RECALL CURVE
    # ========================================================


    precision,recall,_=precision_recall_curve(

        y_true,

        y_prob

    )


    plt.figure(
        figsize=(6,5)
    )


    plt.plot(

        recall,

        precision

    )


    plt.xlabel(
        "Recall"
    )


    plt.ylabel(
        "Precision"
    )


    plt.title(
        "Precision Recall Curve"
    )


    plt.savefig(

        os.path.join(

            RESULT_DIR,

            "precision_recall_curve.png"

        ),

        dpi=300,

        bbox_inches="tight"

    )


    plt.close()



    # ========================================================
    # REPORT
    # ========================================================


    report={

        "threshold":

            float(threshold),


        "roc_auc":

            float(roc_auc),


        "classification_report":

            classification_report(

                y_true,

                y_pred,

                output_dict=True

            )

    }



    with open(

        os.path.join(

            RESULT_DIR,

            "final_report.json"

        ),

        "w"

    ) as f:


        json.dump(

            report,

            f,

            indent=4

        )



    print()

    print(
        "[DONE]"
    )

    print(
        "Results saved in:"
    )

    print(
        RESULT_DIR
    )



if __name__=="__main__":

    main()