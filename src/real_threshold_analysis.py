import os
import json
import csv
import numpy as np
import torch
import torch.nn as nn

import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

from chbmit_pytorch_dataset import CHBMITDataset



# =====================================================
# PATHS
# =====================================================

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


TEST_INDEX_PATH = os.path.join(
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



# =====================================================
# CONFIG
# =====================================================

CHANNELS = 23

BATCH_SIZE = 16


DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)



# =====================================================
# MODEL
# =====================================================


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

            nn.BatchNorm1d(32),

            nn.ReLU(),

            nn.MaxPool1d(4),



            nn.Conv1d(
                32,
                64,
                7,
                padding=3
            ),

            nn.BatchNorm1d(64),

            nn.ReLU(),

            nn.MaxPool1d(4),



            nn.Conv1d(
                64,
                128,
                5,
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

            nn.Dropout(0.4),

            nn.Linear(
                64,
                2
            )

        )


    def forward(
        self,
        x
    ):

        x=self.features(x)

        x=self.classifier(x)

        return x




# =====================================================
# METRICS
# =====================================================


def metrics(
    y_true,
    y_pred
):


    tp=np.sum(
        (y_true==1)&
        (y_pred==1)
    )


    tn=np.sum(
        (y_true==0)&
        (y_pred==0)
    )


    fp=np.sum(
        (y_true==0)&
        (y_pred==1)
    )


    fn=np.sum(
        (y_true==1)&
        (y_pred==0)
    )


    precision = (
        tp/(tp+fp)
        if tp+fp>0
        else 0
    )


    recall = (
        tp/(tp+fn)
        if tp+fn>0
        else 0
    )


    f1 = (

        2*precision*recall/
        (precision+recall)

        if precision+recall>0

        else 0

    )


    accuracy=(tp+tn)/len(y_true)


    specificity=(

        tn/(tn+fp)

        if tn+fp>0

        else 0

    )


    return {

        "precision":float(precision),

        "recall":float(recall),

        "sensitivity":float(recall),

        "specificity":float(specificity),

        "accuracy":float(accuracy),

        "f1":float(f1),

        "tp":int(tp),

        "tn":int(tn),

        "fp":int(fp),

        "fn":int(fn)

    }



# =====================================================
# LOAD MODEL
# =====================================================


def load_model():

    model=EEGCNN()

    model.to(DEVICE)


    checkpoint=torch.load(
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




# =====================================================
# INFERENCE
# =====================================================


def inference(
    model,
    loader
):

    probs=[]

    labels=[]


    with torch.no_grad():

        for x,y in loader:


            x=x.to(DEVICE)


            out=model(x)


            p=torch.softmax(
                out,
                dim=1
            )[:,1]


            probs.extend(
                p.cpu().numpy()
            )


            labels.extend(
                y.numpy()
            )


    return (

        np.array(labels),

        np.array(probs)

    )




# =====================================================
# MAIN
# =====================================================


def main():


    print("="*70)

    print(
        "REAL THRESHOLD ANALYSIS"
    )

    print("="*70)



    dataset=CHBMITDataset(

        X_PATH,

        Y_PATH,

        TEST_INDEX_PATH,

        NORMALIZATION_PATH

    )


    loader=DataLoader(

        dataset,

        batch_size=BATCH_SIZE,

        shuffle=False

    )



    model=load_model()


    y_true, probabilities=inference(

        model,

        loader

    )



    thresholds=np.arange(
        0.01,
        1.00,
        0.01
    )



    results=[]



    for t in thresholds:


        pred=(

            probabilities>=t

        ).astype(int)


        m=metrics(

            y_true,

            pred

        )


        m["threshold"]=float(t)


        results.append(m)




    # SAVE JSON


    with open(

        os.path.join(
            RESULT_DIR,
            "real_threshold_analysis.json"
        ),

        "w"

    ) as f:


        json.dump(

            results,

            f,

            indent=4

        )



    # CSV


    with open(

        os.path.join(
            RESULT_DIR,
            "real_threshold_table.csv"
        ),

        "w",

        newline=""

    ) as f:


        writer=csv.DictWriter(

            f,

            fieldnames=results[0].keys()

        )


        writer.writeheader()

        writer.writerows(results)



    # BEST VALUES


    best_f1=max(
        results,
        key=lambda x:x["f1"]
    )


    best_recall=max(
        results,
        key=lambda x:x["recall"]
    )


    best_precision=max(
        results,
        key=lambda x:x["precision"]
    )



    print()

    print("BEST F1")

    print(best_f1)


    print()

    print("BEST RECALL")

    print(best_recall)


    print()

    print("BEST PRECISION")

    print(best_precision)




    # PLOTS


    th=[r["threshold"] for r in results]


    plt.figure(
        figsize=(8,5)
    )


    plt.plot(
        th,
        [r["precision"] for r in results],
        label="Precision"
    )


    plt.plot(
        th,
        [r["recall"] for r in results],
        label="Recall"
    )


    plt.plot(
        th,
        [r["f1"] for r in results],
        label="F1"
    )


    plt.xlabel(
        "Threshold"
    )


    plt.ylabel(
        "Score"
    )


    plt.grid()

    plt.legend()


    plt.savefig(

        os.path.join(
            RESULT_DIR,
            "real_threshold_metrics.png"
        ),

        dpi=300

    )


    plt.close()



    print()

    print(
        "[DONE]"
    )



if __name__=="__main__":

    main()