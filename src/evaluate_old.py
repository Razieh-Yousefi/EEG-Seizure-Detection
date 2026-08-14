import os
import sys
import json
import numpy as np

import torch
import torch.nn as nn

from torch.utils.data import DataLoader


# ============================================================
# FIX PYTHON PATH
# ============================================================

CURRENT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

PROJECT_DIR = os.path.dirname(
    CURRENT_DIR
)

sys.path.append(
    CURRENT_DIR
)


from chbmit_pytorch_dataset import CHBMITDataset



# ============================================================
# CHB-MIT EEG MODEL EVALUATION
# ============================================================


DATA_DIR = os.path.join(
    PROJECT_DIR,
    "data"
)


MODEL_DIR = os.path.join(
    PROJECT_DIR,
    "models"
)



# ============================================================
# DATA FILES
# ============================================================


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



# ============================================================
# MODEL FILE
# ============================================================


MODEL_PATH = os.path.join(
    MODEL_DIR,
    "best_chbmit_model.pt"
)



# ============================================================
# OUTPUT
# ============================================================


OUTPUT_JSON = os.path.join(
    CURRENT_DIR,
    "evaluation_results.json"
)



# ============================================================
# CONFIGURATION
# ============================================================


BATCH_SIZE = 16


CHANNELS = 23


SAMPLES_PER_WINDOW = 1280



DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)



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


            nn.BatchNorm1d(
                32
            ),


            nn.ReLU(),


            nn.MaxPool1d(
                kernel_size=4
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
                kernel_size=4
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




# ============================================================
# METRICS
# ============================================================


def calculate_metrics(
    y_true,
    y_pred
):


    y_true = np.asarray(
        y_true,
        dtype=np.int64
    )


    y_pred = np.asarray(
        y_pred,
        dtype=np.int64
    )



    tp = int(
        np.sum(
            (y_true == 1)
            &
            (y_pred == 1)
        )
    )


    tn = int(
        np.sum(
            (y_true == 0)
            &
            (y_pred == 0)
        )
    )


    fp = int(
        np.sum(
            (y_true == 0)
            &
            (y_pred == 1)
        )
    )


    fn = int(
        np.sum(
            (y_true == 1)
            &
            (y_pred == 0)
        )
    )


    total = len(
        y_true
    )


    accuracy = (

        (tp + tn) / total

        if total > 0

        else 0.0

    )


    precision = (

        tp / (tp + fp)

        if (tp + fp) > 0

        else 0.0

    )


    recall = (

        tp / (tp + fn)

        if (tp + fn) > 0

        else 0.0

    )


    specificity = (

        tn / (tn + fp)

        if (tn + fp) > 0

        else 0.0

    )


    f1 = (

        2 * precision * recall
        /
        (precision + recall)

        if (precision + recall) > 0

        else 0.0

    )


    return {

        "accuracy":
            float(accuracy),

        "precision":
            float(precision),

        "recall":
            float(recall),

        "sensitivity":
            float(recall),

        "specificity":
            float(specificity),

        "f1":
            float(f1),

        "tn":
            tn,

        "fp":
            fp,

        "fn":
            fn,

        "tp":
            tp

    }
# ============================================================
# THRESHOLD SEARCH
# ============================================================


def find_best_threshold(
    y_true,
    probabilities
):

    thresholds = np.arange(
        0.05,
        0.951,
        0.01
    )


    best_threshold = 0.50

    best_f1 = -1.0

    best_metrics = None



    for threshold in thresholds:


        predictions = (
            probabilities >= threshold
        ).astype(
            np.int64
        )


        metrics = calculate_metrics(
            y_true,
            predictions
        )



        if metrics["f1"] > best_f1:


            best_f1 = metrics["f1"]


            best_threshold = float(
                threshold
            )


            best_metrics = metrics



    return (

        best_threshold,

        best_metrics

    )



# ============================================================
# ROC-AUC
# ============================================================


def calculate_roc_auc(
    y_true,
    probabilities
):


    y_true = np.asarray(
        y_true
    )


    probabilities = np.asarray(
        probabilities
    )



    positives = np.sum(
        y_true == 1
    )


    negatives = np.sum(
        y_true == 0
    )



    if positives == 0 or negatives == 0:

        return None



    order = np.argsort(
        probabilities
    )



    sorted_labels = y_true[
        order
    ]



    positive_ranks = np.where(
        sorted_labels == 1
    )[0] + 1



    auc = (

        np.sum(
            positive_ranks
        )

        -

        positives *
        (positives + 1)
        /
        2

    ) / (

        positives *
        negatives

    )



    return float(
        auc
    )



# ============================================================
# PR-AUC
# ============================================================


def calculate_pr_auc(
    y_true,
    probabilities
):


    y_true = np.asarray(
        y_true,
        dtype=np.int64
    )


    probabilities = np.asarray(
        probabilities,
        dtype=np.float64
    )



    order = np.argsort(
        -probabilities
    )



    sorted_labels = y_true[
        order
    ]



    total_positive = np.sum(
        sorted_labels == 1
    )



    if total_positive == 0:

        return None



    tp = 0

    fp = 0



    precisions = []

    recalls = []



    for label in sorted_labels:



        if label == 1:

            tp += 1


        else:

            fp += 1



        precision = (

            tp /
            (tp + fp)

        )


        recall = (

            tp /
            total_positive

        )


        precisions.append(
            precision
        )


        recalls.append(
            recall
        )



    recalls = np.asarray(
        recalls
    )


    precisions = np.asarray(
        precisions
    )



    recalls = np.concatenate(
        [
            np.array(
                [0.0]
            ),

            recalls
        ]
    )


    precisions = np.concatenate(
        [
            np.array(
                [1.0]
            ),

            precisions
        ]
    )



    auc = np.trapz(
        precisions,
        recalls
    )



    return float(
        auc
    )




# ============================================================
# MAIN
# ============================================================


def main():


    print("=" * 70)

    print(
        "CHB-MIT EEG MODEL EVALUATION"
    )

    print("=" * 70)



    print()

    print(
        "Project directory:"
    )

    print(
        PROJECT_DIR
    )



    print()

    print(
        "Data directory:"
    )

    print(
        DATA_DIR
    )



    print()

    print(
        "Device:"
    )

    print(
        DEVICE
    )



    print()

    print(
        "PyTorch version:"
    )

    print(
        torch.__version__
    )



    # ========================================================
    # 1. CHECK FILES
    # ========================================================


    print()

    print("=" * 70)

    print(
        "1. CHECKING INPUT FILES"
    )

    print("=" * 70)



    required_files = [


        X_PATH,


        Y_PATH,


        TEST_INDICES_PATH,


        NORMALIZATION_PATH,


        MODEL_PATH


    ]



    for path in required_files:



        if not os.path.isfile(path):


            raise FileNotFoundError(

                f"Required file not found:\n{path}"

            )



        print(

            "[OK]",

            path

        )



    # ========================================================
    # 2. LOAD TEST DATASET
    # ========================================================


    print()

    print("=" * 70)

    print(
        "2. LOADING TEST DATASET"
    )

    print("=" * 70)



    test_dataset = CHBMITDataset(

        X_PATH,

        Y_PATH,

        TEST_INDICES_PATH,

        NORMALIZATION_PATH

    )



    print()

    print(
        "Test samples:",
        len(test_dataset)
    )



    # ========================================================
    # 3. TEST LABEL DISTRIBUTION
    # ========================================================


    print()

    print("=" * 70)

    print(
        "3. TEST LABEL DISTRIBUTION"
    )

    print("=" * 70)



    test_indices = np.load(
        TEST_INDICES_PATH
    )


    y_all = np.load(
        Y_PATH
    )



    test_labels = y_all[
        test_indices
    ]



    class_values, class_counts = np.unique(

        test_labels,

        return_counts=True

    )



    for label, count in zip(

        class_values,

        class_counts

    ):


        print(

            f"Class {int(label)}: {int(count)}"

        )



    test_seizures = int(

        np.sum(
            test_labels == 1
        )

    )



    test_non_seizures = int(

        np.sum(
            test_labels == 0
        )

    )



    print()

    print(
        "Seizure samples:",
        test_seizures
    )


    print(
        "Non-seizure samples:",
        test_non_seizures
    )
# ========================================================
# 4. CREATE TEST DATALOADER
# ========================================================


    print()

    print("=" * 70)

    print(
        "4. CREATING TEST DATALOADER"
    )

    print("=" * 70)



    test_loader = DataLoader(

        test_dataset,

        batch_size=BATCH_SIZE,

        shuffle=False,

        num_workers=0,

        pin_memory=torch.cuda.is_available()

    )



    print(
        "[OK] Test DataLoader created."
    )




    # ========================================================
    # 5. CREATE MODEL
    # ========================================================


    print()

    print("=" * 70)

    print(
        "5. CREATING MODEL"
    )

    print("=" * 70)



    model = EEGCNN()



    model = model.to(
        DEVICE
    )



    print(
        model
    )




    total_parameters = sum(

        p.numel()

        for p in model.parameters()

    )



    print()

    print(
        "Total parameters:",
        total_parameters
    )





    # ========================================================
    # 6. LOAD BEST MODEL
    # ========================================================


    print()

    print("=" * 70)

    print(
        "6. LOADING BEST MODEL"
    )

    print("=" * 70)




    checkpoint = torch.load(

        MODEL_PATH,

        map_location=DEVICE

    )




    if isinstance(

        checkpoint,

        dict

    ):


        if "model_state_dict" in checkpoint:



            model.load_state_dict(

                checkpoint[

                    "model_state_dict"

                ]

            )


        elif "state_dict" in checkpoint:



            model.load_state_dict(

                checkpoint[

                    "state_dict"

                ]

            )


        else:



            model.load_state_dict(

                checkpoint

            )



    else:



        model.load_state_dict(

            checkpoint

        )





    model.eval()



    print(
        "[OK] Best model loaded."
    )



    if isinstance(checkpoint, dict):


        if "epoch" in checkpoint:


            print(

                "Best epoch:",

                checkpoint["epoch"]

            )



        if "metrics" in checkpoint:



            if "f1" in checkpoint["metrics"]:



                print(

                    "Validation F1:",

                    checkpoint["metrics"]["f1"]

                )






    # ========================================================
    # 7. RUN TEST INFERENCE
    # ========================================================



    print()

    print("=" * 70)

    print(
        "7. RUNNING TEST INFERENCE"
    )

    print("=" * 70)





    all_probabilities = []

    all_predictions = []

    all_true_labels = []



    total_batches = len(
        test_loader
    )




    with torch.no_grad():



        for batch_number, (x, y) in enumerate(


            test_loader,

            start=1

        ):



            x = x.to(
                DEVICE
            )



            outputs = model(
                x
            )



            probabilities = torch.softmax(

                outputs,

                dim=1

            )[:, 1]



            predictions = (

                probabilities >= 0.50

            ).long()





            all_probabilities.extend(

                probabilities.cpu()

                .numpy()

                .tolist()

            )



            all_predictions.extend(

                predictions.cpu()

                .numpy()

                .tolist()

            )



            all_true_labels.extend(

                y.numpy()

                .tolist()

            )





            if (

                batch_number == 1

                or batch_number % 20 == 0

                or batch_number == total_batches

            ):



                print(

                    f"Processed "

                    f"{batch_number}/"

                    f"{total_batches} batches"

                )







    y_true = np.asarray(

        all_true_labels,

        dtype=np.int64

    )



    probabilities = np.asarray(

        all_probabilities,

        dtype=np.float64

    )



    predictions_050 = np.asarray(

        all_predictions,

        dtype=np.int64

    )





    print()

    print(

        "[OK] Test inference completed."

    )



    print()

    print(

        "Total evaluated samples:",

        len(y_true)

    )
    # ========================================================
    # 8. RESULTS AT THRESHOLD = 0.50
    # ========================================================


    print()

    print("=" * 70)

    print(
        "8. RESULTS AT THRESHOLD = 0.50"
    )

    print("=" * 70)



    metrics_050 = calculate_metrics(

        y_true,

        predictions_050

    )



    print()

    print(
        f"Accuracy    : {metrics_050['accuracy']:.4f}"
    )

    print(
        f"Precision   : {metrics_050['precision']:.4f}"
    )

    print(
        f"Recall      : {metrics_050['recall']:.4f}"
    )

    print(
        f"F1          : {metrics_050['f1']:.4f}"
    )

    print(
        f"Sensitivity : {metrics_050['sensitivity']:.4f}"
    )

    print(
        f"Specificity : {metrics_050['specificity']:.4f}"
    )

    print(
        f"TN={metrics_050['tn']} "
        f"FP={metrics_050['fp']} "
        f"FN={metrics_050['fn']} "
        f"TP={metrics_050['tp']}"
    )




    # ========================================================
    # 9. ROC-AUC
    # ========================================================


    print()

    print("=" * 70)

    print(
        "9. ROC-AUC"
    )

    print("=" * 70)



    roc_auc = calculate_roc_auc(

        y_true,

        probabilities

    )



    if roc_auc is None:


        print(
            "ROC-AUC unavailable"
        )


    else:


        print(
            f"ROC-AUC: {roc_auc:.6f}"
        )





    # ========================================================
    # 10. PR-AUC
    # ========================================================


    print()

    print("=" * 70)

    print(
        "10. PR-AUC"
    )

    print("=" * 70)



    pr_auc = calculate_pr_auc(

        y_true,

        probabilities

    )



    if pr_auc is None:


        print(
            "PR-AUC unavailable"
        )


    else:


        print(
            f"PR-AUC: {pr_auc:.6f}"
        )






    # ========================================================
    # 11. BEST THRESHOLD SEARCH
    # ========================================================


    print()

    print("=" * 70)

    print(
        "11. BEST THRESHOLD SEARCH"
    )

    print("=" * 70)




    best_threshold, best_threshold_metrics = find_best_threshold(

        y_true,

        probabilities

    )



    print()

    print(
        "Best threshold:",
        f"{best_threshold:.2f}"
    )


    print(
        f"Precision   : {best_threshold_metrics['precision']:.4f}"
    )

    print(
        f"Recall      : {best_threshold_metrics['recall']:.4f}"
    )

    print(
        f"F1          : {best_threshold_metrics['f1']:.4f}"
    )

    print(
        f"Sensitivity : {best_threshold_metrics['sensitivity']:.4f}"
    )

    print(
        f"Specificity : {best_threshold_metrics['specificity']:.4f}"
    )

    print(
        f"TN={best_threshold_metrics['tn']} "
        f"FP={best_threshold_metrics['fp']} "
        f"FN={best_threshold_metrics['fn']} "
        f"TP={best_threshold_metrics['tp']}"
    )






    # ========================================================
    # 12. SAVE RESULTS
    # ========================================================



    print()

    print("=" * 70)

    print(
        "12. SAVING RESULTS"
    )

    print("=" * 70)




    results = {


        "device":
            str(DEVICE),


        "model_path":
            MODEL_PATH,


        "test_samples":
            int(len(y_true)),


        "test_seizure_samples":
            int(test_seizures),


        "test_nonseizure_samples":
            int(test_non_seizures),


        "threshold_0.50":
            metrics_050,


        "roc_auc":
            roc_auc,


        "pr_auc":
            pr_auc,


        "best_threshold":
            float(best_threshold),


        "best_threshold_metrics":
            best_threshold_metrics

    }





    with open(

        OUTPUT_JSON,

        "w",

        encoding="utf-8"

    ) as f:


        json.dump(

            results,

            f,

            indent=4

        )



    print()

    print(
        "[OK] Evaluation saved:"
    )

    print(
        OUTPUT_JSON
    )






    # ========================================================
    # FINAL SUMMARY
    # ========================================================



    print()

    print("=" * 70)

    print(
        "FINAL EVALUATION SUMMARY"
    )

    print("=" * 70)



    print()

    print(
        "TEST RESULTS"
    )


    print(
        f"Accuracy    : {metrics_050['accuracy']:.4f}"
    )


    print(
        f"Precision   : {metrics_050['precision']:.4f}"
    )


    print(
        f"Recall      : {metrics_050['recall']:.4f}"
    )


    print(
        f"F1          : {metrics_050['f1']:.4f}"
    )


    print(
        f"Sensitivity : {metrics_050['sensitivity']:.4f}"
    )


    print(
        f"Specificity : {metrics_050['specificity']:.4f}"
    )



    if roc_auc is not None:

        print(
            f"ROC-AUC     : {roc_auc:.4f}"
        )


    if pr_auc is not None:

        print(
            f"PR-AUC      : {pr_auc:.4f}"
        )



    print()

    print(
        "BEST THRESHOLD"
    )


    print(
        f"Threshold   : {best_threshold:.2f}"
    )


    print(
        f"F1          : {best_threshold_metrics['f1']:.4f}"
    )


    print(
        f"Sensitivity : {best_threshold_metrics['sensitivity']:.4f}"
    )


    print(
        f"Specificity : {best_threshold_metrics['specificity']:.4f}"
    )



    print()

    print("=" * 70)

    print(
        "[SUCCESS] CHB-MIT MODEL EVALUATION COMPLETED"
    )

    print("=" * 70)



    print()

    print(
        "DONE"
    )




# ============================================================
# RUN
# ============================================================


if __name__ == "__main__":


    main()