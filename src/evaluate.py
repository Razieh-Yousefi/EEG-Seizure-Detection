import os
import json
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from chbmit_pytorch_dataset import CHBMITDataset


# ============================================================
# CHB-MIT EEG MODEL EVALUATION V2
# Validation threshold selection + final test evaluation
# ============================================================


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)


SRC_DIR = os.path.dirname(
    os.path.abspath(__file__)
)


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


TRAIN_INDICES_PATH = os.path.join(
    DATA_DIR,
    "train_indices.npy"
)


VAL_INDICES_PATH = os.path.join(
    DATA_DIR,
    "val_indices.npy"
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
# MODEL
# ============================================================

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "best_chbmit_model.pt"
)


# ============================================================
# OUTPUT
# ============================================================

OUTPUT_JSON = os.path.join(
    SRC_DIR,
    "evaluation_results_v2.json"
)


# ============================================================
# CONFIG
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
# MODEL DEFINITION
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


    accuracy = (
        (tp + tn)
        /
        len(y_true)
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
        2
        *
        precision
        *
        recall
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
# THRESHOLD SEARCH ON VALIDATION SET
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
# INFERENCE FUNCTION
# ============================================================

def run_inference(
    model,
    loader
):

    model.eval()


    all_probabilities = []

    all_labels = []


    total_batches = len(
        loader
    )


    with torch.no_grad():


        for batch_index, (
            x,
            y
        ) in enumerate(
            loader,
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
            )[:,1]


            all_probabilities.extend(
                probabilities.cpu().numpy()
            )


            all_labels.extend(
                y.numpy()
            )


            if (
                batch_index == 1
                or batch_index % 20 == 0
                or batch_index == total_batches
            ):

                print(
                    f"Processed "
                    f"{batch_index}/"
                    f"{total_batches} batches"
                )


    return (

        np.asarray(
            all_labels,
            dtype=np.int64
        ),

        np.asarray(
            all_probabilities,
            dtype=np.float64
        )
    )



# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    model = EEGCNN()


    model = model.to(
        DEVICE
    )


    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )


    if isinstance(
        checkpoint,
        dict
    ):


        if (
            "model_state_dict"
            in checkpoint
        ):


            model.load_state_dict(
                checkpoint[
                    "model_state_dict"
                ]
            )


        elif (
            "state_dict"
            in checkpoint
        ):


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


    return model



# ============================================================
# CHECK FILES
# ============================================================

def check_files():


    required_files = [

        X_PATH,

        Y_PATH,

        VAL_INDICES_PATH,

        TEST_INDICES_PATH,

        NORMALIZATION_PATH,

        MODEL_PATH

    ]


    for path in required_files:


        if not os.path.isfile(
            path
        ):

            raise FileNotFoundError(
                f"Missing file:\n{path}"
            )


        print(
            "[OK]",
            path
        )
# ============================================================
# MAIN FUNCTION
# ============================================================

def main():

    print("=" * 70)
    print("CHB-MIT EEG MODEL EVALUATION V2")
    print("=" * 70)


    print()

    print("Project directory:")
    print(PROJECT_DIR)


    print()

    print("Data directory:")
    print(DATA_DIR)


    print()

    print("Device:")
    print(DEVICE)


    print()

    print("PyTorch version:")
    print(torch.__version__)



    # ========================================================
    # 1. CHECK FILES
    # ========================================================

    print()

    print("=" * 70)
    print("1. CHECKING INPUT FILES")
    print("=" * 70)


    check_files()



    # ========================================================
    # 2. LOAD LABELS
    # ========================================================

    print()

    print("=" * 70)
    print("2. LOADING DATA INFORMATION")
    print("=" * 70)



    y_all = np.load(
        Y_PATH
    )


    val_indices = np.load(
        VAL_INDICES_PATH
    )


    test_indices = np.load(
        TEST_INDICES_PATH
    )


    print()

    print(
        "Total samples:",
        len(y_all)
    )


    print(
        "Validation samples:",
        len(val_indices)
    )


    print(
        "Test samples:",
        len(test_indices)
    )



    # ========================================================
    # 3. CREATE VALIDATION DATASET
    # ========================================================

    print()

    print("=" * 70)
    print("3. CREATING VALIDATION DATASET")
    print("=" * 70)



    val_dataset = CHBMITDataset(
        X_PATH,
        Y_PATH,
        VAL_INDICES_PATH,
        NORMALIZATION_PATH
    )


    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )


    print(
        "Validation samples:",
        len(val_dataset)
    )



    # ========================================================
    # 4. CREATE TEST DATASET
    # ========================================================

    print()

    print("=" * 70)
    print("4. CREATING TEST DATASET")
    print("=" * 70)



    test_dataset = CHBMITDataset(
        X_PATH,
        Y_PATH,
        TEST_INDICES_PATH,
        NORMALIZATION_PATH
    )


    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )


    print(
        "Test samples:",
        len(test_dataset)
    )



    # ========================================================
    # 5. LOAD MODEL
    # ========================================================

    print()

    print("=" * 70)
    print("5. LOADING TRAINED MODEL")
    print("=" * 70)



    model = load_model()


    print(
        "[OK] Model loaded"
    )



    # ========================================================
    # 6. VALIDATION INFERENCE
    # ========================================================

    print()

    print("=" * 70)
    print("6. VALIDATION INFERENCE")
    print("=" * 70)



    y_val, val_probabilities = run_inference(
        model,
        val_loader
    )


    print()

    print(
        "[OK] Validation inference completed"
    )



    # ========================================================
    # 7. FIND BEST THRESHOLD
    # ========================================================

    print()

    print("=" * 70)
    print("7. FINDING BEST THRESHOLD FROM VALIDATION")
    print("=" * 70)



    best_threshold, val_best_metrics = find_best_threshold(
        y_val,
        val_probabilities
    )


    print()

    print(
        "Best threshold:",
        f"{best_threshold:.2f}"
    )


    print(
        "Validation F1:",
        f"{val_best_metrics['f1']:.4f}"
    )


    print(
        "Validation Precision:",
        f"{val_best_metrics['precision']:.4f}"
    )


    print(
        "Validation Recall:",
        f"{val_best_metrics['recall']:.4f}"
    )
    # ========================================================
    # 8. TEST INFERENCE
    # ========================================================

    print()

    print("=" * 70)
    print("8. TEST INFERENCE")
    print("=" * 70)


    y_test, test_probabilities = run_inference(
        model,
        test_loader
    )


    print()

    print(
        "[OK] Test inference completed"
    )


    # ========================================================
    # 9. APPLY VALIDATION THRESHOLD ON TEST
    # ========================================================

    print()

    print("=" * 70)
    print("9. FINAL TEST RESULTS")
    print("=" * 70)



    test_predictions = (
        test_probabilities >= best_threshold
    ).astype(
        np.int64
    )


    test_metrics = calculate_metrics(
        y_test,
        test_predictions
    )



    print()

    print(
        "Threshold used:",
        f"{best_threshold:.2f}"
    )


    print(
        f"Accuracy    : {test_metrics['accuracy']:.4f}"
    )

    print(
        f"Precision   : {test_metrics['precision']:.4f}"
    )

    print(
        f"Recall      : {test_metrics['recall']:.4f}"
    )

    print(
        f"F1          : {test_metrics['f1']:.4f}"
    )

    print(
        f"Sensitivity : {test_metrics['sensitivity']:.4f}"
    )

    print(
        f"Specificity : {test_metrics['specificity']:.4f}"
    )

    print(
        f"TN={test_metrics['tn']} "
        f"FP={test_metrics['fp']} "
        f"FN={test_metrics['fn']} "
        f"TP={test_metrics['tp']}"
    )



    # ========================================================
    # 10. SAVE RESULTS
    # ========================================================

    print()

    print("=" * 70)
    print("10. SAVING RESULTS")
    print("=" * 70)



    results = {

        "device":
            str(DEVICE),


        "model_path":
            MODEL_PATH,


        "threshold_source":
            "validation_set",


        "selected_threshold":
            float(best_threshold),


        "validation_metrics_at_threshold":
            val_best_metrics,


        "test_metrics":
            test_metrics,


        "test_samples":
            int(len(y_test)),


        "validation_samples":
            int(len(y_val))

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
        "[OK] Results saved:"
    )

    print(
        OUTPUT_JSON
    )



    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()

    print("=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)


    print()

    print(
        "Validation selected threshold:",
        f"{best_threshold:.2f}"
    )


    print()

    print(
        "FINAL TEST PERFORMANCE"
    )


    print(
        f"Accuracy    : {test_metrics['accuracy']:.4f}"
    )

    print(
        f"Precision   : {test_metrics['precision']:.4f}"
    )

    print(
        f"Recall      : {test_metrics['recall']:.4f}"
    )

    print(
        f"F1          : {test_metrics['f1']:.4f}"
    )

    print(
        f"Sensitivity : {test_metrics['sensitivity']:.4f}"
    )

    print(
        f"Specificity : {test_metrics['specificity']:.4f}"
    )


    print()

    print("=" * 70)
    print("[SUCCESS] EVALUATION V2 COMPLETED")
    print("=" * 70)



# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()