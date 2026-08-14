import os
import json
import time
import random
import numpy as np

import torch
import torch.nn as nn

from torch.utils.data import DataLoader, WeightedRandomSampler

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# ============================================================
# PATH CONFIGURATION
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

os.makedirs(
    MODEL_DIR,
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


BEST_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "best_chbmit_model.pt"
)


HISTORY_PATH = os.path.join(
    PROJECT_DIR,
    "training_history.json"
)


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

CHANNELS = 23

SAMPLES_PER_WINDOW = 1280

BATCH_SIZE = 16

NUM_EPOCHS = 20

LEARNING_RATE = 0.001

WEIGHT_DECAY = 1e-4

PATIENCE = 5

NUM_WORKERS = 0


DEVICE = torch.device(
    "cpu"
)



# ============================================================
# SEED
# ============================================================

def set_seed(seed=42):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)



# ============================================================
# NORMALIZATION CREATION
# ============================================================

def create_normalization_file():

    print(
        "Normalization file not found."
    )

    print(
        "Creating normalization parameters from training data..."
    )


    X = np.load(
        X_PATH,
        mmap_mode="r"
    )


    train_indices = np.load(
        TRAIN_INDICES_PATH
    )


    train_data = X[
        train_indices
    ]


    mean = np.mean(
        train_data,
        axis=(0,2)
    )


    std = np.std(
        train_data,
        axis=(0,2)
    )


    std[
        std == 0
    ] = 1.0


    np.savez(
        NORMALIZATION_PATH,
        channel_mean=mean.astype(
            np.float32
        ),
        channel_std=std.astype(
            np.float32
        )
    )


    print(
        "Normalization created:"
    )

    print(
        NORMALIZATION_PATH
    )



# ============================================================
# DATASET
# ============================================================


class CHBMITDataset(
    torch.utils.data.Dataset
):

    def __init__(
        self,
        X_path,
        y_path,
        indices_path,
        normalization_path
    ):


        self.X = np.load(
            X_path,
            mmap_mode="r"
        )


        self.y = np.load(
            y_path
        )


        self.indices = np.load(
            indices_path
        )


        normalization = np.load(
            normalization_path
        )


        self.mean = np.asarray(
            normalization["channel_mean"],
            dtype=np.float32
        )


        self.std = np.asarray(
            normalization["channel_std"],
            dtype=np.float32
        )


        self.validate()



    def validate(self):


        if self.X.ndim != 3:

            raise ValueError(
                "X must be 3 dimensional"
            )


        if self.X.shape[1] != CHANNELS:

            raise ValueError(
                f"Expected {CHANNELS} channels"
            )


        if self.X.shape[2] != SAMPLES_PER_WINDOW:

            raise ValueError(
                "Wrong window size"
            )


        if len(self.X) != len(self.y):

            raise ValueError(
                "X/y mismatch"
            )


        if self.mean.shape != (
            CHANNELS,
        ):

            raise ValueError(
                "Invalid mean shape"
            )


        if self.std.shape != (
            CHANNELS,
        ):

            raise ValueError(
                "Invalid std shape"
            )


        if np.any(
            self.std <= 0
        ):

            raise ValueError(
                "Invalid std values"
            )



    def __len__(self):

        return len(
            self.indices
        )



    def __getitem__(
        self,
        idx
    ):

        real_idx = int(
            self.indices[idx]
        )


        sample = np.asarray(
            self.X[real_idx],
            dtype=np.float32
        )


        sample = (
            sample -
            self.mean[:,None]
        ) / self.std[:,None]


        sample = np.ascontiguousarray(
            sample
        )


        x = torch.from_numpy(
            sample
        )


        y = torch.tensor(
            int(self.y[real_idx]),
            dtype=torch.long
        )


        return x,y
# ============================================================
# MODEL
# ============================================================


class EEGCNN(nn.Module):

    def __init__(self):

        super().__init__()


        self.features = nn.Sequential(

            nn.Conv1d(
                in_channels=CHANNELS,
                out_channels=32,
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
# CLASS WEIGHTS
# ============================================================


def calculate_class_weights(
    y,
    train_indices
):


    labels = y[
        train_indices
    ]


    counts = np.bincount(
        labels,
        minlength=2
    )


    print()
    print("="*70)
    print("TRAIN CLASS DISTRIBUTION")
    print("="*70)


    print(
        "Class 0:",
        int(counts[0])
    )


    print(
        "Class 1:",
        int(counts[1])
    )



    if counts[1] == 0:

        raise RuntimeError(
            "No seizure samples in training set"
        )



    weights = (
        counts.sum()
        /
        (
            2.0 *
            counts.astype(
                np.float32
            )
        )
    )



    weights = torch.tensor(
        weights,
        dtype=torch.float32
    )



    print(
        "Class weights:",
        weights.tolist()
    )


    return weights




# ============================================================
# WEIGHTED RANDOM SAMPLER
# ============================================================


def create_train_sampler(
    y,
    train_indices
):


    labels = y[
        train_indices
    ]


    counts = np.bincount(
        labels,
        minlength=2
    )


    class_weights = (
        1.0 /
        counts.astype(
            np.float64
        )
    )


    sample_weights = (
        class_weights[
            labels
        ]
    )


    sampler = WeightedRandomSampler(

        weights=torch.tensor(
            sample_weights,
            dtype=torch.double
        ),

        num_samples=len(
            train_indices
        ),

        replacement=True

    )


    return sampler





# ============================================================
# METRICS
# ============================================================


def calculate_metrics(
    y_true,
    y_pred
):


    accuracy = accuracy_score(
        y_true,
        y_pred
    )


    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )


    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )


    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )


    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[
            0,
            1
        ]
    )


    tn = int(
        cm[0,0]
    )

    fp = int(
        cm[0,1]
    )

    fn = int(
        cm[1,0]
    )

    tp = int(
        cm[1,1]
    )



    sensitivity = (
        tp /
        (tp+fn)
        if (tp+fn)>0
        else 0
    )


    specificity = (
        tn /
        (tn+fp)
        if (tn+fp)>0
        else 0
    )



    return {

        "accuracy":
            float(accuracy),

        "precision":
            float(precision),

        "recall":
            float(recall),

        "f1":
            float(f1),

        "sensitivity":
            float(sensitivity),

        "specificity":
            float(specificity),

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
# PRINT METRICS
# ============================================================


def print_metrics(
    name,
    metrics
):


    print()
    print(name)


    print(
        f"Accuracy    : {metrics['accuracy']:.4f}"
    )

    print(
        f"Precision   : {metrics['precision']:.4f}"
    )

    print(
        f"Recall      : {metrics['recall']:.4f}"
    )

    print(
        f"F1          : {metrics['f1']:.4f}"
    )

    print(
        f"Sensitivity : {metrics['sensitivity']:.4f}"
    )

    print(
        f"Specificity : {metrics['specificity']:.4f}"
    )


    print(
        f"TN={metrics['tn']} "
        f"FP={metrics['fp']} "
        f"FN={metrics['fn']} "
        f"TP={metrics['tp']}"
    )
# ============================================================
# TRAIN ONE EPOCH
# ============================================================


def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device
):

    model.train()


    total_loss = 0.0

    total = 0


    all_targets = []

    all_predictions = []



    for x,y in loader:


        x = x.to(
            device
        )

        y = y.to(
            device
        )


        optimizer.zero_grad(
            set_to_none=True
        )


        output = model(
            x
        )


        loss = criterion(
            output,
            y
        )


        loss.backward()


        optimizer.step()



        batch_size = y.size(0)


        total_loss += (
            loss.item()
            *
            batch_size
        )


        total += batch_size



        pred = torch.argmax(
            output,
            dim=1
        )



        all_targets.extend(
            y.detach()
            .cpu()
            .numpy()
            .tolist()
        )


        all_predictions.extend(
            pred.detach()
            .cpu()
            .numpy()
            .tolist()
        )



    metrics = calculate_metrics(
        all_targets,
        all_predictions
    )


    metrics["loss"] = (
        total_loss /
        total
    )


    return metrics






# ============================================================
# EVALUATION
# ============================================================


@torch.no_grad()

def evaluate(
    model,
    loader,
    criterion,
    device
):


    model.eval()


    total_loss = 0.0

    total = 0


    all_targets = []

    all_predictions = []



    for x,y in loader:


        x = x.to(
            device
        )


        y = y.to(
            device
        )


        output = model(
            x
        )


        loss = criterion(
            output,
            y
        )


        batch_size = y.size(0)


        total_loss += (
            loss.item()
            *
            batch_size
        )


        total += batch_size



        pred = torch.argmax(
            output,
            dim=1
        )


        all_targets.extend(
            y.cpu()
            .numpy()
            .tolist()
        )


        all_predictions.extend(
            pred.cpu()
            .numpy()
            .tolist()
        )



    metrics = calculate_metrics(
        all_targets,
        all_predictions
    )


    metrics["loss"] = (
        total_loss /
        total
    )


    return metrics






# ============================================================
# SAVE CHECKPOINT
# ============================================================


def save_checkpoint(
    model,
    optimizer,
    epoch,
    metrics
):


    checkpoint = {


        "epoch":
            epoch,


        "model_state_dict":
            model.state_dict(),


        "optimizer_state_dict":
            optimizer.state_dict(),


        "metrics":
            metrics,


        "config":

        {

            "channels":
                CHANNELS,

            "samples":
                SAMPLES_PER_WINDOW,

            "batch_size":
                BATCH_SIZE

        }

    }



    torch.save(
        checkpoint,
        BEST_MODEL_PATH
    )







# ============================================================
# MAIN
# ============================================================


def main():


    set_seed(
        SEED
    )


    print("="*70)

    print(
        "CHB-MIT EEG TRAINING"
    )

    print("="*70)



    print()

    print(
        "Project:",
        PROJECT_DIR
    )

    print(
        "Data:",
        DATA_DIR
    )


    print(
        "Device:",
        DEVICE
    )


    print(
        "Torch:",
        torch.__version__
    )





    # --------------------------------------------------------
    # CHECK FILES
    # --------------------------------------------------------


    print()

    print("="*70)

    print(
        "1. CHECKING FILES"
    )

    print("="*70)



    required_files = [


        X_PATH,

        Y_PATH,

        TRAIN_INDICES_PATH,

        VAL_INDICES_PATH,

        TEST_INDICES_PATH

    ]



    for f in required_files:


        if not os.path.exists(f):

            raise FileNotFoundError(
                f
            )


        print(
            "[OK]",
            f
        )



    if not os.path.exists(
        NORMALIZATION_PATH
    ):


        create_normalization_file()


    else:

        print(
            "[OK] Normalization exists"
        )





    # --------------------------------------------------------
    # LOAD METADATA
    # --------------------------------------------------------


    print()

    print("="*70)

    print(
        "2. LOADING METADATA"
    )

    print("="*70)



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



    print(
        "Total:",
        len(y)
    )


    print(
        "Train:",
        len(train_indices)
    )


    print(
        "Validation:",
        len(val_indices)
    )


    print(
        "Test:",
        len(test_indices)
    )






    # --------------------------------------------------------
    # DATASETS
    # --------------------------------------------------------


    print()

    print("="*70)

    print(
        "3. CREATING DATASETS"
    )

    print("="*70)



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



    print(
        "Train:",
        len(train_dataset)
    )


    print(
        "Val:",
        len(val_dataset)
    )


    print(
        "Test:",
        len(test_dataset)
    )





    # --------------------------------------------------------
    # LOADERS
    # --------------------------------------------------------


    sampler = create_train_sampler(
        y,
        train_indices
    )



    train_loader = DataLoader(

        train_dataset,

        batch_size=BATCH_SIZE,

        sampler=sampler,

        num_workers=NUM_WORKERS

    )



    val_loader = DataLoader(

        val_dataset,

        batch_size=BATCH_SIZE,

        shuffle=False,

        num_workers=NUM_WORKERS

    )



    test_loader = DataLoader(

        test_dataset,

        batch_size=BATCH_SIZE,

        shuffle=False,

        num_workers=NUM_WORKERS

    )



    print(
        "[OK] DataLoaders created"
    )
# ============================================================
# MODEL CREATION
# ============================================================


    print()

    print("="*70)

    print(
        "4. CREATING MODEL"
    )

    print("="*70)



    model = EEGCNN()



    model = model.to(
        DEVICE
    )



    print(model)



    total_params = sum(
        p.numel()
        for p in model.parameters()
    )


    print()

    print(
        "Total parameters:",
        total_params
    )





    # --------------------------------------------------------
    # CLASS WEIGHTS
    # --------------------------------------------------------


    class_weights = calculate_class_weights(
        y,
        train_indices
    )


    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(
            DEVICE
        )
    )



    print()

    print(
        "Loss:",
        criterion
    )





    # --------------------------------------------------------
    # OPTIMIZER
    # --------------------------------------------------------


    optimizer = torch.optim.AdamW(

        model.parameters(),

        lr=LEARNING_RATE,

        weight_decay=WEIGHT_DECAY

    )




    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(

        optimizer,

        mode="max",

        factor=0.5,

        patience=2

    )





    # --------------------------------------------------------
    # TRAINING LOOP
    # --------------------------------------------------------


    print()

    print("="*70)

    print(
        "5. START TRAINING"
    )

    print("="*70)



    history = []


    best_f1 = -1


    best_epoch = 0


    patience_counter = 0



    start_time = time.time()



    for epoch in range(
        1,
        NUM_EPOCHS+1
    ):


        print()

        print("-"*70)

        print(
            f"EPOCH {epoch}/{NUM_EPOCHS}"
        )

        print("-"*70)



        epoch_start = time.time()



        train_metrics = train_one_epoch(

            model,

            train_loader,

            optimizer,

            criterion,

            DEVICE

        )



        val_metrics = evaluate(

            model,

            val_loader,

            criterion,

            DEVICE

        )




        scheduler.step(
            val_metrics["f1"]
        )



        print()

        print_metrics(
            "TRAIN",
            train_metrics
        )


        print_metrics(
            "VALIDATION",
            val_metrics
        )



        current_lr = optimizer.param_groups[0]["lr"]



        print()

        print(
            "Learning rate:",
            current_lr
        )



        epoch_time = (
            time.time()
            -
            epoch_start
        )


        print(
            "Epoch time:",
            f"{epoch_time:.2f}s"
        )




        history.append(

            {

                "epoch":
                    epoch,

                "train":
                    train_metrics,

                "validation":
                    val_metrics,

                "lr":
                    current_lr

            }

        )





        # ----------------------------------------------------
        # SAVE BEST MODEL
        # ----------------------------------------------------


        if val_metrics["f1"] > best_f1:


            best_f1 = (
                val_metrics["f1"]
            )


            best_epoch = epoch


            patience_counter = 0



            save_checkpoint(

                model,

                optimizer,

                epoch,

                val_metrics

            )



            print()

            print(
                "[OK] Best model saved"
            )

            print(
                "Best validation F1:",
                best_f1
            )



        else:


            patience_counter += 1


            print()

            print(
                "No improvement"
            )


            print(
                "Patience:",
                patience_counter,
                "/",
                PATIENCE
            )






        with open(

            HISTORY_PATH,

            "w",

            encoding="utf-8"

        ) as f:


            json.dump(

                history,

                f,

                indent=2

            )






        if patience_counter >= PATIENCE:


            print()

            print(
                "EARLY STOPPING"
            )


            break







    total_time = (
        time.time()
        -
        start_time
    )






    # --------------------------------------------------------
    # LOAD BEST MODEL
    # --------------------------------------------------------


    print()

    print("="*70)

    print(
        "6. LOADING BEST MODEL"
    )

    print("="*70)



    checkpoint = torch.load(

        BEST_MODEL_PATH,

        map_location=DEVICE

    )



    model.load_state_dict(

        checkpoint[
            "model_state_dict"
        ]

    )



    print(
        "[OK] Best model loaded"
    )



    print(
        "Best epoch:",
        checkpoint["epoch"]
    )


    print(
        "Best F1:",
        checkpoint["metrics"]["f1"]
    )






    # --------------------------------------------------------
    # FINAL TEST
    # --------------------------------------------------------


    print()

    print("="*70)

    print(
        "7. FINAL TEST"
    )

    print("="*70)




    test_metrics = evaluate(

        model,

        test_loader,

        criterion,

        DEVICE

    )



    print_metrics(

        "TEST RESULTS",

        test_metrics

    )






    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------


    print()

    print("="*70)

    print(
        "FINAL SUMMARY"
    )

    print("="*70)



    print()

    print(
        "Best epoch:",
        best_epoch
    )


    print(
        "Best validation F1:",
        best_f1
    )


    print()

    print(
        "Test accuracy:",
        test_metrics["accuracy"]
    )


    print(
        "Test F1:",
        test_metrics["f1"]
    )


    print(
        "Sensitivity:",
        test_metrics["sensitivity"]
    )


    print(
        "Specificity:",
        test_metrics["specificity"]
    )


    print()

    print(
        "Confusion Matrix:"
    )


    print(

        f"TN={test_metrics['tn']} "
        f"FP={test_metrics['fp']} "
        f"FN={test_metrics['fn']} "
        f"TP={test_metrics['tp']}"

    )



    print()

    print(
        "Training time:",
        f"{total_time/60:.2f} minutes"
    )


    print()

    print(
        "Best model:"
    )

    print(
        BEST_MODEL_PATH
    )



    print()

    print("="*70)

    print(
        "[SUCCESS] TRAINING COMPLETED"
    )

    print("="*70)





# ============================================================
# RUN
# ============================================================


if __name__ == "__main__":

    main()