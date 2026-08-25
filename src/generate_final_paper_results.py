import json
import os
import csv


RESULT_DIR = "results"


print("="*70)
print("GENERATING FINAL PAPER RESULTS")
print("="*70)



# -----------------------------
# Load files
# -----------------------------

with open(
    os.path.join(
        RESULT_DIR,
        "final_report.json"
    ),
    "r"
) as f:
    window_results = json.load(f)



with open(
    os.path.join(
        RESULT_DIR,
        "final_test_persistence_evaluation.json"
    ),
    "r"
) as f:
    patient_results = json.load(f)



# -----------------------------
# Extract window level
# -----------------------------

window_summary = {

    "threshold":
        window_results["threshold"],

    "roc_auc":
        window_results["roc_auc"],

    "accuracy":
        window_results["classification_report"]["accuracy"],

    "normal_class":{

        "precision":
            window_results["classification_report"]["0"]["precision"],

        "recall":
            window_results["classification_report"]["0"]["recall"],

        "f1":
            window_results["classification_report"]["0"]["f1-score"]

    },


    "seizure_class":{

        "precision":
            window_results["classification_report"]["1"]["precision"],

        "recall":
            window_results["classification_report"]["1"]["recall"],

        "f1":
            window_results["classification_report"]["1"]["f1-score"]

    }

}



# -----------------------------
# Patient level
# -----------------------------

patient_summary = {

    "frozen_rule":
        patient_results["frozen_validation_rule"],


    "metrics":
        patient_results["test_metrics"],


    "baseline_q95":
        patient_results["q95_baseline_metrics"]

}



# -----------------------------
# Final JSON
# -----------------------------


final = {

    "experiment":
        "EEG Seizure Detection Final Evaluation",


    "window_level":
        window_summary,


    "patient_level":
        patient_summary,


    "optimization_status":{

        "test_used_for_optimization":False,

        "model_modified":False,

        "dataset_modified":False

    }

}



with open(
    os.path.join(
        RESULT_DIR,
        "PAPER_FINAL_RESULTS.json"
    ),
    "w"
) as f:

    json.dump(
        final,
        f,
        indent=4
    )



# -----------------------------
# Patient CSV Table
# -----------------------------

csv_file = os.path.join(
    RESULT_DIR,
    "patient_level_results_table.csv"
)


with open(
    csv_file,
    "w",
    newline=""
) as f:

    writer = csv.writer(f)


    writer.writerow([
        "Patient",
        "True Label",
        "Prediction",
        "Positive Fraction",
        "Cluster Max Size",
        "Q95"
    ])


    for name,p in patient_results["patients"].items():

        writer.writerow([

            name,

            p["true_label"],

            p["prediction"],

            p["positive_fraction"],

            p["cluster_max_size"],

            p["q95"]

        ])



# -----------------------------
# Text summary
# -----------------------------

txt_file=os.path.join(
    RESULT_DIR,
    "final_summary.txt"
)


with open(txt_file,"w") as f:

    f.write(
f"""
EEG SEIZURE DETECTION FINAL RESULTS

Window-level evaluation
-----------------------

ROC-AUC:
{window_summary['roc_auc']:.4f}

Accuracy:
{window_summary['accuracy']:.4f}

Seizure class:

Precision:
{window_summary['seizure_class']['precision']:.4f}

Recall:
{window_summary['seizure_class']['recall']:.4f}

F1-score:
{window_summary['seizure_class']['f1']:.4f}



Patient-level frozen evaluation
-------------------------------

Sensitivity:
{patient_summary['metrics']['sensitivity']:.4f}

Specificity:
{patient_summary['metrics']['specificity']:.4f}

Precision:
{patient_summary['metrics']['precision']:.4f}

F1-score:
{patient_summary['metrics']['f1']:.4f}



Temporal persistence improvement
--------------------------------

Baseline q95:

Sensitivity:
{patient_summary['baseline_q95']['sensitivity']:.4f}

Specificity:
{patient_summary['baseline_q95']['specificity']:.4f}

F1:
{patient_summary['baseline_q95']['f1']:.4f}


Frozen temporal rule:

Sensitivity:
{patient_summary['metrics']['sensitivity']:.4f}

Specificity:
{patient_summary['metrics']['specificity']:.4f}

F1:
{patient_summary['metrics']['f1']:.4f}


"""
)



print()
print("[DONE]")
print()
print("Created:")
print("results\\PAPER_FINAL_RESULTS.json")
print("results\\patient_level_results_table.csv")
print("results\\final_summary.txt")