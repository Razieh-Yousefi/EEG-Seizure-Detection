import json
import os


RESULT_DIR = "results"


print("="*70)
print("FINAL FROZEN PATIENT RULE TEST")
print("="*70)


# Frozen rule obtained from validation
RULE = {
    "fraction_threshold": 0.005,
    "min_cluster_size": 4,
    "max_gap": 2,
    "min_runs": 0
}


# Load test temporal information
with open(
    os.path.join(
        RESULT_DIR,
        "final_test_patient_level_report.json"
    ),
    "r"
) as f:
    data=json.load(f)



patients=data["patient_results"]



def apply_rule(p):

    positive_fraction = p["positive_fraction"]
    cluster_size = p.get(
        "cluster_max_size",
        0
    )


    prediction = (
        positive_fraction >= RULE["fraction_threshold"]
        and
        cluster_size >= RULE["min_cluster_size"]
    )

    return prediction



tp=tn=fp=fn=0

patient_outputs=[]


for p in patients:

    prediction = apply_rule(p)

    true_label = p["true_label"]


    if true_label==1 and prediction:
        tp+=1

    elif true_label==1 and not prediction:
        fn+=1

    elif true_label==0 and prediction:
        fp+=1

    else:
        tn+=1



    patient_outputs.append({

        "patient":p["patient"],
        "true_label":true_label,
        "prediction":int(prediction),
        "positive_fraction":p["positive_fraction"],
        "cluster_max_size":p.get(
            "cluster_max_size",
            None
        ),
        "q95":p.get(
            "q95",
            None
        )

    })



sensitivity = tp/(tp+fn) if tp+fn else 0
specificity = tn/(tn+fp) if tn+fp else 0
precision = tp/(tp+fp) if tp+fp else 0

f1 = (
    2*precision*sensitivity/(precision+sensitivity)
    if precision+sensitivity
    else 0
)



result={

    "experiment":
    "Final frozen patient temporal rule evaluation",


    "optimization_on_test":False,

    "rule":RULE,


    "metrics":{

        "tp":tp,
        "tn":tn,
        "fp":fp,
        "fn":fn,

        "sensitivity":sensitivity,
        "specificity":specificity,
        "precision":precision,
        "f1":f1

    },


    "patients":patient_outputs

}



with open(
    os.path.join(
        RESULT_DIR,
        "FINAL_FROZEN_RULE_TEST_RESULT.json"
    ),
    "w"
) as f:

    json.dump(
        result,
        f,
        indent=4
    )



print()
print("FINAL TEST RESULT")
print("-----------------")

print(json.dumps(
    result["metrics"],
    indent=4
))


print()
print("[DONE]")
print(
    "results\\FINAL_FROZEN_RULE_TEST_RESULT.json"
)