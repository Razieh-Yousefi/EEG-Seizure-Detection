import json
import itertools
import os


RESULT_DIR = "results"


print("="*70)
print("PATIENT LEVEL RULE OPTIMIZATION")
print("="*70)


with open(
    os.path.join(
        RESULT_DIR,
        "validation_patient_level_aggregation_analysis.json"
    ),
    "r"
) as f:
    data=json.load(f)


patients=data["patient_statistics"]


q95_values=[
    0.5,
    0.7,
    0.8,
    0.85,
    0.9,
    0.95,
    0.97,
    0.99
]


fraction_values=[
    0.005,
    0.01,
    0.02,
    0.05,
    0.1
]


max_probability_values=[
    0.8,
    0.9,
    0.95,
    0.99
]


def evaluate(rule):

    tp=tn=fp=fn=0

    for name,p in patients.items():

        fraction = (
            p["positive_windows"]
            /
            p["total_windows"]
        )


        prediction = (
            p["q95"] >= rule["q95"]
            and
            fraction >= rule["fraction"]
            and
            p["max"] >= rule["max_probability"]
        )


        true=p["has_true_seizure"]


        if true and prediction:
            tp+=1

        elif true and not prediction:
            fn+=1

        elif not true and prediction:
            fp+=1

        else:
            tn+=1


    sensitivity=tp/(tp+fn) if tp+fn else 0
    specificity=tn/(tn+fp) if tn+fp else 0
    precision=tp/(tp+fp) if tp+fp else 0

    f1=(
        2*precision*sensitivity/(precision+sensitivity)
        if precision+sensitivity
        else 0
    )


    return {
        "tp":tp,
        "tn":tn,
        "fp":fp,
        "fn":fn,
        "sensitivity":sensitivity,
        "specificity":specificity,
        "precision":precision,
        "f1":f1
    }



best=None
safe=[]


for q,f,m in itertools.product(
        q95_values,
        fraction_values,
        max_probability_values):


    rule={
        "q95":q,
        "fraction":f,
        "max_probability":m
    }


    metrics=evaluate(rule)


    if metrics["sensitivity"]>=0.9:

        result={
            "rule":rule,
            "metrics":metrics
        }

        safe.append(result)


        if (
            best is None
            or
            metrics["specificity"] >
            best["metrics"]["specificity"]
        ):
            best=result



print()
print("SAFE RULES:",len(safe))

print()
print("BEST RULE:")
print(best)



output={
    "best_rule":best,
    "number_of_safe_rules":len(safe),
    "all_safe_rules":safe
}


with open(
    os.path.join(
        RESULT_DIR,
        "patient_rule_optimization.json"
    ),
    "w"
) as f:
    json.dump(
        output,
        f,
        indent=4
    )


print()
print("[DONE]")
print(
    "results\\patient_rule_optimization.json"
)