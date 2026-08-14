import numpy as np
import json
import os


BASE = os.path.dirname(__file__)

PROB_FILE = os.path.join(
    BASE,
    "test_window_probabilities.npz"
)

RESULT_FILE = os.path.join(
    BASE,
    "probability_smoothing_results.json"
)


# ==============================
# Load probabilities
# ==============================

data = np.load(PROB_FILE)

labels = data["labels"]
probabilities = data["probabilities"]


threshold = 0.56


print("="*60)
print("PROBABILITY SMOOTHING ANALYSIS")
print("="*60)

print("Samples:", len(labels))


# ==============================
# Evaluation function
# ==============================

def evaluate(pred, labels):

    tp = np.sum((pred==1)&(labels==1))
    tn = np.sum((pred==0)&(labels==0))
    fp = np.sum((pred==1)&(labels==0))
    fn = np.sum((pred==0)&(labels==1))

    sensitivity = tp/(tp+fn) if tp+fn else 0
    specificity = tn/(tn+fp) if tn+fp else 0
    precision = tp/(tp+fp) if tp+fp else 0

    f1 = (
        2*precision*sensitivity/
        (precision+sensitivity)
        if precision+sensitivity else 0
    )

    return {
        "tp":int(tp),
        "tn":int(tn),
        "fp":int(fp),
        "fn":int(fn),
        "sensitivity":float(sensitivity),
        "specificity":float(specificity),
        "precision":float(precision),
        "f1":float(f1)
    }



results={}


# ==============================
# Baseline
# ==============================

baseline = probabilities >= threshold

results["baseline"] = evaluate(
    baseline,
    labels
)


# ==============================
# Moving averages
# ==============================

for window in [3,5,7]:

    smooth = np.convolve(
        probabilities,
        np.ones(window)/window,
        mode="same"
    )

    pred = smooth >= threshold


    results[
        f"moving_average_{window}"
    ] = evaluate(
        pred,
        labels
    )


# ==============================
# Save
# ==============================

with open(
    RESULT_FILE,
    "w"
) as f:

    json.dump(
        results,
        f,
        indent=2
    )


print("\nRESULTS")
print(json.dumps(results,indent=2))


print("\nSaved:")
print(RESULT_FILE)