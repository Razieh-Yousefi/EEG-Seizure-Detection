import os
import json
import numpy as np


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)


SALIENCY_FILE = os.path.join(
    BASE_DIR,
    "fp_saliency_full_results.json"
)


TEMPORAL_FILE = os.path.join(
    BASE_DIR,
    "fp_temporal_analysis_results.json"
)


CHANNEL_FILE = os.path.join(
    BASE_DIR,
    "fp_channel_importance_results.json"
)


OUTPUT_FILE = os.path.join(
    BASE_DIR,
    "fp_root_cause_analysis.json"
)



print("="*70)
print("FALSE POSITIVE ROOT CAUSE ANALYSIS")
print("="*70)



# ============================================================
# LOAD FILES
# ============================================================

print()
print("Loading files...")


with open(
    SALIENCY_FILE,
    "r"
) as f:
    saliency = json.load(f)



with open(
    TEMPORAL_FILE,
    "r"
) as f:
    temporal = json.load(f)



with open(
    CHANNEL_FILE,
    "r"
) as f:
    channel = json.load(f)



print("[OK] Files loaded")



# ============================================================
# SALIENCY ANALYSIS
# ============================================================


print()
print("="*70)
print("1. SALIENCY ANALYSIS")
print("="*70)



fp_records = saliency[
    "false_positive"
]


tp_records = saliency[
    "true_positive"
]



fp_saliency = np.array(
    [
        x["channel_saliency"]
        for x in fp_records
    ],
    dtype=np.float32
)


tp_saliency = np.array(
    [
        x["channel_saliency"]
        for x in tp_records
    ],
    dtype=np.float32
)



fp_mean = fp_saliency.mean(axis=0)

tp_mean = tp_saliency.mean(axis=0)



saliency_ratio = (
    fp_mean /
    (tp_mean + 1e-12)
)



saliency_rank = np.argsort(
    saliency_ratio
)[::-1]



top_saliency_channels = []

for ch in saliency_rank[:10]:

    top_saliency_channels.append({

        "channel":
            int(ch+1),

        "fp_mean":
            float(fp_mean[ch]),

        "tp_mean":
            float(tp_mean[ch]),

        "ratio":
            float(saliency_ratio[ch])

    })



print("Top suspicious channels:")

for x in top_saliency_channels:

    print(
        x
    )



# ============================================================
# CHANNEL IMPORTANCE
# ============================================================


print()
print("="*70)
print("2. CHANNEL IMPORTANCE")
print("="*70)



channel_result = {}



if isinstance(channel, dict):

    for k,v in channel.items():

        if isinstance(v,(int,float)):

            channel_result[k]=v



# ============================================================
# TEMPORAL
# ============================================================


print()
print("="*70)
print("3. TEMPORAL PATTERN")
print("="*70)



temporal_summary = {


    "threshold":
        temporal.get(
            "validation_threshold"
        ),


    "window_duration_seconds":
        temporal.get(
            "window_duration_seconds"
        ),


    "global":
        temporal.get(
            "global"
        ),


    "longest_runs":
        temporal.get(
            "top_20_longest_runs"
        )

}



print(
    "Longest FP runs:"
)


print(
    temporal_summary["longest_runs"][:5]
)



# ============================================================
# FINAL REPORT
# ============================================================


report = {


    "summary": {


        "false_positive_samples":
            len(fp_records),


        "true_positive_samples":
            len(tp_records)

    },


    "saliency_analysis": {


        "top_channels":
            top_saliency_channels

    },


    "channel_importance":
        channel_result,


    "temporal_analysis":
        temporal_summary,


    "interpretation": [

        "Channels with ratio > 1 receive higher attention in false positives.",

        "Long temporal FP runs indicate sustained non-seizure patterns confused by the model.",

        "Combined channel and temporal evidence identifies possible causes of false alarms."

    ]

}



with open(
    OUTPUT_FILE,
    "w"
) as f:

    json.dump(
        report,
        f,
        indent=2
    )



print()
print("[OK] Saved:")
print(OUTPUT_FILE)


print()
print("="*70)
print("ROOT CAUSE ANALYSIS COMPLETED")
print("="*70)