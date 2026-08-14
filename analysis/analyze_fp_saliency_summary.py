import os
import json
import numpy as np


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

INPUT_PATH = os.path.join(
    BASE_DIR,
    "fp_saliency_full_results.json"
)

OUTPUT_PATH = os.path.join(
    BASE_DIR,
    "fp_saliency_summary.json"
)


CHANNELS = 23


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("FP / TP SALIENCY CHANNEL SUMMARY")
print("=" * 70)


# ============================================================
# LOAD
# ============================================================

with open(
    INPUT_PATH,
    "r"
) as f:
    data = json.load(f)


print()
print("Loaded:")
print(INPUT_PATH)

print()
print("Keys:")
print(list(data.keys()))



# ============================================================
# EXTRACT CHANNEL SALIENCY
# ============================================================

def extract_saliency(records):

    all_values = []

    for item in records:

        if "channel_saliency" not in item:

            raise RuntimeError(
                "channel_saliency field missing."
            )

        values = np.asarray(
            item["channel_saliency"],
            dtype=np.float32
        )

        if values.shape != (CHANNELS,):

            raise RuntimeError(
                f"Invalid channel shape: {values.shape}"
            )

        all_values.append(
            values
        )


    if len(all_values) == 0:

        return np.zeros(
            CHANNELS,
            dtype=np.float32
        )


    return np.mean(
        np.stack(all_values),
        axis=0
    )



# ============================================================
# CALCULATE
# ============================================================

fp_saliency = extract_saliency(
    data["false_positive"]
)

tp_saliency = extract_saliency(
    data["true_positive"]
)



# ============================================================
# RANK CHANNELS
# ============================================================

ratios = (
    fp_saliency /
    (tp_saliency + 1e-12)
)


ranking = []


for ch in np.argsort(
    ratios
)[::-1]:


    ranking.append({

        "channel":
            int(ch + 1),

        "ratio":
            float(ratios[ch]),

        "fp_mean_saliency":
            float(fp_saliency[ch]),

        "tp_mean_saliency":
            float(tp_saliency[ch])

    })



# ============================================================
# PRINT
# ============================================================

print()
print("=" * 70)
print("TOP FP / TP SALIENCY CHANNELS")
print("=" * 70)


for i,item in enumerate(
    ranking[:10],
    start=1
):

    print(
        f"{i:02d}. "
        f"Channel {item['channel']:02d} | "
        f"ratio={item['ratio']:.4f} | "
        f"FP={item['fp_mean_saliency']:.8f} | "
        f"TP={item['tp_mean_saliency']:.8f}"
    )



# ============================================================
# SAVE
# ============================================================

output = {

    "num_fp":
        len(data["false_positive"]),

    "num_tp":
        len(data["true_positive"]),

    "ranking":
        ranking,

    "fp_mean_channel_saliency":
        fp_saliency.tolist(),

    "tp_mean_channel_saliency":
        tp_saliency.tolist()

}


with open(
    OUTPUT_PATH,
    "w"
) as f:

    json.dump(
        output,
        f,
        indent=2
    )



print()
print("[OK] Saved:")
print(OUTPUT_PATH)


print()
print("=" * 70)
print("SALIENCY SUMMARY COMPLETED")
print("=" * 70)