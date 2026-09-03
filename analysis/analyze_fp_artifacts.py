from pathlib import Path
import json

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_DIR / "data"
RESULTS_DIR = PROJECT_DIR / "results"


X_PATH = DATA_DIR / "X_chbmit_full.npy"

Y_PATH = DATA_DIR / "y_chbmit_full.npy"

TEST_INDICES_PATH = DATA_DIR / "test_indices.npy"

NORMALIZATION_PATH = DATA_DIR / "normalization_params.npz"

PROBABILITY_PATH = DATA_DIR / "test_window_probabilities.npz"


OUTPUT_JSON = RESULTS_DIR / "fp_artifact_analysis.json"

OUTPUT_DIR = RESULTS_DIR / "fp_artifact_plots"



# ============================================================
# SETTINGS
# ============================================================

THRESHOLD = 0.95

HIGH_CONFIDENCE_THRESHOLD = 0.95


EXPECTED_CHANNELS = 23

EXPECTED_SAMPLES = 1280


TOP_CHANNELS = 8



# ============================================================
# FEATURE FUNCTIONS
# ============================================================


def calculate_rms(signal):

    return float(
        np.sqrt(
            np.mean(
                np.square(signal)
            )
        )
    )



def calculate_zero_crossing_rate(signal):

    centered = signal - np.mean(signal)

    signs = np.sign(centered)

    crossings = np.sum(
        signs[:-1] != signs[1:]
    )

    return float(
        crossings /
        max(
            1,
            len(signal) - 1
        )
    )



def calculate_line_length(signal):

    return float(
        np.sum(
            np.abs(
                np.diff(signal)
            )
        )
    )



def calculate_high_frequency_ratio(
    signal,
    sampling_rate=256.0,
    cutoff_frequency=20.0
):

    signal = signal - np.mean(signal)


    frequencies = np.fft.rfftfreq(
        len(signal),
        d=1.0 / sampling_rate
    )


    spectrum = (
        np.abs(
            np.fft.rfft(signal)
        )
        ** 2
    )


    total_power = np.sum(
        spectrum
    )


    if total_power <= 0:

        return 0.0


    high_frequency_power = np.sum(
        spectrum[
            frequencies >= cutoff_frequency
        ]
    )


    return float(
        high_frequency_power /
        total_power
    )



def calculate_band_power(
    signal,
    low_frequency,
    high_frequency,
    sampling_rate=256.0
):

    signal = signal - np.mean(signal)


    frequencies = np.fft.rfftfreq(
        len(signal),
        d=1.0 / sampling_rate
    )


    spectrum = (
        np.abs(
            np.fft.rfft(signal)
        )
        ** 2
    )


    total_power = np.sum(
        spectrum
    )


    if total_power <= 0:

        return 0.0


    mask = (

        (frequencies >= low_frequency)

        &

        (frequencies < high_frequency)

    )


    band_power = np.sum(
        spectrum[mask]
    )


    return float(
        band_power /
        total_power
    )



def calculate_channel_features(window):


    channel_features = []


    for channel_index in range(
        window.shape[0]
    ):


        signal = window[channel_index]


        channel_features.append(

            {

                "channel":
                    channel_index + 1,


                "rms":
                    calculate_rms(signal),


                "zero_crossing_rate":
                    calculate_zero_crossing_rate(signal),


                "line_length":
                    calculate_line_length(signal),


                "high_frequency_ratio":
                    calculate_high_frequency_ratio(signal),


                "beta_power":
                    calculate_band_power(
                        signal,
                        13.0,
                        30.0
                    ),


                "gamma_power":
                    calculate_band_power(
                        signal,
                        30.0,
                        80.0
                    )

            }

        )


    return channel_features



# ============================================================
# WINDOW SUMMARY
# ============================================================


def summarize_window_features(
    channel_features
):

    rms_values = np.array(
        [
            item["rms"]
            for item in channel_features
        ]
    )


    zcr_values = np.array(
        [
            item["zero_crossing_rate"]
            for item in channel_features
        ]
    )


    line_values = np.array(
        [
            item["line_length"]
            for item in channel_features
        ]
    )


    hf_values = np.array(
        [
            item["high_frequency_ratio"]
            for item in channel_features
        ]
    )


    beta_values = np.array(
        [
            item["beta_power"]
            for item in channel_features
        ]
    )


    gamma_values = np.array(
        [
            item["gamma_power"]
            for item in channel_features
        ]
    )


    return {

        "mean_rms":
            float(np.mean(rms_values)),


        "max_rms":
            float(np.max(rms_values)),


        "mean_zero_crossing_rate":
            float(np.mean(zcr_values)),


        "max_zero_crossing_rate":
            float(np.max(zcr_values)),


        "mean_line_length":
            float(np.mean(line_values)),


        "max_line_length":
            float(np.max(line_values)),


        "mean_high_frequency_ratio":
            float(np.mean(hf_values)),


        "max_high_frequency_ratio":
            float(np.max(hf_values)),


        "mean_beta_power":
            float(np.mean(beta_values)),


        "max_beta_power":
            float(np.max(beta_values)),


        "mean_gamma_power":
            float(np.mean(gamma_values)),


        "max_gamma_power":
            float(np.max(gamma_values))

    }
# ============================================================
# LOAD DATA
# ============================================================


print("=" * 70)
print("FP ARTIFACT AND MORPHOLOGY ANALYSIS")
print("=" * 70)


print()

print("Loading files...")



required_files = [

    X_PATH,

    Y_PATH,

    TEST_INDICES_PATH,

    NORMALIZATION_PATH,

    PROBABILITY_PATH

]


for file_path in required_files:

    if not file_path.exists():

        raise FileNotFoundError(
            f"Missing file:\n{file_path}"
        )



X = np.load(
    X_PATH,
    mmap_mode="r"
)



y = np.load(
    Y_PATH
)



test_indices = np.load(
    TEST_INDICES_PATH
).reshape(-1)



normalization = np.load(
    NORMALIZATION_PATH
)



channel_mean = np.asarray(

    normalization["channel_mean"],

    dtype=np.float32

)



channel_std = np.asarray(

    normalization["channel_std"],

    dtype=np.float32

)



prediction_data = np.load(

    PROBABILITY_PATH,

    allow_pickle=True

)



probabilities = np.asarray(

    prediction_data["probabilities"]

).reshape(-1)



labels = np.asarray(

    prediction_data["labels"]

).reshape(-1)




print()

print(
    "X shape:",
    X.shape
)


print(
    "Labels:",
    len(labels)
)


print(
    "Probabilities:",
    len(probabilities)
)


print(
    "Test indices:",
    len(test_indices)
)



# ============================================================
# VALIDATION
# ============================================================


if X.ndim != 3:

    raise ValueError(
        "X must be 3 dimensional"
    )



if X.shape[1] != EXPECTED_CHANNELS:

    raise ValueError(
        f"Expected {EXPECTED_CHANNELS} channels"
    )



if X.shape[2] != EXPECTED_SAMPLES:

    raise ValueError(
        f"Expected {EXPECTED_SAMPLES} samples"
    )



if not (

    len(test_indices)

    ==

    len(labels)

    ==

    len(probabilities)

):

    raise ValueError(
        "Prediction arrays length mismatch"
    )



# ============================================================
# CLASSIFICATION
# ============================================================


predictions = (

    probabilities >= THRESHOLD

).astype(int)



tp_mask = (

    (predictions == 1)

    &

    (labels == 1)

)



fp_mask = (

    (predictions == 1)

    &

    (labels == 0)

)



high_fp_mask = (

    fp_mask

    &

    (

        probabilities

        >=

        HIGH_CONFIDENCE_THRESHOLD

    )

)



tp_positions = np.where(
    tp_mask
)[0]



fp_positions = np.where(
    fp_mask
)[0]



high_fp_positions = np.where(
    high_fp_mask
)[0]



print()

print("=" * 70)

print("CLASSIFICATION")

print("=" * 70)



print(
    "TP:",
    len(tp_positions)
)


print(
    "FP:",
    len(fp_positions)
)


print(
    "High confidence FP:",
    len(high_fp_positions)
)




# ============================================================
# NORMALIZED WINDOW LOADING
# ============================================================


def load_normalized_window(
    sample_index
):


    raw_window = np.asarray(

        X[int(sample_index)],

        dtype=np.float32

    )


    normalized_window = (

        raw_window

        -

        channel_mean[:, None]

    ) / (

        channel_std[:, None]

    )


    return normalized_window




# ============================================================
# WINDOW ANALYSIS
# ============================================================


def analyze_position(position):


    sample_index = int(

        test_indices[position]

    )



    window = load_normalized_window(

        sample_index

    )



    channel_features = calculate_channel_features(

        window

    )



    summary = summarize_window_features(

        channel_features

    )



    return {


        "position":

            int(position),



        "sample_index":

            sample_index,



        "label":

            int(labels[position]),



        "probability":

            float(probabilities[position]),



        "summary":

            summary,



        "channels":

            channel_features

    }




# ============================================================
# EXTRACT TP / FP FEATURES
# ============================================================


print()

print("=" * 70)

print("EXTRACTING FEATURES")

print("=" * 70)



print(
    "Processing TP windows..."
)



tp_results = [

    analyze_position(position)

    for position in tp_positions

]



print(
    "Processing FP windows..."
)



fp_results = [

    analyze_position(position)

    for position in fp_positions

]



print(
    "Processing High-confidence FP windows..."
)



high_fp_results = [

    analyze_position(position)

    for position in high_fp_positions

]



print()

print(
    "TP feature windows:",
    len(tp_results)
)



print(
    "FP feature windows:",
    len(fp_results)
)



print(
    "High FP feature windows:",
    len(high_fp_results)
)
# ============================================================
# GROUP SUMMARY
# ============================================================


SUMMARY_FEATURES = [

    "mean_rms",

    "max_rms",

    "mean_zero_crossing_rate",

    "max_zero_crossing_rate",

    "mean_line_length",

    "max_line_length",

    "mean_high_frequency_ratio",

    "max_high_frequency_ratio",

    "mean_beta_power",

    "max_beta_power",

    "mean_gamma_power",

    "max_gamma_power"

]



def summarize_group(results):


    output = {}



    for feature in SUMMARY_FEATURES:


        values = np.array(

            [

                item["summary"][feature]

                for item in results

            ],

            dtype=np.float32

        )



        if len(values) == 0:


            output[feature] = {

                "count": 0

            }


        else:


            output[feature] = {


                "count":

                    int(len(values)),



                "mean":

                    float(np.mean(values)),



                "median":

                    float(np.median(values)),



                "std":

                    float(np.std(values)),



                "min":

                    float(np.min(values)),



                "max":

                    float(np.max(values))

            }



    return output




tp_summary = summarize_group(

    tp_results

)



fp_summary = summarize_group(

    fp_results

)



high_fp_summary = summarize_group(

    high_fp_results

)




# ============================================================
# FEATURE COMPARISON
# ============================================================


print()

print("=" * 70)

print("ARTIFACT FEATURE COMPARISON")

print("=" * 70)



for feature in SUMMARY_FEATURES:


    print()

    print(feature)



    tp_mean = tp_summary[feature].get(

        "mean",

        0

    )


    fp_mean = fp_summary[feature].get(

        "mean",

        0

    )


    high_mean = high_fp_summary[feature].get(

        "mean",

        0

    )



    print(

        f"TP       = {tp_mean:.6f}"

    )


    print(

        f"FP       = {fp_mean:.6f}"

    )


    print(

        f"High FP  = {high_mean:.6f}"

    )





# ============================================================
# CHANNEL CONCENTRATION ANALYSIS
# ============================================================


print()

print("=" * 70)

print("CHANNEL CONCENTRATION ANALYSIS")

print("=" * 70)




def calculate_channel_mean(

    results,

    feature

):


    if len(results) == 0:


        return np.zeros(

            EXPECTED_CHANNELS

        )



    values = np.array(

        [

            [

                channel[feature]

                for channel in item["channels"]

            ]

            for item in results

        ],

        dtype=np.float32

    )



    return np.mean(

        values,

        axis=0

    )





CHANNEL_FEATURES = [

    "rms",

    "zero_crossing_rate",

    "line_length",

    "high_frequency_ratio",

    "beta_power",

    "gamma_power"

]




channel_comparison = {}




for feature in CHANNEL_FEATURES:


    tp_values = calculate_channel_mean(

        tp_results,

        feature

    )



    fp_values = calculate_channel_mean(

        fp_results,

        feature

    )



    high_values = calculate_channel_mean(

        high_fp_results,

        feature

    )



    relative_difference = (

        fp_values - tp_values

    ) / (

        np.abs(tp_values) + 1e-8

    )



    ranking = np.argsort(

        np.abs(relative_difference)

    )[::-1]



    print()

    print(feature.upper())



    for channel_index in ranking[:TOP_CHANNELS]:


        print(

            f"Channel {channel_index+1:02d} | "

            f"TP={tp_values[channel_index]:.6f} | "

            f"FP={fp_values[channel_index]:.6f} | "

            f"High-FP={high_values[channel_index]:.6f} | "

            f"Diff={relative_difference[channel_index]:+.4f}"

        )



    channel_comparison[feature] = {


        "tp_mean":

            tp_values.tolist(),



        "fp_mean":

            fp_values.tolist(),



        "high_fp_mean":

            high_values.tolist(),



        "relative_difference":

            relative_difference.tolist()

    }




# ============================================================
# DOMINANT HIGH FREQUENCY CHANNELS
# ============================================================


print()

print("=" * 70)

print("HIGH FREQUENCY DOMINANCE")

print("=" * 70)




def dominant_channel(

    result,

    feature

):


    values = np.array(

        [

            channel[feature]

            for channel in result["channels"]

        ]

    )



    return int(

        np.argmax(values)

    ) + 1




fp_dominant_channels = []


high_fp_dominant_channels = []




for item in fp_results:


    fp_dominant_channels.append(

        dominant_channel(

            item,

            "high_frequency_ratio"

        )

    )





for item in high_fp_results:


    high_fp_dominant_channels.append(

        dominant_channel(

            item,

            "high_frequency_ratio"

        )

    )





def count_channels(values):


    counts = {}



    for channel in values:


        channel = str(channel)



        counts[channel] = (

            counts.get(channel, 0)

            +

            1

        )



    return dict(

        sorted(

            counts.items(),

            key=lambda x: x[1],

            reverse=True

        )

    )




fp_dominant_counts = count_channels(

    fp_dominant_channels

)



high_fp_dominant_counts = count_channels(

    high_fp_dominant_channels

)




print()

print("FP dominant channels:")



for channel, count in list(

    fp_dominant_counts.items()

)[:10]:


    print(

        f"Channel {int(channel):02d}: {count}"

    )




print()

print("High-confidence FP dominant channels:")



for channel, count in list(

    high_fp_dominant_counts.items()

)[:10]:


    print(

        f"Channel {int(channel):02d}: {count}"

    )
# ============================================================
# VISUALIZATION
# ============================================================


OUTPUT_DIR.mkdir(

    parents=True,

    exist_ok=True

)



def plot_feature(

    feature,

    title,

    filename

):


    tp_values = np.array(

        [

            item["summary"][feature]

            for item in tp_results

        ]

    )



    fp_values = np.array(

        [

            item["summary"][feature]

            for item in fp_results

        ]

    )



    high_values = np.array(

        [

            item["summary"][feature]

            for item in high_fp_results

        ]

    )



    plt.figure(

        figsize=(8,5)

    )



    plt.boxplot(

        [

            tp_values,

            fp_values,

            high_values

        ],

        tick_labels=[

            "TP",

            "FP",

            "High-FP"

        ]

    )



    plt.title(

        title

    )


    plt.ylabel(

        feature

    )


    plt.grid(

        axis="y",

        alpha=0.3

    )


    plt.tight_layout()



    plt.savefig(

        OUTPUT_DIR / filename,

        dpi=150

    )



    plt.close()




plot_feature(

    "mean_high_frequency_ratio",

    "High Frequency Ratio: TP vs FP",

    "high_frequency_ratio_comparison.png"

)



plot_feature(

    "mean_zero_crossing_rate",

    "Zero Crossing Rate: TP vs FP",

    "zero_crossing_comparison.png"

)



plot_feature(

    "mean_line_length",

    "Line Length: TP vs FP",

    "line_length_comparison.png"

)



plot_feature(

    "mean_rms",

    "RMS: TP vs FP",

    "rms_comparison.png"

)




# ============================================================
# SAVE FINAL JSON
# ============================================================



results = {


    "settings": {


        "threshold":

            THRESHOLD,


        "high_confidence_threshold":

            HIGH_CONFIDENCE_THRESHOLD

    },



    "classification": {


        "tp":

            int(len(tp_positions)),



        "fp":

            int(len(fp_positions)),



        "high_confidence_fp":

            int(len(high_fp_positions))

    },



    "tp_summary":

        tp_summary,



    "fp_summary":

        fp_summary,



    "high_fp_summary":

        high_fp_summary,



    "channel_comparison":

        channel_comparison,



    "fp_dominant_high_frequency_channels":

        fp_dominant_counts,



    "high_fp_dominant_high_frequency_channels":

        high_fp_dominant_counts

}





RESULTS_DIR.mkdir(

    parents=True,

    exist_ok=True

)



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




# ============================================================
# FINAL OUTPUT
# ============================================================


print()

print("=" * 70)

print("ANALYSIS COMPLETED")

print("=" * 70)



print()

print("Results saved:")

print(

    OUTPUT_JSON

)



print()

print("Plots saved:")

print(

    OUTPUT_DIR

)



print()

print("DONE")