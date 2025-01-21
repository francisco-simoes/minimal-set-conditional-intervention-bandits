import os
import pickle
import re

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

DATASETS_ROUNDS: dict[str, int] = {
    "asia": 1000,
    # "cancer": 1000 ,
    # "sachs": 5000,
    # "child":10000,
    # "water": 10000,
}

DATASET_TARGET_PAIRS: dict[str, str] = {
    "asia": "dysp",
    # "cancer": "Dyspnoea",
    # "sachs": "Akt",
    # "child": "LowerBody2",
    # "water": "CBODD_12_45",
}


def _contains_n_rounds_substring(s, n_rounds):
    # Create a regex pattern that looks for the exact number followed by 'rounds'
    # Use a lookbehind to ensure it's not preceded by another digit,
    # and a lookahead to ensure it's followed by 'rounds'
    pattern = r"(?<!\d){}rounds(?!\d)".format(re.escape(str(n_rounds)))
    return bool(re.search(pattern, s))


def extract_metrics_from_files(
    datasets_rounds, directory="./"
) -> dict[str, dict[str, NDArray]]:
    """
    Extracts metrics from pickle files for given dataset names and numbers of rounds.

    This function scans the specified directory for pickle files that contain
    metrics data for the given datasets. It extracts numpy arrays from the keys
    "avg_cumulative_regrets_bf", "avg_cumulative_regrets_mgiss",
    "probs_best_arm_bf", and "probs_best_arm_mgiss" in the pickle files.

    Args:
        dataset_names (dict): A dictionary of dataset names and round numbers to filter and extract
                                                            the corresponding metrics.
        directory (str): The directory to scan for pickle files. Defaults to the current directory ('./').

    Returns:
        dict: A dictionary where the keys are dataset names and the values are dictionaries containing:
              - "avg_cumulative_regrets_bf",
              - "avg_cumulative_regrets_mgiss",
              - "probs_best_arm_bf",
              - "probs_best_arm_mgiss".
              If no metrics files are found for a dataset, the dictionary for that dataset will be empty.
    """

    extracted_data = {}
    for dataset, n_rounds in datasets_rounds.items():
        dataset_data = {
            "avg_cumulative_regrets_bf": np.empty(shape=(0, n_rounds)),
            "avg_cumulative_regrets_mgiss": np.empty(shape=(0, n_rounds)),
            "probs_best_arm_bf": np.empty(shape=(0, n_rounds)),
            "probs_best_arm_mgiss": np.empty(shape=(0, n_rounds)),
        }

        # Loop through the files in the directory
        for filename in os.listdir(directory):
            # Check if the filename matches the dataset name, n_rounds and if it's a "metrics" file
            if (
                dataset in filename
                and _contains_n_rounds_substring(filename, n_rounds)
                and "metrics" in filename
                and filename.endswith(".pkl")
            ):
                filepath = os.path.join(directory, filename)

                # Load file
                with open(filepath, "rb") as f:
                    current_file_data = pickle.load(f)

                # Ensure the dictionary contains the required keys
                for metric_name in dataset_data.keys():
                    if metric_name in current_file_data:
                        # Add data to array
                        # TODO This is wrong! need to re-compute averages!
                        # Need to: collect all arrays and register their n_runs,
                        # to then perform a weighted average
                        # OR: NOTE: Maybe easier: get histories instead, concatenate
                        # them and compute averages metric curves from scratch. can
                        # also extract numbers of nodes that way!
                        # OR: NOTE: just re-run everything with the right amount of runs.
                        dataset_data[metric_name] = np.concatenate(
                            (
                                dataset_data[metric_name],
                                current_file_data[metric_name].reshape(-1, n_rounds),
                            ).flatten()
                        )
                    else:
                        print(f"Warning: Key '{metric_name}' not found in {filename}")

        # Store the data for this dataset
        extracted_data[dataset] = dataset_data

    return extracted_data


def plot_cond_bandits_metrics(
    dataset: str,
    target: str,
    dataset_metrics: dict[str, NDArray],
):
    # == Plotting ==
    # Set a global style for consistency and better aesthetics
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 10,  # Title font size
            "axes.labelsize": 9,  # Label font size
            "legend.fontsize": 8,  # Legend font size
            "lines.linewidth": 1.5,
        }
    )

    # Create a figure with two subplots (1 row, 2 columns)
    fig, axs = plt.subplots(2, 1, figsize=(4, 6))  # Adjust figure size as needed
    fig.suptitle(f"{dataset} dataset with target '{target}'")

    # First subplot: Cumulative Regret Curves
    # axs[0].set_title(
    #     f"Cumulative Regret Curves - {name} Dataset ({len(bn.nodes)} nodes) with target '{target}'"
    # )

    averaged_cum_regrets_bf = dataset_metrics["avg_cumulative_regrets_bf"]
    averaged_cum_regrets_mgiss = dataset_metrics["avg_cumulative_regrets_mgiss"]
    axs[0].plot(
        averaged_cum_regrets_bf,
        # label=f"Brute-force ({n_nodes_bf} nodes)",
        label=f"Brute-force",
        color="red",
    )
    axs[0].plot(
        averaged_cum_regrets_mgiss,
        # label=f"mGISS (fraction: {mGISS_fraction})",
        label=f"mGISS",
        color="darkblue",
    )
    axs[0].legend(frameon=False, loc="lower right")
    # axs[0].set_xlabel("Rounds")
    axs[0].set_ylabel("Cumulative Regret")
    axs[0].tick_params(axis="x", labelbottom=False)  # Hide x-axis tick labels
    axs[0].spines["top"].set_visible(False)
    axs[0].spines["right"].set_visible(False)

    # Second subplot: Optimal Arm Selection Probability
    # axs[1].set_title(
    #     f"Optimal Arm Selection Probability - {name} Dataset ({len(bn.nodes)} nodes) with target '{target}'"
    # )
    mgiss_probs_best_arm = dataset_metrics["probs_best_arm_mgiss"]
    bf_probs_best_arm = dataset_metrics["probs_best_arm_bf"]
    # fmt:off
    import ipdb; ipdb.set_trace() # noqa
    # fmt:on
    axs[1].plot(
        bf_probs_best_arm,
        # label=f"Brute-force ({n_nodes_bf} nodes)",
        label=f"Brute-force",
        color="red",
    )
    axs[1].plot(
        mgiss_probs_best_arm,
        # label=f"mGISS (fraction: {mGISS_fraction})",
        label=f"mGISS",
        color="darkblue",
    )
    # axs[1].legend(frameon=False, loc="lower right")
    axs[1].set_xlabel("Rounds")
    axs[1].set_ylabel("Probability of Best Arm")
    axs[1].spines["top"].set_visible(False)
    axs[1].spines["right"].set_visible(False)

    # Adjust layout and spacing
    plt.tight_layout()
    fig.subplots_adjust(wspace=0.3)  # space between supplots

    try:
        plt.show()
    except RuntimeError as e:  # Avoid backend-related errors
        print(e)


if __name__ == "__main__":
    directory = "./"  # Modify with the directory where your pickle files are located

    extracted_data = extract_metrics_from_files(DATASETS_ROUNDS, directory)

    for dataset, target in DATASET_TARGET_PAIRS.items():
        plot_cond_bandits_metrics(dataset, target, extracted_data[dataset])
