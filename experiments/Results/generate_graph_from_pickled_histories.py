import os
import pickle
import re

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

DATASET_FILE_DICT: dict[str, str] = {  # Choose histories files to load
    "asia": "./conducb_histories_asia_500runs_5000rounds_27_00_45_32.pkl",
    "sachs": "./conducb_histories_sachs_500runs_5000rounds_22_12_37_40.pkl",
    "child": "./conducb_histories_child_500runs_5000rounds_25_18_03_53.pkl",
    "pathfinder": "./conducb_histories_pathfinder_300runs_7000rounds_05_05_18_03.pkl",
}

DATASET_TARGET_DICT: dict[str, str] = {
    "asia": "dysp",
    "sachs": "Akt",
    "child": "LowerBody2",
    "pathfinder": "F70",
}

DATASET_N_NODES_DICT: dict[str, int] = {
    "asia": 8,
    "sachs": 11,
    "child": 20,
    "pathfinder": 109,
}


def plot_cumulative_regret_curves(
    dataset: str, file_path: str, n_nodes: int, target: str, legend_on=False, save=False
):
    """Plot CR curves (and their standard deviations)."""
    # Load histories
    with open(file_path, "rb") as handle:
        histories = pickle.load(handle)

    # Plotting
    plt.figure(figsize=(4.5, 4))
    colors = {"bf": "red", "mgiss": "darkblue"}
    labels = {"bf": "brute-force", "mgiss": "mGISS"}
    for strategy in colors.keys():
        histories_this_strategy = histories[strategy]
        cum_regrets_array = np.array(
            [history["cumulative_regrets"] for history in histories_this_strategy]
        )  # one row per run

        x = range(len(cum_regrets_array.mean(axis=0)))
        y = cum_regrets_array.mean(axis=0)
        std_devs = cum_regrets_array.std(axis=0)
        plt.fill_between(
            x,
            y - std_devs,
            y + std_devs,
            color=colors[strategy],
            alpha=0.2,
            edgecolor=None,
            # label="Standard Deviation",
        )
        n_nodes_this_strategy = len(histories_this_strategy[0]["node_expected_rewards"])
        plt.plot(
            x,
            y,
            label=f"{labels[strategy]} ({n_nodes_this_strategy} nodes)",
            color=colors[strategy],
        )
        ax = plt.gca()
        if legend_on:
            ax.legend(frameon=False, loc="lower right")
        ax.set_xlabel("Rounds")
        ax.set_ylabel("Cumulative Regret")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    plt.title(f"{dataset} ({n_nodes} nodes). Target: {target}")
    plt.tight_layout()
    if save:
        plt.savefig(f"../Images/cumulative_regret_curves_{dataset}.png")

    # try:
    #     plt.show()
    # except RuntimeError as e:  # Avoid backend-related errors
    #     print(e)
    plt.clf()


if __name__ == "__main__":
    # Set a global style for consistency and better aesthetics
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.titlesize": 14,  # Title font size
            "axes.labelsize": 10,  # Label font size
            "legend.fontsize": 10,  # Legend font size
            "lines.linewidth": 1.5,
        }
    )

    legend_on = True
    for dataset, file_path in DATASET_FILE_DICT.items():
        n_nodes = DATASET_N_NODES_DICT[dataset]
        target = DATASET_TARGET_DICT[dataset]
        # legend_on = True if dataset == "asia" else False
        plot_cumulative_regret_curves(
            dataset, file_path, n_nodes, target, legend_on=legend_on, save=True
        )
