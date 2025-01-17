import logging
from typing import Any, Callable, Optional

import numpy as np
from numpy.typing import NDArray
from pgmpy.global_vars import logger
from pgmpy.utils import get_example_model

from _cond_int_cbn_mab import CondIntCBN_MAB
from _cond_int_ucb import CondIntUCB
from _split_algo import SPLIT_on_target
from _utils import get_node_with_most_ancestors, most_common_list_element

logger.setLevel(logging.ERROR)  # No warnings from pgmpy, only errors

DATASETS_TARGETS_DCT = {
    # Smaller:
    "asia": "dysp",
    "cancer": "Cancer",
    "sachs": "Akt",
    # Larger:
    # "survey": "T",  # categorical
    # "alarm": "BP",
    # "barley": "protein",  # too big
    # "child": "LowerBodyO2",  # too big
    # "insurance": "PropCost",  # too big
    # "mildew": "dm4",
    # "water": "CBODD_12_45",
}

# # Uncomment to check that chosen targets are indeed nodes with most ancestors
# # (and with more than one parent)
# for name, target in DATASETS_TARGETS_DCT.items():
#     bn = get_example_model(name)
#     chosen = get_node_with_most_ancestors(bn.to_directed(), no_single_children=True)
#     print(name, chosen, target)

# N_RUNS = 100  # graphs will average over the N_RUNS runs.
N_RUNS = 2  # graphs will average over the N_RUNS runs.
# N_ROUNDS = 1000  # number of rounds in each run
N_ROUNDS = 2  # number of rounds in each run


def generate_reward_converter(bn, target):
    target_states: list = bn.states[target]
    n = len(target_states)
    if n < 2:
        raise ValueError("Target should have at least 2 possible states.")
    # Create dictionary state: value based on state's index.
    state_value_dict = {state: i / (n - 1) for i, state in enumerate(target_states)}
    # print(state_value_dict)

    def reward_converter(state: Any):
        return state_value_dict[state]

    return reward_converter


# TODO will actually want to run 100 times each one, and plot average! + plot prob of best arm selection
# based on those 100 runs
if __name__ == "__main__":
    for name, target in DATASETS_TARGETS_DCT.items():
        print(f"\n\n=== Dataset: {name}; Target: {target}===")
        bn = get_example_model(name)

        reward_converter = generate_reward_converter(bn, target)

        print("\nCreating cond int mab using brute force")
        mab_bf = CondIntCBN_MAB(bn, target=target, verbose_search_space=True)

        cond_ucb_bf = CondIntUCB(mab_bf, reward_to_float_converter=reward_converter)
        n_nodes_bf = len(mab_bf.candidate_nodes)

        print("\nCreating cond int mab using mGISS")
        mab_mgiss = CondIntCBN_MAB(
            bn,
            target=target,
            search_space_reduction_func=SPLIT_on_target,
            verbose_search_space=True,
        )
        cond_ucb_mgiss = CondIntUCB(
            mab_mgiss, reward_to_float_converter=reward_converter
        )
        n_nodes_mgiss = len(mab_mgiss.candidate_nodes)

        mGISS_fraction = round(n_nodes_mgiss / n_nodes_bf, 2)

        histories_bf = []
        histories_mgiss = []
        for run in range(N_RUNS):
            print(f"\n\n==Run {run}/{N_RUNS} for dataset {name} and target {target}==")

            print("\nBrute-force")
            history_bf = cond_ucb_bf.run(N_ROUNDS)
            print("Total Reward:", sum(history_bf["observed_rewards"]))
            print("Cumulative Regret:", history_bf["cumulative_regrets"][-1])

            print("\nmGISS")
            history_mgiss = cond_ucb_mgiss.run(N_ROUNDS)
            print("Total Reward:", sum(history_mgiss["observed_rewards"]))
            print("Cumulative Regret:", history_mgiss["cumulative_regrets"][-1])

            histories_bf += [history_bf]
            histories_mgiss += [history_mgiss]

        # Get average cumulative regret curves
        cum_regrets_array_bf = np.array(
            [history["cumulative_regrets"] for history in histories_bf]
        )  # one row per run
        averaged_cum_regrets_bf = np.mean(cum_regrets_array_bf, axis=0)

        cum_regrets_array_mgiss = np.array(
            [history["cumulative_regrets"] for history in histories_mgiss]
        )  # one row per run
        averaged_cum_regrets_mgiss = np.mean(cum_regrets_array_mgiss, axis=0)

        # TODO best arm selection prob curves

        bf_sel_nodes: NDArray = np.array(
            [history["selected_nodes"] for history in histories_bf]
        )
        bf_best_node = most_common_list_element(
            [history["best_node_and_policy"][0] for history in histories_bf]
        )
        bf_mask = bf_sel_nodes == bf_best_node
        bf_probs_best_arm = bf_mask.mean(
            axis=0
        )  # Fraction of the N_RUNS where the best arm was chosen, for each index

        mgiss_sel_nodes: NDArray = np.array(
            [history["selected_nodes"] for history in histories_mgiss]
        )
        mgiss_best_node = most_common_list_element(
            [history["best_node_and_policy"][0] for history in histories_mgiss]
        )
        mgiss_mask = mgiss_sel_nodes == mgiss_best_node
        mgiss_probs_best_arm = mgiss_mask.mean(
            axis=0
        )  # Fraction of the N_RUNS where the best arm was chosen, for each index

        import pickle

        import matplotlib.pyplot as plt

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
        fig.suptitle(f"{name} dataset ({len(bn.nodes)} nodes) with target '{target}'")

        # First subplot: Cumulative Regret Curves
        # axs[0].set_title(
        #     f"Cumulative Regret Curves - {name} Dataset ({len(bn.nodes)} nodes) with target '{target}'"
        # )
        axs[0].plot(
            averaged_cum_regrets_bf,
            label=f"Brute-force ({n_nodes_bf} nodes)",
            color="red",
        )
        axs[0].plot(
            averaged_cum_regrets_mgiss,
            label=f"mGISS (fraction: {mGISS_fraction})",
            color="darkblue",
        )
        # axs[0].legend()
        # axs[0].set_xlabel("Rounds")
        axs[0].set_ylabel("Cumulative Regret")
        axs[0].tick_params(axis="x", labelbottom=False)  # Hide x-axis tick labels
        axs[0].spines["top"].set_visible(False)
        axs[0].spines["right"].set_visible(False)

        # Second subplot: Optimal Arm Selection Probability
        # axs[1].set_title(
        #     f"Optimal Arm Selection Probability - {name} Dataset ({len(bn.nodes)} nodes) with target '{target}'"
        # )
        axs[1].plot(
            bf_probs_best_arm,
            label=f"Brute-force ({n_nodes_bf} nodes)",
            color="red",
        )
        axs[1].plot(
            mgiss_probs_best_arm,
            label=f"mGISS (fraction: {mGISS_fraction})",
            color="darkblue",
        )
        axs[1].legend(frameon=False, loc="lower right")
        axs[1].set_xlabel("Rounds")
        axs[1].set_ylabel("Probability of Best Arm")
        axs[1].spines["top"].set_visible(False)
        axs[1].spines["right"].set_visible(False)

        # Adjust layout and spacing
        plt.tight_layout()
        fig.subplots_adjust(wspace=0.3)  # space between supplots

        # Save fig
        with open(  # Save Figure object for last minute changes
            f"./Images/ucb_results_{name}_{N_ROUNDS}runs_{N_ROUNDS}rounds.pkl", "wb"
        ) as handle:
            pickle.dump(fig, handle)
        plt.savefig(f"./Images/ucb_results_{name}_{N_ROUNDS}runs_{N_ROUNDS}rounds.png")

        try:
            plt.show()
        except RuntimeError as e:  # Avoid backend-related errors
            print(e)

        # # Free up memory
        # del bn
        # del mab_bf
        # del cond_ucb_bf
        # del mab_mgiss
        # del cond_ucb_mgiss
