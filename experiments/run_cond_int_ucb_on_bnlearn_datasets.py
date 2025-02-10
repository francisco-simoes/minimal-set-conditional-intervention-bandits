import concurrent.futures
import logging
from typing import Any, Callable, Optional

import numpy as np
from numpy.typing import NDArray
from pgmpy.global_vars import logger
from pgmpy.utils import get_example_model

from _c4_algo import C4_on_target
from _cond_int_cbn_mab import CondIntCBN_MAB
from _cond_int_ucb import CondIntUCB
from _utils import get_node_with_most_ancestors, most_common_list_element

logger.setLevel(logging.ERROR)  # No warnings from pgmpy, only errors

MAX_WORKERS = 2  # For parallelization (LOCAL)
# MAX_WORKERS = 30  # For parallelization (CLUSTER)

DATASETS_TARGETS_DCT = {
    "asia": "dysp",
    "sachs": "Akt",
    "child": "LowerBodyO2",
    "pathfinder": "F70",
}

# # Uncomment to check that chosen targets are indeed nodes with most ancestors
# # (and with more than one parent), except for the cancer model.
# for name, target in DATASETS_TARGETS_DCT.items():
#     bn = get_example_model(name)
#     chosen = get_node_with_most_ancestors(bn.to_directed(), no_single_children=True)
#     print(name, chosen, target)

N_RUNS = 5  # graphs will average over the N_RUNS runs.
# N_RUNS = 500  # graphs will average over the N_RUNS runs.
# N_RUNS = 2  # graphs will average over the N_RUNS runs.

# N_ROUNDS = 2000  # number of rounds in each run (SMALLER datasets)
# N_ROUNDS = 50000  # number of rounds in each run (LARGER datasets)
# N_ROUNDS = 5000  # number of rounds in each run (LARGER datasets)
N_ROUNDS = 100  # number of rounds in each run


def generate_reward_converter(bn, target):
    target_states: list = bn.states[target]
    n = len(target_states)
    if n < 2:
        raise ValueError("Target should have at least 2 possible states.")
    # Create dictionary state: value based on state's index.
    state_value_dict = {state: i / (n - 1) for i, state in enumerate(target_states)}
    # print(state_value_dict)

    # def reward_converter(state: Any):
    #     return state_value_dict[state]

    return RewardConverter(state_value_dict)


# Must define a generic reward conversion callable on top-level,
# otherwise multiprocessing fails due to pickling problems
class RewardConverter:
    def __init__(self, state_value_dict: Callable):
        self.state_value_dict = state_value_dict

    def __call__(self, state: Any):
        return self.state_value_dict[state]


reward_converters = {}
# (These need to be defined in global scope, otherwise multiprocessing fails
#        due to peculiarities of pickle package.)
for name, target in DATASETS_TARGETS_DCT.items():
    bn = get_example_model(name)
    func_name = f"{name}_reward_converter"
    # Dynamically add reward converter functions to the global namespace
    # globals()[func_name] = generate_reward_converter(bn, target)
    # reward_converters[name] = globals()[func_name]

    reward_converters[name] = generate_reward_converter(bn, target)

del bn  # Free up memory


if __name__ == "__main__":
    for name, target in DATASETS_TARGETS_DCT.items():
        print(f"\n\n=== Dataset: {name}; Target: {target}===")
        bn = get_example_model(name)

        # reward_converter = generate_reward_converter(bn, target)

        def single_run(
            run_number,
            N_RUNS,
            dataset,
            target,
            N_ROUNDS,
            cond_ucb_instance,
            node_selection_method: str,
        ):
            assert node_selection_method in ("bf", "mgiss")
            print(
                f"\n\n==Run {run_number}/{N_RUNS} for dataset {dataset} and target {target} [{node_selection_method}]=="  # noqa
            )

            results = {}

            history = cond_ucb_instance.run(N_ROUNDS)
            print("Total Reward:", sum(history["observed_rewards"]))
            print("Cumulative Regret:", history["cumulative_regrets"][-1])
            results[node_selection_method] = history

            return results

        # === Brute-Force ===
        print("\nCreating cond int mab using brute force")
        mab_bf = CondIntCBN_MAB(bn, target=target, verbose_search_space=True)

        print("\nCreating cond int ucb instance using brute force")
        cond_ucb_bf = CondIntUCB(
            mab_bf, reward_to_float_converter=reward_converters[name]
        )
        n_nodes_bf = len(mab_bf.candidate_nodes)

        histories_bf = []

        # Parallel execution for brute-force
        print(
            f"Starting parallel execution for brute-force method on {MAX_WORKERS} workers."
        )
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=MAX_WORKERS
        ) as executor:
            # Submit tasks to the executor
            futures = []
            for run in range(N_RUNS):
                # Submit the task and store the future
                future = executor.submit(
                    single_run,
                    run,
                    N_RUNS,
                    name,
                    target,
                    N_ROUNDS,
                    cond_ucb_bf,
                    "bf",
                )
                futures.append(future)
                # Print the submitted task information
                print(f"Run {run} has been submitted to the executor.")

            # Gather results
            for future in concurrent.futures.as_completed(futures):
                history = future.result()
                histories_bf.append(history["bf"])
        print("Parallelized execution completed for brute-force method.")

        # Free up memory
        del mab_bf
        del cond_ucb_bf

        # === mGISS ===
        print("\nCreating cond int mab using mGISS")
        mab_mgiss = CondIntCBN_MAB(
            bn,
            target=target,
            search_space_reduction_func=C4_on_target,
            verbose_search_space=True,
        )
        print("\nCreating cond int ucb instance using mGISS")
        cond_ucb_mgiss = CondIntUCB(
            mab_mgiss, reward_to_float_converter=reward_converters[name]
        )
        n_nodes_mgiss = len(mab_mgiss.candidate_nodes)

        mGISS_fraction = round(n_nodes_mgiss / n_nodes_bf, 2)

        histories_mgiss = []

        # Parallel execution for mgiss
        print(f"Starting parallel execution for mgiss method on {MAX_WORKERS} workers.")
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=MAX_WORKERS
        ) as executor:
            # Submit tasks to the executor
            futures = []
            for run in range(N_RUNS):
                # Submit the task and store the future
                future = executor.submit(
                    single_run,
                    run,
                    N_RUNS,
                    name,
                    target,
                    N_ROUNDS,
                    cond_ucb_mgiss,
                    "mgiss",
                )
                futures.append(future)
                # Print the submitted task information
                print(f"Run {run} has been submitted to the executor.")

            # Gather results
            for future in concurrent.futures.as_completed(futures):
                history = future.result()
                histories_mgiss.append(history["mgiss"])
        print("Parallelized execution completed for mgiss method.")

        # Free up memory
        del mab_mgiss
        del cond_ucb_mgiss

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
        from datetime import datetime

        import matplotlib.pyplot as plt

        # == Save results ==
        timestamp = datetime.now().strftime("%d_%H_%M_%S")
        with open(  # Save histories
            f"./Results/conducb_histories_{name}_{N_RUNS}runs_{N_ROUNDS}rounds_{timestamp}.pkl",
            "wb",
        ) as handle:
            pickle.dump({"bf": histories_bf, "mgiss": histories_mgiss}, handle)

        with open(  # Save cumulative regrets and best arm probs
            f"./Results/conducb_metrics_{name}_{N_RUNS}runs_{N_ROUNDS}rounds_{timestamp}.pkl",
            "wb",
        ) as handle:
            pickle.dump(
                {
                    "avg_cumulative_regrets_bf": averaged_cum_regrets_bf,
                    "avg_cumulative_regrets_mgiss": averaged_cum_regrets_mgiss,
                    "probs_best_arm_bf": bf_probs_best_arm,
                    "probs_best_arm_mgiss": mgiss_probs_best_arm,
                },
                handle,
            )

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
        # axs[1].legend(frameon=False, loc="lower right")
        axs[1].set_xlabel("Rounds")
        axs[1].set_ylabel("Probability of Best Arm")
        axs[1].spines["top"].set_visible(False)
        axs[1].spines["right"].set_visible(False)

        # Adjust layout and spacing
        plt.tight_layout()
        fig.subplots_adjust(wspace=0.3)  # space between supplots

        # Save fig
        with open(  # Save Figure object for last minute changes
            f"./Images/conducb_results_{name}_{N_RUNS}runs_{N_ROUNDS}rounds_{timestamp}.pkl",
            "wb",
        ) as handle:
            pickle.dump(fig, handle)
        plt.savefig(
            f"./Images/conducb_results_{name}_{N_RUNS}runs_{N_ROUNDS}rounds_{timestamp}.png"
        )

        try:
            plt.show()
        except RuntimeError as e:  # Avoid backend-related errors
            print(e)
