from itertools import accumulate
from typing import Any, Callable, Optional

import numpy as np
from scipy.stats import bernoulli
from tqdm import tqdm

from _cond_int_cbn_mab import CondIntCBN_MAB
from _fixed_node_contextual_ucb import FixedNodeContextualUCB
from _split_algo import SPLIT_on_target
from _utils import rowdf_to_dict


class CondIntUCB:
    """
    A class to run a Conditional Intervention UCB algorithm where, in each iteration,
    a node is selected using UCB, and a step is run on the FixedNodeContextualUCB
    instance corresponding to that node.
    """

    def __init__(
        self,
        mab: CondIntCBN_MAB,
        reward_to_float_converter: Optional[Callable] = None,
        # optimal_expected_reward: Optional[list[float]] = None,
    ):
        self.mab = mab
        self.target = mab.target
        self.reward_to_float_converter = reward_to_float_converter

        # Initialize FixedNodeContextualUCB instances for each node
        self.candidate_nodes = list(self.mab.candidate_nodes)

        self.fixed_node_ucbs: dict[Any, FixedNodeContextualUCB] = {}
        for node in self.candidate_nodes:
            # for node in tqdm(self.candidate_nodes):
            print(f"Creating UCBs for node {node}")
            print(f"\tstates: {mab.bn.states[node]}")
            self.fixed_node_ucbs[node] = FixedNodeContextualUCB(
                node, mab, reward_to_float_converter
            )
        # self.fixed_node_ucbs: dict[Any, FixedNodeContextualUCB] = {
        #     node: FixedNodeContextualUCB(node, mab, reward_to_float_converter)
        #     for node in tqdm(self.candidate_nodes)
        # }

        self.n_nodes = len(self.candidate_nodes)

        #         if optimal_expected_reward is None:
        #             print(
        #                 """
        # Optimal expected rewards not given. I will compute cumulative regret values retroactively,
        # using the empirical estimation of the optimal rewards.
        # """
        #             )
        # self.optimal_expected_reward = optimal_expected_reward
        self._initialize_run()

    def _initialize_run(self):
        for contextual_ucb in self.fixed_node_ucbs.values():
            contextual_ucb._initialize_run()
        self.total_pulls = 0  # Total number of pulls during run
        self.node_pull_counts = np.zeros(self.n_nodes)
        self.node_rewards = np.zeros(self.n_nodes)
        self.node_expected_rewards = np.zeros(self.n_nodes)

        self.selected_nodes = []
        self.selected_nodes_idxs = []
        # self.selected_arms = []  # Pulled arms during run
        self.optimal_expected_reward = None
        self.observed_rewards = []
        self.instant_regrets = []
        self.cumulative_regrets = []
        self.best_node_and_policy: tuple[
            Any, dict[Any, Any]
        ]  # [0]: best node; [1]: best policy for that node

    def _update_ucb_score(self, node_index):
        """Compute the UCB score for a node, given its index."""
        if self.node_pull_counts[node_index] == 0:
            return np.inf  # Encourage exploration for unselected nodes
        avg_reward = self.node_rewards[node_index] / self.node_pull_counts[node_index]
        return avg_reward + np.sqrt(
            2 * np.log(self.total_pulls + 1) / self.node_pull_counts[node_index]
        )

    def _select_node(self):
        """Select a node based on the UCB scores."""
        ucb_scores = [self._update_ucb_score(i) for i in range(self.n_nodes)]
        return np.argmax(ucb_scores)

    def _step(self):
        # Select node based on UCB
        node_index = self._select_node()
        node = self.candidate_nodes[node_index]

        # Run one step of FixedNodeContextualUCB for the selected node
        contextual_ucb = self.fixed_node_ucbs[node]
        contextual_ucb._step()

        # Get the reward from the last step
        reward = contextual_ucb.observed_rewards[-1]

        # Track progress
        self.node_pull_counts[node_index] += 1
        self.node_rewards[node_index] += reward
        self.total_pulls += 1
        self.selected_nodes.append(node)
        self.selected_nodes_idxs.append(node_index)
        # self.selected_arms.append(contextual_ucb.selected_arms[-1])
        self.observed_rewards.append(reward)

    def _record_details(self):

        # Record run details into Contextual UCB instances' attributes
        for contextual_ucb in self.fixed_node_ucbs.values():
            contextual_ucb.record_details()

        # Find best node,policy pair
        self.node_expected_rewards = np.nan_to_num(
            self.node_rewards / self.node_pull_counts,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        self.expected_rewards = [
            self.node_expected_rewards[i].item() for i in self.selected_nodes_idxs
        ]  # Expected rewards of selected nodes
        best_node_index = np.argmax(self.node_expected_rewards).item()
        assert len(self.candidate_nodes) == len(self.node_expected_rewards)
        best_node = self.candidate_nodes[best_node_index]
        best_policy = self.fixed_node_ucbs[best_node].best_policy
        # NOTE: we find the best node,policy pair under the assumption that the
        # best_node empirically found is indeed the best node of this pair.
        # Alternatively, we could esitmate the average reward of the best policy
        # found by each fixed node UCB (by using the frequencies of each context
        # and the rewards for when the "right action" was taken for those contexts),
        # but this would demand high numbers of ietrations, to guarantee that __all__ = [
        # context, best action pairs where sampled often enough.
        self.best_node_and_policy = (best_node, best_policy)

        # For regret: compare expected reward of chosen node with the
        # expected reward of the best node
        # Best node is the one with largest (empirical) average reward
        # NOTE: our optimal_expect_reward is wrt to the node only (not assumming
        # the best policy was chosen), since our task here is only to select the
        # best node.
        if self.optimal_expected_reward is None:
            # Empirical optimal reward estimate, assuming best_node is actually the best
            self.optimal_expected_reward = self.node_expected_rewards[best_node_index]

        # Compute cumulative regret(s)
        # for i in range(n_rounds):
        #     chosen_node_index = self.selected_nodes_idxs[i]
        #     chosen_node = self.candidate_nodes[chosen_node_index]
        #     contextual_ucb = self.fixed_node_ucbs[chosen_node]
        self.instant_regrets = self.optimal_expected_reward - self.expected_rewards
        assert np.all(
            self.instant_regrets >= -0.001
        ), "Instant regret should never be negative."
        self.cumulative_regrets = list(accumulate(self.instant_regrets))

        history = {
            "selected_nodes": self.selected_nodes,
            "observed_rewards": self.observed_rewards,
            "instant_regrets": self.instant_regrets,
            "cumulative_regrets": self.cumulative_regrets,
            "best_node_and_policy": self.best_node_and_policy,
            "node_expected_rewards": self.node_expected_rewards,
        }

        return history

    def run(self, n_rounds: int, fresh_start: bool = True):
        """Run the Conditional Intervention UCB algorithm for a specified number of rounds."""
        if fresh_start:
            self._initialize_run()

        for _ in tqdm(range(n_rounds)):
            self._step()

        history = self._record_details()

        return history


# Example usage
if __name__ == "__main__":
    from pgmpy.utils import get_example_model

    bn = get_example_model("asia")

    def yes_is_zero_converter(s: str):
        if s == "yes":
            return 0
        elif s == "no":
            return 1
        else:
            raise ValueError(
                "Reward to convert must be yes or no if one uses this converter."
            )

    mab_bf = CondIntCBN_MAB(bn, target="dysp")
    mab_mgiss = CondIntCBN_MAB(
        bn, target="dysp", search_space_reduction_func=SPLIT_on_target
    )
    cond_ucb_bf = CondIntUCB(mab_bf, reward_to_float_converter=yes_is_zero_converter)
    cond_ucb_mgiss = CondIntUCB(
        mab_mgiss, reward_to_float_converter=yes_is_zero_converter
    )

    n_rounds = 1000
    history_bf = cond_ucb_bf.run(n_rounds)
    print("Total Reward:", sum(history_bf["observed_rewards"]))
    print("Cumulative Regret:", history_bf["cumulative_regrets"][-1])

    history_mgiss = cond_ucb_mgiss.run(n_rounds)
    print("Total Reward:", sum(history_mgiss["observed_rewards"]))
    print("Cumulative Regret:", history_mgiss["cumulative_regrets"][-1])

    from matplotlib.pyplot import legend, plot, show, title

    title("Cumulative Regret Curves - Lung Cancer Dataset")
    plot(history_bf["cumulative_regrets"], label="Brute-force")
    plot(history_mgiss["cumulative_regrets"], label="mGISS")
    legend()
    try:
        show()
    except RuntimeError as e:  # Avoid backend-related errors
        print(e)

    # # fmt:off
    # import ipdb; ipdb.set_trace() # noqa
    # # fmt:on
