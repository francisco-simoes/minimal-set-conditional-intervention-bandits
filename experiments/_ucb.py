# TODO: the "arm pulling" must, in our case, correspond to sampling
# from a post-intervention dist. So, I should create another file with an SCM class
# with a do() method creating distributions with a sample() method. These distributions
# can be fed to this UCB class as elements of the "reward_distributions" list.
# TODO: the "arm pulling" must, in our case, be conditional on a context.
# We will have one UCB per context. Will create a ContextualUCB class corresopnding
# to one UCB per context.

from itertools import accumulate
from typing import Any, Callable, Optional

import numpy as np
from scipy.stats import bernoulli
from tqdm import tqdm

from _cond_int_cbn_mab import CondIntCBN_MAB
from _samplers import RewardSamplerBase
# from _utils import BernoulliRV
from _utils import RandomVariable, rowdf_to_dict


class NodeUCB:
    """Run UCB on a bandit problem corresponding to a (node, context) pair.

    This bandit problem is characterized by a chosen node, a stochastic reward sampler
    which depends the context and chosen atomic intervention on the node,
    and a chosen context. The reward sampler will be a method of an instance of
    the RewardSamplerBase class.
    """

    def __init__(
        self,
        node: str,
        node_states: list[Any],
        context,
        reward_sampler_base: RewardSamplerBase,
        reward_to_float_converter: Optional[Callable] = None,
        optimal_expected_reward: Optional[float] = None,
    ):
        self.node = node
        self.context = context
        self.reward_sampler = reward_sampler_base.sample_reward
        self.arm_names = node_states  # each arm correspond to a state/name
        self.reward_converter = (
            reward_to_float_converter  # Needed in case reward is not float
        )
        self.n_arms = len(node_states)
        if optimal_expected_reward is None:
            print(
                """\nOptimal expected reward not given. I will compute cumulative regret
values retroactively, using the empirical estimation of
the optimal reward.\n"""
            )
        self.optimal_expected_reward = optimal_expected_reward
        self._initialize_run()

    def _initialize_run(self):
        self.arm_counts = np.zeros(
            self.n_arms
        )  # Number of times each arm has been pulled
        self.arm_rewards = np.zeros(self.n_arms)  # Sum of rewards for each arm
        self.total_pulls = 0  # Total number of pulls during run
        self.unpulled_arms_exist = True
        self.selected_arms = []  # Pulled arms during a run
        self.observed_rewards = []  # Rewards observed during run
        self.arm_expected_rewards = np.zeros(
            self.n_arms
        )  # Expected reward for each arm
        # self.observed_rewards_probs = []  # Probabilities of observed rewards
        self.cumulative_regrets = []  # (Instantaneous) cumulative regrets
        self.best_arm = None

    def _select_arm(self):
        # If an arm hasn't been pulled yet, select it
        if self.unpulled_arms_exist:
            for i in range(self.n_arms):
                if self.arm_counts[i] == 0:
                    return i
            # If code reaches this, no need to search for unpulled arms anymore
            self.unpulled_arms_exist = False

        # Calculate UCB values for each arm
        ucb_values = np.zeros(self.n_arms)
        for i in range(self.n_arms):
            avg_reward = self.arm_rewards[i] / self.arm_counts[i]
            exploration_bonus = np.sqrt(
                (2 * np.log(self.total_pulls)) / self.arm_counts[i]
            )
            ucb_values[i] = avg_reward + exploration_bonus

        # Select the arm with the highest UCB value
        return np.argmax(ucb_values).item()

    def _update_arm(self, chosen_arm, reward):
        self.arm_counts[chosen_arm] += 1
        self.arm_rewards[chosen_arm] += reward
        self.total_pulls += 1

    def step(self):
        chosen_arm = self._select_arm()
        intervention = {self.node: self.arm_names[chosen_arm]}
        # Pull arm
        reward = self.reward_sampler(
            do=intervention,
            context=self.context,
            state_to_float_converter=self.reward_converter,
        )[0]
        self._update_arm(chosen_arm, reward)
        self.selected_arms += [chosen_arm]
        self.observed_rewards += [reward]

    def record_details(self):
        # Estimate expected rewards for each arm, and then for each chosen arm
        self.arm_expected_rewards = self.arm_rewards / self.arm_counts
        self.expected_rewards = [
            self.arm_expected_rewards[i].item() for i in self.selected_arms
        ]

        # best_arm is the one with largest estimated expected reward
        self.best_arm = np.argmax(self.arm_expected_rewards).item()

        if self.optimal_expected_reward is None:
            # Empirical optimal reward estimate, assuming best arm is actually best
            self.optimal_expected_reward = self.arm_expected_rewards[self.best_arm]

        # Computation of cumulative regret
        instant_regrets = self.optimal_expected_reward - self.expected_rewards
        self.cumulative_regrets = list(accumulate(instant_regrets))

        history = {
            "selected_arms": self.selected_arms,
            "observed_rewards": self.observed_rewards,
            "cum_regrets": self.cumulative_regrets,
            "best_arm": self.best_arm,
            "arm_expected_rewards": self.arm_expected_rewards,
        }

        return history

    def run(self, n_rounds, fresh_start=True):
        """Run UCB algorithm.

        n_rounds: Number of rounds to play.
        fresh_start: Reset class attributes before running.
        """
        if fresh_start:
            self._initialize_run()

        for _ in tqdm(range(n_rounds)):
            self.step()

        history = self.record_details()

        return history


# Example usage
if __name__ == "__main__":
    from pgmpy.utils import get_example_model

    # Test NodeUCB in Asia dataset
    bn = get_example_model("asia")
    target = "dysp"

    def yes_is_zero_converter(s: str):
        if s == "yes":
            return 0
        elif s == "no":
            return 1
        else:
            raise ValueError(
                "Reward to convert must be yes or no if one uses this converter."
            )

    mab = CondIntCBN_MAB(bn, target)
    node = "either"
    node_states: list[str] = mab.bn.states[node]
    context_vars = mab.node_contexts[node]
    contextdf = mab.bn.simulate(n_samples=1, show_progress=False)[context_vars]
    context: dict[str, Any] = rowdf_to_dict(contextdf)

    print("Context:", context)

    n_rounds = 1000
    ucb = NodeUCB(
        node, node_states, context, mab, reward_to_float_converter=yes_is_zero_converter
    )
    history = ucb.run(n_rounds)

    print("Total Reward:", sum(history["observed_rewards"]))
    print("Number of times each arm was pulled:", ucb.arm_counts)
    print("Estimated expected reward for each arm:", ucb.arm_expected_rewards)
    print(
        "Best arm:",
        f"{ucb.best_arm}, that is, '{node}' = '{node_states[ucb.best_arm]}'.",
    )

    from matplotlib.pyplot import plot, show

    plot(ucb.cumulative_regrets)
    show()
