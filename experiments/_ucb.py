# TODO: the "arm pulling" must, in our case, correspond to sampling
# from a post-intervention dist. So, I should create another file with an SCM class
# with a do() method creating distributions with a sample() method. These distributions
# can be fed to this UCB class as elements of the "reward_distributions" list.
# TODO: the "arm pulling" must, in our case, be conditional on a context.
# We will have one UCB per context. Will create a ContextualUCB class corresopnding
# to one UCB per context.

from itertools import accumulate
from typing import Optional

import numpy as np
from scipy.stats import bernoulli

# from _utils import BernoulliRV
from _utils import RandomVariable


class UCB:
    """Run UCB on a bandit problem and store results.

    The (finite, stochastic) bandit problem is characterized
    by a set of reward distributions, one for each action/arm.
    """

    def __init__(
        self,
        reward_distributions: list[RandomVariable],
        optimal_expected_reward: Optional[float] = None,
    ):
        self.reward_distributions = reward_distributions
        self.n_arms = len(self.reward_distributions)
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
        self.selected_arms = []  # Pulled arms during a run
        self.observed_rewards = []  # Rewards observed during run
        self.pulled_arm_probs = []  # Probabilities of pulled arms
        self.arm_expected_rewards = np.zeros(
            self.n_arms
        )  # Expected reward for each arm
        self.cumulative_regrets = []  # (Instantaneous) cumulative regrets
        self.best_arm = None

    def _select_arm(self):
        # If any arm hasn't been pulled yet, select it
        for i in range(self.n_arms):
            if self.arm_counts[i] == 0:
                return i

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
        # Pull arm
        reward, arm_prob = self.reward_distributions[chosen_arm].sample()
        reward = reward.item()
        arm_prob = arm_prob.item()
        self._update_arm(chosen_arm, reward)
        self.selected_arms += [chosen_arm]
        self.observed_rewards += [reward]
        self.pulled_arm_probs += [arm_prob]

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
        # instant_regret = self.optimal_expected_reward - reward
        # self.cumulative_regrets += self.cumulative_regrets[-1] + instant_regret

        history = {
            "selected_arms": self.selected_arms,
            "observed_rewards": self.observed_rewards,
            "arm_probs": self.pulled_arm_probs,  # Probabilities of pulled arms
            "cum_regrets": self.cumulative_regrets,
            # "best_arm": best_arm,  # The last chosen arm
            "arm_expected_rewards": self.arm_expected_rewards,
        }

        return history

    def run(self, n_rounds, fresh_start=True):
        """Run UCB algorithm.

        bandit_probs: List of probabilities for each bandit's reward (Bernoulli rewards).
        n_rounds: Number of rounds to play.
        fresh_start: Reset class attributes before running.
        """
        if fresh_start:
            self._initialize_run()

        for _ in range(n_rounds):
            self.step()

        history = self.record_details()

        return history


# Example usage
if __name__ == "__main__":
    reward_distributions = [
        RandomVariable(bernoulli, 0.7),
        RandomVariable(bernoulli, 0.5),
        RandomVariable(bernoulli, 0.1),
    ]
    n_rounds = 10000

    ucb = UCB(reward_distributions)
    history = ucb.run(n_rounds)

    print("Total Reward:", sum(history["observed_rewards"]))
    print("Number of times each arm was pulled:", ucb.arm_counts)
    print("Estimated expected reward for each arm:", ucb.arm_expected_rewards)

    from matplotlib.pyplot import plot, show

    plot(ucb.cumulative_regrets)
    show()
