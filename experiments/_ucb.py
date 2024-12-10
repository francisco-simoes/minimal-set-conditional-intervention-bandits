# TODO: the "arm pulling" must, in our case, correspond to sampling
# from a post-intervention dist. So, I should create another file with an SCM class
# with a do() method creating distributions with a sample() method. These distributions
# can be fed to this UCB class as elements of the "reward_distributions" list.

from itertools import accumulate

import numpy as np
from scipy.stats import bernoulli

# from _utils import BernoulliRV
from _utils import RandomVariable


class UCB:
    def __init__(self, reward_distributions, optimal_expected_reward=None):
        self.reward_distributions = reward_distributions
        self.n_arms = len(self.reward_distributions)
        if optimal_expected_reward is not None:
            self.optimal_expected_reward = optimal_expected_reward
        else:
            print(
                """Optimal expected reward not given. I will compute cumulative regret
                     values retroactively, using the empirical estimation of
                     the optimal reward."""
            )
            self.optimal_expected_reward = None
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

    def run(self, n_rounds):
        """
        bandit_probs: List of probabilities for each bandit's reward (Bernoulli rewards).
        n_rounds: Number of rounds to play.
        """
        self._initialize_run()

        for _ in range(n_rounds):
            chosen_arm = self._select_arm()
            # Pull arm
            reward, arm_prob = self.reward_distributions[chosen_arm].sample()
            reward = reward.item()
            arm_prob = arm_prob.item()
            self._update_arm(chosen_arm, reward)
            self.selected_arms += [chosen_arm]
            self.observed_rewards += [reward]
            self.pulled_arm_probs += [arm_prob]

        # Computation of cumulative regret
        best_arm = self.selected_arms[-1]

        # Estimate expected rewards for each arm, and then for each chosen arm
        self.arm_expected_rewards = self.arm_rewards / self.arm_counts
        self.expected_rewards = [
            self.arm_expected_rewards[i].item() for i in self.selected_arms
        ]

        if self.optimal_expected_reward is None:
            # Empirical optimal reward estimate, assuming best arm is actually best
            self.optimal_expected_reward = self.arm_expected_rewards[best_arm]

        instant_regrets = self.optimal_expected_reward - self.expected_rewards
        self.cumulative_regrets = list(accumulate(instant_regrets))
        # instant_regret = self.optimal_expected_reward - reward
        # self.cumulative_regrets += self.cumulative_regrets[-1] + instant_regret

        history = {
            "selected_arms": self.selected_arms,
            "observed_rewards": self.observed_rewards,
            "arm_probs": self.pulled_arm_probs,  # Probabilities of pulled arms
            "cum_regrets": self.cumulative_regrets,
            "best_arm": best_arm,  # The last chosen arm
            "arm_expected_rewards": self.arm_expected_rewards,
        }
        return history


# Example usage
if __name__ == "__main__":
    reward_distributions = [
        RandomVariable(bernoulli, 0.7),
        RandomVariable(bernoulli, 0.5),
        RandomVariable(bernoulli, 0.1),
    ]
    n_rounds = 1000

    ucb = UCB(reward_distributions)
    history = ucb.run(n_rounds)

    print("Total Reward:", sum(history["observed_rewards"]))
    print("Number of times each arm was pulled:", ucb.arm_counts)
    print("Estimated reward for each arm:", ucb.arm_rewards / ucb.arm_counts)

    # fmt:off
    import ipdb; ipdb.set_trace() # noqa
    # fmt:on
