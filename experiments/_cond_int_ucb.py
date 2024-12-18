from typing import Optional

import numpy as np
from scipy.stats import bernoulli, rv_discrete
from tqdm import tqdm

from _cond_int_cbn_mab import CondIntCBN_MAB
from _ucb import UCB
from _utils import RandomVariable


class CondIntUCB:
    """Run CondIntUCB on a conditional intervention CBN-MAB problem and store results.

    The CondIntUCB algorithm treats each candidate node as a contextual bandit problem,
    thus running a contextual ucb algorithm on each, resulting in one optimal policy
    per node, from which the best is selected. The output is thus a (node, policy) pair.

    There will be a contextual UCB instance for each context c (we assume discrete
    contexts). At each iteration, a node is selected (using UCB at the node level),
    after which one iteration of the contextual UCB algorithm is performed.
    """

    # TODO
    def __init__(
        self,
        mab: CondIntCBN_MAB,
    ):
        self.cont_ucbs = [
            ContextualUCB(reward_cpd[c, :]) for c in range(self.n_contexts)
        ]
        self._initialize_run()

    def _initialize_run(self):
        for i in range(len(self.ucbs)):
            self.ucbs[i]._initialize_run()
        self.total_pulls = 0  # Total number of pulls during run
        self.contexts = []  # Contexts observed during run
        # A run is characterized by tuples (c_t, p(c_t), a_t, p(a_t), x_t).
        self.sampled_contexts_probs = []  # Probability of each observed context
        self.selected_arms = []  # Pulled arms during a run
        self.pulled_arm_probs = []  # Probabilities of the pulled arms
        self.observed_rewards = []  # Rewards observed during run
        self.cumulative_regrets = []  # (Instantaneous) cumulative regrets
        self.best_policy = None

    def run(self, n_rounds, fresh_start=True):
        if fresh_start:
            self._initialize_run()

        for _ in tqdm(range(n_rounds)):
            # Sample context for this round
            context, context_prob = self.context_distribution.sample()
            context = context.item()
            context_prob = context_prob.item()
            # One UCB step, for the UCB for this context
            ucb = self.ucbs[context]
            ucb.step()
            # TODO extract reward to record that! also select arms
            # Update tracking class's attributes
            self.total_pulls += 1
            self.contexts += [context]
            self.sampled_contexts_probs += [context_prob]
            self.selected_arms += [ucb.selected_arms[-1]]
            self.pulled_arm_probs += [ucb.pulled_arm_probs[-1]]
            self.observed_rewards += [ucb.observed_rewards[-1]]

        # Record run details into UCB instances' attributes
        for ucb in self.ucbs:
            ucb.record_details()

        # Compute cumulative regret(s)
        for i in range(n_rounds):
            c = self.contexts[i]
            ucb = self.ucbs[c]
            optimal_arm = ucb.best_arm  # Each context has a best arm
            pulled_arm = self.selected_arms[i]
            instant_regret = (
                ucb.arm_expected_rewards[optimal_arm]
                - ucb.arm_expected_rewards[pulled_arm]
            )
            if i == 0:
                cum_regret = instant_regret
            else:
                cum_regret = self.cumulative_regrets[-1] + instant_regret
            self.cumulative_regrets += [cum_regret]

        # The best policy corresponds to pulling the best arm for each context
        # Thus: we need the best arm from each UCB instance
        self.best_policy = np.zeros(self.n_contexts)
        for c in range(self.n_contexts):
            self.best_policy[c] = self.ucbs[c].best_arm

        history = {
            "sampled_contexts": self.contexts,
            "selected_arms": self.selected_arms,
            "observed_rewards": self.observed_rewards,
            "context_probs": self.sampled_contexts_probs,
            "arm_probs": self.pulled_arm_probs,
            "cum_regrets": self.cumulative_regrets,
            "best_policy": self.best_policy,
        }

        return history


# Example usage
if __name__ == "__main__":
    context_distribution = RandomVariable(
        rv_discrete, values=([0, 1, 2], [0.2, 0.3, 0.5])  # 3 contexts
    )
    n_contexts = 3
    n_actions = 2
    reward_cpd = np.empty((n_contexts, n_actions), dtype=RandomVariable)
    p_success = np.array([[0.5, 0.25], [0.9, 0.75], [0.2, 0.5]])
    for c, a in np.ndindex(reward_cpd.shape):
        reward_cpd[c, a] = RandomVariable(bernoulli, p_success[c, a])
    # Note that optimal policy is then 0->0, 1->0, 2->1.
    n_rounds = 10000

    cont_ucb = ContextualUCB(context_distribution, reward_cpd)
    history = cont_ucb.run(n_rounds)

    print("Total Reward:", sum(history["observed_rewards"]))
    print("Best Policy:", history["best_policy"])

    from matplotlib.pyplot import plot, show

    plot(cont_ucb.cumulative_regrets)
    show()

    # # fmt:off
    # import ipdb; ipdb.set_trace() # noqa
    # # fmt:on
