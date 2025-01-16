from copy import deepcopy
from itertools import product as cartesian_product
from typing import Any, Callable, Optional

import numpy as np
from scipy.stats import bernoulli, rv_discrete
from tqdm import tqdm

from _cond_int_cbn_mab import CondIntCBN_MAB
from _fixed_context_node_ucb import FixedContextNodeUCB
from _samplers import ContextSamplerBase, RewardSamplerBase
from _utils import rowdf_to_dict


class FixedNodeContextualUCB:
    """Run (uncoupled) contextual UCB on a node of a cond. int. CBN-MAB problem.

    The uncoupled contextual UCB algorithm simply consists of treating the (finite,
    stochastic) contextual bandit problem as a set of bandit problems, one for each
    context, and running UCB on each of those.

    The contextual bandit problem is characterized by a context (vector) distribution
    and a reward conditional distribution p(r|c,a) giving the probability of reward r
    given a context c and an action a.
    Additionally, the 'node' part of this problem refers to the fact that the context
    is dependent on the choice of a 'node', so that the `mab` should also be a
    ContextSamplerBase instance, supplied should have a method `sample_context`
    with `node` as an argument.

    There will be a FixedContextNodeUCB instance `ucb` for each context c (we assume
    discrete contexts). At each iteration, a context c is sampled, after which
    ucbs[c] performs an iteration.
    """

    def __init__(
        self,
        node: str,
        mab: CondIntCBN_MAB,
        reward_to_float_converter: Optional[Callable] = None,
        # optimal_expected_rewards: Optional[list[float]] = None,
    ):
        self.node = node
        self.bn_states: dict = mab.bn.states
        self.reward_to_float_converter = reward_to_float_converter
        # In our case, both the context and reward samplers are defined by the mab
        self.context_sampler: Callable = mab.sample_context
        self.reward_sampler: Callable = mab.sample_reward
        # Let's get all contexts (a.k.a. context states) for this node
        self.context_vars: list[Any] = mab.node_contexts[self.node]
        # self.context_states = list(
        #     cartesian_product(*[mab.bn.states[var] for var in self.context_vars])
        # )

        # Construct list of all contexts (dictionaries)
        self.context_states: list[dict] = self._construct_context_states()

        self.n_contexts = len(self.context_states)
        self.ucbs: list[FixedContextNodeUCB] = []
        for context in self.context_states:
            node_states = mab.bn.states[self.node]
            self.ucbs += [
                FixedContextNodeUCB(
                    self.node,
                    node_states,
                    context,
                    mab,
                    reward_to_float_converter=self.reward_to_float_converter,
                )
            ]
        #         if optimal_expected_rewards is None:
        #             print(
        #                 """\nOptimal expected rewards not given. I will compute cumulative regret
        # values retroactively, using the empirical estimation of
        # the optimal rewards.\n"""
        #             )
        # self.optimal_expected_rewards = optimal_expected_rewards
        self._initialize_run()

    def _initialize_run(self):
        for i in range(self.n_contexts):
            self.ucbs[i]._initialize_run()
        self.total_pulls = 0  # Total number of pulls during run
        self.contexts_idxs: list[int] = []  # Contexts observed during run
        # A run is characterized by tuples (c_t, a_t, x_t).
        self.selected_arms = []  # Pulled arms during a run
        self.observed_rewards = []  # Rewards observed during run
        # self.optimal_expected_reward = None
        self.cumulative_regrets = []  # (Instantaneous) cumulative regrets
        self.best_policy: dict[Any, Any] = None

    def _step(self):
        # Sample context for this round
        context_df = self.context_sampler(self.node)
        context: dict[str, Any] = rowdf_to_dict(context_df)
        # context: list[Any] = tuple((context_dict[var] for var in self.context_vars))
        context_index: int = self.context_states.index(context)
        # One UCB step, for the UCB for this context
        ucb = self.ucbs[context_index]
        ucb.step()
        # Update tracking class's attributes
        self.total_pulls += 1
        self.contexts_idxs += [context_index]
        self.selected_arms += [ucb.selected_arms[-1]]
        self.observed_rewards += [ucb.observed_rewards[-1]]

    def record_details(self):
        # Record run details into UCB instances' attributes
        for ucb in self.ucbs:
            ucb.record_details()

        # Compute cumulative regret(s)
        for i in range(self.total_pulls):
            context_index = self.contexts_idxs[i]
            ucb = self.ucbs[context_index]
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
        for i in range(len(self.context_states)):
            self.best_policy[i] = self.ucbs[i].best_arm

        history = {
            "sampled_contexts": self.contexts_idxs,
            "contexts list": self.context_states,
            "selected_arms": self.selected_arms,
            "observed_rewards": self.observed_rewards,
            "cum_regrets": self.cumulative_regrets,
            "best_policy": self.best_policy,
        }

        return history

    def run(self, n_rounds, fresh_start=True):
        if fresh_start:
            self._initialize_run()

        for _ in tqdm(range(n_rounds)):
            self._step()

        history = self.record_details()

        return history

    def _construct_context_states(self):
        state_lists: list[list] = [
            self.bn_states[var] for var in self.context_vars
        ]  # Each element is the list of all possible states for var
        context_states_lsts: list[tuple] = list(
            cartesian_product(*state_lists)
        )  # Each list is a context of form [var1_val1, var2_value1, ...]
        context_states: list[dict] = [
            dict(zip(self.context_vars, states_list))
            for states_list in context_states_lsts
        ]  # Each dict is a context of form {'var1': var1_val1, 'var2': var2_val1, ...}

        return context_states

    @staticmethod
    def __fixed_context_decorator(
        sample_reward_func: Callable, fixed_context
    ) -> Callable:
        """A decorator that fixes the context for the reward sampler function."""

        def wrapper(self, do: dict[str, Any], context: dict[str, Any], *args, **kwargs):
            # Here we override the context with the fixed one
            return sample_reward_func(self, do, fixed_context, *args, **kwargs)

        return wrapper


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

    n_rounds = 1000
    contextual_ucb = FixedNodeContextualUCB(
        node, mab, reward_to_float_converter=yes_is_zero_converter
    )
    history = contextual_ucb.run(n_rounds)

    print("Total Reward:", sum(history["observed_rewards"]))
    print("Best Policy:", history["best_policy"])
    print("With names:")
    for context_idx, val in enumerate(history["best_policy"]):
        print(
            f"Context {history['contexts list'][context_idx]} ---> Set '{node}' to {val}."
        )

    from matplotlib.pyplot import plot, show

    plot(contextual_ucb.cumulative_regrets)
    show()
