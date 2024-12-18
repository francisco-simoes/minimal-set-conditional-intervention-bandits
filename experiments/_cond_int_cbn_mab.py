from typing import Any

from numpy.typing import NDArray
# from _cbn import CausalBayesianNetwork
from pgmpy.models import BayesianNetwork


class CondIntCBN_MAB:
    """Encodes CBN-MAB problem for conditional interventions."""

    # TODO: This class needs to contain information about:
    #   1. the nodes that we can select from.
    #   2. context for each node (ancestors or something maybe more sophisticated later on).
    #   3. A reward CPD for each node, to be fed to a ContextualUCB per node.
    #   OR:
    #   3. A method allowing sampling given a certain intervention on the node + context
    #
    #   NOTE: This seems to mean that the __init__ method should have the BN (from pgmpy)
    #   as an argument to extract the nodes and the ancestors of each node; OR: this class
    #   could subclass the BN class. The reward CPD should also be easy to extract from
    #   the CBN.
    def __init__(self, bn: BayesianNetwork, target: str):
        self.bn = bn
        assert target in self.nodes, "The `target` must be one of the nodes in `cbn`."
        self.target = target
        self.candidate_nodes = self._find_candidates(bn)  # Reduce search space.
        # Each candidate node will have a context.
        self.node_contexts: dict[str, list[str]] = self._find_contexts(bn)
        # Each candidate node will have a reward CPD.
        # self.reward_cpds: dict[str, NDArray] = #TODO
        # NOTE: Need to:
        # 1. use the cbn method for hard interventions to intervene for each value of each node
        # 2. use get_state_probability(Y=1) method on intervened graph for (node, value) pair,
        #       using evidence .. NOTE: NO!! would have to do this for every context (An(X) config).
        #       This is too much! Instead, may be better to not use cbn after all (just bn), and
        #       alter contextual_ucb class so that it can sample using the simulate() method of
        #       the BN. (TODO!!!)

    def _find_candidates(self, cbn):
        # NOTE: For now, take all ancestors of target.
        # Replace this with search space reduction algo later.
        return list(cbn._get_ancestors_of([self.target]))

    def _find_contexts(self, cbn):
        for node in self.candidate_nodes:
            # NOTE: for now, all ancestors of node is the context.
            # May replace later with backdoor set or ancestors that are also in An(Y).
            self.node_contexts[node, list[cbn._get_ancestors_of([node])]]

    def sample_reward(
        self,
        do: dict[str, Any],
        context: dict[str, Any],
        n_samples=1,
        show_progress=False,
        seed=None,
    ) -> list:
        """Sample the target variable, after intervening and given a certain context."""
        df = self.bn.simulate(
            n_samples=n_samples,
            do=do,
            evidence=context,
            seed=seed,
            show_progress=show_progress,
        )
        reward_samples = list(df[self.target])
        return reward_samples
