from numpy.typing import NDArray
from pgmpy.models import BayesianNetwork


class CondIntCBN_MAB:
    """Encodes CBN-MAB problem for conditional interventions."""

    # TODO: This class needs to contain information about:
    #   1. the nodes that we can select from.
    #   2. context for each node (ancestors or something maybe more sophisticated later on).
    #   3. A reward CPD for each node, to be fed to a ContextualUCB per node.
    #
    #   NOTE: This seems to mean that the __init__ method should have the BN (from pgmpy)
    #   as an argument to extract the nodes and the ancestors of each node; OR: this class
    #   could subclass the BN class. The reward CPD should also be easy to extract from
    #   the CBN.
    def __init__(self, cbn: BayesianNetwork, target: str):
        assert target in self.nodes, "The `target` must be one of the nodes in `cbn`."
        self.target = target
        self.candidate_nodes = self.__find_candidates(cbn)  # Reduce search space.
        # Each candidate node will have a context.
        self.node_contexts: dict[str, list[str]] = self.__find_contexts(cbn)
        # Each candidate node will have a reward CPD.
        self.reward_cpds: dict[str, NDArray] = #TODO

    def __find_candidates(self, cbn):
        # NOTE: For now, take all ancestors of target.
        # Replace this with search space reduction algo later.
        return list(cbn._get_ancestors_of([self.target]))

    def __find_contexts(self, cbn):
        for node in self.candidate_nodes:
            # NOTE: for now, all ancestors of node is the context.
            # May replace later with backdoor set or ancestors that are also in An(Y).
            self.node_contexts[node, list[cbn._get_ancestors_of([node])]]
