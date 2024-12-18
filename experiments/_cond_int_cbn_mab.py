from typing import Any

# from numpy.typing import NDArray
from pandas import DataFrame
from pgmpy.models import BayesianNetwork

from _samplers import ContextSamplerBase, RewardSamplerBase


class CondIntCBN_MAB(RewardSamplerBase, ContextSamplerBase):
    """Encodes CBN-MAB problem for conditional interventions."""

    # NOTE: This class needs to contain information about:
    #   1. the nodes that we can select from
    #   2. context for each node
    #   3. A method allowing sampling given a certain intervention on the node + context
    #   4. A method allowing sampling of contexts
    def __init__(self, bn: BayesianNetwork, target: str):
        self.bn = bn
        assert target in self.bn.nodes, "The `target` must be one of the nodes in `bn`."
        self.target = target
        self.candidate_nodes = self._find_candidates()  # Reduce search space.
        # Each candidate node will have a context.
        self.node_contexts: dict[str, list[str]] = self._find_contexts()

    def _find_candidates(self):
        # TODO: For now, take all non-trivial ancestors of target.
        # Replace this with search space reduction algo later.
        target_ancestors = list(self.bn._get_ancestors_of([self.target]))
        target_ancestors.remove(self.target)
        return target_ancestors

    def _find_contexts(self):
        node_contexts = {}
        for node in self.candidate_nodes:
            # NOTE: for now, all non-trivial ancestors of node is the context.
            # May replace later with backdoor set or ancestors that are also in An(Y).
            node_ancestors = list(self.bn._get_ancestors_of([node]))
            node_ancestors.remove(node)
            node_contexts[node] = node_ancestors
        return node_contexts

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

    def sample_context(
        self,
        node,
        n_samples: int = 1,
        show_progress: bool = False,
        seed: Any = None,
    ) -> DataFrame:
        """Sample the node's context."""
        context_vars = self.node_contexts[node]
        df = self.bn.simulate(
            n_samples=n_samples,
            seed=seed,
            show_progress=show_progress,
        )
        # context_samples: NDArray = df[context_vars].values
        context_samples: DataFrame = df[context_vars]
        return context_samples


# Example
if __name__ == "__main__":
    from pgmpy.utils import get_example_model

    bn = get_example_model("asia")
    target = "dysp"
    mab = CondIntCBN_MAB(bn, target)

    print("\n=== Testing sample_reward() method on Asia dataset: ===")
    n_samples = 100
    print(
        "Fraction of samples with dysp=yes given that one does do(lung=yes)"
        + " and observes bronc=yes:",
        mab.sample_reward({"lung": "yes"}, {"bronc": "yes"}, n_samples=n_samples).count(
            "yes"
        )
        / n_samples,
    )
    print(
        "Fraction of samples with dysp=yes given that one does do(lung=no)"
        + " and observes bronc=no:",
        mab.sample_reward({"lung": "no"}, {"bronc": "no"}, n_samples=n_samples).count(
            "yes"
        )
        / n_samples,
    )

    print("\n=== Testing sample_context() method on Asia dataset: ===")
    n_samples = 10
    print(
        "Some samples of the context of the variable either:\n",
        mab.sample_context("either", n_samples=n_samples),
    )
