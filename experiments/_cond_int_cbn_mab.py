from typing import Any, Callable, Optional

# from numpy.typing import NDArray
from pandas import DataFrame
from pgmpy.models import BayesianNetwork

from _samplers import ContextSamplerBase, RewardSamplerBase
from _split_algo import SPLIT_on_target


class CondIntCBN_MAB(RewardSamplerBase, ContextSamplerBase):
    """Encodes CBN-MAB problem for conditional interventions."""

    # NOTE: This class needs to contain information about:
    #   1. the nodes that we can select from
    #   2. context for each node
    #   3. A method allowing sampling given a certain intervention on the node + context
    #   4. A method allowing sampling of contexts
    def __init__(
        self,
        bn: BayesianNetwork,
        target: str,
        search_space_reduction_func: Optional[Callable] = None,
        find_contexts_func: Optional[Callable] = None,
    ):
        """search_space_reduction_func must take a BayesianNetwork instance and the target
        as input and output a set of nodes (strings).
        search_space_reduction_func must take a BayesianNetwork instance and a list
        of candidate nodes as input, and output a dictionary with keys
        node: conditioning set.
        """
        self._find_contexts_func = find_contexts_func
        self._search_space_reduction_func = search_space_reduction_func

        self.bn = bn
        assert target in self.bn.nodes, "The `target` must be one of the nodes in `bn`."
        self.target = target
        self.candidate_nodes = self._find_candidates()  # Reduce search space.
        # Each candidate node will have a context.
        self.node_contexts: dict[str, list[str]] = self._find_contexts()

    def _find_candidates(self):
        if self._search_space_reduction_func is None:
            # Default: All non-trivial ancestors of target.
            target_ancestors = list(self.bn._get_ancestors_of([self.target]))
            target_ancestors.remove(self.target)
            return target_ancestors
        else:
            return list(self._search_space_reduction_func(self.bn, self.target))

    def _find_contexts(self):
        "Returns a dictionary with key-value pairs node: conditioning set."
        node_contexts = {}
        if self._find_contexts_func is None:
            # Default: condset_X = An(X)
            for node in self.candidate_nodes:
                # Z_X = An(Y)\De(X)
                node_ancestors = list(self.bn._get_ancestors_of([node]))
                node_ancestors.remove(node)
                node_contexts[node] = node_ancestors
            return node_contexts
        else:
            node_contexts: self._find_contexts_func(self.bn, self.candidate_nodes)
            assert node_contexts is dict
            return node_contexts

    def sample_reward(
        self,
        do: dict[str, Any],
        context: dict[str, Any],
        state_to_float_converter: Optional[Callable] = None,
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

        numeric_reward_samples = []
        if state_to_float_converter is not None:
            for sample in reward_samples:
                numeric_reward_samples += [state_to_float_converter(sample)]
        else:
            numeric_reward_samples = reward_samples
        return numeric_reward_samples

    # def sample_context(
    #     self,
    #     node,
    #     n_samples: int = 1,
    #     show_progress: bool = False,
    #     seed: Any = None,
    # ) -> DataFrame:
    #     # NOTE: this uses uniform sampling - just for sanity checks
    #     """Sample the node's context."""
    #     context_vars = self.node_contexts[node]
    #     all_var_states: dict = self.bn.states
    #     data = {
    #         var: np.random.choice(states, size=n_samples)
    #         for var, states in all_var_states.items()
    #     }
    #     samples = pd.DataFrame(data)
    #     context_samples: DataFrame = samples[context_vars]
    #     return context_samples

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
    mab = CondIntCBN_MAB(bn, target, search_space_reduction_func=SPLIT_on_target)

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

    print("\n=== Testing using SPLIT for search space reduction on Asia dataset: ===")
    mGISS = mab.candidate_nodes
    print(f"Candidate nodes: {set(mGISS)}")
    print("Expected: {'either', 'bronc', 'smoke'}")
