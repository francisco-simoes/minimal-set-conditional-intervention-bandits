from copy import deepcopy
from typing import Any

import numpy as np
from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import BayesianNetwork


class CausalBayesianNetwork(BayesianNetwork):
    """
    NOTE: hard interventions are not implemented in pgmpy's class.
    It has a do() method, but it does not set the distribution of
    the intervened variable(s) to a value.
    """

    def __init__(self, *args, bn: BayesianNetwork = None, **kwargs):
        if bn is None:  # No base BayesianNetwork instance provided
            super().__init__(*args, **kwargs)
        else:
            # Copy all attributes of bn
            self.__dict__ = deepcopy(bn.__dict__)

    def hard_intervention(self, do: dict[str, Any]):
        nodes = list(do.keys())
        mutilated_network = self.do(nodes)
        # Construct deterministic cpds for intervened nodes
        cpds = []
        for node, value in do.items():
            states = self.states[node]
            cpd_table = self.__one_hot_vector_from_list(states, value)[:, np.newaxis]
            cpd = TabularCPD(
                node,
                len(states),
                cpd_table,
                state_names={node: states},
            )
            cpds += [cpd]
        mutilated_network.add_cpds(*cpds)
        return mutilated_network

    @staticmethod
    def __one_hot_vector_from_list(lst: list, element):
        result = np.zeros(len(lst), dtype=int)

        try:
            # Find the index of the chosen element
            index = lst.index(element)
            # Set the value at the found index to 1
            result[index] = 1
        except ValueError:
            # Handle the case where the element is not in the list
            print(f"Warning: Element {element} not found in the list.")

        return result


# Example
if __name__ == "__main__":
    from pgmpy.utils import get_example_model

    model = get_example_model("asia")
    cbn = CausalBayesianNetwork(bn=model)

    print("=== Testing hard interventions method ===")
    intervened_model = cbn.hard_intervention(do={"lung": "yes", "bronc": "no"})
    print(
        "Probability that lung=yes and bronc=no before interventions:"
        + f"\n\t {cbn.get_state_probability({'lung': 'yes', 'bronc': 'no'})}"
    )
    print(
        "Probability that lung=yes and bronc=no after intervention lung=yes, bronc=no:"
        + f"\n\t {intervened_model.get_state_probability({'lung': 'yes', 'bronc': 'no'})}"  # noqa
    )
