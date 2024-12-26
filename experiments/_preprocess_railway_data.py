from math import gamma as gamma_function

import numpy as np
import pandas as pd
from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import BayesianNetwork


def cbn_from_railway_data(
    path_csv: str,
    buffer: float = 3,
    k: float = 1,
    theta: float = 1,
    max_delay: int = 60,
) -> BayesianNetwork:
    df = pd.read_csv(path_csv, header=0, sep=";")
    # df=pd.read_csv('data/scm.csv',header=0,sep=';')

    print("csv read")
    # Split kortestop
    df.replace("K_A", "A", inplace=True)
    df.replace("K_V", "V", inplace=True)
    # Combine multiple columns into one for Node ID
    df["VanNode"] = df[df.columns[0:1].to_list() + df.columns[2:5].to_list()].apply(
        lambda x: "_".join(x.dropna().astype(str)), axis=1
    )
    df["NaarNode"] = df[df.columns[5:6].to_list() + df.columns[7:10].to_list()].apply(
        lambda x: "_".join(x.dropna().astype(str)), axis=1
    )
    # df = df.astype("str")

    nodes: list[str] = (
        pd.concat((df["VanNode"], df["NaarNode"])).drop_duplicates().to_list()
    )
    cbn = BayesianNetwork()
    cbn.add_nodes_from(nodes)

    for _, row in df.iterrows():
        cbn.add_edge(row["VanNode"], row["NaarNode"])
        # (, , row["GeplandeProcesstijd_sec"])

    # Build CPDs
    # Underlying SCMs: delay from parents + Gamma noise, and a buffer system
    node: str
    for node in cbn.nodes:
        parents: list[str] = cbn.get_parents(node)
        if len(parents) == 0:
            node_cpd_array = _normalized_discrete_gamma(k, theta, x_card=max_delay)
            node_cpd_table = node_cpd_array.reshape(-1, 1)
            evidence = []
            evidence_card = []
        if len(parents) == 1:
            node_cpd_array = np.zeros((max_delay,) * 2)  # (node, parent)
            for p in range(node_cpd_array.shape[1]):
                node_cpd_array[:, p] = _normalized_discrete_gamma(
                    k, theta, x_card=node_cpd_array.shape[0], shift=p
                )
            node_cpd_table = node_cpd_array.reshape(max_delay, -1)
            evidence = parents
            evidence_card = [max_delay]
        elif len(parents) == 2:
            node_time: float = _time_to_float(node[-5:])
            same_train_parent_idx: int = _get_same_train_parent_idx(node, parents)
            other_train_parent: str = parents[0 if same_train_parent_idx == 1 else 1]
            other_train_time: float = _time_to_float(other_train_parent[-5:])
            node_cpd_array = np.zeros((max_delay,) * 3)  # (node, parent, parent)
            for p0, p1 in np.ndindex(*node_cpd_array.shape[1:]):
                if same_train_parent_idx == 1:  # p1 is value of same_train_parent
                    same_train_delay = p1
                    other_train_delay = p0
                else:  # p0 is value of same_train_parent
                    same_train_delay = p0
                    other_train_delay = p1
                # CPD cases:
                if node_time - (other_train_time + other_train_delay) < buffer:
                    # Need to ensure buffer
                    node_cpd_array[:, p0, p1] = _normalized_discrete_gamma(
                        k,
                        theta,
                        x_card=node_cpd_array.shape[0],
                        shift=max(buffer, same_train_delay),
                    )
                else:
                    node_cpd_array[:, p0, p1] = _normalized_discrete_gamma(
                        k, theta, x_card=node_cpd_array.shape[0], shift=same_train_delay
                    )
            node_cpd_table = node_cpd_array.reshape(max_delay, -1)
            evidence = parents
            evidence_card = [max_delay] * 2
        cpd = TabularCPD(
            variable=node,
            variable_card=max_delay,
            values=node_cpd_table,
            evidence=evidence,
            evidence_card=evidence_card,
        )
        cbn.add_cpds(cpd)

    return cbn


def _time_to_float(time_str):
    hours, minutes = map(int, time_str.split(":"))
    return hours + minutes / 60


def _get_same_train_parent_idx(node, parents):
    for i, parent in enumerate(parents):
        if parent[:5] == node[:5]:  # Same train code
            return i


def _gamma_pdf(x, k, theta):
    """Compute the Gamma PDF for given x, shape (k), and scale (theta)."""
    if x < 0:
        return 0
    return (x ** (k - 1) * np.exp(-x / theta)) / (gamma_function(k) * theta**k)


def _normalized_discrete_gamma(k, theta, x_card, shift=0):
    """Discretized version of the Gamma distribution."""
    # Compute unnormalized probabilities
    probabilities = np.array([_gamma_pdf(x - shift, k, theta) for x in range(x_card)])
    # Normalize to make the sum equal to 1
    normalized_probs = probabilities / probabilities.sum()
    return normalized_probs


if __name__ == "__main__":
    cbn = cbn_from_railway_data("./data/railway_data.csv")

    from PIL import Image

    graph = cbn.to_graphviz()
    graph.draw("railway_graph.png", prog="dot")
    Image.open("railway_graph.png").show()

# TEST for way I'm mapping array to table
# def map_index_to_number(n):
#     """
#     Maps all pairs of indices (i, j) where 0 <= i, j < n to unique numbers
#     in the order they appear in a lexicographical manner.

#     :param n: Maximum index value (the range is 0 to n-1)
#     :return: Dictionary mapping (i, j) to corresponding number
#     """
#     # Create all index pairs (i, j) where 0 <= i, j < n
#     pairs = [(i, j) for i in range(n) for j in range(n)]

#     # Create a dictionary mapping each pair to its corresponding number
#     pair_map = {pair: idx for idx, pair in enumerate(pairs)}

#     return pair_map
#
# test = node_cpd_table[:5, :5, :5]

# for i, j in np.ndindex(*test.shape[1:]):
#     print('=====')
#     print(i,j)
#     print(test[:, i, j], '\n')
#     print(test.reshape(5,-1)[:,to_idxs[(i,j)]])
# to_idxs = map_index_to_number(5)
