from itertools import product

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from networkx import DiGraph
from pgmpy.models import BayesianNetwork
from tqdm import tqdm

from _split_algo import SPLIT_on_target
from _utils import generate_dag, get_dag_leaves, get_node_with_most_ancestors

N_GRAPHS = 100
N_NODES_LST = [20, 100, 300, 500]
DEGREES = [2, 4, 9]


def generate_hist_of_mGISS_fractions(
    n_nodes, expected_degree, n_graphs=N_GRAPHS, num_bins=20
):
    p_edge = expected_degree / n_nodes  # Probability that an edge is added
    fractions: list[float] = []  # Each fraction is len(mGISS_Y(dag))/len(An(Y)\Y)
    for _ in tqdm(range(n_graphs)):
        dag: DiGraph = generate_dag(n_nodes, p_edge)

        # # Add node with an edge from each leaf
        # leaves = get_dag_leaves(dag)
        # # Y = N_NODES + 1
        # Y = "Y"
        # dag.add_node(Y)
        # for leaf in leaves:
        #     dag.add_edge(leaf, Y)
        # Y_ancestors_count = len(nx.ancestors(dag, Y))

        # Find the leaf node with the most ancestors
        Y, Y_ancestors_count = get_node_with_most_ancestors(dag)

        # Find mGISS for dag w.r.t. Y
        bn = BayesianNetwork(dag.edges)
        mGISS: set = SPLIT_on_target(bn, Y)
        fraction = len(mGISS) / (
            Y_ancestors_count
        )  # Note that nx.ancestors contains PROPER ancestors
        fractions += [fraction]

        # # Uncomment for prints and graph plots
        # print(fraction)
        # print(Y)
        # print(nx.ancestors(dag, Y))
        # print(mGISS)
        # pos = nx.shell_layout(dag)
        # plt.figure(figsize=(10, 6))
        # nx.draw(dag, pos, with_labels=True)
        # plt.show()

    # Compute histogram bins and heights manually for relative frequencies
    bins = np.linspace(0, 1, num_bins + 1)
    counts, _ = np.histogram(fractions, bins=bins)
    relative_counts = counts / counts.sum()

    # Return histogram data
    return {
        "bins": bins,
        "relative_counts": relative_counts,
        "title": f"{n_nodes} nodes, Expected degree: {expected_degree}. Average = {np.mean(fractions):.2f}",  # noqa
    }

    # # Compute histogram bins and heights manually for relative frequencies
    # # Define bins from 0 to 1
    # bins = np.linspace(0, 1, num_bins + 1)
    # counts, bins = np.histogram(fractions, bins=bins)
    # # Normalize counts to relative frequency
    # relative_counts = counts / counts.sum()
    # # # fmt:off
    # # import ipdb; ipdb.set_trace() # noqa
    # # # fmt:on
    # # Plot the histogram with normalized heights
    # plt.bar(
    #     bins[:-1],
    #     relative_counts,
    #     width=np.diff(bins),
    #     edgecolor="k",
    #     alpha=0.7,
    #     align="edge",
    # )
    # # plt.hist(fractions, bins=30, edgecolor="k", alpha=0.7, density=True)
    # plt.title(f"{n_nodes} nodes. Expected degree: {expected_degree}")
    # plt.xlabel("Fraction mGISS/An(Y)\\Y")
    # plt.ylabel("Frequency")
    # plt.grid(True)
    # plt.show()


if __name__ == "__main__":
    print(N_NODES_LST)
    for n_nodes in N_NODES_LST:
        for degree in DEGREES:
            generate_hist_of_mGISS_fractions(n_nodes, degree)

    histograms = [
        generate_hist_of_mGISS_fractions(n_nodes, degree, N_GRAPHS)
        for n_nodes, degree in product(N_NODES_LST, DEGREES)
    ]

    # Create a grid for plots
    row_number = len(N_NODES_LST)
    col_number = int(len(histograms) / len(N_NODES_LST))
    fig, axes = plt.subplots(
        row_number,
        col_number,
        figsize=(5 * row_number, 3 * col_number),
    )
    if len(histograms) == 1:
        axes = [axes]  # Ensure axes is iterable for a single plot

    for ax, hist_data in zip(axes.flatten(), histograms):
        bins = hist_data["bins"]
        relative_counts = hist_data["relative_counts"]
        ax.bar(
            bins[:-1],
            relative_counts,
            width=np.diff(bins),
            edgecolor="k",
            alpha=0.7,
            align="edge",
        )
        ax.set_title(hist_data["title"])
        ax.set_xlabel("Fraction mGISS/An(Y)\\Y")
        ax.set_ylabel("Frequency")
        ax.grid(True)

    plt.tight_layout()
    plt.show()
