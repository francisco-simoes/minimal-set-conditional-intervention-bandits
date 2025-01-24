from itertools import product

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from networkx import DiGraph
from pgmpy.models import BayesianNetwork
from tqdm import tqdm

from _c4_algo import C4_on_target
from _utils import generate_dag, get_dag_leaves, get_node_with_most_ancestors

N_GRAPHS = 1000
N_NODES_LST = [20, 100, 300, 500]
# N_NODES_LST = [10, 10, 10, 10]
DEGREES = [2, 5, 8, 11]


def generate_hist_of_mGISS_fractions(
    n_nodes, expected_degree, n_graphs=N_GRAPHS, num_bins=20
):
    p_edge = expected_degree / n_nodes  # Probability that an edge is added
    fractions: list[float] = []  # Each fraction is len(mGISS_Y(dag))/len(An(Y)\Y)
    for _ in tqdm(range(n_graphs)):
        dag: DiGraph = generate_dag(n_nodes, p_edge)

        # Find the node with the most ancestors
        Y, Y_ancestors_count = get_node_with_most_ancestors(dag)

        # Find mGISS for dag w.r.t. Y
        bn = BayesianNetwork(dag.edges)
        mGISS: set = C4_on_target(bn, Y)
        fraction = len(mGISS) / (
            Y_ancestors_count
        )  # Note that nx.ancestors contains PROPER ancestors (no need to subract 1)
        fractions += [fraction]

        # # Uncomment for prints and visualizing graphs
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


if __name__ == "__main__":
    print(N_NODES_LST)
    for n_nodes in N_NODES_LST:
        for degree in DEGREES:
            generate_hist_of_mGISS_fractions(n_nodes, degree)

    histograms = [
        generate_hist_of_mGISS_fractions(n_nodes, degree, N_GRAPHS)
        for n_nodes, degree in product(N_NODES_LST, DEGREES)
    ]

    plt.rcParams.update({"text.usetex": True, "font.family": "Helvetica"})

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

    for i, (ax, hist_data) in enumerate(zip(axes.flatten(), histograms)):
        bins = hist_data["bins"]
        relative_counts = hist_data["relative_counts"]
        ax.bar(
            bins[:-1],
            relative_counts,
            color="darkblue",
            width=np.diff(bins),
            edgecolor="k",
            alpha=0.7,
            align="edge",
        )
        ax.set_title(hist_data["title"])
        if i % col_number == 0:  # first subplot of a row
            ax.set_ylabel("Frequency")
        if i in range(
            row_number * col_number - col_number, row_number * col_number
        ):  # last row
            ax.set_xlabel("Fraction $|$mGISS$|$ / $|$An$(Y)$-$\{Y\}|$ ")
        ax.grid(True)

    plt.tight_layout()
    plt.show()
