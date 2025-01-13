import random
from typing import Any

import networkx as nx


def generate_dag(num_nodes, p):
    """
    Generates a DAG with num_nodes nodes.
    Adds a directed edge v -> u with fixed probability p whenever v < u.

    Args:
        num_nodes (int): Number of nodes in the graph.
        p (float): Probability of adding an edge between two nodes.

    Returns:
        nx.DiGraph: A directed acyclic graph.
    """
    # Create an empty directed graph
    dag = nx.DiGraph()

    dag.out_degree
    # Add nodes and edges
    edges = [
        (v, u)
        for v in range(num_nodes)
        for u in range(v + 1, num_nodes)
        if random.random() < p
    ]
    dag.add_edges_from(edges)

    return dag


def get_dag_leaves(dag: nx.DiGraph):
    return [node for node in dag.nodes if dag.out_degree(node) == 0]


def rowdf_to_dict(df):
    # Ensure the DataFrame has only one row
    if df.shape[0] != 1:
        raise ValueError("The DataFrame should have exactly one row.")

    # Convert the first row to a dictionary
    return df.iloc[0].to_dict()


def get_node_with_most_ancestors(dag: nx.DiGraph) -> tuple[Any, int]:
    max_ancestors_leaf = None
    max_ancestors_count = 0

    leaf_nodes = get_dag_leaves(dag)
    for leaf in leaf_nodes:
        # Get ancestors of the leaf node
        ancestors: set = nx.ancestors(dag, leaf)
        # Count the number of ancestors
        count = len(ancestors)
        # Update the maximum if necessary
        if count > max_ancestors_count:
            max_ancestors_count = count
            max_ancestors_leaf = leaf
    return max_ancestors_leaf, max_ancestors_count
