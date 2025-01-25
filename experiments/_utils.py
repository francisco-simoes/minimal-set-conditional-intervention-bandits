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


def get_node_with_most_ancestors(
    dag: nx.DiGraph, no_single_children=False
) -> tuple[Any, int]:
    """Find node with most ancestors.

    If no_single_children is True, then find the non-single-child node with most ancestors.
    """
    # Update the maximum if necessary
    max_ancestors_leaf = None
    max_ancestors_count = 0

    if no_single_children:
        # Need to consider all nodes
        candidate_nodes = dag.nodes
    else:
        # Most-ancestor node will be within leaves
        candidate_nodes = get_dag_leaves(dag)
    for node in candidate_nodes:

        # Test whether it is single child, if no_single_children=True
        if no_single_children:
            n_parents = dag.in_degree(node)
            if n_parents == 1:
                continue  # skip rest

        # Get ancestors of the leaf node
        ancestors: set = nx.ancestors(dag, node)
        # Count the number of ancestors
        count = len(ancestors)

        # Update the maximum if necessary
        if count > max_ancestors_count:
            max_ancestors_count = count
            max_ancestors_leaf = node

    return max_ancestors_leaf, max_ancestors_count


def most_common_list_element(lst):
    return max(lst, key=lst.count)


def round_2_or_until_nonzero(number):
    # Try rounding to 2 decimal places
    rounded = round(number, 2)

    # If rounded value is zero but the number is non-zero, find minimal non-zero precision
    if rounded == 0 and number != 0:
        precision = 3
        while round(number, precision) == 0:
            precision += 1
        return round(number, precision)
    return rounded
