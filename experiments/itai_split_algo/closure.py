import copy
from typing import Set

import networkx as nx


def SPLIT(G: nx.DiGraph, U: Set) -> Set:
    """
    Computes L^\infty(U).
    :param G: A DAG.
    :param U: A set of nodes in G.
    :return: L^\infty(U).
    """

    connector = {node: None for node in G.nodes()}
    S = copy.copy(U)

    for v in U:
        connector[v] = v

    L = list(reversed(list(nx.topological_sort(G))))
    for v in L:
        if v in U:
            continue

        for c in G.successors(v):
            if connector[c] is not None:
                if connector[v] is None:
                    connector[v] = connector[c]
                elif connector[v] != connector[c]:
                    connector[v] = v
                    S.add(v)
                    break

    return S


def split_node(
    G, node, in_suffix="_in", out_suffix="_out"
):  # from chatgpt, slightly modified
    """
    Splits a node `node` in the graph `G` into two nodes:
    `node_in` and `node_out`, with all incoming edges
    directed to `node_in`, all outgoing edges directed from
    `node_out`, and an additional edge from `node_in` to `node_out`.
    Node names are assumed to be strings.

    Parameters:
    - G: NetworkX graph
    - node: Node to split
    - in_suffix: Suffix for the `node_in` node (default: "_in")
    - out_suffix: Suffix for the `node_out` node (default: "_out")

    Returns:
    - G: The modified graph
    """

    G = copy.deepcopy(G)

    # Create new nodes
    node_in = f"{node}{in_suffix}"
    node_out = f"{node}{out_suffix}"

    # Add the new nodes
    G.add_node(node_in)
    G.add_node(node_out)

    # Redirect incoming edges to `node_in`
    for pred, _ in G.in_edges(node, data=False):
        G.add_edge(pred, node_in)

    # Redirect outgoing edges from `node_out`
    for _, succ in G.out_edges(node, data=False):
        G.add_edge(node_out, succ)

    # Add the connecting edge `node_in -> node_out`
    G.add_edge(node_in, node_out)

    # Remove the original node
    G.remove_node(node)

    return G


def max_flow_single_node(G: nx.DiGraph, U: Set, v: str) -> bool:
    """
    Checks whether v is in L^\infty(U), meaning whether or not it has a Lambda-structure to U.
    :param G: DAG
    :param U: Set of nodes in G
    :param v: Node to check
    :return: True if v is in L^\infty(U), else False
    """
    G = copy.deepcopy(G)

    # converting node names to strings in case the user didn't
    mapping = {node: str(node) for node in G.nodes()}
    G = nx.relabel_nodes(G, mapping)
    U = {str(u) for u in U}
    v = str(v)

    if "target" in G.nodes():
        raise ValueError("input nodes cannot be named 'target'")

    G.add_node("target")
    G.add_edges_from((u, "target") for u in U)

    for x in G.nodes():
        if x != "target":
            G = split_node(G, x)

    nx.set_edge_attributes(G, 1, name="capacity")

    G.edges[v + "_in", v + "_out"]["capacity"] = 2

    flow_value, _ = nx.maximum_flow(G, v + "_in", "target")

    return True if flow_value == 2 else False


def max_flow_all_nodes(G: nx.DiGraph, U: Set) -> Set:
    """
    Computes L^\infty(U).
    :param G: A DAG.
    :param U: Set of nodes in G.
    :return: L^\infty(U).
    """
    S = copy.copy(U)
    for v in G.nodes():
        if v not in U and max_flow_single_node(G, U, str(v)):
            S.add(v)
    return S
