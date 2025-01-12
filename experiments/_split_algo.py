import copy
from typing import Set

import networkx as nx
from pgmpy.models import BayesianNetwork


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


def SPLIT_on_target(bn: BayesianNetwork, target: str):
    """Apply SPLIT on (graph of bn, Pa(target))."""
    Pa_Y = bn.predecessors(target)
    digraph: nx.DiGraph = bn.to_directed()
    mGISS = SPLIT(digraph, set(Pa_Y))
    return mGISS
