import random

import matplotlib.pyplot as plt
import networkx as nx

from closure import SPLIT, max_flow_all_nodes

num_nodes = 10
U_cardinality = 3
density_before_removal = 0.2


def random_dag(num_nodes, density_before_removal):
    # don't use these for experiments: for experiments use a more disciplined method to generate random graphs.
    G = nx.gnp_random_graph(num_nodes, density_before_removal, directed=True)
    return nx.DiGraph([(u, v) for (u, v) in G.edges() if u < v])


G = random_dag(
    num_nodes, 0.35
)  # if you want to try your own digraph object, put it here
U = set(
    random.sample(list(G.nodes()), U_cardinality)
)  # if you want to try your own U, put it
print("U:", U)
S = SPLIT(G, U)  # this is the closure
print("S:", S)
print(
    "Same as max flow:", S == max_flow_all_nodes(G, U)
)  # if this ever returns false, then at least one of my implementations is wrong. So far so good, though! :)

# from this point on, the code just draws the graph for visualization. The closure is the non-gray nodes. Within the closure, I color the nodes from U and nodes not from U in different colors.
# plt.clf()
node_colors = []
for node in G.nodes():
    if node in U:
        node_colors.append("blue")  # Color for nodes in U
    elif node in S - U:
        node_colors.append("green")  # Color for nodes in S-U
    else:
        node_colors.append("gray")  # Default color (uncolored)

# Draw the graph
plt.figure(figsize=(20, 20))
pos = nx.spring_layout(G)  # Choose a layout
nx.draw(
    G,
    pos,
    with_labels=True,  # Show node labels
    node_color=node_colors,  # Apply colors
    edge_color="black",  # Edge color
    font_weight="bold",  # Label font weight
    font_size=10,  # Label font size
)

plt.show()
