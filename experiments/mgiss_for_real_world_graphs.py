import matplotlib.pyplot as plt
import networkx as nx
from pgmpy.utils import get_example_model

from _c4_algo import C4_on_target
from _railway_data_to_dag import dag_from_railway_data
from _utils import get_node_with_most_ancestors, round_2_or_until_nonzero

# All discrete and Gaussian models in bnlearn
MODELS = [
    "asia",
    "cancer",
    "earthquake",
    "sachs",
    "survey",
    "alarm",
    "barley",
    "child",
    "insurance",
    "mildew",
    "water",
    "hailfinder",
    "hepar2",
    "win95pts",
    "andes",
    "diabetes",
    "link",
    "pathfinder",
    "pigs",
    "munin1",
    "munin2",
    "munin3",
]


# # All discrete and Gaussian models in bnlearn
# MODELS = [
#     "sachs",
#     "child",
#     "asia",
# ]

fractions: dict[str, float] = {}
n_nodes: list[int] = []
avg_degrees: list[int] = []
n_target_ancestors: list[int] = []
n_mgiss: list[int] = []

for model_name in MODELS:
    print(f"Computing fraction for model {model_name}")
    try:
        # Load the model using bnlearn
        bn = get_example_model(model_name)
        graph = bn.to_directed()

        # Find the node with the most ancestors
        Y, Y_ancestors_count = get_node_with_most_ancestors(
            graph,
            no_single_children=True
            # no_single_children=False,
        )

        # Apply the C4 algorithm
        mgiss = C4_on_target(graph, Y)

        # Compute and store fraction
        fraction = len(mgiss) / (
            Y_ancestors_count
        )  # Note that nx.ancestors contains PROPER ancestors (no need to subract 1)
        fractions[model_name] = fraction

        print(model_name, Y, mgiss, fraction)

        avg_degree = round_2_or_until_nonzero(
            sum([graph.degree(node) for node in graph.nodes]) / len(graph.nodes)
        )

        n_nodes += [len(graph.nodes)]
        avg_degrees += [avg_degree]
        n_mgiss += [len(mgiss)]
        n_target_ancestors += [Y_ancestors_count]

    except Exception as e:
        print(f"Error processing model {model_name}: {e}")
        fractions[model_name] = 0  # Assign 0 if there's an error
        n_nodes += [0]
        avg_degrees += [0]
        n_mgiss += [0]
        n_target_ancestors += [0]

# Add railway model

model_name = "railway"
MODELS += [model_name]
graph = dag_from_railway_data("./data/railway_data.csv")

# Find the node with the most ancestors
Y, Y_ancestors_count = get_node_with_most_ancestors(graph, no_single_children=True)

# Apply the C4 algorithm
mgiss = C4_on_target(graph, Y)

# Store fraction
fraction = len(mgiss) / (
    Y_ancestors_count
)  # Note that nx.ancestors contains PROPER ancestors (no need to subract 1)

print(model_name, Y, mgiss, fraction)

avg_degree = round_2_or_until_nonzero(
    sum([graph.degree(node) for node in graph.nodes]) / len(graph.nodes)
)

fractions[model_name] = fraction
n_nodes += [len(graph.nodes)]
avg_degrees += [avg_degree]
n_mgiss += [len(mgiss)]
n_target_ancestors += [Y_ancestors_count]

# Sort according to n_nodes
# Get the indices of the sorted order of n_nodes
sorted_indices = sorted(range(len(n_nodes)), key=lambda i: n_nodes[i])
# Sort model characteristics and fractions list
sorted_characteristics = [
    (MODELS[i], n_nodes[i], avg_degrees[i]) for i in sorted_indices
]
sorted_fractions_lst = [
    fractions[characteristics[0]] for characteristics in sorted_characteristics
]

# Create bar plot
# plt.rcParams.update({"text.usetex": True, "font.family": "Helvetica"})
plt.figure(figsize=(18, 12))
bars = plt.bar(
    [
        characteristics[0] for characteristics in sorted_characteristics
    ],  # model names, sorted
    sorted_fractions_lst,
    color="darkblue",
    width=1.0,
    edgecolor="k",
    alpha=0.7,
    align="center",  # xticks in center of bars
)
plt.title("Fraction of Nodes Outputted by C4 for Each Model")
plt.xlabel("Model name (number of nodes, average degree)")
plt.ylabel("Fraction $|$mGISS$|$ / $|$An$(Y)$-$\{Y\}|$ ")

# x-ticks will be labeled with the model names and numbers of nodes
# Create list with "<model> (<n_nodes_for_this_model>)"
xtickslabels = [
    f"{model} ({n_nodes}, {avg_degree})"
    for model, n_nodes, avg_degree in sorted_characteristics
]
plt.xticks(ticks=range(len(MODELS)), labels=xtickslabels, rotation=45)

# Add values on top of bars
plt.ylim(0, 1.1)  # extra padding on top for bar labels
# Will need to sort numbers of mgiss and ancestors_Y nodes
n_target_ancestors_sorted = [n_target_ancestors[i] for i in sorted_indices]
n_mgiss_sorted = [n_mgiss[i] for i in sorted_indices]
for i, bar in enumerate(bars):
    height = bar.get_height()
    if height > 0:  # Only display labels for non-zero bars
        plt.text(  # display fraction
            bar.get_x() + bar.get_width() / 2,  # X position: center of bar
            height,  # Y position: bar height
            f"{height:.2f}",  # Format the height to 2 decimal places
            ha="center",  # Horizontal alignment
            va="bottom",  # Vertical alignment
            fontsize=10,  # Font size for the text
        )
        plt.text(  # display actual numbers
            bar.get_x() + bar.get_width() / 2,  # X position: center of bar
            height + 0.03,  # Y position: bar height + a bit
            f"({n_mgiss_sorted[i]}/{n_target_ancestors_sorted[i]})",  # Format the height to 2 decimal places
            ha="center",  # Horizontal alignment
            va="bottom",  # Vertical alignment
            fontsize=10,  # Font size for the text
            color="darkred",
        )
plt.grid(color="gray", linestyle="-", linewidth=0.5, alpha=0.7)


plt.subplots_adjust(bottom=0.2)  # Increase space at the bottom
plt.show()
