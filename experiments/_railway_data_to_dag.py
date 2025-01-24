from math import gamma as gamma_function

import numpy as np
import pandas as pd
from networkx import DiGraph
from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import BayesianNetwork


def dag_from_railway_data(
    path_csv: str,
    buffer: float = 3,
    k: float = 1,
) -> DiGraph:
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
    dag = DiGraph()
    dag.add_nodes_from(nodes)

    for _, row in df.iterrows():
        dag.add_edge(row["VanNode"], row["NaarNode"])

    return dag


if __name__ == "__main__":
    dag = dag_from_railway_data("./data/railway_data.csv")

    import networkx as nx
    from PIL import Image

    # Use Graphviz for visualization
    agraph = nx.nx_agraph.to_agraph(dag)
    agraph.draw("Images/railway_graph.png", prog="dot")
    Image.open("Images/railway_graph.png").show()
