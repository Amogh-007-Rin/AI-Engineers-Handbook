"""Small DGL message-passing contract."""

import dgl
import torch


def graph_and_features():
    graph = dgl.graph(([0, 1, 2], [1, 2, 0]), num_nodes=3)
    features = torch.tensor([[1.0], [2.0], [4.0]])
    graph.ndata["x"] = features
    return graph


def incoming_sum(graph):
    if "x" not in graph.ndata or graph.ndata["x"].shape[0] != graph.num_nodes():
        raise ValueError("one aligned feature row per node required")
    graph = graph.local_var()
    graph.update_all(dgl.function.copy_u("x", "m"), dgl.function.sum("m", "sum"))
    return graph.ndata["sum"]
