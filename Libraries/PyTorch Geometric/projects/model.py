"""PyG graph and batching contract."""

import torch
from torch_geometric.data import Batch, Data


def make_graph(values, edges):
    x = torch.tensor(values, dtype=torch.float32).reshape(-1, 1)
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edges must be source/destination pairs")
    if edge_index.numel() and (edge_index.min() < 0 or edge_index.max() >= len(values)):
        raise ValueError("edge index outside node range")
    return Data(x=x, edge_index=edge_index)


def batch_graphs(graphs):
    if not graphs:
        raise ValueError("at least one graph required")
    return Batch.from_data_list(graphs)
