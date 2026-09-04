"""Deterministic graph feature and schema fixture."""

import json
import networkx as nx


def build(edges):
    graph = nx.Graph()
    for left, right, weight in edges:
        if left == right or weight <= 0:
            raise ValueError("self-loops and non-positive weights are rejected")
        graph.add_edge(str(left), str(right), weight=float(weight))
    return graph


def features(graph):
    if not isinstance(graph, nx.Graph):
        raise TypeError("graph must be a NetworkX graph")
    return {node: {"degree": graph.degree(node), "weighted_degree": graph.degree(node, weight="weight")}
            for node in sorted(graph.nodes)}


def round_trip(graph):
    payload = nx.node_link_data(graph, edges="links")
    return nx.node_link_graph(json.loads(json.dumps(payload)), edges="links")
