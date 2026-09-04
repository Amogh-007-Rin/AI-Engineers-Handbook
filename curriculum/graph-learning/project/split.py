"""Dependency-free graph split integrity checks."""


def audit_split(edges, train_nodes, test_nodes):
    train, test = set(train_nodes), set(test_nodes)
    overlap = train & test
    if overlap:
        raise ValueError(f"node partitions overlap: {sorted(overlap)}")
    known = train | test
    unknown = sorted({node for edge in edges for node in edge} - known)
    crossing = [edge for edge in edges if (edge[0] in train) != (edge[1] in train)]
    return {"unknown_nodes": unknown, "crossing_edges": crossing}
