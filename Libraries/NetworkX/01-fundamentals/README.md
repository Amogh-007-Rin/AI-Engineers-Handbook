---
title: NetworkX graph construction traversal and structural features
slug: networkx-foundations
level: foundation
stage: graph-learning
estimated_hours: 10
prerequisites:
  - python-foundations
learning_objectives:
  - Model nodes, edges, attributes, direction, and multigraph semantics
  - Compute traversals and centrality without changing graph state
  - Detect disconnected and adversarial graph inputs
  - Serialize a graph with an explicit schema and stable identifiers
formats:
  - lesson
  - project
  - assessment
compute: cpu
status: draft
last_verified: 2026-09-03
library: NetworkX
supported_versions: 3.4.x
---

# NetworkX foundations

Treat a graph as a data model before treating it as an algorithm. Decide whether
edges are directed, weighted, temporal, or multi-relational; define node
identity and attribute types; and specify whether self-loops are valid. NetworkX
will accept many Python objects, so validate these invariants at the boundary.

Traversal, shortest paths, connected components, degree, and centrality have
different assumptions and costs. A directed graph may have no path where an
undirected projection does; a weighted shortest path is not a hop count. Test
empty, singleton, disconnected, duplicate, and cyclic graphs, and reject NaN or
negative weights when an algorithm requires it.

For ML, split graph data by node, edge, component, or time according to the
prediction task. Random edge splits can leak shared neighbors. Persist stable
identifiers and graph schema beside serialized data, and record algorithm,
version, and approximation choices.

## Completion criteria

- [ ] Graph invariants and identifier policy are tested.
- [ ] Algorithm assumptions and complexity are documented.
- [ ] Disconnected and malformed inputs have explicit outcomes.
- [ ] Serialization preserves structure and attributes.
