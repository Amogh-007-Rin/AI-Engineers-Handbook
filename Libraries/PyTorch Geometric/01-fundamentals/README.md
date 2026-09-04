---
title: PyTorch Geometric data batching convolutions sampling and graph evaluation
slug: pytorch-geometric-foundations
level: practitioner
stage: graph-learning
estimated_hours: 14
prerequisites:
  - pytorch-foundations
  - networkx-foundations
learning_objectives:
  - Build Data objects with valid edge indices and aligned features
  - Understand batching offsets and graph-level assignment vectors
  - Implement and test message passing on small graphs
  - Prevent leakage in node edge and graph prediction splits
formats:
  - lesson
  - project
  - assessment
compute: gpu-optional
status: draft
last_verified: 2026-09-04
library: PyTorch Geometric
supported_versions: 2.x
---

# PyTorch Geometric foundations

`Data` couples node features, edge indices, labels, and optional edge attributes.
The edge matrix has shape `[2, E]`; IDs must be in range and every feature/label
row must align with its entity. Validate isolated nodes, self-loops, duplicate
edges, direction, dtype, and device before a layer sees the graph.

Batching concatenates graphs and offsets indices while a batch vector maps nodes
back to graph examples. Pool only with the intended assignment. Neighbor loaders
sample a changing computation graph, so seed and record the sampler and design
the train/validation split before sampling.

Evaluate degree, component, class, and temporal slices; compare a topology-free
baseline to determine whether the graph adds value. Persist data schema, ID maps,
transforms, split, model/optimizer state, and package versions.

## Completion criteria

- [ ] Edge index and feature alignment are validated.
- [ ] Batch offsets and pooling are tested on two graphs.
- [ ] Split and sampler avoid shared-entity/future leakage.
- [ ] Checkpoint and data schema reload together.
