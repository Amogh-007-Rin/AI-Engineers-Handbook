---
title: DGL graphs message passing batching sampling and evaluation
slug: dgl-foundations
level: practitioner
stage: graph-learning
estimated_hours: 14
prerequisites:
  - networkx-foundations
learning_objectives:
  - Construct homogeneous and heterogeneous graphs with typed features
  - Implement message passing while preserving node and edge alignment
  - Batch and sample neighborhoods without label leakage
  - Evaluate graph models with task-aware splits and artifact contracts
formats:
  - lesson
  - project
  - assessment
compute: gpu-optional
status: draft
last_verified: 2026-09-04
library: DGL
supported_versions: 2.x
---

# DGL foundations

DGL stores topology separately from node and edge feature tensors. Define edge
direction, node/edge types, feature dimensions, dtype, device, and identifier
mapping before training. A tensor row is meaningful only while it remains
aligned with the corresponding graph ID.

Message passing consists of message construction, aggregation, and node update.
Test these steps on a hand-computable graph before using a convolution layer.
Batching relabels graph-local nodes; neighborhood sampling changes the observed
graph and can leak labels or future edges if the split is designed afterward.

For node, edge, and graph tasks, choose component-, entity-, or time-aware splits
that match deployment. Report isolated nodes, degree slices, class imbalance,
sampling variance, memory, and latency. Persist topology schema, ID mappings,
feature transforms, model state, and DGL/PyTorch versions together.

## Completion criteria

- [ ] Topology, feature alignment, dtype, device, and ID mapping are tested.
- [ ] Message passing matches a hand-computed fixture.
- [ ] Sampling and split policy prevent neighborhood leakage.
- [ ] Artifact reload preserves graph and prediction contracts.
