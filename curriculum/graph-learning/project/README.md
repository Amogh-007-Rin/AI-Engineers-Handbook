---
title: Leakage-aware graph split project
slug: graph-split-project
level: practitioner
stage: graph-learning
estimated_hours: 10
prerequisites:
  - graph-learning-foundations
learning_objectives:
  - Detect node overlap and cross-split edges in graph evaluation
  - Construct component-aware partitions for inductive evaluation
  - Document transductive versus inductive assumptions explicitly
formats:
  - project
  - assessment
compute: cpu
status: draft
last_verified: 2026-09-04
---

# Leakage-aware graph split project

Validate node partitions and quantify crossing edges. Run `python3 -m unittest
-v`, then extend the fixture with connected components, temporal edges, a
feature-only baseline, and degree/cold-start slice metrics.
