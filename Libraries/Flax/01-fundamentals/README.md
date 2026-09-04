---
title: Flax modules variables state and training foundations
slug: flax-foundations
level: practitioner
stage: deep-learning
estimated_hours: 14
prerequisites:
  - jax-foundations
learning_objectives:
  - Initialize and apply Flax modules with explicit variable collections
  - Build pure compiled training steps over parameters and optimizer state
  - Test random keys mutable state serialization and shape contracts
formats:
  - lesson
  - project
compute: cpu
status: draft
last_verified: 2026-09-03
library: Flax
supported_versions: 0.x
---

# Flax modules, variables, state, and training

Flax modules declare computation while `init` creates variable collections and `apply` consumes them. Parameters, batch statistics, caches, and other collections are explicit data. Mutable collections must be requested and returned deliberately; accidental omission can freeze state or make evaluation update it.

Random streams are named and supplied with split JAX keys. Shapes/dtypes determine initialization and compilation. Training steps should accept variables, optimizer state, batches, and keys, then return updated state and metrics as pytrees. Keep host logging and checkpoint I/O outside compiled functions.

Serialize state dictionaries with architecture/configuration and verify restore into a freshly initialized target structure. Test training/evaluation mode, batch-stat updates, dropout keys, shape errors, gradient finiteness, and multi-device axis assumptions.

## Completion criteria

- [ ] Variable collections and mutability are explicit.
- [ ] Every stochastic operation receives a unique named key.
- [ ] Compiled steps are pure and return all updated state.
- [ ] Checkpoint restore and train/eval behavior have tests.
