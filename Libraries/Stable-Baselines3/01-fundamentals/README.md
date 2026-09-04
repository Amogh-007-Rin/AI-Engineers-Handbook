---
title: Stable Baselines3 algorithms vector environments callbacks and evaluation
slug: stable-baselines3-foundations
level: practitioner
stage: reinforcement-learning
estimated_hours: 14
prerequisites:
  - gymnasium-foundations
learning_objectives:
  - Match on-policy and off-policy algorithms to action spaces and sample needs
  - Validate environments and vectorization semantics
  - Evaluate across seeds with callbacks and held-out conditions
  - Save load and resume models with normalization state
formats:
  - lesson
  - project
  - assessment
compute: gpu-optional
status: draft
last_verified: 2026-09-04
library: Stable-Baselines3
supported_versions: 2.x
---

# Stable-Baselines3 foundations

Stable-Baselines3 supplies reliable implementations, not automatic experimental
validity. Match algorithm to discrete/continuous action space, on/off-policy
data use, and compute budget. Run the environment checker and a random policy;
distinguish termination from truncation and wrap training/evaluation identically
except for reward/observation statistics that must not update during evaluation.

Report return distributions over multiple seeds and enough episodes, plus sample
efficiency, wall time, success rate, and safety constraints. Tune only on a
validation environment; retain shifted conditions for final robustness. Save
model, replay buffer if resuming, VecNormalize statistics, seed/config, and
environment version, then reload them together.

## Completion criteria

- [ ] Algorithm/action-space and wrapper choices are justified.
- [ ] Environment checker and termination/truncation tests pass.
- [ ] Evaluation uses multiple seeds and frozen normalization.
- [ ] Save/load/resume includes all required state.
