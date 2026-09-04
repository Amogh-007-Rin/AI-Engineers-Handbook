---
title: Gymnasium environments spaces wrappers and reproducible episodes
slug: gymnasium-foundations
level: foundation
stage: reinforcement-learning
estimated_hours: 12
prerequisites:
  - rl-foundations
learning_objectives:
  - Distinguish observation/action spaces, termination, and truncation
  - Seed environments and action spaces for reproducible episodes
  - Compose wrappers without changing the declared contract
  - Validate agents against deterministic and invalid-action cases
formats:
  - lesson
  - project
  - assessment
compute: cpu
status: draft
last_verified: 2026-09-03
library: Gymnasium
supported_versions: 1.1.x
---

# Gymnasium foundations

An environment is a contract: `reset` returns an observation and info; `step`
returns observation, reward, terminated, truncated, and info. `terminated` is
the task's absorbing condition, while `truncated` is an external time or
resource cutoff. Bootstrapping across these flags incorrectly biases learning.

Declare bounded observation and action spaces, dtype, and shape. Seed the
environment and action space explicitly; seed does not make an unseeded agent
or external simulator deterministic. Wrappers should preserve or explicitly
transform spaces and info, and their order must be documented. Validate random,
boundary, and invalid actions before running expensive training.

## Completion criteria

- [ ] Termination and truncation semantics are tested separately.
- [ ] Reset/step shapes, dtypes, and bounds are explicit.
- [ ] Seeds reproduce an episode trace.
- [ ] Wrapper order and reward transformations are documented.
