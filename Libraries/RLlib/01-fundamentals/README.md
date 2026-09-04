---
title: RLlib algorithms environments scaling checkpoints and evaluation
slug: rllib-foundations
level: advanced
stage: reinforcement-learning
estimated_hours: 16
prerequisites:
  - gymnasium-foundations
  - ray-foundations
learning_objectives:
  - Configure algorithms environments rollout workers learners and resources
  - Separate training exploration from deterministic evaluation
  - Scale sampling without changing reproducibility or policy semantics
  - Checkpoint restore and evaluate under failure and environment shifts
formats:
  - lesson
  - project
  - assessment
compute: gpu-optional
status: draft
last_verified: 2026-09-04
library: RLlib
supported_versions: 2.x
---

# RLlib foundations

RLlib connects environment runners, replay/sampling, learners, policies, and
Ray resources. Start with a single-process deterministic configuration and
verify Gymnasium semantics before adding workers. Declare CPUs/GPUs per learner
and runner; oversubscription can look like algorithm instability.

Training metrics reflect exploratory trajectories. Create a separate evaluation
configuration with fixed seeds, disabled exploration where appropriate, enough
episodes, and confidence intervals. Track environment steps separately from
learner updates. Scaling changes sample ordering and nondeterminism, so compare
distributions rather than a single exact return.

Checkpoint policy weights, optimizer, counters, config, environment version, and
framework. Restore in a clean process and resume/evaluate. Inject worker failure,
bound retries, and ensure environment side effects are idempotent.

## Completion criteria

- [ ] Algorithm/environment/resource config is explicit and validated.
- [ ] Training and evaluation policies are separated.
- [ ] Scaling and seed variance are measured.
- [ ] Checkpoint restore and worker failure are tested.
