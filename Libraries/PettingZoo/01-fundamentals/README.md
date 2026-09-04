---
title: PettingZoo multi-agent environments AEC parallel APIs and validation
slug: pettingzoo-foundations
level: practitioner
stage: reinforcement-learning
estimated_hours: 12
prerequisites:
  - gymnasium-foundations
learning_objectives:
  - Distinguish agent-environment-cycle and parallel interaction semantics
  - Define per-agent spaces rewards termination truncation and lifecycle
  - Seed and validate multi-agent environments reproducibly
  - Evaluate cooperation competition fairness and emergent failure
formats:
  - lesson
  - project
  - assessment
compute: cpu
status: draft
last_verified: 2026-09-04
library: PettingZoo
supported_versions: 1.x
---

# PettingZoo foundations

PettingZoo standardizes multi-agent environments through sequential AEC and
simultaneous Parallel APIs. Specify when agents enter or leave, whose turn it is,
how actions are masked, and whether rewards are immediate or accumulated.
Termination is task completion; truncation is an external cutoff, per agent.

Observation/action spaces may vary by agent but must remain consistent with the
declared agent set. Dead agents still have lifecycle obligations under the AEC
contract. Seed environment and policies, test invalid/masked actions, and run
the provided API tests before training.

Multi-agent evaluation needs individual and system metrics: return, variance,
cooperation, exploitation, fairness, policy cycling, and robustness to opponent
or teammate shifts. Keep evaluation opponents fixed and versioned.

## Completion criteria

- [ ] Agent lifecycle, turns, spaces, rewards, and flags are explicit.
- [ ] Seeded episode traces and invalid actions are tested.
- [ ] API conformance tests pass.
- [ ] Evaluation covers individual, system, and fairness outcomes.
