---
title: Value policy and deep reinforcement learning
slug: rl-algorithms
level: practitioner
stage: reinforcement-learning
estimated_hours: 16
prerequisites:
  - rl-foundations
learning_objectives:
  - Compare dynamic programming temporal-difference and policy-gradient methods
  - Diagnose instability off-policy error and reward exploitation
  - Design reproducible and safety-aware RL experiments
formats:
  - lesson
  - project
compute: free-gpu
status: draft
last_verified: 2026-09-03
---

# Value, policy, and deep reinforcement learning

Dynamic programming uses a known model. Monte Carlo methods learn from complete returns. Temporal-difference methods bootstrap from estimates; Q-learning is off-policy while SARSA follows the behavior policy. Function approximation generalizes across states but can become unstable when combined with bootstrapping and off-policy data.

Policy gradients optimize expected return through sampled trajectories; actor-critic methods use a learned value baseline to reduce variance. Deep RL adds replay, target networks, normalization, clipping, entropy, or trust-region ideas depending on the algorithm. Each mechanism addresses a failure mode and introduces assumptions.

Evaluate environment versions, wrappers, observation/reward scaling, time limits, seeds, training budget, and deterministic evaluation separately. Compare random and heuristic baselines. Inspect learned behavior: high reward may reveal exploitation of a flawed proxy.

## Completion criteria

- [ ] Algorithm choice is connected to on/off-policy data and action space.
- [ ] Training and evaluation environments are separated.
- [ ] Curves show seeds and uncertainty, not only smoothed means.
- [ ] Reward hacking, unsafe exploration, and deployment boundaries are reviewed.
