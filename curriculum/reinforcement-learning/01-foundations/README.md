---
title: Bandits Markov decision processes and returns
slug: rl-foundations
level: foundation
stage: reinforcement-learning
estimated_hours: 12
prerequisites:
  - deep-learning-optimization
learning_objectives:
  - Define states actions rewards policies returns and value functions
  - Explain exploration exploitation and temporal credit assignment
  - Evaluate agents against random and heuristic baselines across seeds
formats:
  - lesson
  - exercise
compute: cpu
status: draft
last_verified: 2026-09-03
---

# Bandits, Markov decision processes, and returns

An agent observes state, chooses an action under a policy, receives reward, and transitions. An MDP assumes the current state contains the information needed to predict the next-state/reward distribution. Partial observability violates that assumption and may require memory or belief state.

The discounted return sums future rewards with factor `gamma`. A state value is expected return under a policy; an action value conditions on the first action. Bellman equations relate values recursively. They are expectation identities, not learning algorithms by themselves.

Bandits isolate exploration without state transitions. Greedy selection exploits current estimates; random exploration gathers information. Regret compares accumulated reward with an oracle action. Report distributions across independent seeds, not one attractive curve.

## Exercise

Implement a Bernoulli multi-armed bandit, random policy, and epsilon-greedy policy. Pre-generate reward tables so policies face comparable randomness. Plot or tabulate cumulative reward and regret across at least 30 seeds with uncertainty. Test epsilon zero/one, ties, invalid probabilities, and deterministic arms.

## Completion criteria

- [ ] Environment and policy randomness are independently controlled.
- [ ] Comparisons share problem instances.
- [ ] Results show uncertainty across seeds.
- [ ] Reward assumptions and possible proxy failures are explicit.
