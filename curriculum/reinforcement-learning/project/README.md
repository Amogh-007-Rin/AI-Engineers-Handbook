---
title: Reproducible bandit evaluation project
slug: rl-bandit-project
level: practitioner
stage: reinforcement-learning
estimated_hours: 14
prerequisites:
  - rl-algorithms
learning_objectives:
  - Implement and evaluate exploration policies across shared problem instances
  - Report cumulative reward regret and uncertainty across seeds
formats:
  - project
  - assessment
compute: cpu
status: draft
last_verified: 2026-09-03
---

# Reproducible bandit evaluation project

Run `python3 -m unittest -v test_bandit.py`. Add random, greedy, epsilon-greedy, and upper-confidence policies. Compare them on shared pre-generated reward tables across at least 30 seeds. Report cumulative reward, regret, uncertainty, sensitivity to horizon and reward gap, and failure cases. Then define a real decision scenario and document delayed feedback, nonstationarity, safety, privacy, and proxy-reward risks.

Passing requires 80/100 across correctness, fair evaluation, statistical reporting, experiments, and risk communication. Single-seed conclusions or policies evaluated on different random problem instances require revision.
