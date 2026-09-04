---
title: LightGBM leaf wise boosting foundations
slug: lightgbm-foundations
level: practitioner
stage: machine-learning
estimated_hours: 12
prerequisites:
  - sklearn-foundations
  - optuna-foundations
learning_objectives:
  - Explain histogram and leaf-wise tree growth tradeoffs
  - Control leaves depth bins sampling and categorical features
  - Evaluate persist and operate LightGBM reproducibly
formats:
  - lesson
  - project
compute: cpu
status: draft
last_verified: 2026-09-03
library: LightGBM
supported_versions: 4.x
---

# LightGBM leaf-wise boosting foundations

LightGBM bins continuous features and commonly grows the leaf with greatest loss reduction. This can be efficient and expressive, but unconstrained leaf-wise growth can overfit small data. Treat `num_leaves`, depth, minimum leaf data, bins, learning rate, rounds, sampling, and regularization as coupled controls.

Native categorical handling depends on integer/category representation and stable category meaning. Unknown/missing categories, ordered codes, and train-serving mappings require tests. Early stopping uses training-owned validation only. Constrain CPU threads, inspect evaluation history, and compare wall time and memory fairly against baselines.

Save a native booster and its schema/category contract. Split/gain importance is not causal and can favor features with more split opportunities; pair it with held-out permutation and error analysis.

## Completion criteria

- [ ] Leaf count and minimum-data constraints fit dataset size.
- [ ] Category codes and unknown behavior are stable.
- [ ] Evaluation and early stopping remain inside training data.
- [ ] Resource, persistence, and schema behavior are tested.
