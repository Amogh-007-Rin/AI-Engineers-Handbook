---
title: XGBoost gradient boosting foundations
slug: xgboost-foundations
level: practitioner
stage: machine-learning
estimated_hours: 12
prerequisites:
  - sklearn-foundations
  - optuna-foundations
learning_objectives:
  - Explain additive trees gradients Hessians and regularization
  - Train and evaluate XGBoost without split or early-stopping leakage
  - Control missing values resources persistence and feature contracts
formats:
  - lesson
  - project
compute: cpu
status: draft
last_verified: 2026-09-03
library: XGBoost
supported_versions: 3.x
---

# XGBoost gradient boosting foundations

Gradient boosting builds an additive model: each tree reduces error under the objective’s local gradient information. XGBoost regularizes leaf weights and tree structure and uses second-order information where available. Depth, leaves, learning rate, rounds, row/column sampling, and regularization jointly control capacity; tune them as a system against a simple baseline.

Missing values follow learned default directions, which is useful only when missingness at deployment resembles training. Split before preprocessing, pass a validation set only from training data for early stopping, and evaluate the untouched test set once. Class weights alter the fitted objective; they do not replace threshold selection or calibration.

Set thread and device budgets, inspect training/validation curves, report rounds selected, and save the native model plus feature schema and library version. Feature importance is model-specific association, sensitive to correlation and split opportunity, and not causal evidence.

## Completion criteria

- [ ] Baseline, objective, split, metric, and early-stop set are explicit.
- [ ] Capacity and regularization are tuned inside training data.
- [ ] Missing, categorical, class-weight, and threshold behavior are tested.
- [ ] Native serialization reloads under a validated feature contract.
