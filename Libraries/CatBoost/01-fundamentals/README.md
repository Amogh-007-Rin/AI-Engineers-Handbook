---
title: CatBoost ordered categorical boosting foundations
slug: catboost-foundations
level: practitioner
stage: machine-learning
estimated_hours: 12
prerequisites:
  - sklearn-foundations
  - optuna-foundations
learning_objectives:
  - Explain ordered target statistics and symmetric tree tradeoffs
  - Train CatBoost with stable categorical and missing-value contracts
  - Evaluate resources persistence calibration and feature effects
formats:
  - lesson
  - project
compute: cpu
status: draft
last_verified: 2026-09-03
library: CatBoost
supported_versions: 1.x
---

# CatBoost ordered categorical boosting foundations

CatBoost can transform categorical features using ordered target statistics designed to reduce target leakage and prediction shift. This does not make arbitrary preprocessing safe: identify categorical columns consistently, preserve string/category meaning, and test unseen values. Symmetric trees offer predictable inference structure while depth, learning rate, iterations, sampling, and regularization control capacity.

Use training-owned validation for best-iteration selection, constrain threads, record random seeds, and compare against simple and one-hot baselines. Missing numeric values are supported under defined modes; missing categorical values still need a stable representation and domain policy.

Persist the native model with feature order/names and category contract. Feature effects remain associational; inspect errors, calibration, and relevant slices before relying on importance explanations.

## Completion criteria

- [ ] Categorical identity and unseen/missing behavior are tested.
- [ ] Ordered statistics are explained without claiming leakage immunity.
- [ ] Best iteration, randomness, threads, and resources are recorded.
- [ ] Native reload preserves predictions under the schema contract.
