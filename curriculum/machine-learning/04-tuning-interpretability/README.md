---
title: Tuning calibration and interpretability
slug: ml-tuning-interpretability
level: advanced
stage: machine-learning
estimated_hours: 10
prerequisites:
  - classical-model-families
learning_objectives:
  - Design bounded reproducible hyperparameter searches
  - Evaluate probability calibration and decision thresholds
  - Use interpretability methods without making causal claims
formats:
  - lesson
  - exercise
compute: cpu
status: draft
last_verified: 2026-09-03
---

# Tuning, calibration, and interpretability

Hyperparameter search is an experiment over a validation protocol. Define the search space, budget, metric, seed policy, stopping rule, and retained artifacts before running it. Searching harder can overfit validation data; reserve the final test set for one honest estimate after model choices are fixed.

Random search is a strong default when only some dimensions matter. Bayesian methods may use a limited budget efficiently but do not repair a noisy objective. Early stopping must be nested within the training process without peeking at the final test set.

## Probabilities and decisions

A discriminative model may rank cases well while producing poor probabilities. Inspect calibration curves and proper scoring rules. Select thresholds from explicit error costs and capacity constraints, then report behavior across relevant groups. Revalidate when prevalence or costs change.

## Interpretation boundaries

Global importance describes model behavior over a dataset; local explanations describe one prediction under an explainer’s assumptions. Permutation importance can be distorted by correlated features. Additive explanation methods depend on background data and feature-dependence assumptions. None proves a feature causes an outcome.

## Exercise

Write a preregistered tuning protocol capped at 30 trials. Include baseline, search distributions, folds, seeds, primary metric, tie breaker, early-stop rule, artifacts, and test-set policy. Add a threshold table with false-positive/negative costs and an interpretation report that explicitly lists correlation, leakage, and causality limitations.

## Completion criteria

- [ ] Search cost and stopping rules are bounded in advance.
- [ ] The final test set is not used for selection.
- [ ] Thresholds map to stated decision costs.
- [ ] Explanations are verified against controlled examples and avoid causal language.
