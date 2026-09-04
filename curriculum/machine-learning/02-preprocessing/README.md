---
title: Leakage-safe preprocessing and feature pipelines
slug: ml-preprocessing-pipelines
level: practitioner
stage: machine-learning
estimated_hours: 8
prerequisites:
  - ml-framing-evaluation
learning_objectives:
  - Fit preprocessing only on training observations
  - Build composable pipelines for numeric categorical and missing data
  - Test feature contracts and prevent train-serving skew
formats:
  - lesson
  - exercise
compute: cpu
status: draft
last_verified: 2026-09-03
---

# Leakage-safe preprocessing and feature pipelines

A learned preprocessing step is part of the model. Means, vocabularies, imputation values, selected features, and dimensionality reductions must be fitted using training data only. Fitting before splitting lets validation information influence training even when labels are not passed explicitly.

## Fit and transform

`fit` learns state from permitted observations. `transform` applies fixed state. A safe evaluation loop splits first, fits every learned step on the training fold, applies those steps to validation data, and evaluates the complete pipeline. Cross-validation must repeat this process independently for each fold.

Numeric scaling can help distance- and gradient-based models, but does little for many tree models. Categorical encoding must define unknown-category behavior. Missingness requires a domain policy: absence may be unknown, not applicable, delayed, or informative. Add missing indicators only when they have a plausible interpretation.

## Feature contracts

Record feature names, order, dtype, units, valid ranges, availability time, missing policy, and transformation version. Use the same tested transformation artifact for training and serving. Reimplemented online logic invites train-serving skew.

## Exercise

Design a pipeline for numeric, categorical, Boolean, and timestamp fields. Draw the fit/transform boundary, identify all learned state, and write tests for unseen categories, all-missing columns, extra/missing fields, changed units, and values unavailable at prediction time. Run the pipeline inside cross-validation rather than preparing the full dataset once.

## Completion criteria

- [ ] No learned state is fitted before a split.
- [ ] Unknown and missing values have explicit policies.
- [ ] Feature order, units, and availability are tested.
- [ ] Training and serving reuse the same transformation logic.
