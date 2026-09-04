---
title: Scikit Learn estimators pipelines and evaluation
slug: sklearn-foundations
level: practitioner
stage: machine-learning
estimated_hours: 14
prerequisites:
  - ml-preprocessing-pipelines
  - classical-model-families
learning_objectives:
  - Compose preprocessing and estimators into leakage-safe pipelines
  - Evaluate models with splitters scorers and baselines matched to deployment
  - Inspect fitted state feature names calibration and failure behavior
formats:
  - lesson
  - project
compute: cpu
status: draft
last_verified: 2026-09-03
library: Scikit-Learn
supported_versions: 1.x
---

# Scikit-Learn estimators, pipelines, and evaluation

Estimators expose parameter configuration before `fit` and learned attributes afterward. Transformers map input features; predictors produce scores, probabilities, or labels. A `Pipeline` makes preprocessing part of the estimator so every cross-validation fold learns transformations from training observations only.

Use `ColumnTransformer` for heterogeneous schemas and declare unknown-category and missing-value policies. Select splitters from deployment: grouped entities, ordered time, stratification, or nested validation. A scorer’s sign, averaging, sample weighting, and probability/label input are part of the experiment contract.

Start with `DummyClassifier` or `DummyRegressor`, then a simple model. Inspect `get_params`, fitted attributes, transformed feature names, convergence warnings, learning/validation curves, calibration, and slice errors. Persist the complete pipeline with library versions; serialized Python objects are trusted-code artifacts, not safe untrusted data.

## Completion criteria

- [ ] All learned preprocessing occurs inside validation folds.
- [ ] Split and metric match deployment and error costs.
- [ ] Baselines, feature schema, and fitted state are inspectable.
- [ ] Inference rejects schema drift and documents serialization trust.
