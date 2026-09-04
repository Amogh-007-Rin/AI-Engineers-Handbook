---
title: Classical machine learning model families
slug: classical-model-families
level: practitioner
stage: machine-learning
estimated_hours: 14
prerequisites:
  - ml-preprocessing-pipelines
learning_objectives:
  - Explain inductive biases of major classical model families
  - Select baselines from data constraints rather than popularity
  - Diagnose underfitting overfitting calibration and threshold errors
formats:
  - lesson
  - exercise
compute: cpu
status: draft
last_verified: 2026-09-03
---

# Classical machine-learning model families

Models differ in the patterns they can represent, assumptions they encode, data they need, and operational costs they impose. Selection begins with those properties—not a leaderboard.

## Linear and generalized linear models

Linear models combine features additively. They are fast, stable baselines and often interpretable when features and correlations are handled carefully. Regularization constrains parameters: L2 shrinks smoothly; L1 can create sparse solutions. Logistic regression models log-odds and requires calibration/threshold decisions separate from fitting.

## Neighbors, kernels, and naive Bayes

Nearest-neighbor methods defer learning until prediction and depend heavily on distance, scaling, and dimension. Kernel methods represent nonlinear boundaries through similarities but can scale poorly with observations. Naive Bayes makes strong conditional-independence assumptions yet remains a valuable fast baseline, especially for sparse text.

## Trees and ensembles

Trees partition feature space with human-readable rules but are unstable and prone to overfitting. Bagging and random forests reduce variance by averaging diverse trees. Gradient boosting builds corrections sequentially and is often strong on tabular data, but requires careful validation, regularization, early stopping, and probability checks.

## Unsupervised models

Clustering finds structure under a chosen similarity and objective; labels such as “customer type” are interpretations, not discovered truth. Dimensionality reduction may support compression or visualization, but visual separation is not proof of real classes. Evaluate stability, reconstruction or neighborhood preservation, and downstream usefulness.

## Exercise

For five scenarios—sparse text, small regulated tabular data, nonlinear tabular data, high-dimensional similarity search, and customer exploration—choose a simple baseline and a stronger candidate. State assumptions, preprocessing, metric, scaling behavior, explanation needs, and a result that would make you reject the candidate.

## Completion criteria

- [ ] Every choice identifies an inductive bias and operational cost.
- [ ] Baselines are simpler than candidates.
- [ ] Rejection criteria are stated before results.
- [ ] Predictive explanation is not confused with causality.
