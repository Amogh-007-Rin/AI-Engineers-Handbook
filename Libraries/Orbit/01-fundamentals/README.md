---
title: Orbit Bayesian structural forecasting foundations
slug: orbit-foundations
level: advanced
stage: machine-learning
estimated_hours: 12
prerequisites:
  - prophet-foundations
learning_objectives:
  - Explain local trend seasonality regression and posterior uncertainty
  - Build rolling origin Orbit evaluations against shared baselines
  - Diagnose priors regressors convergence and structural change
formats:
  - lesson
  - project
compute: cpu
status: draft
last_verified: 2026-09-03
library: Orbit
supported_versions: 1.x on Python 3.11
---

# Orbit Bayesian structural forecasting foundations

Orbit provides Bayesian structural time-series models such as DLT and LGT. Components describe assumptions about trend, seasonality, regression, and noise; they are not automatically causal explanations. Choose estimator, seasonality, trend flexibility, priors, and regression signs from domain knowledge and backtests.

Input requires a regular date column and response. Audit gaps, duplicates, timezones, aggregation, zeros, and outliers. Regressors must exist at every prediction horizon or have their own forecast uncertainty. Use rolling origins shared with seasonal-naive, drift, ARIMA, and Prophet candidates.

Inspect posterior/optimization diagnostics, residual dependence, interval coverage/width by horizon, parameter sensitivity, runtime, and structural breaks. Approximate estimators trade posterior fidelity for speed; document the choice. Pin Stan/runtime dependencies and cache compilation outside learner artifacts.

## Completion criteria

- [ ] Structural components and priors have domain justification.
- [ ] Rolling origins and baselines match other forecasting academies.
- [ ] Future regressors and their uncertainty are available honestly.
- [ ] Diagnostics and interval behavior accompany point accuracy.
