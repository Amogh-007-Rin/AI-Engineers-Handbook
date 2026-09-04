---
title: Statsmodels inference diagnostics and forecasting foundations
slug: statsmodels-foundations
level: practitioner
stage: mathematics
estimated_hours: 14
prerequisites:
  - ml-framing-evaluation
learning_objectives:
  - Specify interpretable statistical models and reference categories
  - Diagnose residual dependence heteroskedasticity and influence
  - Report estimates intervals assumptions and limitations
formats:
  - lesson
  - project
compute: cpu
status: draft
last_verified: 2026-09-03
library: Statsmodels
supported_versions: 0.x
---

# Statsmodels inference, diagnostics, and forecasting foundations

Statistical models connect a data-generating assumption to estimates and uncertainty. Write the estimand, observational unit, formula, transformations, interactions, reference categories, missing policy, and dependence structure before fitting. A small p-value is not an effect size, practical importance, causal proof, or guarantee assumptions hold.

Ordinary least squares assumes a linear conditional mean and errors suitable for the chosen standard-error calculation. Inspect residual patterns, leverage/influence, heteroskedasticity, dependence, and specification sensitivity. Use robust or clustered uncertainty only when its sampling assumptions match the data.

Time-series models require ordered data, frequency, leakage-safe backtests, stationarity/seasonality reasoning, and residual autocorrelation diagnostics. Prediction intervals depend on model and innovation assumptions; structural change can invalidate them.

## Completion criteria

- [ ] Estimand and model specification precede results.
- [ ] Coefficients include units, intervals, and reference definitions.
- [ ] Diagnostics lead to sensitivity analysis, not ritual screenshots.
- [ ] Association is not described as causation without identification.
