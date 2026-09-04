---
title: Prophet additive forecasting foundations
slug: prophet-foundations
level: practitioner
stage: machine-learning
estimated_hours: 12
prerequisites:
  - arima-sarima-foundations
learning_objectives:
  - Explain trend changepoint seasonality holiday and regressor components
  - Build time-safe Prophet backtests and baseline comparisons
  - Test frequency timezone future-regressor and uncertainty contracts
formats:
  - lesson
  - project
compute: cpu
status: draft
last_verified: 2026-09-03
library: Prophet
supported_versions: 1.x
---

# Prophet additive forecasting foundations

Prophet models trend plus seasonal, holiday, regressor, and error components. Its convenience does not remove specification decisions: growth form, changepoint flexibility, seasonal period/mode, holiday windows, priors, and future regressor availability encode assumptions.

Input uses `ds` timestamps and `y` targets. Define frequency and timezone before removing timezone information. Missing dates differ from zero demand. Extra regressors must be known or separately forecast at the prediction horizon; using realized future values leaks information.

Compare last, seasonal-naive, and drift baselines over rolling origins matching operational retraining and horizons. Diagnose errors by horizon and calendar regime. Prophet uncertainty depends on trend/observation assumptions and normally excludes upstream regressor uncertainty.

## Completion criteria

- [ ] Frequency, gaps, zeros, timezones, and forecast horizon are explicit.
- [ ] Changepoints/seasonality are tuned inside training windows.
- [ ] Every future regressor has an availability contract.
- [ ] Components are explanations of model structure, not causal effects.
