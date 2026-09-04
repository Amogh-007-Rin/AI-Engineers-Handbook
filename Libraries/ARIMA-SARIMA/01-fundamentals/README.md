---
title: ARIMA SARIMA forecasting foundations
slug: arima-sarima-foundations
level: practitioner
stage: machine-learning
estimated_hours: 14
prerequisites:
  - statsmodels-foundations
learning_objectives:
  - Explain autoregression differencing moving-average and seasonal terms
  - Diagnose stationarity residual dependence and forecast uncertainty
  - Evaluate forecasts with rolling origins and naive seasonal baselines
formats:
  - lesson
  - project
compute: cpu
status: draft
last_verified: 2026-09-03
library: ARIMA-SARIMA
supported_versions: Statsmodels 0.x
---

# ARIMA/SARIMA forecasting foundations

ARIMA combines autoregressive lags, differencing, and moving-average error terms. SARIMA adds seasonal versions with a declared period. Differencing targets changing level or seasonal structure; excessive differencing adds noise and changes interpretation. ACF/PACF plots suggest structure but do not mechanically select a model.

Establish last-value, seasonal-naive, and drift baselines before fitting. Use expanding or sliding rolling-origin backtests that preserve time order and mirror forecast horizon. Never calculate preprocessing, missing-value fills, or regressor values using observations beyond each origin.

Inspect residual mean, autocorrelation, variance, outliers, and distribution. Prediction intervals are conditional on the model and future assumptions, not guarantees. Structural breaks, interventions, changing seasonality, aggregation, and unavailable future regressors need explicit scenarios.

## Completion criteria

- [ ] Frequency, timestamp convention, horizon, and availability delay are explicit.
- [ ] Naive and seasonal-naive baselines share rolling origins with candidates.
- [ ] Orders are justified and selected inside training periods.
- [ ] Residual and interval behavior are reported across horizons.
