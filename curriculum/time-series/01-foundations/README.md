---
title: Time-series framing forecasting evaluation and operations
slug: time-series-foundations
level: practitioner
stage: time-series
estimated_hours: 18
prerequisites:
  - ml-framing-evaluation
learning_objectives:
  - Define timestamp horizon frequency availability and forecasting target contracts
  - Construct leakage-safe lag rolling calendar and external features
  - Evaluate forecasts with rolling origins baselines and calibrated intervals
  - Monitor drift freshness latency and retraining without future information
formats:
  - lesson
  - exercise
  - project
  - assessment
compute: cpu
status: draft
last_verified: 2026-09-04
---

# Time-series foundations

A forecasting problem is defined by the observation timestamp, prediction
origin, horizon, sampling frequency, target availability delay, and decision the
forecast supports. Event time and ingestion time are different. Sort and
deduplicate explicitly, localize timezones, represent missing periods, and
record whether a value was knowable at the prediction origin.

Lag and rolling features must be shifted so the current or future target cannot
enter its own prediction. Fit scaling, imputation, seasonality, and category
state inside each training window. External regressors are valid only if their
future values are genuinely known or separately forecast. Random splitting is
usually invalid because it leaks adjacent history and deployment regime.

Start with last-value, seasonal-naive, and simple trend baselines. Use expanding
or sliding rolling origins that reproduce retraining cadence and horizon.
Report MAE/RMSE or scaled errors by horizon and relevant segment; inspect bias,
turning points, holidays, missing periods, and regime changes. Evaluate interval
coverage and width, not only point accuracy.

Operations require data freshness, schema and calendar checks, error after labels
arrive, interval calibration, drift, compute cost, and a fallback forecast.
Backtests must be reproducible from a cutoff manifest, dataset hash, feature
availability table, code/environment revision, and random seeds.

## Exercises

1. Write an availability table for five features and identify future leakage.
2. Implement a shifted lag and rolling mean, testing the first valid timestamp.
3. Compare naive and seasonal-naive forecasts over at least four rolling origins.
4. Create an interval-coverage table by horizon and regime.
5. Draft freshness, fallback, retraining, and rollback rules.

## Assessment

Pass at 80/100: 20 temporal contract, 25 leakage-safe backtest, 20 baselines and
metrics, 20 uncertainty/error analysis, 15 reproducibility and operations.
Random splitting or future-derived features are automatic failures.
