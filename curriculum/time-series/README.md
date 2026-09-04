# Time-series learning path

Time series require evaluation and deployment contracts that respect temporal
order. This path covers time identity, leakage-safe features, forecasting
baselines, rolling-origin evaluation, uncertainty, monitoring, and retraining.

1. Study [time-series foundations](01-foundations/README.md).
2. Complete the [rolling backtest project](project/README.md).
3. Continue with the ARIMA/SARIMA, Prophet, Orbit, and Statsmodels academies.

Do not advance on random train/test results. Passing evidence must use only
information available at each prediction timestamp and compare with a naive or
seasonal baseline.
