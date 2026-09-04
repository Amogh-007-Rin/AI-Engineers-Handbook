# Prophet time-safe forecasting project

Run `python3 -m unittest -v test_model.py`. Forecast an open calendar-driven series with rolling origins and shared naive baselines. Audit missing dates, zeros, frequency, timezone, changepoints, seasonality, holidays, future-regressor availability, interval coverage, horizon errors, and structural shifts. Compare with ARIMA/SARIMA or another justified model. Passing requires 80/100 and no future regressor, holiday, scaling, or model-selection leakage.
