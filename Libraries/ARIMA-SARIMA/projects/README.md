# ARIMA/SARIMA rolling forecast project

Run `python3 -m unittest -v test_forecast.py`. Use an open seasonal series with a declared business horizon. Implement last, seasonal-naive, and drift baselines; compare ARIMA/SARIMA candidates over shared rolling origins; select orders without final-window leakage; and report horizon-wise errors, interval coverage/width, residual diagnostics, runtime, structural breaks, and failure cases. Passing requires 80/100 and no shuffled split or future-derived feature.
