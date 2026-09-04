# Orbit structural forecast project

Always run `python3 -m unittest -v test_time_contract.py`; run `test_orbit_model.py` in the Python 3.11 Orbit environment. Fit DLT/LGT on open seasonal data, compare shared naive/ARIMA/Prophet baselines over identical rolling origins, justify priors/components/estimator, and report convergence, interval calibration, horizon error, structural shifts, runtime, and regressor availability. Passing requires 80/100 and no time leakage or unexamined diagnostics.
