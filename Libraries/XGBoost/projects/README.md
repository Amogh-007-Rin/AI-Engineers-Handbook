# XGBoost evaluated boosting project

Run `python3 -m unittest -v test_model.py`. Replace the fixture with an open tabular dataset; compare dummy, linear, and XGBoost pipelines under identical splits. Tune within training data, use a separate early-stop fold, report uncertainty, calibration, threshold costs, slices, missing-value behavior, training time, inference latency, and native-model round trip. Passing requires 80/100 and no validation/test leakage.
