# CatBoost evaluated categorical boosting project

Run `python3 -m unittest -v test_model.py`. Add real categorical columns and compare one-hot linear/tree baselines with native CatBoost handling. Test unseen/missing categories, tune inside training data, report uncertainty, calibration, slices, best iteration, resources, and native round trip. Passing requires 80/100 with stable category semantics and no target-derived preprocessing.
