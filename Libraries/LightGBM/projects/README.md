# LightGBM evaluated boosting project

Run `python3 -m unittest -v test_model.py`. Compare dummy, linear, and LightGBM models on one open dataset. Bound leaf growth relative to observations, test native categorical/unknown/missing behavior, tune and early-stop inside training data, report uncertainty/calibration/slices/resources, and verify native persistence. Passing requires 80/100 with no leakage or unstable category mapping.
