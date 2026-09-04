# PyTorch reproducible training project

Run `python3 -m unittest -v test_model.py`. Extend the fixture into a two-layer classifier with Dataset/DataLoader, train/validation/test isolation, tiny-batch overfit test, gradient checks, nonfinite detection, best-checkpoint restoration, multi-seed results, calibration, profiling, and CPU inference benchmark. Add a compiled comparison only after correctness. Passing requires 80/100 and no test tuning, mode error, unsafe artifact load, or unverified gradient.
