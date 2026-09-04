# Optuna reproducible search project

Run `python3 -m unittest -v test_study.py`. Optimize a complete Scikit-Learn pipeline with nested or held-out validation, typed conditional parameters, fixed budget, seeded sampler, persistent SQLite storage, trial metadata, and intentional failure handling. Compare random search, pruning, and one default baseline under equal compute; evaluate the untouched test set once. Passing requires 80/100 and no test leakage, unbounded range, missing failed-trial audit, or unsupported importance claim.
