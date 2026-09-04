# W&B tracking contract

Validate recursively secret-redacted configs and bounded sweep definitions
without requiring network access. In the declared environment, the native test
runs W&B in offline mode, scopes all state directories to a disposable path,
logs a fixed metric, and attaches a model-card artifact. Run `python -W error
-m unittest -v`; extend it with fixed split metrics, artifact provenance, and a
report containing failed runs.
