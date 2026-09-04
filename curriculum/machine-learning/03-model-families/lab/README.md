# Classical model-family lab

This lab exposes the mechanics and biases of four representative methods. Run:

```bash
python3 curriculum/machine-learning/03-model-families/lab/model_families.py
python3 -m unittest discover -s curriculum/machine-learning/03-model-families/lab -v
```

Expected output contains line slope `2`, prediction `8`, neighbor class `warm`,
a zero-error stump at `1.5`, and centroids `0.5` and `9.5`. Eight tests pass.

## Investigation

Add one extreme target to the linear data, rescale nearest-neighbor features,
flip one stump label, and run k-means with several seeds. Predict which fitted
quantity changes before executing. Compare model behavior rather than declaring
one family universally best.

## Limitations

These implementations prioritize inspectability, not performance or full
statistical guarantees. The one-dimensional data omit feature interactions,
regularization, probabilistic calibration, robust solvers, vectorization, and
production persistence. Use maintained libraries after understanding the
contracts and verify their version-specific defaults.
