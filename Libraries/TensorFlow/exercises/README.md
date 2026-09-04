# TensorFlow exercises

1. Convert the eager training loop to `tf.function` and measure retracing for
   two batch sizes and one incompatible feature shape.
2. Build a `tf.data` train/validation pipeline with deterministic shuffling and
   train-only normalization; assert identifier disjointness.
3. Add early stopping with best-state restoration and compare against a constant
   regressor on held-out data.
4. Invoke the SavedModel signature from a fresh process and validate names,
   shapes, dtypes, numeric tolerance, and malformed-request behavior.
