---
title: NumPy vectorization broadcasting and reliability
slug: numpy-vectorization
level: practitioner
stage: data
estimated_hours: 6
prerequisites:
  - numpy-foundations
learning_objectives:
  - Replace elementwise Python loops with readable array operations
  - Predict broadcasting compatibility before execution
  - Test numerical code for finite values tolerance and edge cases
formats:
  - lesson
  - exercise
compute: cpu
status: published
last_verified: 2026-09-03
library: NumPy
supported_versions: 2.x
---

# Vectorization, broadcasting, and numerical reliability

Vectorization expresses work as array operations implemented in optimized loops. It often improves speed and clarity, but large temporary arrays can make a vectorized expression slower or less memory-efficient.

Broadcasting compares shapes from the right. Dimensions are compatible when equal or when either is `1`.

```python
import numpy as np

x = np.arange(12, dtype=float).reshape(4, 3)
mean = x.mean(axis=0, keepdims=True)  # (1, 3)
scale = x.std(axis=0, keepdims=True)  # (1, 3)
safe_scale = np.where(scale == 0, 1.0, scale)
z = (x - mean) / safe_scale
np.testing.assert_allclose(z.mean(axis=0), 0.0, atol=1e-12)
```

`keepdims=True` records intent and preserves a shape that broadcasts across samples.

## Reliability checklist

- Specify reduction axes rather than trusting defaults.
- Handle empty inputs, zero denominators, NaN, infinity, and constant features.
- Use `np.testing.assert_allclose` for floating point; exact equality is often inappropriate.
- Seed a local `np.random.default_rng`, not mutable global randomness.
- Measure before optimizing and include allocations in the measurement.

## Exercise

Implement feature standardization for a 2D floating array. Reject non-2D and nonfinite input, preserve constant columns as zeros, return learned mean/scale, and never mutate input. Test one row, constant columns, large magnitudes, NaN, and invalid shapes. Compare against a loop implementation for correctness before timing both.

## Completion criteria

- [ ] Broadcasting is explained from concrete shapes.
- [ ] Edge-case tests pass without warnings.
- [ ] Timing includes a warm-up and verifies equal results.
- [ ] Performance claims include measured sizes and environment.
