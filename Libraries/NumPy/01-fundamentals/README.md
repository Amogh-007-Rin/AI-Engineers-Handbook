---
title: NumPy arrays shapes dtypes and indexing
slug: numpy-foundations
level: foundation
stage: data
estimated_hours: 5
prerequisites:
  - linear-algebra-foundations
learning_objectives:
  - Construct arrays with intentional shapes and dtypes
  - Predict the shape and values produced by indexing and reshaping
  - Distinguish basic-indexing views from advanced-indexing copies
formats:
  - lesson
  - exercise
compute: cpu
status: published
last_verified: 2026-09-03
library: NumPy
supported_versions: 2.x
---

# Arrays, shapes, dtypes, and indexing

An `ndarray` stores same-typed values across named-by-position axes. Its `shape` gives axis lengths, `dtype` determines representation, and `ndim` counts axes. Always attach domain meaning to axes: `(samples, features)` is more informative than “two-dimensional.”

```python
import numpy as np

x = np.array([[1.0, 2.0], [3.0, 4.0]])
assert x.shape == (2, 2)
assert x.dtype == np.float64
assert x[0].shape == (2,)
assert x[:, :1].shape == (2, 1)
```

Notice that `x[:, 0]` drops an axis while `x[:, :1]` preserves it. Many ML bugs are shape bugs hidden by arrays that happen to broadcast.

## Views and copies

Basic slicing usually returns a view sharing memory. Integer-array and Boolean indexing return copies. Test the behavior rather than relying on intuition:

```python
view = x[:, :1]
copy = x[[0, 1], [0, 1]]
assert np.shares_memory(x, view)
assert not np.shares_memory(x, copy)
```

## Exercise

Create an array representing 4 samples, 3 time steps, and 2 features. Without running code, write the shape produced by each operation: one sample, one feature preserving its axis, the last two time steps, transpose to time-major order, and flatten only the time/feature axes. Then implement and assert every prediction.

Add tests showing integer overflow with an intentionally small dtype and precision loss when converting floating-point values to integers. Explain when each conversion is acceptable.

## Completion criteria

- [ ] Every operation has an axis-level explanation and shape assertion.
- [ ] You demonstrate one view and one copy with `shares_memory`.
- [ ] Unsafe dtype conversion is detected before corrupting data.
