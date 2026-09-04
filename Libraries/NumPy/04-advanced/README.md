---
title: NumPy memory performance and interoperability
slug: numpy-advanced
level: advanced
stage: data
estimated_hours: 5
prerequisites:
  - numpy-vectorization
learning_objectives:
  - Diagnose array memory ownership layout and unintended copies
  - Profile alternative implementations without changing semantics
  - Define safe array boundaries between numerical libraries
formats:
  - lesson
  - exercise
compute: cpu
status: draft
last_verified: 2026-09-03
library: NumPy
supported_versions: 2.x
---

# Memory, performance, and interoperability

An array combines a data buffer with dtype, shape, and strides. Transposes can be cheap views with non-contiguous strides; downstream native code may silently copy them into contiguous storage.

Inspect `flags`, `strides`, `base`, `nbytes`, and `np.shares_memory`. Prefer profiling the full pipeline over micro-optimizing one expression. In-place operations reduce allocation only when mutation is safe and dtype casting remains valid.

At library boundaries, document dtype, shape, device, ownership, mutability, and lifetime. Zero-copy exchange is valuable only when both sides agree about these properties.

## Exercise

Benchmark row-wise and column-wise reductions on C- and Fortran-contiguous arrays. Confirm equal results, record shapes/dtypes, repeat measurements, and explain behavior using strides. Then design a boundary function that accepts array-like input and returns a finite, C-contiguous `float32` matrix without copying when the input already satisfies the contract.

## Completion criteria

- [ ] Correctness is verified before timing.
- [ ] Conclusions reference layout and measured evidence.
- [ ] Boundary behavior includes shape, dtype, finite-value, copy, and mutation tests.
