---
title: Vectors matrices and linear transformations
slug: linear-algebra-foundations
level: foundation
stage: mathematics
estimated_hours: 8
prerequisites:
  - python-foundations
learning_objectives:
  - Interpret vectors and matrices as data and transformations
  - Compute and verify shapes dot products and matrix products
  - Explain how linear algebra represents an ML prediction
formats:
  - lesson
  - exercise
compute: cpu
status: draft
last_verified: 2026-09-03
---

# Vectors, matrices, and linear transformations

A vector may represent one observation, a direction, or parameters. A matrix may represent observations or a transformation. Meaning comes from the problem; shape records organization.

For data matrix `X` with `n` rows and `d` features, weights `w` produce `n` predictions:

```text
X: (n, d)  ·  w: (d,)  →  y: (n,)
```

The shared dimension must agree. Each output is the dot product between one observation and the weights: the computational heart of a linear model.

## Practice

Without NumPy, implement vector addition, dot product, transpose, and matrix-vector multiplication with Python lists. Validate dimensions and test valid, empty, and incompatible inputs. Calculate a three-observation, two-feature prediction by hand and confirm the implementation.

Explain why elementwise multiplication differs from matrix multiplication, what each axis means, how changing one weight changes predictions, and why shape checks belong at interfaces.

## Completion criteria

- [ ] Hand calculations and program outputs agree.
- [ ] Invalid shapes produce intentional errors.
- [ ] You can explain multiplication without syntax.
- [ ] You can connect the calculation to linear prediction.
