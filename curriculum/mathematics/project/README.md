---
title: Numerical linear algebra and gradient verification project
slug: mathematics-verification-project
level: foundation
stage: mathematics
estimated_hours: 12
prerequisites:
  - linear-algebra-foundations
learning_objectives:
  - Implement vectors matrices dot products and matrix-vector multiplication
  - Verify analytical derivatives with finite differences
  - Detect shape mismatch non-finite input and ill-conditioned reasoning
  - Explain numerical tolerance complexity and modeling interpretation
formats:
  - project
  - assessment
compute: cpu
status: draft
last_verified: 2026-09-04
---

# Numerical reasoning project

Implement small linear-algebra operations without NumPy, then verify the
derivative of a quadratic using central differences. Run `python3 -m unittest
-v`. Extend with matrix multiplication, norms, a least-squares example, gradient
tolerance across step sizes, and a written geometric interpretation.
