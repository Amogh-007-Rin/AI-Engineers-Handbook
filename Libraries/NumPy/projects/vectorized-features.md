---
title: NumPy vectorized feature engine
slug: numpy-feature-project
level: practitioner
stage: data
estimated_hours: 8
prerequisites:
  - numpy-vectorization
learning_objectives:
  - Build and test a reusable numerical feature pipeline
  - Evaluate correctness stability performance and memory behavior
formats:
  - project
  - assessment
compute: cpu
status: published
last_verified: 2026-09-03
library: NumPy
supported_versions: 2.x
---

# Project: vectorized feature engine

Build a small package that fits and applies standardization, polynomial pair interactions, and cosine similarity to 2D numeric data.

## Requirements

- Expose `fit`, `transform`, and `fit_transform` without global state.
- Validate dimensions, finite values, feature counts, and fitted state.
- Avoid mutating caller-owned arrays and document copy behavior.
- Handle constant columns and zero-norm rows deliberately.
- Compare vectorized outputs with a simple loop reference.
- Include unit tests, type hints, docstrings, and a reproducible benchmark.

## Assessment

Score 20 points each for correctness/edge cases, shape and dtype reasoning, API/tests, performance evidence, and explanation/tradeoffs. A pass requires 80/100 with no correctness or mutation defect. For advanced credit, profile memory and implement a chunked similarity path.
