---
title: JAX transformations arrays state and compilation
slug: jax-foundations
level: practitioner
stage: deep-learning
estimated_hours: 16
prerequisites:
  - deep-learning-autodiff
  - deep-learning-optimization
learning_objectives:
  - Compose grad jit vmap and array programs with pure functions
  - Manage parameters state and random keys explicitly
  - Diagnose tracing recompilation donation and numerical behavior
formats:
  - lesson
  - project
compute: cpu
status: draft
last_verified: 2026-09-03
library: JAX
supported_versions: 0.x
---

# JAX transformations, arrays, state, and compilation

JAX transforms pure numerical functions. `grad` differentiates scalar-output functions, `vmap` adds mapped axes, and `jit` traces array operations into compiled programs. Python side effects, data-dependent Python control flow, dynamic shapes, and conversion of tracers to host values break that model or trigger surprising behavior.

Arrays are immutable in user code; updates use functional `.at` operations. Randomness is explicit: split keys so each stochastic operation receives a unique key and return updated state. Parameters, optimizer state, mutable model collections, and metrics should cross function boundaries explicitly.

Compilation cost must be separated from steady-state execution. Shapes, dtypes, static arguments, and control paths can create recompilations. Device execution is asynchronous, so benchmark only after blocking. Configure precision and accelerator before importing/initializing JAX and test numerical tolerances across platforms.

## Completion criteria

- [ ] Core compute is pure and state/randomness is explicit.
- [ ] Gradients are finite-difference checked on a small fixture.
- [ ] Compilation count and steady-state timing are distinguished.
- [ ] Shape/dtype changes and invalid tracer behavior have tests.
