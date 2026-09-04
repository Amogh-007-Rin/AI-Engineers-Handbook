---
title: Keras models layers training and backend portability
slug: keras-foundations
level: practitioner
stage: deep-learning
estimated_hours: 14
prerequisites:
  - deep-learning-framework-parity
learning_objectives:
  - Build functional and subclassed Keras models with explicit contracts
  - Control training evaluation callbacks serialization and randomness
  - Test backend portability and backend-specific boundaries
formats:
  - lesson
  - project
compute: cpu
status: draft
last_verified: 2026-09-03
library: Keras
supported_versions: 3.x
---

# Keras models, layers, training, and backend portability

Keras 3 provides a high-level model/layer interface across supported backends. Sequential models fit simple stacks; the Functional API represents directed acyclic graphs with inspectable inputs/outputs; subclassing handles dynamic behavior but requires more serialization and shape discipline.

`compile` binds optimizer, loss, and metrics; `fit` owns a training loop. Verify loss reduction, sample weighting, metric aggregation, train/eval behavior, callbacks, and best-checkpoint restoration rather than treating convenience as correctness. Use framework-neutral `keras.ops` inside portable layers and identify backend-specific operations explicitly.

Select the backend before importing Keras. Portability means equivalent contracts and tolerances, not bitwise equality. Save in the native `.keras` format, reload in a clean process, and test predictions plus custom objects. Record backend, precision, device, seed, data order, and versions.

## Completion criteria

- [ ] Input/output shapes and train/eval semantics are tested.
- [ ] Best validation state is restored before test evaluation.
- [ ] Native save/load preserves behavior.
- [ ] Backend-neutral and backend-specific code are clearly separated.
