---
title: TensorFlow tensors training pipelines and SavedModel serving
slug: tensorflow-foundations
level: practitioner
stage: deep-learning
estimated_hours: 16
prerequisites:
  - deep-learning-framework-parity
learning_objectives:
  - Build modules with explicit shape and dtype contracts
  - Implement eager and compiled training with gradient validation
  - Create deterministic input pipelines without evaluation leakage
  - Export and reload a typed SavedModel serving signature
formats:
  - lesson
  - project
  - assessment
compute: cpu
status: draft
last_verified: 2026-09-03
library: TensorFlow
supported_versions: 2.20.x
---

# TensorFlow foundations

TensorFlow moves from eager experimentation to traced graphs. Start with
tensors, broadcasting, variables, and automatic differentiation. Treat shape,
dtype, device, and batch semantics as public interfaces. A `tf.function` can
retrace when Python values or shapes change, so accept tensor arguments, add an
input signature where appropriate, and test more than one batch size.

Use `tf.GradientTape` to expose the learning algorithm before adopting a
high-level trainer. Assert the loss is finite, gradients exist for every
trainable variable, and evaluation does not mutate state. With `tf.data`, split
before fitting transforms, shuffle only training data with a seed, batch, and
prefetch. Never let validation or test records fit preprocessing state.

Deployment is another contract. Export a `tf.Module` with a named, typed
serving signature; reload it in a clean process and compare output names,
shapes, dtypes, and numeric tolerances. SavedModel does not contain the data
card, threshold policy, monitoring rules, or rollback decision—ship those too.

## Practice ladder

1. Explain broadcasting and variables versus immutable tensors.
2. Write and test a gradient step in eager mode.
3. Compile it and inspect retracing across batches.
4. Build a deterministic train/validation pipeline.
5. Export, reload, and invoke a named inference signature.

## Completion criteria

- [ ] Two batch sizes satisfy the inference contract.
- [ ] Training loss decreases and every gradient is finite.
- [ ] Test data never influences fitted preprocessing.
- [ ] SavedModel round-trip predictions agree within tolerance.
