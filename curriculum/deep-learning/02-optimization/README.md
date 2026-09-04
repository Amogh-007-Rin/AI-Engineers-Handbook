---
title: Optimization and training diagnostics
slug: deep-learning-optimization
level: practitioner
stage: deep-learning
estimated_hours: 10
prerequisites:
  - deep-learning-autodiff
learning_objectives:
  - Implement minibatch gradient descent with controlled randomness
  - Diagnose optimization generalization and numerical failures
  - Design reproducible training and checkpoint protocols
formats:
  - lesson
  - exercise
compute: cpu
status: draft
last_verified: 2026-09-03
---

# Optimization and training diagnostics

Training searches parameters that reduce an empirical objective; generalization asks whether behavior transfers to unseen data. Keep those questions separate. Monitor training and validation loss, task metrics, learning rate, gradient/parameter norms, throughput, and numerical validity.

Gradient descent follows an aggregate direction. Minibatches add sampling noise that may help exploration but make runs variable. Momentum smooths directions; adaptive optimizers rescale parameter updates. Optimizer choice cannot rescue invalid data, a broken loss, leakage, or an architecture unable to represent the task.

## Diagnostic sequence

1. Overfit a tiny batch to test capacity and wiring.
2. Compare against a simple baseline.
3. Inspect inputs, targets, loss scale, gradients, and parameter updates.
4. Change one variable and retain experiment artifacts.
5. Reproduce across seeds before trusting small gains.

Exploding values suggest scale, initialization, learning-rate, precision, or unstable-operation problems. Flat learning may indicate detached graphs, saturated activations, incorrect labels, excessive regularization, or a near-zero learning rate.

## Exercise

Train a scalar linear model with full-batch and stochastic updates. Plot or tabulate loss and gradient norm, test an unstable learning rate, and implement early stopping that restores the best validation checkpoint. Record data split, seed, initial parameters, configuration, and environment.

## Completion criteria

- [ ] A tiny dataset can be intentionally overfit.
- [ ] Divergence produces a useful diagnostic instead of silent NaN.
- [ ] The best checkpoint is restored and independently evaluated.
- [ ] Claims are stable across declared seeds.
