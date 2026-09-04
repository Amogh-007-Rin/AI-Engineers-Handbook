---
title: From-scratch gradient and training project
slug: deep-learning-scratch-project
level: practitioner
stage: deep-learning
estimated_hours: 14
prerequisites:
  - deep-learning-optimization
learning_objectives:
  - Implement and gradient-check a transparent trainable model
  - Detect divergence and verify convergence with tests
  - Extend scalar reasoning to batched neural computations
formats:
  - project
  - assessment
compute: cpu
status: draft
last_verified: 2026-09-03
---

# From-scratch gradient and training project

Run the reference scalar model:

```bash
python3 -m unittest -v test_linear_model.py
```

Extend it to a two-layer network for a nonlinear binary task using only the standard library or NumPy. Implement stable activation/loss calculations, cached forward values, accumulated backward gradients, seeded initialization, minibatches, validation, early stopping, and checkpoint restoration.

Test gradients with centered differences, prove loss decreases on a tiny deterministic dataset, deliberately trigger divergence, and compare with a constant and linear baseline. Write an experiment report with at least three seeds and an ablation removing the hidden nonlinearity.

## Gate

Score 20 points each for math/correctness, gradient tests, training diagnostics, experimental design, and explanation/reproducibility. Passing requires 80/100 and no unverified gradient, hidden global randomness, test-set tuning, or silent nonfinite value.
