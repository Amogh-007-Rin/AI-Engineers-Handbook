---
title: Computation graphs and backpropagation
slug: deep-learning-autodiff
level: foundation
stage: deep-learning
estimated_hours: 10
prerequisites:
  - linear-algebra-foundations
  - classical-ml-stage-project
learning_objectives:
  - Derive local derivatives and compose them with the chain rule
  - Implement reverse-mode differentiation for scalar graphs
  - Check analytical gradients with finite differences
formats:
  - lesson
  - exercise
compute: cpu
status: draft
last_verified: 2026-09-03
---

# Computation graphs and backpropagation

A computation graph records values produced by operations. Each operation knows a local derivative. Reverse-mode automatic differentiation starts from the output sensitivity `1` and applies the chain rule backward, accumulating contributions when a value influences the output through multiple paths.

For `loss = (w*x + b - y)^2`, let `error = w*x + b - y`. Then `d(loss)/dw = 2*error*x` and `d(loss)/db = 2*error`. Backpropagation is bookkeeping that generalizes this composition to large graphs.

## Gradient checking

Approximate a derivative with the centered difference `(f(x+h)-f(x-h))/(2h)`. Compare with relative and absolute tolerance. Very large `h` is a poor local approximation; extremely small `h` suffers floating-point cancellation. Gradient checks diagnose implementation errors, not model quality.

## Exercise

Draw the scalar graph above, derive every local derivative, implement forward/backward passes, and compare `w` and `b` gradients with finite differences across normal, zero-error, negative, and large inputs. Add a branched graph where one value contributes twice and prove gradients accumulate.

## Completion criteria

- [ ] Hand derivation, implementation, and finite differences agree.
- [ ] Shared-node contributions are accumulated rather than overwritten.
- [ ] You explain why gradients need both graph structure and forward values.
