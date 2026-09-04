---
title: Neural architecture families and inductive biases
slug: neural-architecture-families
level: practitioner
stage: deep-learning
estimated_hours: 12
prerequisites:
  - deep-learning-optimization
learning_objectives:
  - Connect architecture operations to assumptions about data structure
  - Compare convolution recurrence attention and message passing
  - Select a minimal architecture and falsifiable baseline
formats:
  - lesson
  - exercise
compute: cpu
status: draft
last_verified: 2026-09-03
---

# Neural architecture families and inductive biases

Architecture encodes assumptions. Dense networks share little structure. Convolutions reuse local filters and encode translation-related assumptions. Recurrent networks update state across an ordered sequence. Attention creates content-dependent interactions. Graph networks exchange messages along declared relationships.

Transformers combine attention, position information, feed-forward transformations, residual paths, and normalization. Their flexibility and parallel training are powerful, but quadratic attention cost, data requirements, latency, and weak built-in domain structure can make simpler architectures preferable.

Generative families optimize different mechanisms: autoregressive models factor sequences, autoencoders learn representations through reconstruction, GANs learn through an adversarial game, and diffusion models learn to reverse corruption. Evaluation must follow the intended use; likelihood or visual appeal alone is rarely sufficient.

## Exercise

For image classification, irregular sensor sequences, molecular graphs, tabular risk scoring, and long-document retrieval, propose a simple baseline and one neural architecture. Specify input/output shapes, invariance or ordering assumptions, loss, metrics, compute constraint, and an ablation that tests the claimed architectural advantage.

## Completion criteria

- [ ] Architecture choices are tied to data structure.
- [ ] Every neural candidate has a simpler baseline.
- [ ] Shape transitions and computational bottlenecks are explicit.
- [ ] An ablation can falsify the architecture claim.
