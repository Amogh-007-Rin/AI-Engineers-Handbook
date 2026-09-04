---
title: fastai datablocks learners interpretation and export
slug: fastai-foundations
level: practitioner
stage: deep-learning
estimated_hours: 14
prerequisites:
  - pytorch-foundations
learning_objectives:
  - Build leakage-safe DataBlock pipelines and inspect batches
  - Train Learners with explicit metrics and callbacks
  - Diagnose slice errors beyond aggregate accuracy
  - Export learners and test preprocessing parity after reload
formats:
  - lesson
  - project
  - assessment
compute: cpu
status: draft
last_verified: 2026-09-03
library: FastAI
supported_versions: 2.8.x
---

# fastai foundations

fastai is a productive layer over PyTorch, but its defaults remain decisions
you own. A `DataBlock` declares item discovery, splitting, labels, transforms,
and batching. Inspect representative batches, class counts, tensor ranges, and
actual validation membership before training. Split first; fit vocabularies and
normalization on training data only.

A `Learner` combines model, loss, optimizer behavior, metrics, and callback
events. Callbacks can change data, optimization, and control flow, so document
their order and test custom callbacks in isolation. Use learning-rate finding
as a diagnostic rather than an oracle, and retain a final test set while tuning.

Interpretation begins after the headline metric: examine confusion matrices,
high-loss examples, minority slices, calibration, and likely label errors.
Export only the minimum inference artifact. Reload it in a clean process and
verify transforms, category vocabulary, output ordering, and predictions.

## Practice ladder

1. Build a tabular DataBlock and inspect its batches.
2. Prove train and validation identifiers do not overlap.
3. Fit a CPU model against a majority-class baseline.
4. Analyze the largest errors and one meaningful slice.
5. Export, reload, and compare decoded predictions.

## Completion criteria

- [ ] Split and transform boundaries are tested.
- [ ] Metrics include a baseline and minority behavior.
- [ ] Callback side effects are explained.
- [ ] Export/reload preserves vocabulary and predictions.
