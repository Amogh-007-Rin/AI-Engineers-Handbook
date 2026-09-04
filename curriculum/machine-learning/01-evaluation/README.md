---
title: Machine learning problem framing and evaluation
slug: ml-framing-evaluation
level: foundation
stage: machine-learning
estimated_hours: 10
prerequisites:
  - data-quality-foundations
learning_objectives:
  - Translate a product question into an evaluable ML task
  - Design leakage-safe splits baselines and metrics
  - Perform slice-based error analysis and communicate uncertainty
formats:
  - lesson
  - project
compute: cpu
status: draft
last_verified: 2026-09-03
---

# Machine-learning problem framing and evaluation

Begin with the decision being improved, not an algorithm. Define the prediction unit, target, observation time, prediction time, action, error costs, constraints, and a non-ML baseline. If useful labels cannot be obtained at the right time, model sophistication cannot repair the product design.

## Splits are simulations

A test split estimates future behavior only when it resembles deployment. Random splits can leak entities, time, locations, or near-duplicates. Choose temporal, grouped, geographic, or stratified splitting from the data-generating process and keep the final test set untouched until decisions are fixed.

## Metrics encode values

Accuracy hides class imbalance and unequal error costs. Precision asks how often positive predictions are correct; recall asks how many true positives are found. Ranking, probability, regression, and generation tasks require different metrics. Always pair aggregate metrics with uncertainty, slice results, calibration where probabilities drive decisions, and qualitative error inspection.

## Baseline ladder

Compare against chance, a constant predictor, a simple rule, and a simple learned model before complex methods. A baseline tests whether the pipeline and evaluation are meaningful and makes added complexity accountable.

## Project: evaluation protocol

Choose a small public supervised dataset. Write the decision context and failure costs; create a leakage threat model; implement constant, rule-based, and simple learned baselines; justify the split; report uncertainty and at least three meaningful slices; inspect false positives and negatives; and publish dataset/model cards using the repository templates.

## Completion criteria

- [ ] The target and inputs exist at their claimed times.
- [ ] The split reflects deployment and blocks known leakage paths.
- [ ] Metrics follow error costs and include uncertainty.
- [ ] Complexity is compared with simple baselines.
- [ ] The report identifies a concrete next experiment rather than merely ranking models.
