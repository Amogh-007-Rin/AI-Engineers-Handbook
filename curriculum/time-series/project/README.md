---
title: Rolling-origin forecasting project
slug: time-series-backtest-project
level: practitioner
stage: time-series
estimated_hours: 12
prerequisites:
  - time-series-foundations
learning_objectives:
  - Implement expanding-window forecast origins without temporal leakage
  - Compare a naive forecast with a candidate using horizon-aware metrics
  - Produce reproducible per-origin evidence and operational recommendations
formats:
  - project
  - assessment
compute: cpu
status: draft
last_verified: 2026-09-04
---

# Rolling-origin forecasting project

Implement and test expanding-window origins for an ordered series. The starter
uses a last-value baseline; add a seasonal baseline and one justified candidate.
Report per-origin and aggregate errors, interval coverage if applicable, failure
examples, compute cost, and a fallback/retraining policy.

Run `python3 -m unittest -v`. Passing requires disjoint temporal windows, at
least four origins, baseline comparison, no future-derived features, and a
written decision supported by the measured horizon.
