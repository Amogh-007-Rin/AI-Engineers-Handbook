---
title: Leakage-safe baseline evaluation project
slug: classical-ml-stage-project
level: practitioner
stage: machine-learning
estimated_hours: 16
prerequisites:
  - ml-tuning-interpretability
learning_objectives:
  - Build a reproducible baseline-to-candidate evaluation workflow
  - Demonstrate that entity groups cannot leak across data splits
  - Communicate model evidence limitations and deployment decisions
formats:
  - project
  - assessment
compute: cpu
status: draft
last_verified: 2026-09-03
---

# Leakage-safe baseline evaluation project

The included `baselines.py` provides dependency-free reference behavior for a majority baseline, classification metrics, and grouped splitting. Run its tests from this directory:

```bash
python3 -m unittest -v test_baselines.py
```

Choose a small, openly licensed classification dataset with repeated entities or another defensible grouping. Add a rule baseline and one learned candidate using a pinned ML library.

## Deliverables

- Decision brief, dataset card, feature availability table, and leakage threat model.
- Reproducible group-aware split plus proof of zero group overlap.
- Majority, rule, and learned-model results using the same folds and metrics.
- Slice analysis, confidence intervals, calibration/threshold analysis when probabilities are used, and inspected errors.
- Model card, experiment report, tested inference interface, and next-step recommendation.

## Gate

A pass requires 80/100: framing/splits 20, correctness/tests 20, evaluation 20, reproducibility 20, communication/risk 20. Any test-set selection, group leakage, silent row loss, or unreproducible result is an automatic revision.
