---
title: Vision evaluation and robustness project
slug: vision-evaluation-project
level: practitioner
stage: computer-vision
estimated_hours: 16
prerequisites:
  - vision-task-design
learning_objectives:
  - Build an auditable vision evaluation pipeline
  - Analyze geometry recognition robustness and deployment failures
formats:
  - project
  - assessment
compute: free-gpu
status: draft
last_verified: 2026-09-03
---

# Vision evaluation and robustness project

Run the CPU-only geometry tests:

```bash
python3 -m unittest -v test_vision_metrics.py
```

Choose an openly licensed small vision dataset and a pretrained or simple baseline. Publish a dataset card, group-safe split, preprocessing contract, baseline, task metrics, slice analysis, visual error gallery, robustness test, latency/resource measurement, and model card. A pass requires 80/100 across correctness, data/splits, evaluation, reproducibility, and risk/communication; leakage or transformed-label corruption requires revision.
