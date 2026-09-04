---
title: AI paper reproduction project
slug: paper-reproduction-project
level: advanced
stage: research
estimated_hours: 40
prerequisites:
  - research-experimentation
learning_objectives:
  - Reproduce a bounded published claim with traceable artifacts
  - Explain deviations uncertainty and threats to validity
  - Test one original ablation or extension
formats:
  - project
  - assessment
compute: free-gpu
status: draft
last_verified: 2026-09-03
---

# AI paper reproduction project

Run `python3 -m unittest -v test_statistics.py`. Select a primary paper whose central experiment fits a free-GPU budget. Register the claim, acceptance tolerance, baselines, data, metrics, runs, compute ceiling, and stopping rule before implementation.

Publish environment lock, data checksums, configurations, logs, raw per-run results, uncertainty, resource use, deviations, and a reproduction verdict. Add one ablation or extension motivated by the literature review. Passing requires 80/100 across fidelity, experimental control, reproducibility, analysis, and communication. Missing artifacts, result cherry-picking, or unlabelled protocol changes require revision.
