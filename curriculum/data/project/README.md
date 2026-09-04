---
title: Dataset contract and quality audit project
slug: dataset-quality-project
level: foundation
stage: data
estimated_hours: 12
prerequisites:
  - data-quality-foundations
learning_objectives:
  - Validate schemas uniqueness missingness ranges categories and timestamps
  - Detect join multiplication identity overlap and temporal leakage risks
  - Produce a reproducible quality report and dataset card
  - Define quarantine ownership remediation and monitoring behavior
formats:
  - project
  - assessment
compute: cpu
status: draft
last_verified: 2026-09-04
---

# Dataset quality project

Audit a small event fixture with explicit field rules. Run `python3 -m unittest
-v`, then extend the checks with uniqueness, categories, timestamp ordering,
cross-field invariants, train/test identity overlap, and join cardinality. Produce
a dataset card covering source, consent/license, purpose, exclusions, versions,
quality results, limitations, privacy, retention, and owner.
