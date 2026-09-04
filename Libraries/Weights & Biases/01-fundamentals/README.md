---
title: Weights and Biases experiment tracking sweeps artifacts and reports
slug: wandb-foundations
level: practitioner
stage: ml-systems
estimated_hours: 12
prerequisites:
  - ml-production-systems
learning_objectives:
  - Log structured configs, metrics, artifacts, and system context safely
  - Make sweeps reproducible with explicit budgets and seeds
  - Compare runs on fixed splits and meaningful metric definitions
  - Publish reports without leaking secrets or sensitive records
formats:
  - lesson
  - project
  - assessment
compute: cpu
status: draft
last_verified: 2026-09-03
library: Weights & Biases
supported_versions: 0.x
---

# Weights & Biases foundations

Tracking is useful only when an experiment can be understood and repeated.
Log a normalized config, code revision, data reference, split policy, seed,
metrics with units/steps, and artifact lineage. Redact tokens, private rows,
and environment variables before logging; configure offline mode for restricted
or air-gapped training.

Sweeps are search programs, not magic. Define an objective direction, budget,
early stopping, search space, and validation protocol before launching workers.
Use a fixed held-out set and compare against a baseline. Reports should preserve
the decision context, failed runs, uncertainty, and known limitations.

## Completion criteria

- [ ] Config, code, data, seed, and metric semantics are logged.
- [ ] Secrets and sensitive data are redacted or kept offline.
- [ ] Sweep budget and objective are reproducible.
- [ ] Report includes failed runs, baseline, and uncertainty.
