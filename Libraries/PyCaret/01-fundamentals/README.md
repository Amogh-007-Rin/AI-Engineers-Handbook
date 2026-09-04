---
title: PyCaret transparent low code ML foundations
slug: pycaret-foundations
level: practitioner
stage: machine-learning
estimated_hours: 10
prerequisites:
  - sklearn-foundations
  - optuna-foundations
learning_objectives:
  - Inspect and control PyCaret experiment preprocessing and validation
  - Compare models under fair metrics budgets and baselines
  - Export reproducible pipelines without hiding assumptions
formats:
  - lesson
  - project
compute: cpu
status: draft
last_verified: 2026-09-03
library: PyCaret
supported_versions: 3.x on Python 3.11
---

# PyCaret transparent low-code ML foundations

PyCaret coordinates preprocessing, validation, model comparison, tuning, interpretation, and persistence. Low code reduces orchestration, not statistical responsibility. Before `setup`, define target timing, ignored identifiers, feature types, split strategy, groups/time order, missing/category policy, and primary metric. Inspect the configured experiment rather than accepting inferred roles silently.

`compare_models` compares only the included candidates under its configured folds and budget. Retain dummy and simple baselines, avoid ranking on a test set, inspect fold distributions and runtime, and confirm probability/threshold behavior. `finalize_model` retrains on all available development data; use it only after evaluation is fixed.

Save the complete pipeline, environment, feature schema, experiment configuration, and leader table. Generated plots and dashboards are diagnostic aids, not proof. Escape the abstraction to Scikit-Learn when custom splitting, auditing, or deployment contracts cannot be made explicit.

## Completion criteria

- [ ] Inferred types and preprocessing are inspected and corrected.
- [ ] Validation mirrors deployment and test data remains untouched.
- [ ] Model comparison includes simple baselines and equal budgets.
- [ ] Saved pipeline and schema reproduce predictions in a clean environment.
