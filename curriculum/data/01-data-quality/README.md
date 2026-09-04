---
title: Data quality and reproducible analysis
slug: data-quality-foundations
level: foundation
stage: data
estimated_hours: 6
prerequisites:
  - python-foundations
learning_objectives:
  - Define and test a dataset contract before analysis
  - Identify missingness duplication leakage and invalid values
  - Produce a reproducible data quality report
formats:
  - lesson
  - project
compute: cpu
status: draft
last_verified: 2026-09-03
---

# Data quality and reproducible analysis

Models learn from data-generating and measurement processes captured in a dataset, not reality directly. Document what one row represents, fields, units, ranges, missing-value meanings, collection time, target availability, and intended use.

## Quality dimensions

- **Validity:** Types, ranges, and categories follow the contract.
- **Completeness:** Required fields and missingness are measured.
- **Uniqueness:** Entity keys and duplicate meanings are defined.
- **Consistency:** Units, encodings, timestamps, and relationships agree.
- **Timeliness:** Inputs would exist at prediction time.
- **Representativeness:** Sampling fits the intended population and use.

Leakage is information present during training that would not legitimately exist when predicting. It can create excellent test metrics and a useless deployed model.

## Project: dataset health report

Choose a small openly licensed tabular dataset. Produce a dataset card, executable schema, quality report, prediction-time leakage review, deterministic cleaning pipeline, and tests using deliberately corrupted examples. Preserve raw input and count every rejected row.

## Completion criteria

- [ ] A clean checkout reproduces the report.
- [ ] Every cleaning decision is justified and counted.
- [ ] Tests fail on known corruptions.
- [ ] Limitations and unresolved risks remain visible.

Use the [Pandas academy](../../../Libraries/Pandas/README.md) for implementation.
