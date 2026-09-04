---
title: Pandas auditable data quality pipeline
slug: pandas-quality-project
level: practitioner
stage: data
estimated_hours: 10
prerequisites:
  - pandas-workflows
learning_objectives:
  - Build a reproducible schema-driven tabular pipeline
  - Audit joins cleaning decisions and rejected records
formats:
  - project
  - assessment
compute: cpu
status: published
last_verified: 2026-09-03
library: Pandas
supported_versions: 2.x
---

# Project: auditable data quality pipeline

Build a package that reads small customer and transaction CSV files, validates their contracts, cleans deterministic defects, creates customer-level features, and writes data plus a machine-readable quality report.

## Requirements

- Preserve raw files and never silently discard records.
- Declare dtypes, keys, ranges, timestamp policy, and missing-value meanings.
- Validate join cardinality and report unmatched records.
- Separate I/O, validation, transformations, and reporting.
- Make repeated runs deterministic and test normal, boundary, and corrupted fixtures.
- Record input/output row counts and every rejection reason.

## Assessment

Score 20 points each for contracts/correctness, transformations/joins, tests/reproducibility, auditability, and communication/tradeoffs. A pass requires 80/100 and no silent row loss, many-to-many explosion, or leakage. Advanced credit requires a measured alternative implementation in DuckDB or Polars and an evidence-based engine decision.
