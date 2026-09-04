---
title: DuckDB analytical SQL foundations
slug: duckdb-foundations
level: foundation
stage: data
estimated_hours: 8
prerequisites:
  - data-quality-foundations
learning_objectives:
  - Query local analytical data with typed relational operations
  - Validate joins aggregation grain and null behavior
  - Inspect query plans and choose safe parameterization
formats:
  - lesson
  - project
compute: cpu
status: draft
last_verified: 2026-09-03
library: DuckDB
supported_versions: 1.x
---

# DuckDB analytical SQL foundations

DuckDB is an embedded analytical database: the process opens a database or in-memory connection and executes vectorized SQL without operating a separate server. It is strong for local OLAP, files, and reproducible analysis; it is not a replacement for a concurrent transactional service database.

Define table grain, keys, types, null semantics, and units before querying. Use parameter binding for values, never string interpolation. Validate expected join cardinality with pre-query uniqueness checks and post-query row counts. SQL `NULL` is unknown: comparisons yield unknown, and aggregates generally skip nulls.

Use `EXPLAIN`/profiling to inspect scans, filters, joins, and aggregations. Select required columns, filter early when semantics permit, and compare Parquet/CSV/database scans with measured data. Persist databases only when lifecycle and concurrency behavior are understood.

## Exercises

Load a tiny customer/transaction fixture; find orphans; create one row per customer including customers without transactions; distinguish `COUNT(*)` and `COUNT(column)`; parameterize a date boundary; and prove duplicate dimension keys would multiply facts. Inspect the plan before and after early projection/filtering.

## Completion criteria

- [ ] Grain and keys are documented and asserted.
- [ ] Inputs use explicit types and parameters.
- [ ] Join multiplication and null behavior have tests.
- [ ] Performance claims include plans and measurements.
