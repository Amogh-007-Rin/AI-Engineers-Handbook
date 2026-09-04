---
title: Polars expressions lazy queries and streaming foundations
slug: polars-foundations
level: practitioner
stage: data
estimated_hours: 10
prerequisites:
  - data-quality-foundations
learning_objectives:
  - Build typed transformations with expressions and lazy plans
  - Validate joins nulls schemas and aggregation grain
  - Inspect optimized plans and streaming suitability
formats:
  - lesson
  - project
compute: cpu
status: draft
last_verified: 2026-09-03
library: Polars
supported_versions: 1.x
---

# Polars expressions, lazy queries, and streaming foundations

Polars expressions describe column transformations without Python row loops. Eager APIs execute immediately; lazy frames build a logical plan that can optimize projection, predicate placement, and execution. Inspect plans and collect only at deliberate boundaries.

Schemas are contracts: choose integer widths, categorical/string behavior, date/time zones, and null semantics intentionally. Joins can multiply rows regardless of engine speed; test key uniqueness and input/output grain. `null` differs from floating NaN and both need explicit policies.

Streaming can reduce memory for supported plans but not every operation streams. Benchmark total time and peak memory on representative data, including scan and materialization. Compare with Pandas or DuckDB using equivalent types and semantics.

## Completion criteria

- [ ] Transformations use expressions and explicit schemas.
- [ ] Lazy plans show projection/filter pushdown where expected.
- [ ] Join cardinality and null/NaN behavior are tested.
- [ ] Engine claims include correctness, runtime, memory, and maintenance.
