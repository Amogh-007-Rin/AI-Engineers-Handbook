---
title: PySpark DataFrames schemas partitions joins and fault-tolerant pipelines
slug: pyspark-foundations
level: practitioner
stage: data-engineering
estimated_hours: 14
prerequisites:
  - data-quality-foundations
learning_objectives:
  - Define explicit schemas and null semantics for distributed DataFrames
  - Understand lazy plans, shuffles, joins, partitions, and skew
  - Build idempotent feature pipelines with bounded actions
  - Validate reproducibility and data-quality checks at scale
formats:
  - lesson
  - project
  - assessment
compute: cpu
status: draft
last_verified: 2026-09-04
library: PySpark
supported_versions: 4.x
---

# PySpark foundations

PySpark DataFrames are lazily evaluated logical plans. Use explicit schemas at
ingestion; inference can vary with sample order and nulls. Transformations
construct a plan, while actions trigger execution. Inspect the plan before
collecting, and never bring an unbounded dataset to the driver.

Joins and aggregations shuffle data. Choose join keys and null semantics, detect
skew, broadcast only bounded tables, and partition output deliberately. A
failed task may be retried, so sinks must be idempotent and UDF side effects
must not be relied on. Validate counts, uniqueness, ranges, and temporal
coverage before publishing features.

Record Spark/Python versions, SQL configuration, input snapshots, schema, and
output partitions. Test local mode for logic, then a representative cluster for
resource, shuffle, serialization, and failure behavior.

## Completion criteria

- [ ] Input/output schemas and null semantics are explicit.
- [ ] Plan, shuffle, partition, and driver-memory assumptions are measured.
- [ ] Quality checks and idempotent writes are tested.
- [ ] Local correctness is separated from cluster performance evidence.
