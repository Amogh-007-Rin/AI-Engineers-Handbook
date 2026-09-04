---
title: Airflow DAGs scheduling idempotency data intervals and operations
slug: airflow-foundations
level: practitioner
stage: data-engineering
estimated_hours: 14
prerequisites:
  - ml-production-systems
learning_objectives:
  - Model task dependencies with explicit data-interval semantics
  - Make retries, backfills, and external side effects idempotent
  - Separate orchestration from business logic and data processing
  - Test DAG structure, alerts, SLAs, secrets, and failure recovery
formats:
  - lesson
  - project
  - assessment
compute: cpu
status: draft
last_verified: 2026-09-04
library: Airflow
supported_versions: 3.x
---

# Airflow foundations

Airflow orchestrates finite tasks; it is not a streaming engine or a place to
hide data-processing logic. A DAG defines dependencies, schedule, data interval,
retries, concurrency, and ownership. Keep task functions independently testable
and pass small references through metadata rather than large datasets.

Every retry or backfill may repeat side effects. Partition outputs by logical
data interval, write atomically, and use idempotency keys or transactions.
Understand catchup, start date, timezone, and manual-run semantics before
enabling a schedule. Bound retries and timeouts; classify permanent failures.

Parse DAGs without network calls or secret access. Retrieve credentials from a
secret backend at task runtime. Monitor scheduling delay, duration, retry count,
data freshness, and output quality—not only task success. Practice clearing,
rerunning, backfilling, and rolling back a deployment.

## Completion criteria

- [ ] DAG dependencies and data intervals are tested.
- [ ] Tasks are idempotent across retries and backfills.
- [ ] Parsing has no external side effects or secrets.
- [ ] Alerts, freshness, recovery, and ownership are documented.
