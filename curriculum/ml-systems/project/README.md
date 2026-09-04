---
title: Production AI readiness project
slug: production-ai-project
level: advanced
stage: ml-systems
estimated_hours: 24
prerequisites:
  - ml-production-systems
learning_objectives:
  - Package deploy observe and recover a versioned AI service
  - Validate reliability security performance and rollback evidence
formats:
  - project
  - assessment
compute: cpu
status: draft
last_verified: 2026-09-03
---

# Production AI readiness project

Run `python3 -m unittest -v test_reliability.py`. Package an earlier model behind a typed service and container. Add schema tests, health/readiness behavior, structured logs, request IDs, latency/error/resource metrics, model/data monitoring, load test, dependency failure, canary criteria, rollback, threat model, SLO/error budget, cost estimate, runbook, and incident postmortem. Passing requires 80/100 and successful clean deployment plus rollback; secrets, unbounded inputs, or unverifiable artifacts require revision.
