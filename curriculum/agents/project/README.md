---
title: Capability gated tool using agent project
slug: safe-agent-project
level: advanced
stage: agents
estimated_hours: 20
prerequisites:
  - agent-safety-evaluation
learning_objectives:
  - Build a bounded agent with typed least-privilege tools
  - Evaluate success policy compliance injection and recovery
  - Implement approval idempotency budgets and audit traces
formats:
  - project
  - assessment
compute: cpu
status: draft
last_verified: 2026-09-03
---

# Capability-gated tool-using agent project

Run `python3 -m unittest -v test_capabilities.py`. Build a support or research agent with at least two read tools and one simulated write tool. Trusted code must enforce identity, resource scope, approval, arguments, step/call/time budget, and idempotency.

Create deterministic tests for normal completion, denied resource, prompt injection in tool output, approval mismatch, timeout, duplicate write, partial failure, stale state, and budget exhaustion. Publish traces with secrets removed, threat model, success/policy/recovery metrics, cost/latency report, and runbook. Any unauthorized or unapproved side effect is an automatic revision.
