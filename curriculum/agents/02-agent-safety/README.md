---
title: Agent security evaluation and recovery
slug: agent-safety-evaluation
level: advanced
stage: agents
estimated_hours: 14
prerequisites:
  - agent-foundations
  - rag-grounded-evaluation
learning_objectives:
  - Enforce least privilege and human approval outside the model
  - Evaluate task success policy compliance and recovery separately
  - Defend against prompt injection and tool output manipulation
formats:
  - lesson
  - project
compute: cpu
status: draft
last_verified: 2026-09-03
---

# Agent security, evaluation, and recovery

Prompts cannot enforce authorization. Resolve identity, permissions, resource scope, and approval in trusted code at every tool call. Treat retrieved documents, web pages, emails, tool output, and other agents as untrusted data that may contain instructions.

Test task success separately from policy compliance. Include forbidden actions, ambiguous requests, malicious content, unavailable tools, timeouts, duplicate delivery, partial writes, stale state, budget exhaustion, and model refusal. Replay deterministic traces where possible; run stochastic scenarios across seeds/models and report uncertainty.

Recovery begins with idempotent tools, transaction boundaries, checkpoints, compensating actions, and visible escalation. Retries need bounded attempts and classification: retry transient failures, not invalid requests or denied permissions.

## Completion criteria

- [ ] Permission tests fail closed even when model output requests expansion.
- [ ] Human approval binds exact action, arguments, and freshness window.
- [ ] Duplicate and partial execution cannot silently corrupt state.
- [ ] Evaluation records traces, policy violations, budgets, and recovery outcomes.
