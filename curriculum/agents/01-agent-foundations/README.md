---
title: Agent workflows tools state and memory
slug: agent-foundations
level: practitioner
stage: agents
estimated_hours: 12
prerequisites:
  - foundation-model-systems
learning_objectives:
  - Distinguish deterministic workflows from autonomous agent loops
  - Define typed tool contracts state transitions and termination
  - Bound agent cost steps time and side effects
formats:
  - lesson
  - exercise
compute: cpu
status: draft
last_verified: 2026-09-03
---

# Agent workflows, tools, state, and memory

Use a deterministic workflow when steps and branches are known. Add model-directed routing only where interpreting ambiguous input creates measurable value. An agent loop observes state, proposes an action, validates it, executes through a controlled tool, records the result, and terminates on success, failure, or budget.

Tools are security boundaries. Define typed inputs/outputs, authentication context, allowed resources, idempotency, timeout, retry policy, and side effects. Validate outside the model. Separate conversation history, working state, durable memory, and authoritative records; model-generated summaries are not sources of truth.

Every run needs maximum steps, wall time, tokens/cost, tool calls, and side-effect scope. Termination must not depend only on the model saying it is finished.

## Exercise

Model a support-ticket workflow as explicit states and transitions. Implement read-only lookup before any write, require approval for account changes, attach idempotency keys, and define terminal success/failure. Compare with a free-running loop and list which uncertainties actually require model judgment.

## Completion criteria

- [ ] State and termination are externally enforced.
- [ ] Tool schemas and side effects are explicit.
- [ ] Durable memory has provenance and update policy.
- [ ] Deterministic steps remain deterministic.
