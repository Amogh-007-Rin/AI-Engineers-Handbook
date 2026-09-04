---
title: OpenAI SDK responses tools structured output and safety boundaries
slug: openai-sdk-foundations
level: practitioner
stage: agents
estimated_hours: 14
prerequisites:
  - agent-foundations
learning_objectives:
  - Define typed request, response, timeout, retry, and cost contracts
  - Separate model instructions, untrusted content, tools, and application state
  - Validate structured outputs and tool arguments before side effects
  - Test privacy, moderation, rate limits, and graceful provider failure
formats:
  - lesson
  - project
  - assessment
compute: external-service
status: draft
last_verified: 2026-09-04
library: OpenAI SDK
supported_versions: 1.x
---

# OpenAI SDK foundations

An SDK call is a boundary to an external, probabilistic service. Pin model and
API version where supported, set timeouts, budgets, retry policy, and request
IDs, and never place API keys in source or logs. Treat model output as untrusted
data until it satisfies a schema and policy check.

Tool use expands the attack surface: tool names and arguments must be allowlisted,
validated, authorized, and idempotent. Keep retrieved documents and user text
separate from higher-priority instructions, and record provenance without
retaining unnecessary personal data. Streaming, cancellation, rate limits, and
provider errors need explicit state transitions.

Evaluate quality, safety, cost, latency, refusal behavior, and prompt-injection
resistance on a fixed versioned dataset. Mock provider responses in unit tests;
reserve live calls for a small, budgeted integration suite with redacted traces.

## Completion criteria

- [ ] Requests have bounded timeout, budget, and retry contracts.
- [ ] Outputs and tool arguments are schema/policy validated before side effects.
- [ ] Secrets, untrusted content, and personal data are controlled.
- [ ] Offline tests and budgeted live evaluation are separated.
