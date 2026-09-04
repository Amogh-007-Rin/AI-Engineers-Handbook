---
title: Dash layouts callbacks state validation and production dashboards
slug: dash-foundations
level: practitioner
stage: visualization
estimated_hours: 12
prerequisites:
  - plotly-foundations
learning_objectives:
  - Build declarative layouts with stable component identifiers
  - Design callback input, output, state, and error contracts
  - Separate data queries, business logic, and presentation
  - Test accessibility, caching, concurrency, and deployment behavior
formats:
  - lesson
  - project
  - assessment
compute: cpu
status: draft
last_verified: 2026-09-04
library: Dash
supported_versions: 3.x
---

# Dash foundations

Dash connects declarative component trees to server-side callbacks. Treat every
component ID and callback input/output as an API: keep identifiers stable,
validate values, and make business logic an ordinary pure function. Callback
graphs must not contain accidental cycles or ambiguous duplicate outputs.

Do not run expensive queries or model inference inside layout construction.
Bound callback duration and payload size, cache by complete input identity, and
prevent one user's state from leaking to another. Return actionable error and
empty states. Test callback functions directly, then use browser tests for
keyboard navigation, focus, color contrast, loading, and concurrent sessions.

## Completion criteria

- [ ] Layout IDs and callback dependencies are unique and tested.
- [ ] Core computations are pure and input validated.
- [ ] Cache/session boundaries and failures are explicit.
- [ ] Accessibility, latency, and deployment evidence is recorded.
