---
title: Rasa intents entities dialogue policies actions and conversation safety
slug: rasa-foundations
level: practitioner
stage: agents
estimated_hours: 14
prerequisites:
  - nlp-language-systems
learning_objectives:
  - Define intent entity slot and response contracts from user goals
  - Design dialogue stories and rules with explicit fallback behavior
  - Isolate and authorize custom actions before external side effects
  - Evaluate NLU, dialogue, safety, and recovery on held-out conversations
formats:
  - lesson
  - project
  - assessment
compute: cpu
status: draft
last_verified: 2026-09-04
library: Rasa
supported_versions: 3.x
---

# Rasa foundations

A Rasa assistant combines NLU classification/entity extraction, dialogue state,
policies, domain declarations, responses, and custom actions. Start from user
goals and failure costs. Keep intent labels distinguishable, entity roles
explicit, and slots minimal; ambiguous training examples create brittle policy
behavior.

Stories demonstrate paths, rules encode invariant behavior, and fallback is a
designed recovery path rather than a generic apology. Custom actions cross a
trust boundary: validate slots, authenticate, authorize, apply idempotency, and
confirm consequential actions. Never let untrusted text choose an arbitrary
endpoint or command.

Evaluate NLU by class/entity and confusion pair, then test end-to-end dialogue,
out-of-scope inputs, interruptions, corrections, repeated actions, and handoff.
Version training data, domain, configuration, action code, and evaluation set.

## Completion criteria

- [ ] Intent/entity/slot taxonomy is coherent and held-out tested.
- [ ] Dialogue fallback, correction, and handoff paths are explicit.
- [ ] Actions validate authorization and idempotency before effects.
- [ ] NLU, dialogue, safety, and operations metrics are recorded.
