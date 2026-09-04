---
title: Ray tasks actors datasets scheduling and fault tolerance
slug: ray-foundations
level: practitioner
stage: ml-systems
estimated_hours: 14
prerequisites:
  - ml-production-systems
learning_objectives:
  - Choose tasks, actors, and data abstractions from state and lifetime needs
  - Bound resources, object-store memory, retries, and concurrency
  - Make distributed functions serializable, deterministic, and observable
  - Test partial failure, cancellation, and idempotent side effects
formats:
  - lesson
  - project
  - assessment
compute: cpu
status: draft
last_verified: 2026-09-04
library: Ray
supported_versions: 2.x
---

# Ray foundations

Ray tasks are stateless remote functions; actors own state across calls. Choose
the smallest lifetime that matches the computation and declare CPU/GPU/custom
resource needs. Object references are futures, not values; calling `get` too
early serializes a graph and can exhaust the object store. Keep closures small
and ensure arguments/results are serializable.

Retries make transient work robust but can duplicate side effects. Use
idempotency keys or transactional sinks, bound retry counts, and classify
non-retryable errors. Actor restarts change state unless checkpointed. Observe
queue depth, object-store pressure, task latency, failures, and resource
saturation before scaling.

## Completion criteria

- [ ] Resource and serialization contracts are explicit.
- [ ] Futures, retries, cancellation, and side effects are tested.
- [ ] Actor state/checkpoint behavior is documented.
- [ ] Metrics cover queueing, memory, latency, and failure.
