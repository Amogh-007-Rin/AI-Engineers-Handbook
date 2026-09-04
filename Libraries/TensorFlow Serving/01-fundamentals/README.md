---
title: TensorFlow Serving signatures versions batching and rollout
slug: tensorflow-serving-foundations
level: practitioner
stage: ml-systems
estimated_hours: 12
prerequisites:
  - tensorflow-foundations
learning_objectives:
  - Design stable SavedModel signatures and version directories
  - Validate REST/gRPC request shapes, dtypes, and batching behavior
  - Configure model loading, readiness, and resource limits
  - Roll out and roll back versions using measured quality and latency gates
formats:
  - lesson
  - project
  - assessment
compute: cpu
status: draft
last_verified: 2026-09-04
library: TensorFlow Serving
supported_versions: 2.20.x
---

# TensorFlow Serving foundations

Serving consumes SavedModel signatures, not a training notebook. Give every
input/output a stable name, dtype, shape, and versioned path. Validate malformed
requests and dynamic batch behavior before exposing REST or gRPC. Batching can
improve throughput while increasing tail latency; measure both and set explicit
queue, timeout, and concurrency limits.

Model loading and readiness are separate from process liveness. Keep old
versions available for rollback, verify signature compatibility, and gate
promotion on model quality, latency, error rate, resource usage, and drift.
Never log raw sensitive payloads; propagate request IDs and redact errors.

## Completion criteria

- [ ] Signature and request schema are versioned and tested.
- [ ] Batching, timeout, and resource behavior are measured.
- [ ] Readiness/load failures fail closed.
- [ ] Rollout and rollback gates are explicit.
