---
title: Kubernetes workloads probes resources rollout and policy
slug: kubernetes-foundations
level: practitioner
stage: ml-systems
estimated_hours: 14
prerequisites:
  - docker-foundations
learning_objectives:
  - Define workload, service, config, secret, and identity boundaries
  - Configure requests, limits, probes, graceful termination, and autoscaling
  - Roll out immutable images with observable rollback criteria
  - Apply least-privilege and network policy to inference workloads
formats:
  - lesson
  - project
  - assessment
compute: cpu
status: draft
last_verified: 2026-09-03
library: Kubernetes
supported_versions: 1.31.x
---

# Kubernetes foundations

Kubernetes reconciles declared desired state; it does not make an application
stateless, observable, secure, or correct. A deployment needs immutable image
identity, resource requests/limits, a service contract, config/secret sources,
and a service account with only required permissions. Keep secrets out of Git
and distinguish configuration from credentials.

Startup, readiness, and liveness probes answer different questions. Readiness
controls traffic; liveness restarts a stuck process; startup protects slow model
initialization. Pair them with termination grace, preStop behavior, bounded
concurrency, and a PodDisruptionBudget where availability matters. Autoscaling
on CPU alone may miss queue depth, token cost, or GPU memory.

Rollouts require observable gates: error rate, latency, saturation, model
quality, and drift. Use canary or blue/green traffic, retain the previous image
and config, and test rollback. Apply admission, network, and workload policies
before exposing an endpoint.

## Completion criteria

- [ ] Manifest has immutable image, resources, probes, identity, and policy.
- [ ] Readiness/liveness/startup semantics are tested.
- [ ] Rollout and rollback metrics are explicit.
- [ ] Secrets, network access, and disruption behavior are bounded.
